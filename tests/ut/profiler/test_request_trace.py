import asyncio
import json
import os
import pickle
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from vllm_ascend.profiler.frontend_trace import FrontendTrace
from vllm_ascend.profiler.request_trace import RequestTrace
from vllm_ascend.profiler.request_trace_hooks import (
    RequestTraceMiddleware,
    install_frontend_trace,
    install_scheduler_trace,
)


class TestRequestTrace(unittest.TestCase):
    def test_frontend_profiler_covers_renderer_thread(self):
        import torch

        frontend = FrontendTrace(Mock())
        with ThreadPoolExecutor(max_workers=1) as executor:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
            # Installing wraps module methods; restore them after the test.
            with (
                patch("transformers.TokenizersBackend.__call__"),
                patch("transformers.tokenization_utils_base.render_jinja_template"),
            ):
                frontend.install(SimpleNamespace(_executor=executor), profiler)
                # Model AsyncLLM: start/stop from an auxiliary thread, while
                # rendering happens on an existing, separate executor.
                with ThreadPoolExecutor(max_workers=1) as controller:
                    controller.submit(profiler.start).result(timeout=5)
                    token = frontend.request_id.set("profiled-request")
                    try:
                        executor.submit(frontend.wrap_sync(lambda: 7, "chat_template")).result(timeout=5)
                    finally:
                        frontend.request_id.reset(token)
                        controller.submit(profiler.stop).result(timeout=5)
            names = {event.name for event in profiler.events()}
            self.assertIn("vllm.frontend_trace request=profiled-request stage=chat_template", names)
            self.assertIn("vllm.frontend_trace request=profiled-request stage=executor_work", names)

    def test_frontend_executor_preserves_request_identity_and_values(self):
        """Interleaved requests keep their IDs after crossing worker threads."""
        trace = Mock()
        frontend = FrontendTrace(trace)
        barrier = threading.Barrier(2)

        def work(value):
            barrier.wait(timeout=5)
            with frontend.stage("tokenize"):
                return value

        with ThreadPoolExecutor(max_workers=2) as executor:
            frontend.install_executor(executor)
            futures = []
            for request_id in ("a", "b"):
                token = frontend.request_id.set(request_id)
                futures.append(executor.submit(work, request_id))
                frontend.request_id.reset(token)
            self.assertEqual([f.result(timeout=5) for f in futures], ["a", "b"])
            calls_before = trace.emit.call_count
            self.assertEqual(executor.submit(lambda: 3).result(timeout=5), 3)
            self.assertEqual(trace.emit.call_count, calls_before)
        stages = [c.kwargs for c in trace.emit.call_args_list if c.args[0] == "frontend_stage_end"]
        self.assertEqual(
            sorted((s["request_id"], s["stage"]) for s in stages),
            [("a", "executor_work"), ("a", "tokenize"), ("b", "executor_work"), ("b", "tokenize")],
        )
        self.assertTrue(all(s["thread_cpu_ns"] >= 0 and s["elapsed_ns"] >= 0 for s in stages))

    def test_frontend_worker_exception_propagates_and_closes_stage(self):
        trace = Mock()
        frontend = FrontendTrace(trace)

        def fail():
            raise ValueError("encoding failed")

        with ThreadPoolExecutor(max_workers=1) as executor:
            frontend.install_executor(executor)
            token = frontend.request_id.set("failing-request")
            future = executor.submit(frontend.wrap_sync(fail, "tokenize"))
            frontend.request_id.reset(token)
            with self.assertRaisesRegex(ValueError, "encoding failed"):
                future.result(timeout=5)
        ends = [c.kwargs["stage"] for c in trace.emit.call_args_list if c.args[0] == "frontend_stage_end"]
        self.assertEqual(ends, ["tokenize", "executor_work"])

    def test_disabled_trace_is_side_effect_free(self):
        with patch.dict(os.environ, {}, clear=True):
            trace = RequestTrace(rank=0)
            self.assertFalse(trace.enabled)
            trace.emit("ignored", request_ids=["req-0"])
            with trace.phase(1, "forward"):
                pass
            self.assertIsNone(trace._file)

    def test_enabled_trace_writes_clock_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"VLLM_REQUEST_TRACE_DIR": directory}):
            trace = RequestTrace(rank=2)
            trace.emit(
                "worker_step_begin",
                step_id=7,
                request_ids=["req-0", "req-1"],
                scheduled_tokens={"req-0": 4},
                computed_tokens={"req-0": 128},
            )
            trace.close()

            files = list(Path(directory).glob("worker_pid*_rank2.jsonl"))
            self.assertEqual(len(files), 1)
            records = [json.loads(line) for line in files[0].read_text().splitlines()]
            self.assertEqual(records[0]["event"], "clock_sync")
            self.assertEqual(records[1]["step_id"], 7)
            self.assertEqual(records[1]["request_ids"], ["req-0", "req-1"])
            self.assertEqual(records[1]["computed_tokens"]["req-0"], 128)

    def test_phase_always_emits_end_on_exception(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"VLLM_REQUEST_TRACE_DIR": directory}):
            trace = RequestTrace(rank=0)
            with (
                self.assertRaisesRegex(RuntimeError, "phase failure"),
                trace.phase(3, "forward", request_ids=["req-0"]),
            ):
                raise RuntimeError("phase failure")
            trace.close()

            path = next(Path(directory).glob("worker_pid*_rank0.jsonl"))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in records],
                ["clock_sync", "phase_begin", "phase_end"],
            )
            self.assertEqual(records[1]["phase"], "forward")
            self.assertEqual(records[2]["phase"], "forward")

    def test_scheduler_preserves_batch_and_correlates_out_of_order_outputs(self):
        """Async results must keep the key of their own scheduled batch."""
        outputs = [SimpleNamespace(num_scheduled_tokens={f"r{i}": i + 1}, scheduled_new_reqs=[]) for i in range(2)]
        scheduler = SimpleNamespace(
            add_request=Mock(return_value="added"),
            schedule=Mock(side_effect=outputs),
            update_from_output=Mock(return_value={}),
            running=[],
            waiting=[],
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"VLLM_ASCEND_REQUEST_TRACE_DIR": directory}),
        ):
            install_scheduler_trace(scheduler)
            install_scheduler_trace(scheduler)
            self.assertEqual(scheduler.add_request(SimpleNamespace(request_id="r0", num_prompt_tokens=9)), "added")
            first, second = scheduler.schedule(False), scheduler.schedule(False)
            self.assertIs(first, outputs[0])
            self.assertEqual(first.num_scheduled_tokens, {"r0": 1})
            self.assertNotEqual(first._request_trace_key, second._request_trace_key)
            first = pickle.loads(pickle.dumps(first))
            scheduler.update_from_output(second, None)
            scheduler.update_from_output(first, None)
            scheduler._scheduler_request_trace.close()
            records = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            keys = [r["scheduler_key"] for r in records if r["event"] == "scheduler_output_begin"]
            self.assertEqual(keys, [list(second._request_trace_key), list(first._request_trace_key)])

    def test_frontend_preserves_values_and_closes_render_on_cancellation(self):
        """Tracing must not swallow cancellation or change response objects."""
        trace = Mock()
        value = object()
        state = SimpleNamespace(
            openai_serving_chat=SimpleNamespace(render_chat_request=AsyncMock(side_effect=asyncio.CancelledError)),
            engine_client=SimpleNamespace(
                add_request=AsyncMock(return_value=value),
                _add_request=AsyncMock(return_value=value),
                output_processor=SimpleNamespace(process_outputs=Mock(return_value=value)),
            ),
        )
        install_frontend_trace(state, trace)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(state.openai_serving_chat.render_chat_request(SimpleNamespace(request_id="r0")))
        self.assertEqual(trace.emit.call_args_list[-1].args[0], "render_end")
        self.assertIs(asyncio.run(state.engine_client.add_request("r0")), value)
        self.assertIs(
            asyncio.run(
                state.engine_client._add_request(SimpleNamespace(request_id="r0-internal", external_req_id="r0"))
            ),
            value,
        )
        self.assertIs(state.engine_client.output_processor.process_outputs([]), value)

    def test_asgi_trace_handles_split_utf8_and_role_frames_without_changing_bytes(self):
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"id":"r0","choices":[{"delta":{"content":"\xe4',
            b'\xb8\xad"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        sent = []

        async def app(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            for chunk in chunks:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def send(message):
            sent.append(message)

        middleware = RequestTraceMiddleware(app)
        middleware.trace.close()
        middleware.trace = Mock(enabled=True)
        middleware.installed = True
        asyncio.run(
            middleware(
                {"type": "http", "path": "/v1/chat/completions", "headers": [(b"x-request-id", b"r0")]},
                AsyncMock(return_value={"type": "http.request", "body": b"{}"}),
                send,
            )
        )
        events = [call.args[0] for call in middleware.trace.emit.call_args_list]
        self.assertEqual(events.count("http_first_content_sent"), 1)
        self.assertEqual(events[-1], "http_end")
        self.assertEqual(b"".join(m.get("body", b"") for m in sent), b"".join(chunks))

    def test_disabled_scheduler_installs_no_wrappers(self):
        original = Mock()
        scheduler = SimpleNamespace(schedule=original)
        with patch.dict(os.environ, {}, clear=True):
            install_scheduler_trace(scheduler)
        self.assertIs(scheduler.schedule, original)

    def test_profiler_honors_record_shapes_configuration(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = SimpleNamespace(
            profiler="torch",
            torch_profiler_dir="/tmp/profile-test",
            torch_profiler_with_memory=False,
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=True,
        )
        with (
            patch(
                "vllm_ascend.profiler.torch_npu_profiler.get_ascend_config",
                return_value=SimpleNamespace(msmonitor_use_daemon=False),
            ),
            patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.profile") as factory,
        ):
            TorchNPUProfilerWrapper._create_profiler(config, "test")
        self.assertTrue(factory.call_args.kwargs["record_shapes"])


if __name__ == "__main__":
    unittest.main()
