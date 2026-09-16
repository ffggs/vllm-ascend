# SPDX-License-Identifier: Apache-2.0
"""Opt-in API and scheduler observations for request-correlated profiling.

Hooks wrap existing instance methods without replacing scheduling policy.
Only timestamps, IDs and token counts are recorded, never prompt contents.
"""

import codecs
import json
import os
from functools import wraps

from vllm_ascend.profiler.frontend_trace import FrontendTrace
from vllm_ascend.profiler.output_trace import (
    install_api_output_trace,
    install_collector_trace,
    install_engine_output_trace,
)
from vllm_ascend.profiler.request_trace import RequestTrace


def install_scheduler_trace(scheduler):
    if getattr(scheduler, "_scheduler_request_trace", None) is not None:
        return
    trace = RequestTrace(rank=0, process="scheduler")
    if not trace.enabled:
        return
    scheduler._scheduler_request_trace = trace
    install_engine_output_trace(trace)
    trace.emit("scheduler_initialized", scheduler_class=type(scheduler).__qualname__)
    cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if cache_manager is not None:
        original_lookup = cache_manager.get_computed_blocks

        @wraps(original_lookup)
        def lookup(request, *args, **kwargs):
            with trace.phase(None, "kv_cache_lookup", request_id=request.request_id):
                return original_lookup(request, *args, **kwargs)

        cache_manager.get_computed_blocks = lookup
    original_add = scheduler.add_request
    original_schedule = scheduler.schedule
    original_update = scheduler.update_from_output
    step = 0

    @wraps(original_add)
    def add_request(request, *args, **kwargs):
        trace.emit("scheduler_received", request_id=request.request_id, prompt_tokens=request.num_prompt_tokens)
        result = original_add(request, *args, **kwargs)
        trace.emit(
            "scheduler_enqueued",
            request_id=request.request_id,
            running=len(scheduler.running),
            waiting=len(scheduler.waiting),
        )
        return result

    @wraps(original_schedule)
    def schedule(*args, **kwargs):
        nonlocal step
        key = (os.getpid(), step)
        step += 1
        trace.emit(
            "schedule_begin",
            scheduler_key=key,
            waiting_ids=[r.request_id for r in scheduler.waiting],
            running_ids=[r.request_id for r in scheduler.running],
        )
        try:
            output = original_schedule(*args, **kwargs)
        except BaseException as exc:
            trace.emit("schedule_error", scheduler_key=key, error_type=type(exc).__name__)
            raise
        # SchedulerOutput is a pickled dataclass; this tag travels with the
        # exact batch, including when several async steps are in flight.
        output._request_trace_key = key
        trace.emit(
            "schedule_end",
            scheduler_key=key,
            request_ids=list(output.num_scheduled_tokens),
            scheduled_tokens=output.num_scheduled_tokens,
            cached_tokens={r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs},
        )
        return output

    @wraps(original_update)
    def update_from_output(scheduler_output, model_runner_output, *args, **kwargs):
        key = getattr(scheduler_output, "_request_trace_key", None)
        trace.emit("scheduler_output_begin", scheduler_key=key)
        outputs = original_update(scheduler_output, model_runner_output, *args, **kwargs)
        for batch in outputs.values():
            for output in batch.outputs:
                trace.emit(
                    "scheduler_output",
                    scheduler_key=key,
                    request_id=output.request_id,
                    new_tokens=len(output.new_token_ids),
                    finish_reason=output.finish_reason,
                )
        trace.emit("scheduler_output_end", scheduler_key=key)
        return outputs

    scheduler.add_request = add_request
    scheduler.schedule = schedule
    scheduler.update_from_output = update_from_output


def enable_scheduler_trace():
    # Loaded after Ascend's scheduler class patches, before engine creation.
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    if getattr(original_init, "_request_trace_installed", False):
        return

    @wraps(original_init)
    def traced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        install_scheduler_trace(self)

    traced_init._request_trace_installed = True
    Scheduler.__init__ = traced_init


def install_frontend_trace(state, trace):
    from pydantic import TypeAdapter
    from starlette.requests import Request
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionStreamResponse

    serving = state.openai_serving_chat
    engine = state.engine_client
    frontend = FrontendTrace(trace)
    install_api_output_trace(getattr(engine, "engine_core", None), trace)
    install_collector_trace(trace)
    online_renderer = getattr(serving, "online_renderer", None)
    frontend.install(getattr(online_renderer, "renderer", None), getattr(engine, "profiler", None))
    frontend.install_method(online_renderer, "preprocess_chat", "preprocess_chat", asynchronous=True)
    frontend.install_method(serving, "_check_model", "check_model", asynchronous=True)
    frontend.install_method(getattr(engine, "input_processor", None), "process_inputs", "engine_process_inputs")
    frontend.install_method(engine.output_processor, "add_request", "output_state_create")
    frontend.install_method(Request, "json", "http_json_body", asynchronous=True)
    frontend.install_method(TypeAdapter, "validate_python", "http_schema_validation")
    frontend.install_method(ChatCompletionStreamResponse, "model_dump_json", "sse_serialize")
    for name in ("build_tok_params", "build_chat_params", "to_sampling_params"):
        frontend.install_method(ChatCompletionRequest, name, name)
    original_render = serving.render_chat_request
    original_add = engine.add_request
    original_enqueue = engine._add_request
    original_process = engine.output_processor.process_outputs

    @wraps(original_render)
    async def render(request, *args, **kwargs):
        fields = {"request_id": request.request_id}
        context_token = frontend.request_id.set(request.request_id)
        trace.emit("render_begin", **fields)
        try:
            return await original_render(request, *args, **kwargs)
        finally:
            trace.emit("render_end", **fields)
            frontend.request_id.reset(context_token)

    @wraps(original_add)
    async def add_request(request_id, *args, **kwargs):
        context_token = frontend.request_id.set(request_id)
        trace.emit("engine_add_begin", request_id=request_id)
        try:
            return await original_add(request_id, *args, **kwargs)
        finally:
            trace.emit("engine_add_end", request_id=request_id)
            frontend.request_id.reset(context_token)

    @wraps(original_enqueue)
    async def enqueue(request, *args, **kwargs):
        fields = {"request_id": request.request_id, "external_request_id": request.external_req_id}
        trace.emit("engine_enqueue_begin", **fields)
        try:
            return await original_enqueue(request, *args, **kwargs)
        finally:
            trace.emit("engine_enqueue_end", **fields)

    @wraps(original_process)
    def process_outputs(engine_core_outputs, *args, **kwargs):
        for output in engine_core_outputs:
            request_state = getattr(engine.output_processor, "request_states", {}).get(output.request_id)
            if request_state is not None:
                frontend.install_output_state(request_state)
            trace.emit(
                "api_output_received",
                request_id=output.request_id,
                new_tokens=len(output.new_token_ids),
                finish_reason=output.finish_reason,
            )
        # The long-lived output task can inherit the first request's context.
        # A batch must not be attributed to that request; inner wrappers bind
        # the ID of the actual RequestState instead.
        context_token = frontend.request_id.set(None)
        try:
            return original_process(engine_core_outputs, *args, **kwargs)
        finally:
            trace.emit("api_output_processed", request_ids=[o.request_id for o in engine_core_outputs])
            frontend.request_id.reset(context_token)

    original_stats = getattr(engine.output_processor, "_update_stats_from_output", None)
    if original_stats is not None:

        @wraps(original_stats)
        def update_stats(request_state, *args, **kwargs):
            if getattr(request_state, "_ascend_first_content_ready", False):
                return original_stats(request_state, *args, **kwargs)
            token = frontend.request_id.set(request_state.request_id)
            try:
                with frontend.stage("output_stats"):
                    return original_stats(request_state, *args, **kwargs)
            finally:
                frontend.request_id.reset(token)

        engine.output_processor._update_stats_from_output = update_stats

    serving.render_chat_request = render
    engine.add_request = add_request
    engine._add_request = enqueue
    engine.output_processor.process_outputs = process_outputs
    return frontend


class RequestTraceMiddleware:
    """ASGI middleware, enabled explicitly with --middleware.

    The benchmark must send its ID in both request_id and X-Request-Id.
    Records first nonempty content separately from HTTP headers/role frames.
    """

    def __init__(self, app):
        self.app = app
        self.trace = RequestTrace(rank=0, process="api")
        self.installed = False
        self.frontend = None

    async def __call__(self, scope, receive, send):
        if not self.trace.enabled or scope["type"] != "http" or scope.get("path") != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        request_id = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode()
        trace = self.trace
        trace.emit("http_received", request_id=request_id)
        if not self.installed:
            self.frontend = install_frontend_trace(scope["app"].state, trace)
            self.installed = True
        first_content = False
        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")()

        async def traced_receive():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                trace.emit("http_body_received", request_id=request_id)
            return message

        async def traced_send(message):
            nonlocal first_content, buffer
            observing = message["type"] == "http.response.body" and not first_content
            if observing:
                trace.emit("http_body_send_begin", request_id=request_id, bytes=len(message.get("body", b"")))
            await send(message)
            if observing:
                trace.emit("http_body_send_end", request_id=request_id)
            if message["type"] == "http.response.start":
                trace.emit("http_headers_sent", request_id=request_id, status=message["status"])
            if message["type"] == "http.response.body" and not first_content:
                buffer += decoder.decode(message.get("body", b""))
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                    except (ValueError, TypeError):
                        continue
                    if any(c.get("delta", {}).get("content") for c in data.get("choices", [])):
                        first_content = True
                        trace.emit("http_first_content_sent", request_id=request_id, response_id=data.get("id"))
                        if self.frontend is not None:
                            self.frontend.request_id.set(None)
                        buffer = ""
                        break

        try:
            context_token = self.frontend.request_id.set(request_id) if self.frontend else None
            await self.app(scope, traced_receive, traced_send)
        except BaseException as exc:
            trace.emit("http_error", request_id=request_id, error_type=type(exc).__name__)
            raise
        finally:
            trace.emit("http_end", request_id=request_id, had_content=first_content)
            if self.frontend is not None and context_token is not None:
                self.frontend.request_id.reset(context_token)
