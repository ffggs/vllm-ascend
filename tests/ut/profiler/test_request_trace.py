import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_ascend.profiler.request_trace import RequestTrace


class TestRequestTrace(unittest.TestCase):
    def test_disabled_trace_is_side_effect_free(self):
        with patch.dict(os.environ, {}, clear=True):
            trace = RequestTrace(rank=0)
            self.assertFalse(trace.enabled)
            trace.emit("ignored", request_ids=["req-0"])
            with trace.phase(1, "forward"):
                pass
            self.assertIsNone(trace._file)

    def test_enabled_trace_writes_clock_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"VLLM_REQUEST_TRACE_DIR": directory}
        ):
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
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"VLLM_REQUEST_TRACE_DIR": directory}
        ):
            trace = RequestTrace(rank=0)
            with self.assertRaisesRegex(RuntimeError, "phase failure"):
                with trace.phase(3, "forward", request_ids=["req-0"]):
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


if __name__ == "__main__":
    unittest.main()
