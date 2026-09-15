# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead request/step correlation for Ascend profiling.

The trace is intentionally opt-in.  It records CPU timestamps and the request
membership of each worker step, while profiler scopes provide a join key in
``trace_view.json``.  A model forward is shared by all requests in a batch;
the emitted metadata preserves that distinction for offline analysis.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


class RequestTrace:
    """Write request/step events and matching torch profiler scopes."""

    def __init__(self, *, rank: int, process: str = "worker") -> None:
        trace_dir = os.getenv("VLLM_REQUEST_TRACE_DIR")
        self.enabled = bool(trace_dir)
        self._file = None
        self._lock = threading.Lock()
        self.rank = rank
        self.process = process
        if not self.enabled:
            return

        path = Path(trace_dir)  # type: ignore[arg-type]
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{process}_pid{os.getpid()}_rank{rank}.jsonl"
        self._file = (path / filename).open("a", encoding="utf-8", buffering=1)
        self.emit(
            "clock_sync",
            wall_time_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            pid=os.getpid(),
            rank=rank,
            process=process,
        )

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._file is None:
            return
        record = {
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "rank": self.rank,
            "process": self.process,
            **{key: _jsonable(value) for key, value in fields.items()},
        }
        with self._lock:
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    @contextmanager
    def phase(self, step_id: int | None, name: str, **fields: Any) -> Iterator[None]:
        """Emit begin/end events and a profiler scope for one worker phase."""

        if not self.enabled:
            yield
            return

        self.emit("phase_begin", step_id=step_id, phase=name, **fields)
        scope_name = f"vllm.request_trace step={step_id} phase={name}"
        try:
            # Imported lazily so disabled tracing does not import profiler code.
            from torch.profiler import record_function

            with record_function(scope_name):
                yield
        finally:
            self.emit("phase_end", step_id=step_id, phase=name, **fields)

    def close(self) -> None:
        file = self._file
        if file is not None:
            with self._lock:
                file.flush()
                file.close()
            self._file = None

    def __del__(self) -> None:
        # Best-effort flush for worker shutdown. Explicit close remains the
        # preferred lifecycle operation because interpreter teardown is partial.
        try:
            self.close()
        except Exception:
            pass


def disabled_request_trace() -> RequestTrace:
    """Return an inexpensive disabled instance for callers/tests."""

    return RequestTrace(rank=0)
