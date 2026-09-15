# SPDX-License-Identifier: Apache-2.0
"""Request-scoped frontend timings, including work in renderer threads."""

import contextvars
import itertools
import threading
import time
from contextlib import contextmanager
from functools import wraps


class FrontendTrace:
    def __init__(self, trace):
        self.trace = trace
        self.request_id = contextvars.ContextVar("ascend_frontend_request_id", default=None)
        self.tasks = itertools.count()

    @contextmanager
    def stage(self, name, **fields):
        request_id = self.request_id.get()
        if request_id is None:
            yield
            return
        from torch.profiler import record_function

        fields = {"request_id": request_id, "stage": name, "tid": threading.get_native_id(), **fields}
        self.trace.emit("frontend_stage_begin", **fields)
        begin = time.monotonic_ns()
        cpu_begin = time.thread_time_ns()
        try:
            with record_function(f"vllm.frontend_trace request={request_id} stage={name}"):
                yield
        finally:
            self.trace.emit(
                "frontend_stage_end",
                elapsed_ns=time.monotonic_ns() - begin,
                thread_cpu_ns=time.thread_time_ns() - cpu_begin,
                **fields,
            )

    def wrap_sync(self, original, name):
        @wraps(original)
        def traced(*args, **kwargs):
            with self.stage(name):
                return original(*args, **kwargs)

        return traced

    def install_executor(self, executor):
        original_submit = executor.submit

        @wraps(original_submit)
        def submit(fn, /, *args, **kwargs):
            request_id = self.request_id.get()
            if request_id is None:
                return original_submit(fn, *args, **kwargs)
            task_id = next(self.tasks)
            context = contextvars.copy_context()
            self.trace.emit("frontend_task_submitted", request_id=request_id, task_id=task_id)

            def run():
                def call():
                    with self.stage("executor_work", task_id=task_id):
                        return fn(*args, **kwargs)

                return context.run(call)

            return original_submit(run)

        executor.submit = submit

    def install(self, renderer, profiler=None):
        """Observe existing calls without changing the executor or tokenizer."""
        if renderer is None or not hasattr(renderer, "_executor"):
            return
        from transformers import TokenizersBackend, tokenization_utils_base

        if profiler is not None and profiler.experimental_config is None:
            from torch.profiler import _ExperimentalConfig

            # AsyncLLM starts its CPU profiler in asyncio.to_thread. Include
            # renderer workers and the event loop, not only that start thread.
            profiler.experimental_config = _ExperimentalConfig(profile_all_threads=True)
        self.install_executor(renderer._executor)
        tokenization_utils_base.render_jinja_template = self.wrap_sync(
            tokenization_utils_base.render_jinja_template, "chat_template"
        )
        TokenizersBackend.__call__ = self.wrap_sync(TokenizersBackend.__call__, "tokenize")
