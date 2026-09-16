# SPDX-License-Identifier: Apache-2.0
"""Request-scoped frontend timings, including work in renderer threads."""

import contextvars
import itertools
import threading
import time
from contextlib import contextmanager, nullcontext
from functools import partial, wraps


class FrontendTrace:
    def __init__(self, trace):
        self.trace = trace
        self.request_id = contextvars.ContextVar("ascend_frontend_request_id", default=None)
        self.tasks = itertools.count()
        self.spans = itertools.count()
        self.parent_span = contextvars.ContextVar("ascend_frontend_parent_span", default=None)

    @contextmanager
    def stage(self, name, *, async_wait=False, **fields):
        request_id = self.request_id.get()
        if request_id is None:
            yield
            return
        from torch.profiler import record_function

        span_id = next(self.spans)
        fields = {
            "request_id": request_id,
            "stage": name,
            "tid": threading.get_native_id(),
            "span_id": span_id,
            "parent_span_id": self.parent_span.get(),
            "async_wait": async_wait,
            **fields,
        }
        parent_token = self.parent_span.set(span_id)
        self.trace.emit("frontend_stage_begin", **fields)
        begin = time.monotonic_ns()
        cpu_begin = time.thread_time_ns()
        try:
            # A coroutine can suspend while other requests run on this thread.
            # Do not put such intervals on the synchronous torch scope stack.
            scope = (
                nullcontext()
                if async_wait
                else record_function(f"vllm.frontend_trace request={request_id} stage={name}")
            )
            with scope:
                yield
        finally:
            self.trace.emit(
                "frontend_stage_end",
                elapsed_ns=time.monotonic_ns() - begin,
                thread_cpu_ns=None if async_wait else time.thread_time_ns() - cpu_begin,
                **fields,
            )
            self.parent_span.reset(parent_token)

    def wrap_sync(self, original, name):
        @wraps(original)
        def traced(*args, **kwargs):
            with self.stage(name):
                return original(*args, **kwargs)

        return traced

    def wrap_async(self, original, name):
        @wraps(original)
        async def traced(*args, **kwargs):
            with self.stage(name, async_wait=True):
                return await original(*args, **kwargs)

        return traced

    def install_method(self, owner, name, stage, *, asynchronous=False):
        original = getattr(owner, name, None)
        if original is not None:
            wrapper = self.wrap_async if asynchronous else self.wrap_sync
            setattr(owner, name, wrapper(original, stage))

    def install_output_state(self, state):
        """Observe each request through its first nonempty output, including decode."""
        if getattr(state, "_ascend_frontend_traced", False):
            return
        state._ascend_frontend_traced = True
        state._ascend_first_content_ready = False

        def install(owner, method, name, *, output=False):
            original = getattr(owner, method, None)
            if original is None:
                return

            @wraps(original)
            def traced(*args, **kwargs):
                if state._ascend_first_content_ready:
                    return original(*args, **kwargs)
                token = self.request_id.set(state.request_id)
                try:
                    with self.stage(name):
                        result = original(*args, **kwargs)
                    if output and any(getattr(o, "text", "") for o in getattr(result, "outputs", ())):
                        self.trace.emit("api_request_output_ready", request_id=state.request_id)
                        state._ascend_first_content_ready = True
                    return result
                finally:
                    self.request_id.reset(token)

            setattr(owner, method, traced)

        detokenizer = getattr(state, "detokenizer", None)
        for method, name in (
            ("update", "detokenize_update"),
            ("decode_next", "detokenize_next"),
            ("_protected_step", "decode_stream_step"),
            ("get_next_output_text", "output_text_slice"),
        ):
            install(detokenizer, method, name)
        install(getattr(state, "logprobs_processor", None), "update_from_output", "output_logprobs")
        install(state, "make_request_output", "make_request_output", output=True)
        queue = getattr(state, "queue", None)
        if queue is not None:
            original_put = queue.put
            reported = False

            @wraps(original_put)
            def put(value):
                nonlocal reported
                result = original_put(value)
                if not reported:
                    self.trace.emit("api_output_queued", request_id=state.request_id)
                    reported = True
                return result

            queue.put = put

    def install_executor(self, executor):
        original_submit = executor.submit

        @wraps(original_submit)
        def submit(fn, /, *args, **kwargs):
            request_id = self.request_id.get()
            if request_id is None:
                return original_submit(fn, *args, **kwargs)
            task_id = next(self.tasks)
            context = contextvars.copy_context()
            target = fn.func if isinstance(fn, partial) else fn
            function = getattr(target, "__qualname__", type(target).__name__)
            self.trace.emit("frontend_task_submitted", request_id=request_id, task_id=task_id, function=function)

            def run():
                def call():
                    with self.stage("executor_work", task_id=task_id, function=function):
                        return fn(*args, **kwargs)

                return context.run(call)

            return original_submit(run)

        executor.submit = submit

    def install(self, renderer, profiler=None):
        """Observe existing calls without changing the executor or tokenizer."""
        if renderer is None or not hasattr(renderer, "_executor"):
            return
        from tokenizers import Tokenizer
        from transformers import TokenizersBackend, tokenization_utils_base
        from vllm.renderers import hf
        from vllm.renderers.params import ChatParams

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
        for name in ("resolve_chat_template", "resolve_chat_template_content_format", "resolve_chat_template_kwargs"):
            self.install_method(hf, name, name)
        self.install_method(ChatParams, "get_apply_chat_template_kwargs", "template_params")
        self.install_method(TokenizersBackend, "_encode_plus", "tokenize_backend")
        self.install_method(TokenizersBackend, "_convert_encoding", "tokenize_result_conversion")
        self.install_method(Tokenizer, "encode_batch", "tokenize_native_encode_batch")
        self.install_method(hf, "parse_chat_messages_async", "parse_chat_messages", asynchronous=True)
        for name in ("_tokenize_prompt", "_build_tokens_prompt"):
            self.install_method(renderer, name, name.lstrip("_"))
        for name in ("render_messages_async", "tokenize_prompts_async", "process_for_engine_async"):
            self.install_method(renderer, name, name, asynchronous=True)
