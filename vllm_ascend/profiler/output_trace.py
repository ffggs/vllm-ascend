# SPDX-License-Identifier: Apache-2.0
"""Opt-in timestamps for output materialization and engine/API handoff."""

import contextvars
import time
from functools import wraps


def output_fields(outputs):
    return {
        "request_ids": [item.request_id for item in getattr(outputs, "outputs", ())],
        "output_timestamp": getattr(outputs, "timestamp", None),
    }


def trace_async_output(output, trace, step_id, fields):
    if not trace.enabled:
        return output
    original = output.get_output

    @wraps(original)
    def get_output(*args, **kwargs):
        begin = time.thread_time_ns()
        with trace.phase(step_id, "output_materialize", **fields):
            try:
                return original(*args, **kwargs)
            finally:
                trace.emit(
                    "output_materialize_cpu", step_id=step_id, thread_cpu_ns=time.thread_time_ns() - begin, **fields
                )

    output.get_output = get_output
    return output


def install_engine_output_trace(trace):
    # Installed during scheduler construction, before the output IO thread.
    import zmq
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.serial_utils import MsgpackEncoder

    original_thread = EngineCoreProc.process_output_sockets
    if getattr(original_thread, "_ascend_output_trace", False):
        return
    active = contextvars.ContextVar("ascend_output_socket_batch", default=None)
    original_encode = MsgpackEncoder.encode_into
    original_send = zmq.Socket.send_multipart

    @wraps(original_encode)
    def encode(encoder, outputs, *args, **kwargs):
        fields = output_fields(outputs)
        if not fields["request_ids"]:
            return original_encode(encoder, outputs, *args, **kwargs)
        trace.emit("engine_output_encode_begin", **fields)
        try:
            return original_encode(encoder, outputs, *args, **kwargs)
        finally:
            trace.emit("engine_output_encode_end", **fields)
            active.set(fields)

    @wraps(original_send)
    def send(socket, *args, **kwargs):
        fields = active.get()
        if fields is None:
            return original_send(socket, *args, **kwargs)
        trace.emit("engine_output_send_begin", **fields)
        try:
            return original_send(socket, *args, **kwargs)
        finally:
            trace.emit("engine_output_send_end", **fields)
            active.set(None)

    @wraps(original_thread)
    def output_thread(engine, *args, **kwargs):
        original_put = engine.output_queue.put_nowait
        original_get = engine.output_queue.get

        @wraps(original_put)
        def put(value):
            if isinstance(value, tuple) and len(value) == 2:
                trace.emit("engine_output_queued", **output_fields(value[1]))
            return original_put(value)

        @wraps(original_get)
        def get(*args, **kwargs):
            result = original_get(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                trace.emit("engine_output_dequeued", **output_fields(result[1]))
            return result

        engine.output_queue.put_nowait = put
        engine.output_queue.get = get
        return original_thread(engine, *args, **kwargs)

    output_thread._ascend_output_trace = True
    EngineCoreProc.process_output_sockets = output_thread
    MsgpackEncoder.encode_into = encode
    zmq.Socket.send_multipart = send


def install_api_output_trace(engine_core, trace):
    if engine_core is None or not hasattr(engine_core, "decoder"):
        return
    original_decode = engine_core.decoder.decode
    original_put = engine_core.outputs_queue.put_nowait
    original_get = engine_core.get_output_async

    @wraps(original_decode)
    def decode(*args, **kwargs):
        begin_wall, begin_mono = time.time_ns(), time.monotonic_ns()
        output = original_decode(*args, **kwargs)
        fields = output_fields(output)
        trace.emit("api_output_decode_begin", wall_time_ns=begin_wall, monotonic_ns=begin_mono, **fields)
        trace.emit("api_output_decode_end", **fields)
        return output

    @wraps(original_put)
    def put(output):
        trace.emit("api_engine_output_queued", **output_fields(output))
        return original_put(output)

    @wraps(original_get)
    async def get(*args, **kwargs):
        output = await original_get(*args, **kwargs)
        trace.emit("api_engine_output_dequeued", **output_fields(output))
        return output

    engine_core.decoder.decode = decode
    engine_core.outputs_queue.put_nowait = put
    engine_core.get_output_async = get


def install_collector_trace(trace):
    from vllm.v1.engine.output_processor import RequestOutputCollector

    original_get = RequestOutputCollector.get
    original_nowait = RequestOutputCollector.get_nowait

    def received(queue, value):
        if value is not None and not getattr(queue, "_ascend_first_get", False):
            trace.emit("api_request_output_dequeued", request_id=queue.request_id)
            queue._ascend_first_get = True
        return value

    @wraps(original_get)
    async def get(queue):
        return received(queue, await original_get(queue))

    @wraps(original_nowait)
    def nowait(queue):
        return received(queue, original_nowait(queue))

    RequestOutputCollector.get = get
    RequestOutputCollector.get_nowait = nowait
