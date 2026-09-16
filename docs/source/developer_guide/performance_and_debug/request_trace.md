# Request-correlated profiling

Request tracing joins a client's time to first content with frontend rendering,
engine enqueue, scheduler batches, worker phases, and streamed output. It is
disabled by default and records request IDs, timestamps, and token counts, not
prompt text. Use unique request IDs and a new output directory for each run.

Enable `VLLM_ASCEND_REQUEST_TRACE_DIR=/absolute/path/to/request-trace` in the
serving environment. The older `VLLM_REQUEST_TRACE_DIR` spelling is also
accepted. Add these options to the existing model launch command:

```bash
--middleware vllm_ascend.profiler.request_trace_hooks.RequestTraceMiddleware \
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/absolute/path/to/operator-trace","torch_profiler_record_shapes":true,"ignore_frontend":false,"max_iterations":0}'
```

The benchmark must put the same unique ID in the JSON `request_id` field and
the `X-Request-Id` header. Record client send, first nonempty SSE content, and
completion timestamps using both a monotonic clock (for elapsed time) and wall
time (for correlation). HTTP headers and the initial assistant role frame do
not count as first content.

Measure the performance baseline while the operator profiler is inactive.
Save the actual concurrent inputs outside the timing interval, select the
slowest request in the target bucket, and replay its entire concurrent wave
with the same cache warmup. Start operator profiling with `POST /start_profile`
after warmup, then issue the measured wave and call `POST /stop_profile` in a
`finally` block. Keep operator-profiled latency separate from the baseline:
profiling adds overhead.

The request directory contains `api_pid*_rank0.jsonl`,
`scheduler_pid*_rank0.jsonl`, and `worker_pid*_rank*.jsonl`. Each event includes
process, rank, wall and monotonic timestamps. Scheduler keys identify the
originating batch even with asynchronous scheduling; the key is carried on
the batch sent to the worker. Internal and external IDs are joined using the
`engine_enqueue_begin` event. Worker `step_id` and process ID match the
`vllm.request_trace step=... phase=...` scopes in Ascend's `trace_view.json`.

For the HF renderer, `frontend_task_submitted` and the matching `executor_work`
stage (joined by request ID and task ID) separate executor queueing from work.
Nested `chat_template` and `tokenize` stages identify actual template rendering
and encoding, including calls made in worker threads. Stages record native
thread IDs, monotonic elapsed time, and calling-thread CPU time. Thread CPU time
excludes CPU work performed by any additional native tokenizer threads. Nested
stage durations must not be added to their enclosing executor duration.
Each stage has a `span_id` and `parent_span_id`. Asynchronous stages set
`async_wait=true` and `thread_cpu_ns=null`: their elapsed time includes
suspension, and the event-loop thread can execute other requests in between.
Only synchronous stages have calling-thread CPU measurements and torch scopes.
Additional stages cover HTTP JSON/schema processing, chat message parsing,
template resolution/parameters, token encoding and engine input construction.

Output tracing binds the internal ID of each request rather than the context
inherited by the long-lived output task. It records first-content detokenizer
updates, native decode-stream steps, logprobs, output construction and stats.
`api_request_output_ready` and `api_output_queued` precede HTTP content emission.
The enclosing `api_output_received`/`api_output_processed` interval still covers
an entire batch, including other requests, and is not exclusive request cost.
Per-request output stages stop once nonempty output text is ready.
`output_materialize` measures the existing asynchronous token-copy completion
and CPU output conversion; it adds no device synchronization. A worker step end
can precede this operation. Scheduler output, engine queue put/get, Msgpack
encoding, ZMQ send, API Msgpack decoding, API batch queue and per-request
collector dequeue are recorded separately. Batch events retain request IDs and
the engine output timestamp to distinguish consecutive decode outputs. The
gap between send completion and API decoding can include event-loop scheduling
as well as transport; do not label the entire gap as network time.
`sse_serialize` and HTTP body send begin/end cover JSON serialization and ASGI
handoff. ASGI send completion is not an acknowledgment of client receipt.
Frontend CPU and worker CPU+NPU capture can run in the same profiling window;
separate process files are joined using the request timestamps and IDs.
The middleware enables PyTorch's `profile_all_threads` option on the standard
AsyncLLM CPU profiler. This includes renderer threads even though vLLM starts
the profiler from an auxiliary thread. The option requires a PyTorch build
that supports `_ExperimentalConfig(profile_all_threads=True)`.

Include all worker steps overlapping the client's send-to-first-content
window, including preceding work for other requests. A shared batch's kernels
belong to the batch and must not be reported as exclusive per-request cost.
Compare host phase duration with the union of overlapping device execution
intervals; summing nested CPU scopes or overlapping kernels overcounts time.
Report missing stages explicitly instead of calling all unexplained time
queueing. Check graph mode per forward: under `FULL_DECODE_ONLY`, prefill is
normally `NONE` while eligible decode forwards should be `FULL`.

This middleware observes the chat completion API in the standard single
frontend topology. Split frontend deployments and alternate serving endpoints
need their own stage instrumentation. The tracing code does not change the
scheduler policy or enable speculative decoding.
