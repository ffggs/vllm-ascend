# Gemma4 template early exit

Some Gemma4 chat templates search for the next non-tool message for every
history entry. After finding it, the original loop still visits the rest of
the history. This can cause quadratic template-rendering work in long chats.

Enable the equivalent early exit with:

```bash
--additional-config '{"gemma4_template_early_exit":true}'
```

Merge this key with any other additional configuration in your launch command.
The default is false. The patch targets the default tokenizer template for
models with `model_type=gemma4` and requires exactly one occurrence of the
known search loop. Other templates are unchanged. Explicit per-request or CLI
templates continue to take precedence. The model's files and shared tokenizer
are not modified; each enabled renderer passes the optimized default through
a per-call copy of the immutable rendering parameters. This also preserves
the upstream tokenizer pool's copying behavior.

This reduces redundant loop iterations on each render. It does not cache
rendered prompts or token IDs, modify sandbox checks, change tokenization,
or delay requests for batching. It has no effect on decode graph selection.

Validate rendered text and token IDs against the original template, including
tool results, assistant continuations, thinking options, and fresh inputs.
Measure template execution separately from encoding and executor queueing,
then run the complete application benchmark with operator profiling inactive.
Template microbenchmark gains alone do not establish end-to-end TTFT gains.

On Gemma4-26B W8A8 with DP1TP1, FULL_DECODE_ONLY, one renderer worker and
concurrency 4, the complete customer conversation benchmark's 8k-10k bucket
P95 TTFT decreased from 785.6 to 700.9 ms (276 requests in each run).
KV hit rates were 98.60% and 98.59%. The bucket uses each wave's mean input
length. Operator profiling was inactive for these measurements. The target
of 500 ms was not reached by this change alone.

Canonical-template differential validation covered 2102 rendered-text,
token-ID and error comparisons. On fresh inputs, median renderer batch time
at concurrency 4 decreased from 292.26 to 169.26 ms, with identical token IDs.
The source template was the Google Gemma 4 canonical template dated 2026-07-09;
measurements were made on 2026-09-15. Gains depend on conversation length and
the template in use.
