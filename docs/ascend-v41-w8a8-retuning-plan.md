# Ascend V4.1 W8A8 retuning plan

## Scope and evidence

This plan supersedes the optimization order in
`docs/ascend-v41-w8a8-serving-handoff.md`. It keeps the handoff's operational
safety constraints, but changes the technical priority: restore trustworthy
forward attribution before attempting another kernel or topology change.

The current evidence is historical, not a fresh remote inventory. Before any
deployment or benchmark, verify the running commit, worktree diff, container
configuration, model configuration, and service health on all four nodes.
Connection details and credentials must stay outside the repository.

## What the existing measurements establish

The historical deployment is a four-node, 32-card 910B3 service with DP4,
attention/dense TP8, MoE TP1, and EP32. The principal comparison uses short
contexts, W8A8, and wide EP.

| Measurement | Historical result | Interpretation |
| --- | ---: | --- |
| vLLM-Ascend single-stream ITL | about 0.438 s | About 2.29 output tokens/s |
| TokenSpeed eager single-stream ITL | about 1.83 s | About 0.55 output tokens/s |
| vLLM-Ascend concurrency 8 | 14.542 output tokens/s | Fixed 2048-token generation |
| TokenSpeed graph concurrency 8 | 1.942 output tokens/s | About 7.5x below vLLM-Ascend |
| TokenSpeed eager concurrency 8 | 1.801 output tokens/s | Graph alone is not the main gap |
| TokenSpeed split-attention concurrency 8 | 1.950 output tokens/s | Essentially the graph baseline |
| FlatKV gather change | 7.12 to 6.13 ms per call | About 14% local, only about 2.3% `model_fwd` |
| SiLU and ngram changes | about 0.994x | No useful end-to-end win |

The handoff's remaining `model_fwd` time is uncertain rather than a precise
63%: p50-based extrapolation leaves about 63% unattributed, while mean-based
extrapolation leaves about 45%. In addition, the existing V4.1 attention stage
timers synchronize the current stream around individual stages. This can
serialize asynchronous execution and inflate both stage times and the
`model_fwd` measured with those timers enabled. Do not use `stage_p50 * 40` as
a device-time profile.

Consequences for the optimization order:

1. Long-context native indexing is important, but it is not the first fix for
   the historical `max_model_len=1024`, low-concurrency baseline.
2. FlatKV gather remains a cost, but another small gather rewrite is unlikely
   to close a 4–7x engine gap.
3. The 16-to-64 query-head padding at TP8 is real, but SparseFlashMLA itself
   was only about 1.3 ms per layer in the synchronized measurements. Head
   padding alone cannot explain the whole gap.
4. MoE, shared experts, dense projections, Engram, collectives, and framework
   work remain insufficiently attributed.

## Phase 0 — read-only inventory and baseline freeze

Do not restart a healthy service for inventory. Do not run VIZTRACER on the
live mesh, reset NPUs, use `fuser`, or remove containers during weight loading.

Collect from every node:

- hostname and rank-to-address mapping; do not infer rank from the order of
  the externally supplied addresses;
- container state and the complete serving command, including graph, DP/TP/EP,
  Engram, context-length, and speculative-decoding options;
- running commit, worktree diff, deploy-tree file hashes, and imported module
  paths, so all ranks are known to execute the same code;
- CANN, driver, `torch`, `torch_npu`, kernel-package, engine, and scheduler
  versions;
- checkpoint configuration and quantization metadata;
- recent rank logs and read-only `npu-smi info` output;
- health, model listing, and a short greedy correctness response on rank 0.

Freeze one deployable baseline. A worktree with unfinished module moves or
missing imports is not a baseline even if another remote deployment is still
running older code.

## Phase 1 — make the baseline measurement trustworthy

### 1A. Benchmark without debug synchronization

Run TokenSpeed and vLLM-Ascend against the same model, topology, input
distribution, and generation contract. Use unique prompts to avoid accidental
prefix reuse and `ignore_eos` for fixed-length decode.

Initial matrix:

- ISL: 32, 128, 512, and 1024 tokens;
- OSL: 128 tokens;
- concurrency: 1, 4, and 8;
- at least one warmup and three measured repetitions per cell.

Report TTFT, ITL p50/mean/p90, request latency, aggregate output tokens/s,
completion fill ratio, errors, and actual running batch per DP rank. Keep
DSpark and other speculation off for the target-model baseline. Measure graph
and eager separately, with attention and layer debug timers disabled for the
primary end-to-end result.

### 1B. Add event-based forward attribution

The existing `ForwardContext.layer_kind_timer` field is still only a stub.
Implement a debug-only timer around the existing unified forward path:

- use NPU events rather than per-stage stream synchronization;
- do not read device scalars or call `.item()` in the forward path;
- skip event creation and timing during graph capture;
- synchronize once after the eager step, outside the layer scopes, then read
  and aggregate event elapsed times;
- reuse or pool event objects where practical, and throttle logging to one
  selected rank;
- preserve the same eager and graph execution paths when timing is disabled.

Start with non-overlapping layer scopes and these nested categories:

1. Engram staging and Engram forward;
2. attention projections, cache write, selection, gather, and SparseFlashMLA;
3. MoE routing, dispatch, routed grouped GEMM, combine, and shared experts;
4. dense projections and other GEMMs;
5. HC/residual operations;
6. explicit collectives and framework time not covered by a layer.

The sum of nested device scopes is not automatically end-to-end wall time when
multiple streams overlap. Record both scope duration and one whole-layer event
span so overlap and gaps remain visible.

Enable `--allow-reload-forward` at the next already-justified controlled boot,
then use `ts reload-forward` for instrumentation iteration. Validate the final
instrumentation on a fresh server before using its output for optimization
decisions.

## Phase 2 — optimize only a measured dominator

A candidate must have a plausible path to at least 15% improvement in
`model_fwd` or end-to-end throughput before it earns a controlled restart.
Microbenchmark wins alone do not pass the gate.

### Attention-dominant result

1. Fuse or eliminate cache-write/scatter and selection overhead before another
   gather-only tweak.
2. Reuse global gather/dequantization across Reuse layers only when the KV
   owner and selection are identical; never share it with a different Reindex
   selection.
3. Cache static metadata such as identity block tables and `cu_seqlens_q`.
4. Develop paged native index/attention kernels that consume FlatKV directly,
   avoiding full BF16 PA_BBND materialization.
5. For long-context runs, prioritize a native paged lightning indexer over the
   Torch `[T,K,128]` keys and FP32 `[T,H,K]` intermediate path.

### MoE- or GEMM-dominant result

1. Separate BF16 MC2 dispatch/combine from local INT8 grouped GEMM; W8A8 does
   not quantize the communication payload in the current distributed path.
2. Check EP32 utilization at low running batch and the cost of dummy/idle DP
   participants.
3. Evaluate shared-expert fusion or overlap only after measuring its device
   span; the NPU path does not have the same auxiliary-stream coverage as the
   CUDA path.
4. Do not change W4A4 during this attribution pass because it changes weights,
   numerics, and kernels simultaneously.

### Engram- or host-dominant result

1. A/B host table against device residency with the same memory headroom;
   retain `shared` as the first multi-rank host-layout comparison.
2. Measure the hashes D2H, per-layer CPU gather, and codes/scales H2D
   separately.
3. Pack the hash-column gathers and transfers, then evaluate direct NPU lookup
   from host-registered tables if the platform supports it.
4. Keep `prepare_ngram` outside `model_fwd` attribution unless the same-step
   wall measurement proves otherwise.

### Topology- or graph-dominant result

1. Compare TP8 with attention DP4/TP2 and EP32 at concurrency 4 and 8, not only
   batch 1. TP2 avoids the 16-to-64 query-head repeat but changes DP batching.
2. Validate graph replay with debug timers disabled and no host readback in
   capture. Graph is an additional optimization, not a substitute for closing
   the eager device-time gap.
3. Re-enable DSpark only after the target model has a stable baseline, and
   report acceptance rate and draft overhead separately.

## Acceptance and restart protocol

For every candidate:

1. Record the exact commit, diff, configuration, load, and artifact location.
2. Run greedy correctness plus fixed-length decode; require no rank failures
   and completion fill ratio of at least 0.9.
3. Compare at least three runs against the frozen baseline with debug timing
   disabled.
4. Require at least 15% improvement in the agreed primary metric, with no
   correctness regression, before keeping an optimization or spending another
   full restart.
5. Keep changes small and reversible. Add behavior tests for code changes and
   update the relevant design documentation.
6. Leave the service in a known healthy state and record any rollback.

## Immediate deliverables

1. Four-node read-only inventory and a verified rank/configuration map.
2. A no-debug-timing TokenSpeed/vLLM baseline under the fixed load matrix.
3. An event-based layer-attribution patch with behavior tests.
4. One ranked optimization proposal supported by the new attribution, rather
   than another speculative gather, SiLU, or graph change.
