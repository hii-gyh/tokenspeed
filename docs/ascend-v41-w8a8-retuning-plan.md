# Ascend V4.1 W8A8 retuning plan

## Scope and evidence

This plan supersedes the optimization order in
`docs/ascend-v41-w8a8-serving-handoff.md`. It keeps the handoff's operational
safety constraints, but changes the technical priority: restore trustworthy
forward attribution before attempting another kernel or topology change.

Historical performance evidence remains useful only as a baseline hypothesis.
The first automated run has now supplied fresh deployment evidence, summarized
below. Connection details and credentials must stay outside the repository.

## Current execution status

The chronological performance claims below are superseded as of 2026-11-16.
Fair fresh Graph comparisons put TokenSpeed at about 23.4–23.7 output tokens/s
versus 207.2 direct / 208.5 SMG tokens/s for vLLM-Ascend; the earlier 30% TokenSpeed
advantage was an eager/graph and warmup mismatch. The current pure-decode
index-key reuse patch passed three NPU tests and improved a paired same-service
five-repeat mean from 24.2318 to 27.6735 tokens/s (1.14203x), with mean TPOT
falling from 0.311667 to 0.265537 s (14.80%). Pure-decode profiling confirms
that owner 20's large gather falls from five calls to one per step, with total
GatherV3 time dropping from 69.576 to 30.405 ms/step. This is a TokenSpeed
self-improvement, not vLLM parity. Full greedy text is not reproducible across
reloads even after a symmetric rollback, so semantic quality remains an open
gate. See `docs/ascend-v41-w8a8-serving-handoff.md` and
`outputs/v41-debug/index-reuse-after/comparison.json` for current evidence.
All twelve completed PD runs remain `performance_benefit=false`.

The accepted two-node service is now the direct-BF16-page NPU Graph with the
three-dimensional BF16 ND mirror writer. Its strict 55-prompt/128-output-token
comparison is `outputs/v41-debug/20261114T214831Z-48475-strict/`: 23.2870
output tokens/s, 0.3168s mean TPOT, and 3.7110s all-row mean TTFT. It beats the
valid vLLM baseline on throughput, TPOT, mean TTFT, and TTFT p90, but not TTFT
p50. The running service received this final writer through `reload-forward`;
a fresh full-weight start remains an open final-validation gate. Four-node PD
run `20261115T010816Z-53548-pd1p1d` completed warmup, Paris correctness, real
KV transfer, and both 2048-token fixed-load repeats with zero request errors.
It proves request-level PD serving, but not an overall performance benefit:
mean throughput is 15.0481 versus 23.2870 tokens/s for non-PD, while mean TPOT
is 0.3046 versus 0.3168 s (a 3.85% improvement). TTFT mean/p50/p90 regresses
to 20.7583/6.5153/50.6315 s, so `pd-vs-nonpd.json` correctly retains
`performance_benefit=false` despite the runner's correctness `status=passed`.

The prefill timeline shows warm EXTEND spans of only about 1.5–3.7 s; the long
tail is dominated by bootstrap/remote-prefill waiting. Run
`20261115T015710Z-53827-pd1p1d` raised `--max-num-seqs` from 8 to 16 and
completed all requests and cleanup successfully. Throughput improved to
17.8940 tokens/s and TTFT mean/p50/p90 fell to 11.2940/6.0477/30.9871 s, but
throughput remained below non-PD and mean TPOT regressed to 0.3546 s. The full
logs identify the more precise cause: native `bootstrap_room % 2` routing sent
23 of 34 requests to decode DP0 and only 11 to DP1. Skewed 7:1 waves ran near
0.36–0.39 s TPOT, while a balanced 4:4 wave ran near 0.28–0.30 s.

Run `20261115T024934Z-54026-pd1p1d` used eight concurrent four-token warmup
requests to cover the measured prefill/decode shapes before the two unchanged
formal runs; warmup traffic stayed outside the reported metrics. Its rendered
entry did **not** contain `--dp-minimum-tokens-scheduler`, so the improvement
must not be attributed to that routing policy. Throughput rose to
20.7765 tokens/s and TTFT mean/p50/p90 fell to 6.0182/6.1502/7.6991 s, but mean
TPOT was 0.3324 s and the strict benefit gate remained false. Decode DP0 took
18 of 41 requests; the second waves split 3:5 and 2:6, proving residual static
room-routing skew. All four PD containers were removed and original workloads
restored. Run `20261115T033428Z-54393-pd1p1d` then tested `--dp-aware` with
round-robin rank selection and sixteen warmup requests. It regressed to
19.0337 tokens/s and 0.3640 s mean TPOT; decode DP0 accepted 26 of 49 requests,
with measured waves splitting 6:2 and 5:3. Independent P/D round-robin counters
therefore did not balance each client wave, and this policy is rejected. The
next run is the first real execution of `--dp-minimum-tokens-scheduler` on the
single-worker registration path, again with sixteen warmup requests. The local
automation suite passes 49 tests.

The sections below retain the chronological bring-up and optimization record.

Non-interactive SSH key authentication now passes on all four hosts. The
one-time bootstrap completed in
`outputs/v41-debug/20261114T072007Z-11077-ssh-bootstrap`; do not repeat it or
regenerate the key. The read-only inventory found no healthy TokenSpeed service
or historical deployment tree. The running DeepSeek/PD containers on the other
three hosts are vLLM-Ascend services and have not been stopped or modified.

Stage 1A passed in `outputs/v41-debug/20261114T072009Z-11068-stage1`: all three
Python packages imported from the uploaded source tree and the real NPU event
recorder completed its small-matrix smoke test. This did not load model weights
or exercise the native scheduler request path.

The explicit native gate then passed in
`outputs/v41-debug/20261114T073045Z-11484-stage1`. It installed pinned wheels
only into an isolated `/tmp/ts-native` environment and verified engine
`0.1.0.post20261112`, scheduler `0.1.54`, SMG `1.15.0.post20261110`, grpc proto
`0.4.28.post20261109`, and grpc servicer `0.15.0.post20261110`. Imports, `ldd`,
the no-argument `ts-host` usage probe, and the NPU event smoke passed. The wheel
hashes and full multi-card HCCL/request dependency closure are not covered by
that gate.

Stage 2 is the isolated single-node, eight-NPU serving gate on `.107`. Run it
with an explicit execution mode:

```bash
bash test/ci/npu/run_v41_debug.sh --stage2 --execution-mode eager --accept-new-host-key
```

It uses a unique source directory, read-only weights, the native virtual
environment, `/readiness`, unbuffered logs, and the non-blocking event recorder.
A host-side pip cache avoids downloading the same native and Python wheels for
every attempt; pre-serve container logs are retained when dependency or native
op preflight fails. The fixed benchmark remains 16 requests, concurrency 8, and 128 output tokens
per request. Its metadata is explicitly `DP1/EP8`, not the historical DP4/EP32.

The first three stage-2 attempts established and fixed separate startup issues:

1. `20261114T073711Z-11734-stage2` treated the gateway `/health` response as
   worker readiness and sent requests before workers were available. Readiness
   now uses `/readiness`.
2. `20261114T073948Z-11803-stage2` exited because the runtime imports
   `viztracer` unconditionally. The isolated supplemental dependencies now
   include `viztracer`, `uv`, and `py-spy`; this does not enable live profiling.
3. `20261114T074212Z-11929-stage2` exposed the missing
   `tokenspeed_kernel_npu.ops.dsv41` adapter before weight loading.

The current Ubuntu nightly image already contains CANN 9.2.1, the
SparseFlashMla/metadata source and packaged custom operators. Importing
`vllm_ascend.vllm_ascend_C` with a mapped NPU registers
`torch.ops._C_ascend.npu_sparse_flash_mla` and
`npu_sparse_flash_mla_metadata`; their native schemas were verified directly.
`tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/dsv41.py` initially
restored the thin NPU boundary over those vLLM operators and forwarded the
existing TokenSpeed arguments unchanged. Subsequent native execution exposed
the 1024/2048 mismatch described below; the boundary now uses the matching
CANN toolkit operators rather than substituting `npu_sparse_flash_attention`
or introducing another attention algorithm.

The fourth attempt, `20261114T075958Z-12828-stage2`, exited before `serve.log`
was created and its container cleanup discarded the preflight error. The runner
now falls back to `nerdctl logs` when the file is empty. This identified the
fifth attempt, `20261114T080335Z-12887-stage2`, as a GitHub wheel-download
timeout, not an adapter failure. Persistent host-side pip caching and longer
network retries were added before the next attempt.

The sixth attempt, `20261114T080723Z-12966-stage2`, passed native installation,
adapter import/schema checks, and eight-rank distributed initialization. It
then failed while constructing MoE INT8 weights: each NPU had about 57.76 GiB
active and could not allocate another 1.06 GiB. The configuration has 40 layers,
384 routed experts, hidden size 5120, and MoE intermediate size 2304. Routed
expert weights alone require about 63.28 GiB per NPU at EP8, 31.64 GiB at EP16,
and 15.82 GiB at EP32, excluding dense/shared weights, activations, KV, and
runtime workspace. Full-weight single-node EP8 is therefore physically
infeasible; reducing `gpu_memory_utilization` cannot fix model construction.

No successful model readiness, Paris correctness, serving event profile,
fixed-load benchmark, multi-node switch, or PD measurement is implied by the
native and adapter gates above. The next deployment gate must use at least
EP16. Other hosts' vLLM services remain untouched pending the topology decision.

### Two-node EP16 and native metadata evidence

`outputs/v41-debug/20261114T081753Z-13104-cluster2` is the first two-node
DP2/EP16 attempt. Both nodes uploaded the same 1437-file source archive,
initialized global ranks 0–15, and completed all 274 checkpoint shards on rank
0. Post-load HBM use was about 43 GiB per NPU, leaving useful runtime headroom.
Warmup then reached the real V4.1 decode path and failed in
`aclnnSparseFlashMlaMetadata`: the vLLM `_C_ascend` wrapper allocated 1024
int32 entries while the installed CANN 9.2.1 operator required 2048. Both TS
containers were removed in a controlled stop/remove sequence. Starting the old
rank-1 vLLM container only restored a transient process state; it subsequently
exited because its historical four-node DP4 peers were not running. That old
container must not be reported as a healthy two-node baseline.

Single-NPU probes isolated the failure from distributed execution, model
weights, and query-head count. Heads 8, 16, and 64 all failed identically in
the `_C_ascend` metadata wrapper. The same image also ships the CANN-matched
`cann_ops_transformer` bindings; those returned a 2048-element metadata vector
for both tested head counts. The NPU adapter now uses these toolkit bindings,
and the Ascend V4.1 path explicitly supplies TND/PA_BBND layouts and the
2048-element metadata contract.

`outputs/v41-debug/probe_cann_smla_attention-result.json` records a real NPU
numeric gate for one query token, 64 heads, one visible BF16 KV row in a 16-row
PA_BBND block, and a zero attention sink. The native output was finite and
matched an independent PyTorch reference with maximum absolute error
`0.00011299` and mean absolute error `0.00001130`. This validates the minimal
SWA native path only; it does not prove compressed-KV, complete-model, graph,
or serving correctness.

The follow-up run `20261114T090449Z-15913-cluster2` completed checkpoint
loading but failed in the actual attention call: CANN defaulted to BSND while
TokenSpeed supplies a three-dimensional TND query. The metadata call already
specified TND/PA_BBND; the attention call now specifies the same two layouts.
Both TS containers were cleaned up in a controlled sequence, and the original
rank-1 vLLM remained in its preexisting exited state.

The single-NPU probe was then extended to call TokenSpeed's real
`_launch_pa_bbnd_decode` entry, not only the native operator. With the explicit
layouts it returned finite `[1, 64, 512]` output and matched the independent
reference with the same `0.00011299` maximum absolute error as the direct CANN
call. The updated evidence is in
`outputs/v41-debug/probe_cann_smla_attention-result.json`.

`20261114T094028Z-20003-cluster2` used that validated launcher contract and
completed all 274 checkpoint shards on both nodes. It then failed during warmup
in `HcclBackend.token_all_gather`: this Torch 2.10 runtime does not provide the
non-standard `torch.distributed.all_gather_single` alias. Both TS containers
were cleaned up; no readiness, request, benchmark, or serving profile was
obtained. The original rank-1 vLLM remained in its preexisting exited state.

The HCCL backend now uses the public `all_gather_into_tensor` and
`reduce_scatter_tensor` collectives while retaining TokenSpeed's own
`all_gather_single` wrapper semantics. Before another full-weight attempt,
`outputs/v41-debug/probe_hccl_collectives-result.json` exercised the real
`HcclBackend` on eight NPU ranks. Rank-ordered gather, reduce-scatter,
all-to-all, and the signed INT64 MAX fallback all matched exact CPU references
with Torch 2.10.0 and torch-npu 2.10.0.post7. This is a collective-only gate,
not a cross-node or full-model result. The first probe attempt using
`torchrun --standalone` stalled before creating workers and was stopped; the
passing probe used an explicit private-IP master address.

The next full attempt (`20261114T101847Z-22088-cluster2`) did not reach weight
loading: rank 0 repeatedly timed out downloading the pinned engine wheel from
GitHub while rank 1 waited at distributed initialization. Both containers were
stopped before checkpoint loading and cleaned up, and both NPU sets were
verified idle. The exact CPython 3.12 aarch64 engine wheel is now cached on
both serving hosts with SHA-256
`cace479db91ea665ae61bb86df0275fd628833ceef96e5e549409970fd8f3033`.
The serving entry uses that file only after `sha256sum -c` succeeds; without a
cache it installs the same explicitly pinned package version. Other native
wheels still need the same hash-and-cache treatment, and this network fix says
nothing about model readiness or performance.

`20261114T102752Z-22648-cluster2` then used the verified cached engine wheel,
completed all 274 checkpoint shards on both nodes, initialized all 16 model
executors, and finished the 1024-token kernel warmup in about 3.4 seconds. Its
device summary reported about 43.39 GiB Torch-allocated and 9.24 GiB free per
NPU. Startup subsequently failed at the native/Python boundary because engine
`0.1.0.post20261112` reads `HostConfig.ttft_trace_sampling_percentage`, which
this branch did not yet define. The later hot-reload and `npugraph_ex` messages
occurred during signal-driven shutdown and are not the primary error. Both TS
containers were verified absent and all 16 NPUs idle before the next attempt.

The matching upstream contract is an integer sampling percentage in `[0, 100]`;
`0` disables TTFT tracing. The minimal `ServerArgs` / `HostConfig` plumbing,
range validation, documentation, and behavior tests are now present without
backporting the unrelated lifecycle-tracing implementation. The real Ascend
image ran `test_server_args_env.py` with `9 passed` from isolated probe source
`/root/gyh_dev/tokenspeed_probe/hostconfig_20261114T102752Z-fix1`. The resulting
run `20261114T110514Z-24230-cluster2` passed two-node readiness after 1761.3s;
both containers remained running and a direct rank0 probe returned HTTP 200.
The original local runner was stopped after readiness to avoid its 1800s total
benchmark timeout triggering cleanup during two eager repeats. The preserved
service is now measured through independent run
`20261114T114449Z-25773-benchmark-tokenspeed-stream` with a 5400s total timeout
and a unique prompt tag that prevents cross-measurement prefix-cache reuse. The
first independent directory `20261114T113754Z-25448-benchmark-tokenspeed-stream`
is explicitly invalidated because the original remote benchmark client remained
alive after its local runner stopped and overlapped it; both client processes
were terminated without stopping the model service. Correctness, benchmark, and
event-profile conclusions still require final artifacts rather than readiness
or earlier warmup success.

PD is a supported TokenSpeed serving architecture on Ascend, with headless
prefill/decode roles behind SMG and Mooncake KV transfer. The generic
`tokenspeed-mooncake` CPython 3.12 wheel is not an Ascend runtime: its native
libraries have unresolved `libcuda.so.1` and `libcudart.so.13` dependencies
(`outputs/v41-debug/probe_mooncake_native-result.json`). Do not install CUDA
stubs or infer Ascend support from that package's import.

This blocker is now addressed by the official Ascend NPU wheel plus the local
TokenSpeed C ABI adapter. Real same-host and cross-node paged transfers are
byte-exact, and the production Rust loader passes. Six real 1P1D attempts have
isolated and fixed stop/restore behavior, the 2 MiB arena-registration
alignment requirement, the gateway-hosted decode handshake, and SMG's 3 s
cold-prefill wedge. Run 6 proves request-level PD correctness and fixed-length
generation, with a TPOT-only benefit; its throughput and TTFT do not meet the
overall benefit gate. Run 7 tests the measured request-slot bottleneck rather
than changing kernels without attribution.

## Clean two-node TokenSpeed acceptance (2026-11-14)

Run `20261114T114449Z-25773-benchmark-tokenspeed-stream` passed greedy Paris
correctness and two independent fixed-load repeats. Each repeat completed all
16 requests with 2048/2048 completion tokens, concurrency 8, 128 output tokens
per request, `ignore_eos=true`, streaming, and zero request errors. A unique
prompt tag prevented cross-measurement prefix-cache reuse; prompts were 55
tokens in this run. The previous independent directory is invalidated because
it overlapped an orphaned original benchmark client.

| TokenSpeed DP2/EP16 eager metric | Result |
| --- | ---: |
| Output throughput, mean of two runs | 5.7577 tokens/s |
| Output throughput range | 5.7290–5.7865 tokens/s |
| TTFT p50 / p90 | 4.4844 / 5.1561 s |
| TPOT mean / p50 / p90 | 1.3639 / 1.3703 / 1.3789 s |
| Fixed-run wall time | 353.929 / 357.482 s |

The same-topology vLLM comparison is complete. Runner
`20261114T120032Z-26250-vllm-cluster2` was superseded before benchmarking because
it incorrectly reused TokenSpeed `/readiness`; vLLM actually exposes `/health`.
The independent valid measurement is
`20261114T122049Z-26908-benchmark-vllm-ascend-stream`. It passed Paris
correctness and the same two 2048/2048-token fixed-load repeats with zero request
errors. The comparison artifact is
`outputs/v41-debug/20261114T122049Z-26908-ts-vs-vllm-comparison.json`.

| Same-contract mean metric | TokenSpeed | vLLM-Ascend | TS / vLLM |
| --- | ---: | ---: | ---: |
| Output throughput | 5.7577 tokens/s | 17.8043 tokens/s | 0.3234x |
| TPOT mean | 1.3639 s | 0.4143 s | 3.2922x |
| TPOT p50 | 1.3703 s | 0.4017 s | 3.4108x |
| TTFT p50 | 4.4844 s | 0.7744 s | 5.7909x |

TokenSpeed has not yet achieved the target TTFT, TPOT, or throughput advantage.
vLLM TTFT p90 was 20.7216 s because some requests queued; that tail does not
offset its p50, TPOT, and throughput lead. Both engine container pairs were
stopped and preserved after logs were fetched.

The clean rank0 event profile contains six post-warmup samples. Across five
DECODE bs=4 samples, mean forward time was 1344.8 ms, attention 1203.9 ms,
attention core 905.4 ms, and FFN 66.4 ms. These scopes are nested and not
additive, and the sample count is small; they prioritize attention-core and
attention-wrapper attribution, not an immediate kernel replacement. Evidence:
`outputs/v41-debug/20261114T110514Z-24230-cluster2/event-profile.json` and the
benchmark directory above.

Three isolated NPU probes then narrowed the attention work:

- Eight live heads versus 64 padded heads produced bit-identical first-eight-head
  outputs and kernel means of 0.1729 versus 0.1732 ms. Removing head padding has
  no measured kernel-level benefit in this configuration.
- The real torch indexer took 3.34–3.57 ms per source over 64–512 history rows;
  logical-slot gather was about 2.15–2.21 ms. Even eight such sources do not
  explain the 905.4 ms attention-core total.
- With four tokens, 128 SWA rows and 192 global rows, the full
  gather/PA_BBND/metadata/SparseFlashMla wrapper took about 4.50 ms per layer.
  The fixed-input kernel was only about 0.26 ms. Current-token SWA scatter was a
  separate 6.22–6.46 ms hotspot.

The torch codec now combines value and scale bytes into one selected-byte
transfer and replaces integer `argsort` with pairwise winner selection for at
most 64 rows. Public API tests cover duplicate and mixed active/inactive slots;
the remote image passed all 11 codec tests. Real-module NPU validation reduced
current-token scatter to 4.3467 ms and preserved byte-identical cache contents in
the duplicate-slot gate.

The subsequent two-node serving A/B is complete. Run
`20261114T134057Z-29163-benchmark-tokenspeed-stream` passed Paris correctness
and both fixed-load repeats. Relative to the old TokenSpeed build, throughput
improved from 5.7577 to 6.0319 output tokens/s (+4.7628%) and mean TPOT fell
from 1.3639s to 1.2998s (-4.6964%). See
`outputs/v41-debug/20261114T134057Z-29163-ts-codec-ab.json`. Five new DECODE
bs=4 samples average 1259.0ms forward and 815.2ms attention core, versus
1344.8ms and 905.4ms before; the small nested-scope event sample supports the
direction but is not the end-to-end proof. The latest TokenSpeed/vLLM ratios are
0.3388x throughput, 3.1376x mean TPOT, and 5.5638x TTFT p50, so the principal
performance goal remains unmet.

The live run `20261114T130934Z-28436-cluster2` has forward reload enabled.
Fine-grained projection, select, scatter, gather, SparseFlashMla, and output
collective scopes passed 26 real-image tests (18 skipped) and were hot-reloaded
without a service restart. Use only samples timestamped after the successful
reload for the next attribution round; the initial missing-`layer_id` reload was
rejected safely and never replaced the running code.


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

`test/ci/npu/collect_v41_inventory.py` implements the read-only SSH inventory
for WSL. Pass hosts explicitly and provide the container, host/container
worktree, and checkpoint paths as options. It uses the dedicated automation key
by default, supports the legacy password path only when explicitly selected,
and never prompts for credentials. It enforces known-host checking unless
`--accept-new-host-key` is explicitly selected, and writes the JSON outside the
repository by default.
The query set includes NPU mapping, CANN tool paths and environment, and
`asys` architecture evidence when available. Do not infer the SoC from the
`npu-smi` chip-name field. If the historical container name is absent, the
collector selects an existing serving candidate and records it as
`selected_container`; inspect includes image, entrypoint/command, mounts,
state, network mode, and published ports. This is discovery only, not engine
identification or a baseline-health override. The collector does not upload
scripts, change containers, reset devices, or submit inference requests.

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

`ForwardContext.layer_kind_timer` carries a debug-only timer around the
existing unified forward path:

- use NPU events rather than per-stage stream synchronization;
- do not read device scalars or call `.item()` in the forward path;
- skip event creation and timing during graph capture;
- poll event completion non-blockingly on later forwards, then aggregate and
  log elapsed times without ever synchronizing the serving stream;
- keep a bounded pending-sample queue and throttle logging to one selected rank;
- preserve the same eager and graph execution paths when timing is disabled.

Start with non-overlapping layer scopes and these nested categories:

1. Engram staging and Engram forward;
2. attention projections, cache write, selection, gather, and SparseFlashMLA;
3. MoE routing, dispatch, routed grouped GEMM, combine, and shared experts;
4. dense projections and other GEMMs;
5. HC/residual operations;
6. explicit collectives and framework time not covered by a layer.

The initial implementation is enabled with
`TOKENSPEED_LOG_LAYER_KIND_TIMING=1`. It samples the first four eager forwards
and then every 100th forward on DP rank 0's first attention TP rank. The
`v41_layer_timing` JSON log records whole-forward, whole-layer, `attention`,
`attention_core`, `ffn`, `moe_select`, `moe_routed`, `moe_shared`,
`moe_pre_comm`, `moe_post_comm`, `wide_ep_prepare`, `wide_ep_gather`,
`engram_stage`, `engram`, and `hc_post` event spans. If `moe_routed` dominates,
the next instrumentation iteration splits dispatch, grouped GEMM, and combine
inside that existing scope rather than adding a second execution path.

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

## CANN/AscendC skill routing

The installed CANN skills are used as gated engineering workflows, not as a
reason to write a kernel before attribution:

1. `model-infer-fusion` decomposes each measured attention/MoE hotspot and
   checks existing `torch_npu` fused APIs against the deployed version's local
   docstrings, including dtype, shape, layout, cache, quantization, and
   distributed constraints. Each replacement gets its own correctness and
   performance comparison.
2. `model-recommend-fusion-planning-estimation` combines profiled hot chains
   and adaptation failures into one ranked list. It requires local 910B3
   compute/bandwidth anchors, an explicit bound decision, conservative and
   optimistic benefit ranges, and a `hot-chain` or `adaptation-failure` source.
   This ranked list is the only entry into new kernel generation.
3. `model-recommend-kernel-generation` then uses standard AscendC on 910B. A
   kernel is not complete without a design, golden precision tests over the
   serving shapes, benchmark against the original operator sequence, profiling
   evidence that it replaced that sequence, and an end-to-end non-regression
   check. The TokenSpeed acceptance bar remains at least 15% end-to-end gain.
4. `ascendc-env-check` supplies the environment gate. Check `ASCEND_HOME_PATH`,
   `ASCEND_OPP_PATH`, `msopgen`, `msprof`, and the actual architecture with its
   runtime probe. Hardware resources come from `PlatformAscendC`, not hardcoded
   core counts or buffer sizes. The `npu-smi` chip name is not an architecture
   identifier.
5. `ascendc-perf-optimize` is applied only to a selected kernel: tiling model,
   inter-card pipeline for communication operators, inter-core pipeline when
   synchronization exists, and single-core pipeline/bound analysis. MC2 work
   additionally requires real TilingData and isolated communication/compute
   measurements before proposing a rebalance.

## Test execution ladder and stop conditions

### One-command automated entry

From WSL at the repository root, run only:

```bash
bash test/ci/npu/run_v41_debug.sh --accept-new-host-key
```

The wrapper selects or creates `.venv-wsl`, installs only the lightweight
`pytest`/`paramiko` test requirements, and executes the debug pipeline. The
default CPU-only WSL path does not install Torch, TokenSpeed, or a kernel
package; real Torch/NPU checks run inside the remote container. SSH uses the
dedicated automation key by default. The legacy `TS_SSH_PW` path is available
only when no key is used. Credentials are never printed, and the runner never
prompts.
Omit `--accept-new-host-key` after the initial controlled lab connection. Add
`--full` for the complete V4.1 regression set and three benchmark repetitions;
this mode requires an already configured full runtime environment and never
auto-builds a CUDA/ROCm/NPU kernel on WSL. Use `--no-benchmark` for the quick
read-only/NPU smoke loop.

Each run writes `outputs/v41-debug/<timestamp-pid>/` with `summary.json`,
stage logs, `inventory.json`, `baseline.json`, the NPU event result, and the
fixed-load benchmark when its gates pass. `outputs/v41-debug/latest-run.txt`
points to the newest result. The runner does not deploy code, create remote
files, restart containers, or reset NPUs. Local failure still permits read-only
inventory but blocks NPU and load execution. A healthy baseline requires all
four hosts to report the same Git HEAD and worktree state.

### Gate A — local behavior and regression tests

Run from the repository in the existing project virtual environment under
WSL. Do not install packages into the system Python.

```bash
set -euo pipefail
python3 -m pytest -x -q \
  test/runtime/test_layer_kind_timing.py \
  test/ci/npu/test_collect_v41_inventory.py
python3 -m pytest -x -q \
  test/runtime/test_deepseek_v41_config.py \
  test/runtime/test_deepseek_v41_wide_ep.py \
  test/runtime/test_deepseek_v41_engram.py \
  test/runtime/test_deepseek_v41_model.py
python3 -m compileall -q \
  python/tokenspeed/runtime/execution/layer_kind_timing.py \
  python/tokenspeed/runtime/models/deepseek_v41.py \
  python/tokenspeed/runtime/models/deepseek_v41_moe.py \
  test/ci/npu/collect_v41_inventory.py \
  test/ci/npu/check_npu_event_timer.py
```

Stop on the first failure, capture the failing test and current diff, and fix
it before remote execution. Run the exact `pre-commit run --all-files` hook
before any commit; formatter changes are part of the same validation loop.

### Gate B — non-interactive WSL inventory

Use the dedicated SSH automation key or an explicitly supplied key path. Do
not write credentials to the repository, a command file, or the JSON output.
The collector never prompts. Use `--accept-new-host-key` only for the initial controlled lab
connection; subsequent runs use the default rejecting policy.

```bash
set -euo pipefail
python3 test/ci/npu/collect_v41_inventory.py \
  39.156.164.107 39.156.164.108 39.156.164.109 39.156.164.110 \
  --container dsv41-ts-dp \
  --deploy-tree /root/gyh_dev/deploy_tokenspeed_full \
  --container-tree /ts \
  --model-path /mnt/weights/DeepSeek-V4.1-Flash-w8a8 \
  --accept-new-host-key \
  --output "$HOME/tokenspeed_ascend_inventory.json"
```

These paths are historical hypotheses to verify, not facts about the new
address mapping. Stop if a host is unreachable, a worktree differs without an
explanation, the service is unhealthy, or the four nodes do not run the same
intended code/configuration. Do not start a benchmark from an unverified
baseline.

### Gate C — NPU event primitive and serving smoke

After synchronizing only the reviewed files into the verified worktree, run
this standalone check inside one agent container on an explicitly selected
visible NPU:

```bash
nerdctl exec dsv41-ts-dp \
  python3 -B /ts/test/ci/npu/check_npu_event_timer.py --device 0
```

It uses a small tensor and verifies `record`, non-blocking `query`, and
`elapsed_time`; its completion wait is test-only and is not in the serving
path. Stop on import, event, timeout, or non-positive-duration failures.

With timing disabled, use `ts reload-forward` for a code/health/greedy
correctness regression. At the next controlled measurement boot, enable
`--allow-reload-forward` and `TOKENSPEED_LOG_LAYER_KIND_TIMING=1`; do not
assume that changing a shell variable reloads an already imported server.
Confirm `v41_layer_timing` appears only on the selected rank, pending samples
remain bounded, graph capture creates no timing events, and service latency
without debug timing matches the frozen baseline within the noise band.

### Gate D — fixed-load A/B and ranked hotspot report

Run correctness first, then the historical fixed load (16 requests,
concurrency 8, 128 output tokens, `ignore_eos`) and the ISL/concurrency matrix.
Measure timing-disabled baseline and candidate builds in the same session and
configuration. Use the event-enabled run only for attribution, not as the
primary throughput result. Produce mean/median/min/max over at least three
runs, completion fill ratio, actual running batch, and the ranked event spans.

Use `test/ci/npu/run_v41_benchmark.py` against an already healthy endpoint so
a protocol or engine comparison does not redeploy or reload the model. Engine,
protocol, and topology are explicit and are stored with the result:

```bash
python3 test/ci/npu/run_v41_benchmark.py \
  --host 39.156.164.107 --engine tokenspeed --protocol stream \
  --topology 'DP2 attn/dense TP8 MoE TP1 EP16 wide-EP short-ctx' --repeat 3
```

For the vLLM-Ascend comparison, keep the load and topology identical and change
only `--engine`, `--host`, and the actually deployed topology label. The stream
benchmark derives TTFT from SSE text arrival, TPOT from server-reported
completion tokens and end-to-end timing, and ITL from inter-chunk intervals; it
does not treat one SSE chunk as one token. The default `non-stream` mode keeps
the historical throughput comparison.

After a successful TokenSpeed run has been measured and its containers have
been stopped cleanly, `test/ci/npu/run_v41_vllm_cluster.py` starts the isolated
two-node vLLM-Ascend comparison. It refuses to start while either NPU set is
busy, uses unique containers and the same current weights, and never stops the
TokenSpeed containers itself. Both entries now make device prefix caching
explicit: TokenSpeed already defaulted to enabled, so the vLLM entry also uses
`--enable-prefix-caching`. The comparison uses DP2/TP8/EP16, 1024-token context
and prefill budget, eight sequences, host-offloaded shared Engram tables,
overlap/async scheduling, and no DSpark, Mooncake transfer, or speculation:

```bash
python3 test/ci/npu/run_v41_vllm_cluster.py \
  --execution-mode eager --success-action keep-vllm \
  --bench-protocol stream --bench-repeat 2
```

The runner is only a prepared comparison path until its own readiness,
correctness, and benchmark artifacts exist. Its generated rank entries pass
`bash -n`; the focused runner/automation regression is 19 passed. Do not infer
a vLLM result from the CLI help or from the historical DP4 configuration.

Stop optimization if the new profile cannot explain a substantial fraction of
`model_fwd`; split the largest existing scope rather than guessing. Only the
top measured dominator enters an existing-API experiment or the CANN fusion
planning gate.

### Gate E — candidate and kernel acceptance

For an existing `torch_npu` API, verify operator-level output parity first,
then microbenchmark, then serving A/B. For a new AscendC kernel, additionally
complete the CANN planning list, design, shape coverage, golden precision,
original-sequence benchmark, and profiling replacement check. In either case,
retain a change only after correctness and at least 15% improvement in the
agreed end-to-end metric; otherwise record the negative result and roll back.

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

## Measured update — global gather reuse

The fine-grained backend scopes replaced the former guesswork: one initial
bs4 sample attributed 675.827ms to `attention_gather_global`, versus 168.286ms
to SWA gather and 7.203ms to SparseFlashMla. The implementation now stores the
Ascend global gather result on the current `V41Selection`, including the
record captured by `prepare_global_selection`; owner and physical-slot identity
must match, and a new selection cannot reuse the previous forward's rows.
Scoped behavior and prepared-wiring tests passed 5 cases, and the default
attention/timing suite passed 26 with 18 skips.

The canonical two-repeat A/B is
`outputs/v41-debug/20261114T145023Z-31677-ts-global-memo-ab.json`:
throughput improved from 6.0319 to 6.8591 output tokens/s (+13.7125%), mean
TPOT fell from 1.2998 to 1.1432s (-12.0501%), and TTFT p50 fell by 8.7416%.
Paris correctness and both 2048-token repeats passed. This is below the 15%
standalone retain threshold, but required no restart and is retained as a
cumulative live candidate. Ten bs4 event samples now average 134.2866ms global
gather and 1131.5901ms forward. The samples are nested attribution data, not
additive wall-clock components.

Target 2 is still unmet:
`outputs/v41-debug/20261114T145023Z-31677-ts-vs-vllm-comparison.json` reports
0.3852x vLLM throughput and 2.7595x mean TPOT. The next gather experiment must
split address construction, packed-byte transfer, and dequantization on an idle
NPU; do not infer a new kernel from the former global-gather total.

## Measured update — gather geometry and all-selection

`outputs/v41-debug/gather-breakdown-result.json` separates logical-address
construction, packed-byte transfer, `_read_planar`, dequantization, and the full
codec gather on a real NPU. Sequential slots, shared random slots, and
per-request random slots have effectively the same cost. The tested padded
arena also matches a contiguous arena: SWA gather is 1.8019 vs 1.7990 ms, and
global gather is 3.4821 vs 3.4695 ms. Therefore slot randomization and this
padded stride do not explain the much larger in-service attribution; neither a
slot-sort change nor a storage-view rewrite is justified by these data.

The fixed-load history is shorter than top-k 512, so selection takes the `all`
path rather than sparse Top-K. A simple arange/mask replacement is not valid
because the production Triton path compacts missing and invalid pages and also
publishes candidate blocks. A fixed-shape cumsum/scatter alternative with an
extra invalid-destination column passed exact rows, lengths, candidate blocks,
and physical-slot comparisons on strided/hole-page cases, but was slower:
1.8667 ms versus 1.6927 ms for the existing implementation
(`outputs/v41-debug/all-selection-run.log`). It was not promoted. QCP query
partitioning is an extend/CUDA path and is not a decode optimization for this
service.

The next controlled experiment is the already-supported NPU Graph execution
mode. This is a hypothesis that eliminating launch gaps can exceed the 15%
restart threshold, not a claimed result. It requires an independent readiness,
Paris-correctness, fixed-load, and TS/vLLM comparison before acceptance.

The first graph attempt (`20261114T151624Z-32379-cluster2`) loaded all 274
checkpoint shards but exited at the first bs4 capture: `forward_step.py`
imported the missing `tokenspeed.runtime.accelerators` module. The failure was
therefore before graph execution and produced no performance evidence. A
minimal `graph_class_for` contract now selects `CUDAGraph` or `NPUGraph` and
rejects unknown devices or missing graph classes; its four behavior tests pass
in the real Ascend image. Run `20261114T154815Z-33154-cluster2` retries the same
experiment with that fix. The previous eager containers remain stopped and
preserved for diagnosis, not as a running fallback service.

The retry passed two-node readiness in 1904.6 s and Paris correctness. Its
independent benchmark is `20261114T162024Z-33753-benchmark-tokenspeed-stream`.
The first 16-request repeat completed 2048/2048 tokens with 13.9536 output
tokens/s, TTFT p50/p90 3.7954/3.8039 s, and TPOT mean/p50 0.5523/0.5541 s.
This is a useful single-repeat signal (about 2.03x the latest eager throughput
and about half its mean TPOT), but not an accepted two-repeat A/B result. The
second repeat failed in its second batch at about ten completion tokens with a
CANN `IndexCheckKernel<long>` assertion followed by ACL 507011 and scheduler
exit. The entire graph benchmark is therefore marked failed; its first repeat
must not be substituted for a stable service result.

`outputs/v41-debug/probe_v41_graph_codec.py` reproduces that assertion without
model weights or distributed execution. It uses the production-sized 4.38 GB
strided SWA arena (3168 pages, 1,382,400-byte page stride) and random int64
slots, then compares NPU Graph replay with eager. Eager warmup and graph capture
succeed, while replay reports index 2,064,356,352 against a larger
4,378,094,592-byte bound and then ACL 507011. The reported index is in bounds,
so this is not evidence of a bad business slot. The read path now uses
`index_select`. An `index_copy_` write candidate passed eager behavior tests but
did not update the external cache during graph replay, so it was rejected.
Advanced-index writes had the same replay limitation. The accepted write path
uses Ascend-native `npu_scatter_nd_update_` through `_ascend_codec.py`, with
int8 views over the same bytes and no runtime torch_npu import. Inactive writes
use slot zero as a read-modify-write destination and share its winner key, so a
valid slot-zero update cannot be overwritten.

A second graph-only defect remained after the write operator was fixed: SWA and
global packed bytes differed from eager. `probe_v41_graph_pack.py` isolated the
first divergence to integer right shift; bits, mantissa and bool-to-int outputs
were equal, while an expected exponent of -7 replayed as about 60. Runtime
bit-field extraction now uses equivalent floor-divide operations, and runtime
left shifts in the same quantizer use multiplication. All pack stages then
matched eager exactly. The final gate passed 14 codec behavior tests and 100
read/write graph replays for each of SWA, global and index formats on the real
4.38 GB strided arena, with duplicate slot 5, valid slot 0 and inactive slot -1;
all mismatch lists are empty. See `outputs/v41-debug/graph-codec-validation.json`.
This is single-card codec evidence, not yet full-model serving acceptance.

## Measured update — valid NPU Graph service

The third graph run exposed a separate correctness defect in the ND-update
choice. A three-way probe used the old advanced-index writes and the packed
input itself as independent truth. Eager and graph comparisons that only used
the same ND-update implementation on both sides had a common-mode blind spot:
four rows could pass while 9/19/33/55/128-row updates silently diverged, with
the first corruption near 4096 bytes. `_ascend_codec.py` now limits every
`npu_scatter_nd_update_` call to at most 4096 bytes and chunks larger payloads.
SWA/global/index are byte-exact against both old writes and packed truth for
4/9/19/33/55/128 rows on the real 4.38 GB strided arena. The large-payload
eager+graph regression in `tokenspeed-kernel/test/npu/ops/test_dsv41_ascend_codec.py`
passes all three formats. CANN 9.2.1's silent large-update corruption and this
chunking requirement are recorded in the dsv41 kernel README.

After forward reload, valid benchmark
`20261114T181916Z-41918-benchmark-tokenspeed-stream` passed Paris correctness,
two complete 2048-token repeats, zero request errors, and 128 streamed chunks
for every request. Its mean throughput is 14.3756 output tokens/s, TTFT
p50/p90 are 3.7655/6.0743 s, and TPOT mean/p50/p90 are
0.5282/0.5227/0.5379 s. Relative to the valid global-memo eager result, Graph
improves throughput by 109.5850% and reduces mean TPOT by 53.7925%; TTFT p50
improves 4.2313%, while TTFT p90 regresses 35.2644%. The evidence is
`20261114T181916Z-41918-graph-vs-eager-ab.json`.

Target 2 remains open. `20261114T181916Z-41918-graph-ts-vs-vllm-comparison.json`
reports 0.8074x vLLM throughput, 1.2751x mean TPOT, and 4.8626x TTFT p50.
Only TTFT p90 is better (0.2931x), and that metric is strongly affected by
vLLM queueing. It does not establish overall superiority. France and repeated
Hello prompts still return first-token EOS after the codec fix, while Paris
and counting probes pass. The attempted same-prompt vLLM reference on `.109`
failed with DP/Gloo disconnects, so this residual model behavior is not yet
classified.

## Measured update — NPU kernel profiler

The eager event recorder cannot attribute Graph replay internals: Python
scopes run during capture, not on every replay. `runtime/engine/profiler.py`
now has an NPU branch using `torch_npu.profiler`, Level1, PipeUtilization, and
`tensorboard_trace_handler`; `DeviceHandle.device_type` is a read-only property
so existing instances remain compatible after forward reload. Committed
forwards advance the shared schedule, stop performs the required final step,
and prefill/decode produce separate artifact sets. CPU/CUDA behavior is kept.

The real Ascend image passed 35 focused tests with one optional-dependency
skip in regression-7. A real single-NPU smoke then attached the profiler after
warmup and covered both an eager prefill and an already-captured NPU Graph
decode replay. Both outputs contain non-empty `kernel_details.csv` with 46–47
columns, operator/API statistics, and valid `trace_view.json`; see
`npu-profiler-regressions-7.log` and `npu-profiler-smoke-result.json`.
`summarize_v41_npu_profile.py` now supports one report or recursive parent
input and has been validated on both smoke reports.

The first live attempt did not exercise this implementation. `reload-forward`
returned HTTP 200 with an explicit `NOT reloaded, outside the forward code`
message for `runtime.engine.profiler` and `runtime.execution.device`. The old
CPU/CUDA profiler then tried to create `/ts/npu-profile` on the read-only
source mount, raising `OSError: Errno 30` and killing rank1's data plane.
Benchmark `20261114T184219Z-42771` has HTTP 503 correctness and no runs and is
invalid for both performance and profiling conclusions. The failed containers
were stopped normally and removed; no NPU reset or forced removal was used.
Fresh run `20261114T184612Z-43017` loaded the control plane and passed its
independent no-profiler gate: benchmark `20261114T191729Z-44795` completed both
2048-token repeats with Paris correctness, 14.0294 output tokens/s, 4.0354s
TTFT p50, and 0.5362s mean TPOT. The small difference from the earlier valid
14.3756/0.5282s result does not change target 2.

The subsequent live capture produced prefill and decode reports for all eight
ranks on each node. `profile-inventory-107.log` and `profile-inventory-108.log`
show non-empty kernel/operator outputs for every rank. The schedule auto-stops
at ten steps, so the later explicit stop call returns “Profiling is not in
progress”; this is expected auto-completion rather than a failed capture. Full
reports occupy 13 GB on `.107` and 9.4 GB on `.108`. The downloader therefore
selects the process named `scheduler_tp0_ep0_dp0` and streams only its two
reports' CSV files plus a selection manifest. Do not return to a whole-tree
`nerdctl cp` for routine attribution.

## Measured update — rank-0 attribution and paged-cache comparison

The compact rank-0 CSV download and `rank0-profile-summary.json` are complete.
The exact read shapes map to the real SWA/global/index formats: 128 or 512
528-byte SWA rows per request, 512 or 2048 288-byte global rows, and 1088 or
4352 68-byte index rows. This confirms that the avoidable common path is the
per-layer gather plus dequantization, not redundant head padding or an
assumed indexer-only bottleneck. The reports are triggered by prefill/decode
but may contain mixed work; nested operators, AICPU wrappers, static/dynamic
entries, and stream overlap must not be summed as wall time. The single
16.7-second MoE dispatch maximum is a boundary outlier with a matching AICPU
wrapper, not a steady 16.7-second kernel.

The real 910B3 vLLM implementation gives a direct structural comparison.
`AscendDSAV41Impl._forward_attention` passes its paged SWA/global caches,
block tables, logical sparse indices, lengths, and `cmp_ratio` directly to
`npu_sparse_flash_mla`; TokenSpeed instead gathers the selected physical slots
and dequantizes them into compact BF16 blocks before every launch. The default
non-A5 cache specifications use BF16 SWA/global storage. A5 `qsmla` and packed
quantization are separate hardware paths. A real isolated CANN 9.2.1 probe
rejected TokenSpeed's uint8 `ori_kv` with `Invalid_Argument_Tensor_Dtype` (only
FP16/BF16 are accepted), so direct quantized pages are ruled out on this ABI.
The vLLM baseline also enables MRV2 tail-128 bounded replay, but TokenSpeed already
narrows later CED decoder layers through `V41DecoderView`/`V41RowPlan`; do not
reimplement that existing behavior as a new optimization.

The isolated gate now calls the production `ascend_paged_attention_from_bf16`, not a probe-local copy. Ratio-1/ratio-2 batch-4, disjoint shuffled pages, strided arenas, initial Graph replay, and changed-input Graph replay all match compact attention with zero numerical difference. Whole-call isolated timings improve from 1.4852 to 0.8519 ms (ratio 1) and from 1.6291 to 0.9492 ms (ratio 2). The BF16 mirror writer also passes real-NPU quantized-roundtrip, duplicate/inactive-slot, strided-page, and Graph replay checks.

Adding the mirrors to the existing packed arena required a geometry change, not just extra byte accounting. Simply retaining the quantized packing violates compressor-tail padding; the generic ratio solver still exceeds its budget, and iterative inflation of every group causes alignment growth and is rejected. The accepted NPU packing is `{SWA:1, R2:15, R1:45, TAIL:162}` with a 4,147,200-byte parent and less than 5% padding in every group. The final native behavior suite is 145 passed and 60 skipped (`outputs/v41-debug/runtime-direct-test-native.log`).

Full-model Graph A/B is now closed by `20261114T204219Z-48473-cluster2`: readiness, Paris correctness, two complete 2048-token rounds, and zero request errors all passed. Direct attention reaches 19.6707 output tokens/s and 0.3703s mean TPOT versus compact Graph's 14.3756 and 0.5282s (+36.83% throughput, -29.91% TPOT). Against vLLM's 17.8043 and 0.4143s, direct is +10.48% throughput and -10.62% TPOT. TTFT is distribution-dependent: direct improves the all-row mean and p90 over vLLM (4.9414s vs 5.7464s; 6.4573s vs 20.7216s) but not p50 (4.7977s vs 0.7744s). Direct also regresses TTFT over compact Graph (mean 4.9414s vs 4.0905s), and mirror packing lowers cache token capacity from 4,812,287 to 1,180,287 at the same arena budget. Further TTFT or cache-capacity work must be justified by a new measured dominator; do not claim uniform TTFT superiority or long-context capacity from this short-context result.

A combined global writer candidate (pack once, write packed, unpack those rows, write BF16) passed the old-versus-new zero-difference and Graph-update checks; the native runtime suite also remained 145 passed/60 skipped. It was nevertheless rejected: at 128/512/2048 rows its mean speedup was only 1.0043x/1.0075x/1.0028x (`outputs/v41-debug/combined-global-writer-result.json`). Quantization/packing, not packed-cache readback, dominates this chain. The experimental production branch was reverted and no full-model restart was spent on it.

## Measured update — three-dimensional BF16 ND mirror writer

The combined-writer negative result left the flat uint8 mirror transfer as the
next measured target. The old path split every 4096 bytes into a separate ND
update to avoid the CANN 9.2.1 silent-corruption boundary. The new NPU BF16
branch keeps the same packed-row read-back and duplicate/inactive-slot winner
logic, but calls ND update on the three-dimensional BF16 cache with `[page,row]`
indices and 512-channel BF16 rows. This is a different operator geometry; the
flat uint8 path retains its 4096-byte workaround.

The isolated production-code probe matched the previous writer for 4–2048 rows.
At 128/512/2048 rows, eager mean timings fell from
25.1209/94.1137/383.5882ms to 0.3081/0.2983/0.3016ms. The real-NPU regression
then passed 7 cases covering strided arenas, quantized roundtrip, 55/512/2048-row
Graph replay, changed inputs, and duplicate/inactive slots
(`bf16-nd-regression.log`). Native runtime behavior remained 145 passed and 60
skipped. CPU/CUDA and NPU FP16/uint8 paths are unchanged.

`20261114T214015Z-bf16nd-live` applied the two codec files to both nodes,
completed HTTP-200 forward reload, warmup, and Graph recapture in 68.4s, and
passed readiness and Paris correctness. The first two post-reload benchmark
sets had 58 and 56 prompt tokens per request because their unique tags tokenized
differently; they are not used for strict comparison. The accepted run is
`20261114T214831Z-48475-strict`: every request has exactly 55 prompt and 128
output tokens, both rounds complete 2048/2048 tokens, and request errors are
zero.

| Metric | Previous direct Graph | BF16 ND writer | vLLM-Ascend |
|---|---:|---:|---:|
| Output throughput mean | 19.6707 tok/s | **23.2870 tok/s** | 17.8043 tok/s |
| TPOT mean | 0.3703 s | **0.3168 s** | 0.4143 s |
| TTFT all-row mean | 4.9414 s | **3.7110 s** | 5.7464 s |
| TTFT p50 | 4.7977 s | **3.0865 s** | 0.7744 s |
| TTFT p90 | 6.4573 s | **5.1849 s** | 20.7216 s |

Thus the final writer adds 18.38% throughput and removes 14.44% mean TPOT over
the previous direct implementation. Against vLLM it is +30.79% throughput,
-23.52% mean TPOT, -35.42% mean TTFT, and -74.98% TTFT p90. TTFT p50 remains
3.9857x vLLM, and mirror cache-token capacity remains 1,180,287. The strict
machine-checked comparisons are `direct-bf16-nd-ab.json` and
`tokenspeed-vs-vllm.json` in the accepted run directory. A fresh full-weight
start after this final writer change has not yet been performed; do not convert
the successful hot reload into a fresh-start claim.

## Measured update — Ascend Mooncake ABI and PD preparation

The official `mooncake_transfer_engine_npu-0.3.13` wheel resolves against CANN
9.2.1 without CUDA dependencies. The local `mooncake/` adapter implements the
TokenSpeed ABI version-1 entry points and the Rust `PageGrid` layout. Real NPU
probes have passed:

- 2 MiB-aligned device-memory registration and unregistration;
- direct `write` and multi-field, non-contiguous `write_pages` from NPU7 to
  NPU6 on `.110`, with all 8 MiB exactly equal to an independent expectation;
- the same two operations from `.110/NPU6` to `.109/NPU0` over Ascend Direct,
  again with zero differing bytes;
- the production Rust `tokenspeed_mooncake::Library::load` path.

Standalone Rust `Engine::start` is not part of that success boundary: an
independent process lacks Torch's Ascend context, and a hand-written ACL probe
segfaulted during ADXL initialization. Request-level validation must therefore
use the real Torch/engine process rather than infer success from the loader.

The native runtime is packaged as
`outputs/v41-debug/tokenspeed-mooncake-ascend-runtime.tar.gz`; an independent
container verified all seven ABI exports, `$ORIGIN` dependency resolution, and
`mooncake.engine.TransferEngine` import. `run_v41_pd_cluster.py` prepares the
full four-node 1P1D topology and explicit `ascend_direct` environment. Latest
local-only prepare `20261114T220007Z-50342-pd1p1d-prepare` and the complete
lightweight NPU automation suite pass (46 tests). After a real PD benchmark,
the runner writes `pd-vs-nonpd.json` against the strict 55/128-token non-PD
reference and reports a primary benefit only when output throughput increases
and mean TPOT decreases. Restoration now attempts every stopped workload and
fails the run if any restore fails, while retaining partial restore evidence.
No PD request or performance result exists yet. `.110` runs unrelated Qwen3
vLLM/PD workloads: NPU 0–5 each use about 57GB/64GB and only NPU 6–7 are idle.
The V4.1 decode role requires two full EP16 nodes, so coexistence on the two
free cards is not a valid alternative. Execution requires explicit replacement
authorization; both authorization prompts timed out and no container was
stopped. The runner defers cleanup on a startup timeout while model weights
may still be loading.

## Device ledger ceiling (2026-11-17)

The decode gap is not a compute shortage. On rank0, a steady pure-decode step
costs `330.0 ms` stage with `318.7 ms` of kernel time, and the kernel groups
break down as `data-movement/index/layout 154.9 ms (46.9%)`,
`elementwise 63.5 ms (19.3%)`, `MoE dispatch/combine 32.4 ms (9.8%)`,
`other 32.6 ms (9.9%)` and `matmul/attention 16.5 ms (5.0%)`.
Gathering KV, materializing BF16 mirrors and re-laying-out indexer scores cost
more than nine times what the matmuls do. See
`outputs/v41-debug/device-ledger-20261117.md` and
`outputs/v41-debug/benefit-ledger-20261117.json`, produced by
`outputs/v41-debug/build_v41_benefit_ledger.py` from
`outputs/v41-debug/pure-decode-profile-20261116T034256Z-415/pure-decode-summary.json`.

Amdahl limits that follow from the same measurement, assuming removed kernels
become free: dropping all data-movement lands at `175.1 ms (1.89x)`;
dropping data-movement and elementwise lands at `111.5 ms (2.96x)`; dropping
MoE communication lands at `297.6 ms (1.11x)`. The frozen baseline gap is
`8.39x` in TPOT (`267.813 ms` versus `31.887 ms`), so the whole identified
structural work, taken at an unattainable zero cost, still misses the `150 ms`
milestone and stays far from `80 ms`. Reaching `80 ms` requires removing about
`287 ms` of the `318.7 ms` kernel sum, which only a rewrite of the decode path
onto native Ascend fused operators can deliver.

A vLLM-side device kernel ledger could not be produced this round and must not
be implied: `--profiler-config` yields front-end-only kineto traces
(27450 of 27515 events are `python_function`, with no device kernels), and
`msprof --dynamic=on --pid=<worker>` rejects live workers with
`no valid pid values, stop profiling`. Obtaining the same `kernel_details.csv`
format would need `msprof --application` wrapped around a fresh vLLM start, and
rank0 has `4.3G` free of `439G`. Cross-engine attribution therefore rests on
request-level TPOT, which still bounds vLLM's whole device step at `31.9 ms`
against TokenSpeed's `302.0 ms` union busy time.
The one remaining lever is scoped in
`docs/ascend-v41-w8a8-native-decode-path.md`: TokenSpeed materializes BF16 KV
mirrors because the recipe assumes CANN rejects quantized pages, but this
image ships `quant_sparse_flash_mla` and `quant_lightning_indexer`, and
vLLM-Ascend already calls them. That routing change is bounded at the ledger's
`1.89x` ceiling (TPOT around `175 ms`), not at `80 ms`.
