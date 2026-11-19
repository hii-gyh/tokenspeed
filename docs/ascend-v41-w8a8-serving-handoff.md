# Ascend V4.1 W8A8 serving — tuning handoff

> **Start here for remote access and test execution:** `docs/ascend-v41-w8a8-access-and-test-runbook.md` contains the current SSH, container, launch, benchmark, correctness-gate, profiling, artifact, and safety instructions. This document retains the detailed chronology and evidence history.

**Location:** `docs/ascend-v41-w8a8-serving-handoff.md` (moved from `%TEMP%\e2e_v41_HANDOFF.md`)
**Branch:** `cursor/ascend-v41-w8a8-serving` (base `main`)
**Goal:** TokenSpeed vs vLLM-Ascend first on the physically viable **DP2 / TP8 / EP16** two-node topology, then the four-node **DP4 / TP8 / EP32** target and PD if its native runtime is available.
**Target model:** DeepSeek-V4.1-Flash-w8a8
**Remote:** direct key-only SSH to `39.156.164.107` (rank0, `192.168.11.2`) and `39.156.164.108` (rank1, `192.168.11.3`); `.109/.110` are not part of the current two-node run.
**Deploy tree:** run-specific `/root/gyh_dev/tokenspeed_cluster/<run-id>` mounted as `/ts`
**Probe/logs:** `outputs/v41-debug/<run-id>-cluster2/`
**Automation:** `test/ci/npu/run_v41_debug.sh`, `run_v41_cluster.py`, and `run_v41_benchmark.py`
**Python (driver):** WSL `.venv-wsl`; Torch/torch_npu run only in remote containers
**HEAD:** `b6beba56e` plus the uncommitted adapter, recorder, HCCL, benchmark, automation, and documentation changes described in the retuning plan
**JSON/profile artifacts:** `outputs/v41-debug/` (valid TS/vLLM fixed-load benchmarks, comparison JSON, full-model event samples, and isolated NPU probes)

---

## 2026-11-16 corrected hot-Graph acceptance (supersedes the old target-2 conclusion)

The earlier 30.79% throughput win compared TokenSpeed NPU Graph with a vLLM
**eager** baseline and included a short first-use warmup. It must not be used as
the current performance conclusion. `bench_v41_steady.py` now defaults its
concurrent warmup length to the measured `--max-tokens`; the acceptance below
uses 16 warmup requests x 128 tokens, five measured repeats, unique prompt tags,
and the same DP2 / attn TP8 / dense TP8 / MoE TP1 / EP16 Graph topology.

| Engine / path | Throughput mean | TTFT p50 | TTFT p90 | TPOT mean |
|---|---:|---:|---:|---:|
| TokenSpeed fresh Graph + SMG | 23.6990 tok/s | 2.5921 s | 4.7363 s | 0.3157 s |
| vLLM-Ascend fresh Graph, direct | 207.1864 tok/s | 0.7802 s | 1.1080 s | 0.03264 s |
| vLLM-Ascend fresh Graph + same SMG | 208.5003 tok/s | 0.8232 s | 1.1219 s | 0.03205 s |

Every accepted round produced 2048/2048 completion tokens with zero request
errors. vLLM direct versus SMG changes throughput by only +0.63%, so the gateway
does not explain the engine gap. Against vLLM with the same SMG version and a
single-worker `passthrough` policy, TokenSpeed throughput is 0.1137x and TPOT is
9.85x. **The aggregated performance target is currently unmet; PD work is paused
until the aggregated direct-Graph gap is understood.** Sources:
`agg1-hot-tokenspeed/summary.json`, `agg2-hot-vllm-ascend/summary.json`,
`agg4-hot-vllm-ascend/summary.json`,
`agg1-hot-tokenspeed/tokenspeed-vs-vllm-graph-smg.json`, and
`agg4-hot-vllm-ascend/vllm-direct-vs-smg.json`.

Correctness is a separate remaining gate. The vLLM Paris probe passed before
both accepted measurements, but later repeated probes sometimes produced
degenerate text; TokenSpeed's fresh pre-run probe was clean (` Paris. \n`).
Do not interpret token-count completion alone as model-quality equivalence.
Add a post-measurement correctness check before claiming production-quality
performance parity.

The twelve earlier PD runs remain valid negative results: the best PD
throughput was 21.0309 tok/s and the best PD TPOT was 0.3046s in different runs,
neither improving the then-current non-PD operating point as a whole. They also
used the old vLLM-eager comparison context and must not be presented as a win
over the new Graph baseline.

### Direct-profile scope correction

A second fresh TokenSpeed run reproduced the negative aggregate result at
23.4362 tok/s and 0.317973s mean TPOT, with pre/post Paris checks passing
(`agg5-hot-tokenspeed/summary.json`). Its direct rank-0 profile is complete, but
the first ten captured steps are **mixed prefill+decode Graphs**, not pure steady
decode. Model 46 has four current decode rows and Model 49 has one; the prefill
rows account for the large history operations. Therefore these reports cannot
yet attribute the pure-decode TPOT gap to a particular cache operation.

For mixed Model 46 steps 5–9, the per-step means are 285.294ms Computing,
20.285ms non-overlapped communication, 28.034ms Free, and 333.614ms Stage. Its
GatherV3 total is 69.950ms/step, dominated by 48.755ms and 16.258ms groups of
large BF16-history reads. The source candidate is the NPU mirror maintenance in
`write_global()`: replay/old rows are masked out of quantized writes but still
participate in mirror gather/scatter. This is a mixed-path optimization
candidate, not yet a pure-decode root cause or an end-to-end win. Mixed Model 49
steps 0–1 average 195.302ms Computing, 37.743ms non-overlapped communication,
48.071ms Free, and 281.117ms Stage. See
`direct-graph-profile-20261115T223810Z/steady-decode-summary.json` and
`steady-decode-model49-summary.json`; the filenames are retained for artifact
compatibility even though subsequent Model-ID analysis established mixed mode.

A pure steady-decode capture is still required. After the first profile, SMG's
ZMQ health probe continually failed and later start attempts could not complete
the backend handshake; the affected containers were stopped normally. A later
fresh replacement `20261116T034256Z-415` passed two-node readiness after
1784.96s, both containers are running, and `/v1/models` returns the model list.
No steady-decode profile, benchmark result, or new performance benefit has yet
been produced from this replacement.

---

## Hard constraints (do not violate)

1. **Remote-only NPU** — no local Ascend.
2. **No reboot thrash** — weight load ~12–22 min; one controlled restart max per measured win.
3. **No** `fuser` / `npu-smi` reset (killed SSH / hygiene disasters).
4. **No mid-load** `nerdctl rm`.
5. **FlatKV on Ascend = torch SPECIALIZED** — Triton codec capability pinned to `{nvidia,amd}` only. **No Triton/TileLang Ascend FlatKV codec.**
6. **No select `.item()` trim** — breaks CUDA/NPU graph capture (sync during capture).
7. **Never `/start_profile` VIZTRACER on live DP4** without a recover plan — it **fatal’d the mesh** (all ranks Exited).
8. Prefer `--allow-reload-forward` + `ts reload-forward` for kernel/forward iteration; verify the run entry actually enables it. The current direct-Graph run does, and `nerdctl exec` must explicitly receive the service `PYTHONPATH`.
9. Scrape RANK0 with host path `PROBE=/root/gyh_dev/token_speed_dsv41flash` (not container `/probe` over SSH).
10. Gate: **≥15%** measured win before another DP4 restart / Phase D graph.

---

## Topology & baseline competitor

| Engine | Topology | Mode | Throughput | TTFT p50 | TPOT mean |
|--------|----------|------|------------|----------|-----------|
| vLLM-Ascend | DP2/TP8/EP16 | eager, async scheduling | **17.8043 tok/s** | **0.7744 s** | **0.4143 s** |
| TokenSpeed compact attention | DP2/TP8/EP16 | NPU Graph | **14.3756 tok/s** | **3.7655 s** | **0.5282 s** |
| TokenSpeed direct BF16 pages | DP2/TP8/EP16 | NPU Graph | **19.6707 tok/s** | **4.7977 s** | **0.3703 s** |
| TokenSpeed direct + 3D BF16 ND writer | DP2/TP8/EP16 | reloaded NPU Graph | **23.2870 tok/s** | **3.0865 s** | **0.3168 s** |

All rows use the canonical 16-request, concurrency-8, 55-prompt-token,
128-output-token, two-repeat streaming load. The final strict BF16-ND run improves
throughput over vLLM by 30.79% and reduces mean TPOT by 23.52%; mean TTFT across
all request rows falls from 5.7464s to 3.7110s and TTFT p90 from 20.7216s to
5.1849s. TTFT p50 remains a regression against vLLM (0.7744s to 3.0865s), so
target 2 is not uniformly better at every percentile. Against the preceding
direct-Graph implementation, the final writer improves throughput by 18.38%,
mean TPOT by 14.44%, mean TTFT by 24.90%, and TTFT p50 by 35.67%. Sources:
`20261114T214831Z-48475-strict/summary.json`,
`20261114T214831Z-48475-strict/tokenspeed-vs-vllm.json`, and
`20261114T214831Z-48475-strict/direct-bf16-nd-ab.json`. The final change was
loaded through `reload-forward` and Graph recapture; a fresh full-weight restart
after this specific writer change has not yet been run.

---

## Clean execution order (adopted)

```
A  OOM → eager READY
B  RANK0 baseline (same-decode scrape; p50 AND mean)
C  Fix by measured dominator (≥15% gate)
C' Next residual bucket (same gate)
D  Graph validated; live NPU-kernel attribution in progress
```

Plus B2/B3: fix unexplained math; align vs vLLM stages.

---

## Chronology & what shipped

### Early lean ladder (pre Phase A–C)
- Split DECODE: fill / settle / model_fwd (`TOKENSPEED_LOG_MM_TIMING`).
- V4.1 attn stage timers (`TOKENSPEED_LOG_V41_ATTN`): scatter, gather_swa, gather_global, sparse_mla, select_global.
- TileLang Ascend FlatKV probes **failed** (absent / P1–P3) → abandoned.
- **Pin torch FlatKV** codecs for Ascend (`ca284be12`).

### Step5 FlatKV codec (gather / scatter / select)
- **5a gather SHIPPED:** LUT + safe 1D `_transfer_selected` + `_GATHER_ROW_CHUNK`.
  Historical **gather_global p50 ≈ 2.77 ms** (best reference).
- **5b scatter:** profiles/alts; partial.
- **5c select:** `.item()` trim **reverted** (graph sync). Graph trim failed capture.
- Ledger: `%TEMP%\e2e_v41_step5_ledger.json`, `e2e_v41_step5*_*.json`.

### Phase A — OOM
- Ascend densified `reshape(-1)` on padded `as_strided` FlatKV pool (~16 GiB) → OOM under ~11 GiB free.
- Fix: `_as_flat1d` storage view (no densify). → READY + Paris.
- Artifact: `e2e_v41_phaseA_oom.json`.

### Phase B — baseline
- Full RANK0 scrape: model_fwd p50 ≈ **1864 ms**.
- Per-call hot: `gather_global` p50 **7.12** (regressed vs 5a **2.77**).
- Extrapolate FlatKV×40L p50 sum ≈ **663 ms**; **unexplained ≈ 1200 ms**.
- Artifact: `e2e_v41_phaseB_baseline.json`, `e2e_v41_phaseB_rank0_raw.txt`.

### Phase C — gather fix (shipped + measured)
- Code: `_cache_flat1d` — dense → `cache.view(-1)`; padded → `_as_flat1d`; fused planar read (one gather).
- Commit: **`53f58ea12`** (+ tests in `test_dsv41_cache_codec_torch.py`).
- Measured: gather_global **7.12 → 6.12** (−14%, **under 15% gate** for more thrash). Still ≫ 5a 2.77.
- model_fwd **1864 → ~1821**.
- Artifact: `e2e_v41_phaseC_gather.json`. **Keep the code** (already live); gate only blocked further restarts/Phase D.

### B2 — attribution (corrected math)
- model_fwd p50 ≈ **1819**; FlatKV Σ p50×40 ≈ **602 ms (33%)**; unexplained p50 ≈ **1217 ms (67%)**.
- Also report mean-based sums (p50 underestimates; select p50/mean skew ×30+).
- Live VIZTRACER → **mesh fatal** → all `dsv41-ts-dp` Exited.
- Prior profiles (prefer `after_silu`): residual looked like shared_experts / Silu / clamp — later proven **stale** for Silu.
- Artifact: `e2e_v41_b2_attribution.json`.

### B3 — vs vLLM
- Stored vLLM only; TS ITL estimated ~4.2× worse.
- Artifact: `e2e_v41_b3_ts_vs_vllm.json`.

### Recover + C' Silu
- One controlled eager reboot (~23 min) → READY + Paris smoke. Serve left **Up**.
- All `/ts` nodes already on NPU `silu_and_mul(..., limit=...)`; `forward_native`/`clamp_max` = 0 in after_silu profile.
- Prior Silu fix rebench ~**0.994×** — **no ≥15% left**. No second restart.
- Post-recover model_fwd p50 ≈ **1799**; unexplained still ~**63%**.
- Artifact: `e2e_v41_cprime_silu.json`.

### C2 — non-Silu residual (stop)
- Live: `update_runtime` ~2 ms; ngram **commit** already fixed historically; `prepare_ngram` huge in viz but mostly **outside** model_fwd.
- Ranked next: **`model_fwd` unexplained non-FlatKV compute** — ≥15% of step, **no zero-restart fix ready** → **STOP**.
- Artifact: `e2e_v41_c2_residual.json`.

---

## Current live state (at handoff)

- The historical four-node `dsv41-ts-dp` deployment and its original host paths no longer exist. Do not assume that service is healthy or restart its old launchers.
- Run `20261114T102752Z-22648-cluster2` completed both nodes' 274/274 checkpoint shards, all 16 model-executor initializations, and the 1024-token eager warmup. It then exited because the pinned native engine expected `HostConfig.ttft_trace_sampling_percentage`, which this branch lacked; the later hot-reload / `npugraph_ex` messages were shutdown noise. No readiness, request, benchmark, or full-model event sample was obtained.
- The minimal integer `0..100` host-config contract is now backported with default `0` (tracing disabled), validation, docs, and tests. The real Ascend image passed all 9 `test_server_args_env.py` tests from an isolated probe. Both previous TS containers were verified absent and all 16 NPUs idle before restart.
- Run `20261114T110514Z-24230-cluster2` passed two-node readiness after 1761.3s. Clean independent run `20261114T114449Z-25773-benchmark-tokenspeed-stream` then passed Paris correctness and two 16-request streaming repeats with 2048/2048 tokens each: mean 5.7577 output tokens/s, TTFT p50 4.4844s, TPOT mean 1.3639s. Unique prompt tags prevented prefix-cache reuse. The earlier independent directory is invalidated by overlap with the original remote benchmark client. Both TS containers have now been stopped and preserved for the vLLM comparison; rank logs and `event-profile.json` are fetched. Five clean DECODE bs=4 samples average 1344.8ms forward, 1203.9ms attention, 905.4ms attention core, and 66.4ms FFN; scopes are nested and the sample count is small.
- The original `.108` `deepseek-v41-dp` container is preserved but exited. Existing vLLM/PD processes on `.109/.110` are unrelated workloads and must not be treated as free capacity. The valid independent vLLM measurement is `20261114T122049Z-26908-benchmark-vllm-ascend-stream`: mean 17.8043 output tokens/s, TTFT p50 0.7744s, TPOT mean 0.4143s under the same fixed load. The earlier `20261114T120032Z-26250-vllm-cluster2` runner was superseded before benchmarking because it checked `/readiness` instead of vLLM `/health`. Both vLLM containers are stopped and preserved. The comparison JSON is `20261114T122049Z-26908-ts-vs-vllm-comparison.json`; TokenSpeed has not achieved the performance target.
- Isolated probes reject live-head padding as a useful kernel optimization (0.1729ms live8 versus 0.1732ms padded64, exact output equality) and bound the real indexer at 3.34–3.57ms/source. The full attention wrapper is about 4.50ms/layer, while current-token SWA scatter alone was 6.22–6.46ms. The production codec now fuses value/scale transfer and avoids AiCpu integer sorting for at most 64 rows; remote tests passed 11 cases and real-module NPU scatter measured 4.3467ms with byte-exact duplicate/inactive-slot behavior.
- Run `20261114T130934Z-28436-cluster2` was the eager two-node service used through the global-gather-reuse A/B and was started with `allow_reload_forward=True`. Its clean benchmark `20261114T134057Z-29163-benchmark-tokenspeed-stream` passed Paris correctness and both fixed-load repeats. Against the old TokenSpeed build it improved throughput by 4.7628% and reduced mean TPOT by 4.6964%; the evidence is `20261114T134057Z-29163-ts-codec-ab.json`. This is a real but insufficient improvement: the latest `20261114T134057Z-29163-ts-vs-vllm-comparison.json` still reports 6.0319 versus 17.8043 output tokens/s and 1.2998s versus 0.4143s mean TPOT. Do not claim the vLLM performance target is met.
- Fine-grained model and backend attention scopes were added without changing timing-disabled behavior. The real Ascend image passed 26 tests with 18 skips. Hot reload is available through `/tmp/ts-native/bin/python -m tokenspeed.cli reload-forward --url http://127.0.0.1:8000`; the image has no `ts` executable, and rank1 has no gateway on port 8000. An initial reload safely refused a missing `layer_id` argument and kept the previous code serving; after the explicit argument fix, reload patched 388 objects and completed warmup/recapture. Backend scopes then identified `attention_gather_global` as the dominant scope.
- Global KV gather/dequant is now memoized on the current `V41Selection`; prepared selections retain their own record, and a new forward must gather again. The scoped-behavior/prepared-wiring tests passed 5 cases and the default attention/timing suite passed 26 with 18 skips. Rank0 reload patched 389 objects and completed warmup/recapture; one earlier wiring error was safely refused with the previous code still serving. Canonical A/B `20261114T145023Z-31677-ts-global-memo-ab.json` shows throughput 6.0319→6.8591 (+13.7125%), mean TPOT 1.2998→1.1432s (-12.0501%), TTFT p50 -8.7416%, with Paris correctness and two complete 2048-token repeats. The comparison `20261114T145023Z-31677-ts-vs-vllm-comparison.json` still reports only 0.3852x vLLM throughput and 2.7595x mean TPOT, so target 2 remains unmet. Ten post-reload bs4 samples average 1131.6ms forward and 134.3ms global gather; scopes remain nested and are attribution evidence, not additive wall time.
- Follow-up probes found no useful slot-order or padded-stride lever: sequential/shared-random/per-request-random slots are effectively equal, and the tested padded gather matches contiguous storage (`gather-breakdown-result.json`). A hole-safe all-selection cumsum/scatter candidate matched rows, lengths, candidate blocks, and physical slots exactly but was slower (1.8667 vs 1.6927ms); it was not promoted (`all-selection-run.log`). The simple arange/mask variant is invalid for hole pages and must not be used.
- Both containers of run `20261114T130934Z-28436-cluster2` are now stopped and preserved. The first NPU Graph attempt, `20261114T151624Z-32379-cluster2`, loaded all weights but exited during its first bs4 capture because `tokenspeed.runtime.accelerators` was absent; no readiness or performance result was obtained. A minimal CUDA/NPU graph-class contract and four behavior tests now pass in the real Ascend image.
- Graph retry `20261114T154815Z-33154-cluster2` passed readiness in 1904.6s and Paris correctness. In `20261114T162024Z-33753-benchmark-tokenspeed-stream`, repeat 0 completed 2048 tokens at 13.9536 output tokens/s with 0.5523s mean TPOT; repeat 1 then failed in its second batch with CANN index assertions, ACL 507011 and scheduler exit. Both graph containers are stopped and preserved. The single successful repeat is not a valid two-repeat acceptance result and the automatic TS/vLLM comparison was not generated.
- `outputs/v41-debug/probe_v41_graph_codec.py` reproduces the index assertion on one NPU with the real 4.38GB strided SWA arena, without model weights. The final codec uses `index_select` reads and Ascend-native ND-scatter writes; `index_copy_` and advanced-index writes were rejected because replay did not update the external cache. A separate graph right-shift defect changed quantization exponents, so runtime bit extraction now uses floor-divide/multiply. All 14 behavior tests pass, and SWA/global/index each pass 100 read/write replays with no mismatches (`graph-codec-validation.json`). This clears the single-card gate, not full-model acceptance; the third two-node graph run and its two-repeat benchmark are the next gate.
- Graph run `20261114T171508Z-38974-cluster2` passed readiness in 1816.1s and remained free of device assertions, but its first benchmark was invalid: every fixed-load request emitted EOS as its first token, so streamed chunks and TTFT/TPOT were absent despite 2048/2048 nominal completion tokens. Prompt probes showed 5/6-token prompts working, 19/33-token counting failing initially, and several France/Hello prompts returning EOS. SWA pack stages at 4/9/19/33/55 rows all matched eager, excluding quantization as the remaining cause.
- An independent legacy-write/packed-truth probe then proved that CANN 9.2.1 `npu_scatter_nd_update_` silently corrupts updates beyond 4096 bytes. The old advanced-index path matched packed truth; ND scatter began diverging at the same byte boundary. `_ascend_codec.py` now chunks updates to 4096 bytes. SWA/global/index are byte-exact against both legacy writes and packed truth at 4/9/19/33/55/128 rows on the real 4.38GB strided arena (`scatter-semantics-*-fixed.log`), and `tokenspeed-kernel/test/npu/ops/test_dsv41_ascend_codec.py` passes all three large-payload eager+graph cases. Rank0 forward reload returned HTTP 200, patched the helper, and completed warmup/recapture.
- Valid post-fix graph benchmark `20261114T181916Z-41918-benchmark-tokenspeed-stream` passed Paris correctness, both 2048-token repeats, zero errors, and 128 streamed chunks for every request. It measured 14.3756 output tokens/s, 3.7655s TTFT p50, 6.0743s TTFT p90, and 0.5282s mean TPOT. Against the canonical eager result this is +109.5850% throughput and -53.7925% mean TPOT, although TTFT p90 regressed 35.2644% (`20261114T181916Z-41918-graph-vs-eager-ab.json`). Against vLLM, throughput is 0.8074x, mean TPOT is 1.2751x, and TTFT p50 is 4.8626x; only TTFT p90 is lower (0.2931x), so target 2 remains unmet (`20261114T181916Z-41918-graph-ts-vs-vllm-comparison.json`).
- After the scatter fix, 19/33-token counting prompts recovered, but 10/20-token France questions and repeated Hello still returned first-token EOS. These cannot yet be classified as a TokenSpeed defect without a same-prompt reference. The existing `.109` vLLM service accepted `/v1/models` on port 7100 but its first completion probe ended in DP/Gloo peer disconnects and API shutdown; all 7100–7103 endpoints then disappeared. Its container logs are retained and it has not been restarted. Do not use the failed probe as model-quality evidence or disturb the other vLLM/PD workloads on `.109/.110`.
- NPU kernel profiling support now selects `torch_npu.profiler` with Level1 and PipeUtilization, advances only on committed forwards, and exports one artifact set per prefill/decode session. Isolated regression-7 passed 35 tests with one optional-dependency skip. Its real-NPU smoke captured both eager prefill and an already-captured NPU Graph decode replay; both reports contain valid `kernel_details.csv`, operator/API statistics, and `trace_view.json` (`npu-profiler-regressions-7.log`, `npu-profiler-smoke-result.json`).
- The first live profiling attempt is invalid. `reload-forward` returned HTTP 200 but explicitly refused to replace `runtime.engine.profiler` and `runtime.execution.device`, so the running server still used its old CPU/CUDA profiler. That profiler attempted to create `/ts/npu-profile` on the read-only source mount, raised `OSError: Errno 30`, and terminated rank1's data plane. Benchmark `20261114T184219Z-42771` therefore has HTTP 503 correctness and no runs; it must not be used for performance comparison or kernel attribution. `/v1/models` continued to return the static model list and was not a readiness check.
- The failed containers were stopped normally and removed after the engine-core fatal, without `rm -f` or an NPU reset. Fresh graph run `20261114T184612Z-43017` loaded the control-plane changes at process start and became ready. Its no-profiler gate passed Paris correctness and two complete fixed-load repeats, measuring 14.0294 output tokens/s and 0.5362s mean TPOT. The subsequent live capture generated complete prefill/decode reports for every rank on both nodes (`live-npu-profile-20261114T184612Z-43017/profile-inventory-*.log`). The 10-step schedule auto-stopped before the explicit stop call, so `stop-response.log` reports “Profiling is not in progress”; this is not missing-profile evidence. Full reports are large (13 GB on `.107`, 9.4 GB on `.108`), so analysis downloads only `scheduler_tp0_ep0_dp0`'s compact CSV artifacts rather than all traces.
- The compact rank-0 summary is complete at `live-npu-profile-20261114T184612Z-43017/rank0-profile-summary.json`. Exact shapes map the dominant reads to SWA (528-byte rows), global (288-byte rows), and index (68-byte rows) gathers plus dequantization. The two files are prefill-triggered and decode-triggered windows, not guaranteed phase-pure spans; static/dynamic, nested AICPU, and stream times are not additive wall time. In particular, the 16.7-second MoE dispatch outlier and its AICPU wrapper are one boundary wait, not a steady per-call kernel cost.
- Source comparison found the main structural difference: vLLM's non-A5 910B3 `AscendDSAV41Impl._forward_attention` passes the original paged caches, block tables, sparse indices, lengths, and `cmp_ratio` directly to `npu_sparse_flash_mla`; TokenSpeed first gathers selected slots into compact BF16 blocks on every layer. The default non-A5 cache specs use BF16 SWA/global storage, while A5 packed-quant `qsmla` is a separate hardware path. A real isolated CANN 9.2.1 probe rejected TokenSpeed's uint8 pages before tiling (`ori_kv` accepts only FP16/BF16), so direct quantized-cache attention is ruled out on this ABI. vLLM MRV2 tail-128 bounded replay is enabled in the baseline, but TokenSpeed's existing `V41DecoderView`/`V41RowPlan` performs the corresponding CED decoder narrowing; it is not a missing optimization to duplicate.
- The production `ascend_paged_attention_from_bf16` path now reads dequantized BF16 SWA/global mirrors through the existing LCM block tables without materializing compact rows. Its own implementation—not a probe-local copy—passes ratio-1 and ratio-2 batch-4 contiguous, page-strided, initial Graph replay, and updated-input Graph replay comparisons against compact attention with `max_abs=mean_abs=0`. Latest isolated whole-call timings are 1.4852→0.8519ms (ratio 1) and 1.6291→0.9492ms (ratio 2); these are not end-to-end results (`direct-bf16-matrix-r1-result.json`, `direct-bf16-matrix-r2-result.json`).
- Ascend cache recipes add 40 `swa_bf16` and four `global_bf16` fields and include their 44KiB/token overhead in `lcm_block_bytes`; non-NPU layouts are unchanged. The packed fields remain the quantization/index representation. `cache_scatter_bf16_torch` reuses the duplicate/inactive-slot, strided-page and 4096-byte Graph-safe transfer path. Its real-NPU regression passes quantized-roundtrip writes, inactive slots, strided pages, Graph replay, and changed-input replay (`bf16-mirror-codec-test.log`, 4 passed including the three pre-existing payload cases). SWA uses the existing fused writer's dequantized output; global currently gathers only newly written owner rows to guarantee the same quantized roundtrip before writing the mirror.
- Full-model direct-attention A/B passed in `20261114T204219Z-48473-cluster2`: readiness in 1871.9s, Paris correctness passed, both Graph rounds completed 2048/2048 tokens with zero errors and 128 streamed chunks per request. The final native behavior gate reached 145 passed and 60 skipped (`runtime-direct-test-native.log`). The NPU packing is explicit `{SWA:1, R2:15, R1:45, TAIL:162}`, yielding a 4,147,200-byte parent with less than 5% padding in every group. Direct throughput is 19.6707 tok/s and mean TPOT is 0.3703s, beating the valid vLLM baseline by 10.48% and 10.62%, respectively. TTFT mean/p90 also improve over vLLM, but p50 remains slower; the mirror increases TTFT relative to the old compact Graph path and reduces cache token capacity from 4,812,287 to 1,180,287 at the same arena budget. The service remains running for follow-up measurements.
- The direct mirror writer's remaining dominator was the flat uint8 ND update split into 4096-byte launches. An isolated three-dimensional BF16 ND update instead selects `[page,row]` and writes one 512-channel BF16 row per update. It matched the previous writer at 4–2048 rows and reduced 128/512/2048-row eager timings from 25.1209/94.1137/383.5882ms to 0.3081/0.2983/0.3016ms. Production preserves the existing quantized read-back, inactive-slot, and duplicate-winner semantics; only the final mirror write changes. The real-NPU gate passes 7 tests, including 55/512/2048-row strided arenas, duplicate/inactive slots, and changed-input Graph replay; the native runtime suite remains 145 passed and 60 skipped.
- Live run `20261114T214015Z-bf16nd-live` backed up both remote codec files on each node, reloaded 2 target plus 8 dependent modules, patched 484 objects, warmed up, and recaptured Graph in 68.4s. Readiness and Paris correctness passed. Two early follow-up runs used 58- and 56-token prompts and are retained only as stability evidence, not strict comparisons.
- Strict run `20261114T214831Z-48475-strict` restored exactly 55 prompt tokens and 128 output tokens for every request. Both rounds completed 2048/2048 tokens with zero errors. Throughput is 23.2870 tok/s, mean TPOT 0.3168s, all-row mean TTFT 3.7110s, TTFT p50/p90 3.0865/5.1849s. Against vLLM these are +30.79% throughput, -23.52% mean TPOT, -35.42% mean TTFT, and -74.98% TTFT p90; TTFT p50 is still 3.9857x vLLM. Cache-token capacity remains 1,180,287. The service is running the hot-reloaded code; fresh full-weight startup validation is still pending.
- Cleanup removed only this run's non-rank0 `trace_view.json` files on `.107`, preserving both rank-0 traces and every CSV. The root filesystem still reports 100% with zero unreserved available space, although Python temporary-file creation recovered. The current service was rechecked after cleanup: `/readiness` returns HTTP 200.
- **Do not** VIZTRACER on a live mesh. Use the event recorder and isolated probes first.

---

## Numbers cheat-sheet (RANK0, eager, ATTN TP RANK 0)

| Metric | Value |
|--------|-------|
| model_fwd p50 | ~**1800 ms**/decode step |
| settle | ~3 ms/step |
| gather_global p50 | ~**6.1 ms** (debt vs 5a **2.77**) |
| FlatKV timed Σ p50×40L | ~**600–670 ms** (~33% of model_fwd) |
| Unexplained | ~**1100–1200 ms** (~63–67%) |
| vs vLLM ITL | TS ~**4.2×** slower |

**FlatKV debt (non-blocking):** contiguous view+fuse helped mildly; did **not** restore 5a gather. Absolute ceiling if restored ≈ +130 ms/step — not the main lever.

---

## What is NOT the main lever anymore

- Triton/TileLang FlatKV on Ascend
- Another gather/scatter codec thrash without new evidence
- Silu / `forward_native` / `clamp_max` (already on NPU path)
- ngram **commit** (prior fix; mm_timing-light)
- select `.item()` trim
- Live VIZTRACER for attribution

---

## Recommended next work (for successor agent)

### 1) Instrument model_fwd (highest priority)
**Superseded:** the initial event-based implementation now lives in
`runtime/execution/layer_kind_timing.py`; usage and the revised optimization
order are in `docs/ascend-v41-w8a8-retuning-plan.md`. It records eager scopes
and reads them only after non-blocking completion queries on later forwards.
Do not add per-stage or per-step stream synchronization for this attribution.

The remaining instrumentation work is to split dispatch, grouped GEMM, and
combine inside `moe_routed` when the first event sample identifies that scope
as the dominator.

Historical recommendation before that implementation:
- attn (already have v41_attn_ms)
- MoE routed / shared
- engram
- dense proj / gemm
- collectives (without `.item()` / host sync in graph path)

Ship → **one** controlled eager boot (or enable `--allow-reload-forward` on next boot) → same-decode scrape → recompute unexplained.

### 2) Pick new C' by ≥15% of model_fwd
Only then change kernels. Candidates after timers (hypothesis only):
- Attention body beyond FlatKV codecs (sparse MLA already small ~1.3 ms)
- Wide-EP / int8 MoE path
- Engram host/device path
- Unaccounted sync / framework inside forward

### 3) prepare_ngram (secondary, E2E wall)
Large in old viztracer idle/fill; small in mm_timing `update_runtime`. Microbench on **free NPU** if serve stopped; don’t confuse with model_fwd gap.

### 4) Phase D graph
Only after eager shows a clear win. Re-validate no host sync in capture (lesson from 5c).

### 5) Re-run B3 live stream_stages
When serve stable; compare to `e2e_vllm_stage.json` (or re-bench vLLM if NPUs free).

The two-node same-contract comparison is complete; do not rerun it merely to fill this former gap. The original TokenSpeed baseline reached 0.3234x vLLM throughput, 3.2922x mean TPOT, and 5.7909x TTFT p50. After the codec optimization those ratios are 0.3388x, 3.1376x, and 5.5638x; after per-forward global gather reuse they are 0.3852x, 2.7595x, and 5.0774x. Rebenchmark after each implemented optimization, using unique prompt tags and the independent benchmark entry. The vLLM runner checks `/health`; TokenSpeed uses `/readiness`.

### 6) Ascend PD native runtime and deployment gate
The generic CPython 3.12 `tokenspeed-mooncake` wheel remains unusable on Ascend because its native libraries require CUDA (`outputs/v41-debug/probe_mooncake_native-result.json`). That blocker has now been bypassed with the official `mooncake_transfer_engine_npu-0.3.13` wheel and a thin TokenSpeed C ABI adapter under `mooncake/`. The adapter implements ABI version 1, including `write` and the five-field `PageGrid` `write_pages` contract. Real Torch/NPU probes passed device-memory registration, NPU7→NPU6 same-host transfers, and `.110/NPU6`→`.109/NPU0` cross-node transfers; both direct and paged writes matched all 8 MiB with zero differing bytes. Production Rust `Library::load` also passed. These results establish the native/transport boundary, not request-level PD serving.

The self-contained runtime tarball passed an independent-container `ldd`, `$ORIGIN`, seven-symbol, and Python-import check. `test/ci/npu/run_v41_pd_cluster.py` renders a four-node 1P1D deployment: `.107/.108` prefill and `.109/.110` decode, each DP2/EP16 behind SMG, with explicit Mooncake `ascend_direct` configuration. The runner writes `pd-vs-nonpd.json` against the strict 55/128-token non-PD reference, defines the primary benefit as higher output throughput plus lower mean TPOT, and returns failure if any original workload cannot be restored. Actual execution requires `--allow-workload-replacement`; `--success-action restore-original` records and restores previously running containers after acceptance.

Four real attempts have now run, replacing the earlier prepare-only status. `20261114T220153Z-50406-pd1p1d` stopped before PD startup and exposed unsafe reuse of stop-plus-remove cleanup for the original TokenSpeed containers; the runner now stops original workloads without removing them. `20261114T220955Z-50658-pd1p1d` loaded all four ranks but failed Mooncake registration at `0x12cb7c770000`. Controlled large-allocation, HCCL, and misalignment probes proved the root cause: Ascend Direct/HCCS registration requires a 2 MiB-aligned device address, while this arena was offset by 1 MiB. The unified `CacheArena` device-region allocation now aligns each view to 2 MiB; the native arena test passed, and the seven lightweight V4.1 runner test files passed 33 tests.

`20261114T230053Z-51921-pd1p1d` then registered all prefill and decode Mooncake engines successfully, proving that production fix, but decode dialed `.109:31001` instead of the gateway on `.107:31001` and timed out in ZMQ. The handshake was corrected to the gateway-hosted contract in `docs/serving/disaggregation.md`. `20261114T234254Z-52822-pd1p1d` passed four-node readiness in 2049.9 s and delivered a correctness request to both P and D. Its cold prefill forward took 53.067 s; SMG independently aborted it after its `first_response` silence bound of 3000 ms, returning HTTP 504 and cancelling decode. The resulting transfer-failed messages were cancellation effects, not evidence of a primary KV data-plane failure. This run produced no valid benchmark or `pd-vs-nonpd.json`. The fifth run (`20261115T002907Z-53421-pd1p1d`) proved that setting `SMG_TOKENSPEED_SERVICER_SILENCE_TIMEOUT_SECS=300` alone does not change this headless gateway path: it failed at the same `window_ms=3000`. SMG's own launcher help identifies the matching control as `--worker-wedge-secs`, whose default is 3 seconds and whose description covers in-flight requests producing no token or completion. The gateway now sets `--worker-wedge-secs 300`; the benchmark warmup remains a separate record. Runs 4 and 5 cleaned up their own containers and restored all six original `.110` workloads successfully.

The sixth run, `20261115T010816Z-53548-pd1p1d`, is the first complete request-level PD acceptance. Four-node readiness took 1958.96 s; warmup, Paris correctness, KV transfer, and both fixed-load repeats all succeeded. Each repeat completed 2048/2048 completion tokens with zero errors. Throughput was 12.8084 and 17.2878 tokens/s (mean 15.0481), versus 23.2870 for the strict non-PD reference. Mean TPOT improved from 0.3168 to 0.3046 s (3.85% lower), but TTFT mean/p50/p90 regressed to 20.7583/6.5153/50.6315 s from 3.7110/3.0865/5.1849 s. Consequently `pd-vs-nonpd.json` reports `performance_benefit=false`; runner `status=passed` denotes request correctness and load-contract success, not the separate throughput-plus-TPOT benefit gate. Its four PD containers were removed and all six original `.110` workloads restored.

Rank-0 timeline analysis rules out steady prefill compute as the explanation for the long tail. Warm EXTEND records are about 1.5–3.7 s, while requests remain in bootstrap/remote-prefill for tens of seconds. With `max_num_seqs=8` and attention DP2, each DP rank has only four request slots; skewed P/D placement can leave a new request waiting for an earlier decode generation to finish. The runner now accepts an explicit even `--max-num-seqs` and records it in `summary.json`; all nine lightweight V4.1 automation suites pass 44 tests. Run `20261115T015710Z-53827-pd1p1d` completed with `max_num_seqs=16`: throughput improved to 17.8940 tokens/s and TTFT mean/p50/p90 fell to 11.2940/6.0477/30.9871 s, but TPOT regressed to 0.3546 s and the benefit gate remained false. Its decode DP0 accepted 23 of 34 requests; balanced waves ran near 0.28–0.30 s TPOT, while skewed 7:1 waves ran near 0.36–0.39 s. Run `20261115T024934Z-54026-pd1p1d` then used eight concurrent warmup requests and reached 20.7765 tokens/s, 0.3324 s mean TPOT, and TTFT mean/p50/p90 of 6.0182/6.1502/7.6991 s. Its actual entry did not include dynamic token-load routing; second waves still split 3:5 and 2:6, so the strict benefit gate remained false. Both runs completed request correctness, all 2048-token repeats, PD-container cleanup, and original-workload restoration. Run `20261115T033428Z-54393-pd1p1d` tested `--dp-aware` round-robin rank selection plus sixteen warmup requests, but regressed to 19.0337 tokens/s and 0.3640 s mean TPOT; measured waves still split 6:2 and 5:3, so that policy was rejected. The next run is the first actual execution of `--dp-minimum-tokens-scheduler` on the single-worker path, with the same sixteen-request warmup and formal load.

---

## 2026-11-16 pure-decode index-key reuse

This section supersedes the earlier performance claims in this handoff. The old
"30% faster than vLLM" result compared TokenSpeed NPU Graph with vLLM eager and
did not warm both engines to the same output length. Fresh, fully warmed Graph
runs instead measured TokenSpeed at 23.6990/23.4362 output tokens/s with
0.315679/0.317973 s mean TPOT. Fresh vLLM-Ascend Graph measured 207.1864
output tokens/s direct and 208.5003 through SMG, with 0.032645/0.032047 s mean
TPOT. Direct versus SMG changed vLLM throughput by only about 0.63%, so router
overhead cannot explain the remaining engine gap.

A ten-step pure-DECODE profile on run `20261116T034256Z-415-cluster2` used eight
requests per batch and excluded new prefill from the capture window. Stable
steps 2–9 contain eight large index-key gathers per model step: three
`2304x68` gathers and five `4352x68` gathers, totaling about 64.77 ms/step.
The model has KV owners `[2,8,14,20]` but index sources
`[2,8,14,20,24,28,32,36]`; owner 20 was being gathered and dequantized five
times per forward. The neighboring scale decode, FP4 lookup, and batch matmul
nodes identify these as index-topk scoring reads, not BF16 mirror reads or
scatter-writer readback.

The current patch caches dequantized torch keys on `IndexMetadata` for paged
decode/verify only, keyed by index-cache identity. Reindex layers sharing an
owner reuse the first layer's keys; dense prefill retains no large keys. Public
`dsv41.index_topk` signatures are unchanged. Three real-NPU tests cover eager
reuse, graph replay after cache refresh, and dense-prefill non-retention
(`3 passed`). A five-way graph microbenchmark measured 1.889x on 2304 rows and
1.834x on 4352 rows, with exact tensor equality in both cases. These are
subcomponent results, not end-to-end speedups.

The paired production A/B used the same fresh service and graph configuration,
8 concurrency, 16 requests, 128 output tokens, a 16-request/128-token warmup,
and five measured repeats. Before the patch, throughput was
23.9376/24.3148/24.2695/24.3029/24.3345 tokens/s and mean TPOT averaged
0.311667 s. After reload, throughput was
26.8547/27.7888/27.8772/27.9273/27.9195 tokens/s and mean TPOT averaged
0.265537 s. Thus mean throughput increased 1.14203x and mean TPOT decreased
14.80%; every paired repeat improved. Artifacts are under
`outputs/v41-debug/index-reuse-before/`, `index-reuse-after/`, and
`index-key-reuse-micro.json`.

Text quality remains a separate open gate. Three fixed greedy prompts produced
different full texts before patch, after patch, and after a symmetric rollback
reload, so repeated reloads are not text-reproducible and this A/B cannot assign
the differences to the patch. Paris inclusion still passes as a smoke check,
but long counting/Chinese outputs contain obvious degradation; token-count
completion and Paris inclusion must not be represented as semantic acceptance.
The three NPU tests establish the patched op's numerical behavior, not full
model text quality. A follow-up six-step pure-decode profile confirms that the production path now
performs three `2304x68` gathers and only one `4352x68` gather per step (down
from five). GatherV3 fell from 69.576 to 30.405 ms/step and mean profiler stage
time fell from 329.983 to 280.219 ms (15.08%). Profiler timings are attribution
evidence only; the formal performance result is the five-repeat production A/B
above. Raw and summarized comparisons are under
`outputs/v41-debug/index-reuse-pure-decode-profile-20261116T034256Z-415/`.

PD disaggregation has twelve completed real runs, all with
`performance_benefit=false`; the best observed TPOT was 0.304624 s and maximum
throughput was 21.0309 tokens/s. Functional Mooncake and Rust-loader transport
checks do not change that negative performance result. PD optimization remains
behind aggregate competitiveness in priority.

## 2026-11-16 fresh vLLM-Ascend Graph control

The TS fresh `20261116T142323Z-480-cluster2` fixed-load result remains
26.1550 output tokens/s, mean TPOT 0.274143 s, TTFT p50 3.365783 s; its full
greedy text gate still fails. After that evidence was retained, the TS 480
containers were stopped and removed to free all eight NPUs per node. A new
two-node vLLM-Ascend Graph control, run `20261116T152901Z-431-vllm-cluster2`,
became `/health` 200 on rank 0 and completed five warmed stream repeats:
214.3981 output tokens/s, mean TPOT 0.031767 s, TTFT p50 0.762674 s. Against
this current direct Graph control, TokenSpeed is 0.12199x throughput, TPOT is
8.630x higher, and TTFT is 4.413x higher; the rough 49-prompt/128-output
request-level excess is still about 92.20% decode/post-first-token. The paired prefill/decode ledger and optimization
queue are in `outputs/v41-debug/prefill-decode-bottleneck-analysis-20261116.json`.

## 2026-11-16 controlled prefill evidence gap and 481 profiling run

Two historical mixed-traffic prefill trigger windows were re-tallied in
`outputs/v41-debug/prefill-mixed-profile-ledger-20261116.json`; both are
explicitly non-pure and predate the index-key reuse change. The 43017 window
recorded GatherV3 at 152.323 ms/145 calls, while the 223810 window recorded a
52.290 ms boundary AllReduce that must not be treated as steady prefill cost;
ViewCopy and GatherV3 remain candidate entries only.

To collect the missing evidence, the 431 vLLM containers were stopped (not
removed) and a TS profiling-only snapshot `20261116T161724Z-481-prefill` was
created from the retained 480 source tree. The driver uses `max_tokens=1`,
starts the profiler before requests, and stops it after completion; this is a
profiling run, not a steady-state benchmark. The 481 TS service is currently
Up; the 431 vLLM containers remain stopped and preserved. The two services
cannot be kept resident simultaneously on the same two-node NPU/shm footprint.

The controlled TS-only rank0 ledger is
`outputs/v41-debug/pure-prefill-ledger-481-20261116.json`. In the 257-token,
eight-request, one-token-output case, rank0 step 0 had 254.546 ms kernel sum
and 245.607 ms union busy; its leading entries were ViewCopy 31.653 ms (12.44%),
GatherV3 20.619 ms (8.10%), ScatterNdUpdate 19.367 ms (7.61%), and AsStrided
13.373 ms (5.25%); the single AllReduce remains a boundary sample.

In the successful 493-token, eight-request case, the report contains both a
prefill step 0 and a decode-like step 1. Prefill attribution must use step 0
only: 3096.887 ms kernel sum, with ScatterNdUpdate 1832.507 ms / 1684 calls
(59.17%), Index 234.750 ms / 687 calls (7.58%), GatherV3 50.722 ms / 190 calls
(1.64%), Sort AICPU 46.816 ms, MoE dispatch AICPU/kernel 43.758/43.459 ms,
ReduceSum 43.430 ms, Mul 41.347 ms, `_silu_and_mul_kernel` 31.732 ms, and
ViewCopy 31.551 ms. Step 1 is 195.212 ms and its top kernel order matches the
existing steady-decode profile; it is excluded from prefill cost.

These are TS rank0 attribution observations only. There is still no paired
TS/vLLM-Ascend same-batch/same-length prefill ledger, no rank1/other-DP/EP
coverage in this ledger, and no end-to-end prefill speedup. A 2048-token prompt
remains blocked by `max_model_len=1024`; wall TTFT still conflates queueing,
scheduler, host and profiler overhead. The prefill evidence gap is therefore
partial, not closed.

## 2026-11-16 vLLM reference relaunch and paired text-gate evidence

After the 481 controlled prefill collection, the two TS 481 containers were
gracefully stopped and their NPU HBM was released; the stopped containers were
not removed. The retained 431 vLLM-Ascend containers were then restarted on the
same two-node 16-card footprint. At the current 18:05 UTC check, 431 is Up on
both nodes, rank 0 `/health` returns 200, `/v1/models` lists `deepseek-v41`, and a
short 4-token chat completion smoke request returned 4 completion tokens. That
smoke proves the service is resident and generating; it does not pass the
strict semantic text gate. The 16-card deployment still cannot keep two
world-size-16 model services resident at once, so the fair comparison remains
alternating residency, not a co-resident one.

The same three fixed greedy prompts (`The capital of France is`,
`Count from 1 to 20, separated by spaces.`, and the Chinese preheat
question) were sent to both sides at `max_tokens=64`, `temperature=0.0`,
`ignore_eos=true`. The paired result is
`outputs/v41-debug/20261116T174000Z-ts-481-vs-vllm-431-gate-comparison.json`.
Both sides still fail the strict rubric: vLLM completes Paris coherently but
only reaches numerals 1..12 before straying, and its Chinese answer mixes
navigation fragments with 50 CJK characters. TS is materially more degenerate:
Paris repeats/strays, count emits `to10x10...` with no ordered continuation,
and Chinese has only 6 CJK characters beside English/code fragments. This
establishes that the W8A8 weights/tokenizer baseline is not as broken as the
TS output, but it does **not** by itself identify whether the TS fault is in
attention/KV/indexer/MoE/hidden-state routing, the LM-head/logits path, or a
Graph/context reload issue.


## 2026-11-16 local code review: mirror and correctness candidates

This check was a local source review and read-only remote status check; it did
not restart a TS service, run a long benchmark, change code/config, or remove
remote resources. Ignoring CRLF noise, the real V4.1-adjacent uncommitted delta
is small: `device.py` adds a `device_type` profiling property; the V4.1
attention backend adds the NPU BF16 page-mirror path; the model file is mostly
`layer_kind_scope`/`active_forward` timing wrappers, with MoE FFN `ctx`
propagation changes that still need tensor-level validation.

The NPU mirror allocates `swa_bf16` and `global_bf16` alongside the packed
uint8 KV cache. The live global write path gathers selected packed rows,
dequantizes them, and scatters them into `global_bf16`; the SWA path writes the
available `mirror_rows`, otherwise gathers from packed cache and scatters into
`swa_bf16`; decode reads `ascend_paged_attention_from_bf16` from that mirror.
The leading unproven correctness suspect is whether mirror pages are
repopulated/invalidated consistently on prefix-cache restore, eviction, Graph
replay, reload-forward, or cache refresh. A stale mirror would degrade attention
output. Source review found a concrete gap: global writes and the decode SWA
current-row write populate their BF16 mirrors, but the prefill SWA path wrote
packed uint8 rows without a matching `swa_bf16` mirror scatter. A local patch now
repopulates old-prefix and current SWA mirror rows in the NPU prefill path; only
`py_compile` has passed so far. No fresh-server text gate, packed-versus-mirror
tensor equality check, Graph-replay equivalence check, or five-repeat A/B has
run, so this remains a pending correctness/performance candidate rather than a
closed root cause or measured win.

The leading decode performance candidates are still mirror gather/scatter/copy
and the remaining index-scoring elementwise path. The post-reuse TS-only
decode ledger has ViewCopy 31.916 ms/step, GatherV3 30.405 ms/step,
ScatterNdUpdate 25.653 ms/step, AsStrided/IndexPut copies 13.323+6.785
ms/step; MoE dispatch/combine and grouped matmul remain separate groups.
These numbers cannot by themselves prove a vLLM kernel-level gap, and no
backend change should be claimed as a win before the paired ledger and
five-repeat fresh A/B.

The other node pair was checked read-only: `.109` has idle NPU memory, while
`.110` still runs six `pd-pat` prefill/decode processes at about 53.6-53.8 GB
per occupied card, leaving only two apparent free cards. There is therefore no
ready independent 16-card co-resident pool for a second world-size-16 service in
the current snapshot.

## 2026-11-16 evening: 427 TS A/B and fresh vLLM 420 alternating baseline

The TS 427 service carrying the pending prefill SWA mirror candidate was
measured after release and stop. Its five-round fixed-load stream results are
`outputs/v41-debug/20261116T182409Z-427-cluster2/427-benchmark.json`:
throughput mean 27.9932 tok/s, median 28.0384, min 27.4491, max 28.3796; TTFT
p50 2.6493s, p90 2.7301s; TPOT mean 0.266165s, p50 0.266001s, p90 0.267497s.
Relative to the clean 480 TS snapshot this is a small TS-side change
(throughput 1.0703x, TPOT 0.9709x, TTFT p50 0.7871x), but it is **not** a
paired kernel attribution and cannot be claimed as proof that the mirror
candidate caused the improvement. 427's strict fixed-prompt gate remains
`gate_passed=false` (`paris`, `count`, `chinese`).

After releasing TS, fresh vLLM-Ascend Graph run `20261116T191538Z-420` was
brought up on the same two-node footprint; both containers are Up, rank0
`/health` returns 200, and `/v1/models` returns `deepseek-v41`. The built-in
Paris smoke returned degenerate text and stopped before the built-in measured
rounds. A temporary performance-only driver retained five measured rounds in
`420-benchmark-nogate.json`; this does not close the text gate. The five-round
vLLM results are throughput mean 209.5961 tok/s, median 204.8606, min
203.8177, max 220.8404; TTFT p50 0.8122s, p90 1.1703s; TPOT mean 0.031887s,
p50 0.031033s, p90 0.035924s. The 420 strict gate also fails
(`paris`, `count`, `chinese`).

The alternating 427-vs-420 comparison is
`outputs/v41-debug/20261116T191538Z-420-vs-427-comparison.json`. TS throughput
is 0.13356x vLLM (vLLM 7.4874x TS), TS TPOT mean is 8.3470x vLLM, and TS
TTFT p50 is 3.2617x vLLM. Under the rough 127-token post-prefill decomposition,
decode/TPOT excess is 29.7533s and TTFT excess is 1.8370s per batch, so
94.18% of the rough request-level excess remains in the decode/逐 token path;
TTFT/prefill is still a real gap but smaller in this fixed-load estimate.
These are mixed/decode-dominated short-context stream numbers, not paired pure
prefill or pure decode kernel ledgers.

420 occupies about 60.4 GB of the 65.5 GB HBM on each rank0 card, leaving
roughly 5.1 GB free. Together with the prior per-service 56-60 GB measurement,
a second world-size-16 same-spec service still cannot be co-resident on this
16-card footprint; the accepted comparison remains alternating residency.

## 2026-11-16 late retry: TS 418 fresh service and five-round ledger

After the 646 rank-1 weight-load failure (`aclrtMemcpy 107017` / devmm user-pages pin failure), retry run `20261116T201549Z-418-cluster2` passed two-node Graph readiness in 1733.45s. Both 418 containers remain Up; rank0 `/health` returns 200 and `/v1/models` lists `deepseek-v41`. The 646 failure remains unclosed: 418 did not reproduce a new devmm error, but one successful retry does not prove the host-paging/lifecycle root cause is gone.

The strict three-prompt gate still fails for 418 (`outputs/v41-debug/20261116T201549Z-418-cluster2/418-gate.json`, blocking `paris,count,chinese`). The benchmark driver's built-in Paris keyword check reports `ok=true`, but that is only a smoke keyword match and must not be treated as semantic acceptance.

The 418 five-round fixed-load stream ledger is
`outputs/v41-debug/20261116T201549Z-418-cluster2/418-benchmark-nogate.json`. All five rounds had zero errors, `fill_ratio=1.0`, and 2048 completion tokens per round. Results: throughput mean 27.6516 tok/s, median 28.1668, min 26.1188, max 28.2964; TTFT p50 2.62580s, p90 4.41417s; TPOT mean 0.267813s, p50 0.265668s, p90 0.275566s. Round throughputs were 26.1188 / 27.5084 / 28.1675 / 28.1668 / 28.2964 tok/s.

The 418-vs-420 alternating comparison is
`outputs/v41-debug/20261116T201549Z-418-vs-420-comparison.json`. TS 418 throughput is 0.13193x vLLM 420 (vLLM 7.5799x TS), TPOT mean is 8.3987x vLLM, and TTFT p50 is 3.2329x vLLM. The rough 127-post-prefill-token decomposition attributes 29.9625s of excess to decode/TPOT and 1.81358s to TTFT, i.e. about 94.29% decode/post-first-token under this short mixed load. This remains a request-level rough estimate, not a paired kernel ledger.

418 versus the same-family TS 427 shows no meaningful fresh improvement: throughput 0.98780x, TPOT 1.00619x, TTFT p50 0.99114x. The pending prefill SWA BF16 mirror candidate therefore still has no measured end-to-end win or validated numeric consistency. Only three warmup decode timing rows were captured before the benchmark; their host-side split averages attention 441.90ms (36.4%), FFN 647.11ms (53.4%, routed MoE 592.99ms / 48.9%), but this small warmup sample is not a steady paired profile.

## 2026-11-17 phase 0 gate: read-only resource/baseline/probe check

This round stayed read-only for service state: no restart, long benchmark, code/config change, or remote cleanup. The retained TS 418 service is still Up on both ranks; rank0 `/health` returns 200 and `/v1/models` lists the W8A8 model and `deepseek-v41`. vLLM 420 remains `Exited (137)` with its five-round ledger retained. Rank0 root is 99% full (5.4G free), `/dev/shm` is 99% full (3.9G free), and `/mnt` NFS is 100% full; rank1 root has 251G free but `/dev/shm` is 90% full. Old rank0 shm profile products total about 41.8G and rank1 about 19.6G; no deletion was authorized, so a new remote profile should not be written until cleanup ownership/space is confirmed.

The phase 0 gate ledger is `outputs/v41-debug/phase0-gate-20261117T0753Z.json`. It records that the latest alternating fixed-load gap remains decode-dominated at the request level: TS 418 27.6516 tok/s, TPOT mean 0.267813s, TTFT p50 2.62580s versus vLLM 420 209.5961 tok/s, TPOT mean 0.031887s, TTFT p50 0.812222s; TS is 0.13193x throughput, 8.39872x TPOT, and 3.23286x TTFT p50. Rough excess is 94.29% decode/post-first-token, but that remains a request-level estimate, not a paired kernel ledger.

A standalone CPU FlatKV probe ran inside the existing 418 container (`outputs/v41-debug/probe_dsv41_torch_codec_cpu.py`, result `probe-dsv41-torch-codec-cpu-result.json`). On torch 2.10.0+cpu, swa/global/index packed scatter->gather roundtrip is exact (max abs diff 0.0), invalid slots read zero, swa/global BF16 mirror write->read roundtrip is exact, mirror duplicate destinations keep the last active write, and a stored packed snapshot plus rebuilt mirror roundtrip matches. These are CPU/syntax checks only: they do not exercise the NPU scatter kernel, Graph replay, reload-forward, live prefix restore/evict, live packed-versus-mirror equality, hidden-state/router/logits comparison, or a fresh-service text gate; SWA dequantize->repack idempotence was not asserted. Plain packaged pytest collection still fails before tests because the dsv41 package `__init__` imports the unavailable `cann_ops_transformer` CANN adapter in the CPU Python.

No new end-to-end performance result or correctness closure was produced this round. The strict 418 gate still fails (`paris,count,chinese`), and the retained vLLM 420 gate also fails. A stdlib-only correctness ladder schema and validator were added under `outputs/v41-debug/v41-correctness-ladder-20261117.json` and `validate_v41_correctness_ladder.py`; the validator returns `VALID` for the nine-stage request/embed/layer/KV/router/expert/final-hidden/logits/text plan. This is still a capture-and-comparison design only: no hidden, router, logits, selected token-ID, or live packed-versus-mirror tensor has been captured. The next runtime work is an opt-in capture path that does not change production numerics; NPU-side comparisons remain blocked until a controlled fresh-service or replay path is available with enough remote space.

## Key files

| Area | Path |
|------|------|
| FlatKV torch codec | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cache_codec_torch.py` |
| Codec tests | `test/runtime/test_dsv41_cache_codec_torch.py` |
| V4.1 attn timers | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| mm_timing | `python/tokenspeed/runtime/execution/model_executor.py` |
| SiluAndMul | `python/tokenspeed/runtime/layers/mlp.py` (+ NPU `silu_and_mul`) |
| LayerKindTimer stub | `python/tokenspeed/runtime/execution/context.py` (~L222) |
| Cluster helper | `%TEMP%\e2e_v41_ts_dp4.py` |

## Key artifacts

- `outputs/v41-debug/20261114T214831Z-48475-strict/` — accepted BF16-ND service benchmark, direct A/B, and validated TS/vLLM comparison
- `outputs/v41-debug/direct-bf16-nd-result.json` and `bf16-nd-regression.log` — isolated writer timings and real-NPU correctness gate
- `docs/ascend-v41-w8a8-serving-handoff.md` — this file
- `%TEMP%\e2e_v41_phaseA_oom.json` / `phaseB_baseline.json` / `phaseC_gather.json`
- `e2e_v41_b2_attribution.json` / `e2e_v41_b3_ts_vs_vllm.json`
- `e2e_v41_cprime_silu.json` / `e2e_v41_c2_residual.json`
- `e2e_vllm_stage.json`
- `e2e_v41_step5_ledger.json` + step5* profiles
- Prior profiles: `e2e_profile_after_silu.txt`, `e2e_profile_after_ngram.py` outputs

## Agent transcripts (parent)

Cursor agent conversation under project `tokenspeed`; Phase C / B2 agents id `70e12cfa-…`, residual agent `3899843b-…`.

---

## One-sentence status

**Current status: TS 418 is Up on the alternating 16-card footprint; vLLM-Ascend 420 is stopped with its retained five-round ledger and failed strict gate. Fresh 418 five-round results are 27.6516 tok/s, TPOT mean 0.267813s, TTFT p50 2.62580s; against vLLM 420 this is 0.13193x throughput (vLLM 7.5799x TS), 8.3987x TPOT, and 3.2329x TTFT p50. Rough request-level excess remains decode-dominant (~94.29%). The 418 strict fixed-prompt gate still fails (`paris,count,chinese`); the keyword-only benchmark smoke is not correctness closure. The controlled TS rank0 prefill evidence is partial, paired TS/vLLM prefill/decode ledgers remain open, and the BF16-mirror/MoE-`ctx` suspects are unproven. Priority order: close or isolate the fresh text gate, validate mirror tensor/Graph consistency before claiming a win, build paired decode/prefill ledgers, rank and validate backend fixes, then resume PD only after the aggregate path is competitive.**
