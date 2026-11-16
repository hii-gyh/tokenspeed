# Ascend V4.1 W8A8 serving — tuning handoff

**Location:** `docs/ascend-v41-w8a8-serving-handoff.md` (moved from `%TEMP%\e2e_v41_HANDOFF.md`)
**Branch:** `cursor/ascend-v41-w8a8-serving` (base `main`)
**Goal:** TokenSpeed vs vLLM-Ascend on same topology **DP4 / TP8 / EP32**, lean remote-only tuning.
**Target model:** DeepSeek-V4.1-Flash-w8a8
**Remote:** jump → nodes `192.168.11.11` (rank0), `.7` (r1), `.8` (r2), `.9` (r3)
**Deploy tree:** `/ts` mounted into `dsv41-ts-dp`
**Probe/logs:** host `/root/gyh_dev/token_speed_dsv41flash/` (container `/probe`)
**Local helpers:** `%TEMP%\e2e_v41_ts_dp4.py` + many `e2e_v41_*.py`
**Python (Windows driver):** `C:\Users\Lenovo\AppData\Local\Python\bin\python.exe`
**HEAD (as of handoff):** `53f58ea12` — FlatKV gather OOM/view fix (on top of `ca284be12` torch FlatKV pin)
**JSON/profile artifacts:** still under `%TEMP%\e2e_v41_*.json` / `e2e_profile_*` (not committed)

---

## Hard constraints (do not violate)

1. **Remote-only NPU** — no local Ascend.
2. **No reboot thrash** — weight load ~12–22 min; one controlled restart max per measured win.
3. **No** `fuser` / `npu-smi` reset (killed SSH / hygiene disasters).
4. **No mid-load** `nerdctl rm`.
5. **FlatKV on Ascend = torch SPECIALIZED** — Triton codec capability pinned to `{nvidia,amd}` only. **No Triton/TileLang Ascend FlatKV codec.**
6. **No select `.item()` trim** — breaks CUDA/NPU graph capture (sync during capture).
7. **Never `/start_profile` VIZTRACER on live DP4** without a recover plan — it **fatal’d the mesh** (all ranks Exited).
8. Prefer `--allow-reload-forward` + `ts reload-forward` for kernel/forward iteration; current boots often **lack** it → full restart required to load Python changes.
9. Scrape RANK0 with host path `PROBE=/root/gyh_dev/token_speed_dsv41flash` (not container `/probe` over SSH).
10. Gate: **≥15%** measured win before another DP4 restart / Phase D graph.

---

## Topology & baseline competitor

| Engine | Topology | Mode (latest TS) | ITL / tok/s |
|--------|----------|------------------|-------------|
| vLLM-Ascend | DP4/TP8/EP32 | stored `%TEMP%\e2e_vllm_stage.json` | ITL ≈ **0.44 s**, ≈ **2.29** tok/s |
| TokenSpeed | DP4/TP8/EP32 | **eager** (`--enforce-eager`) | ITL ≈ **1.83 s**, ≈ **0.55** tok/s (~**4.2×** slower) |

TS stream_stages live re-run was blocked after viztracer death; B3 used mm_timing + wall cross-check.

---

## Clean execution order (adopted)

```
A  OOM → eager READY
B  RANK0 baseline (same-decode scrape; p50 AND mean)
C  Fix by measured dominator (≥15% gate)
C' Next residual bucket (same gate)
D  Graph last (only after eager win)  ← NOT STARTED
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

- Expect **`dsv41-ts-dp` Up** after recover (verify health 200 before work).
- Eager only; Phase D / CUDA graph **not** validated for e2e win.
- Codec on disk `/ts` matches shipped FlatKV fix; workers loaded it via recover restart.
- **Do not** VIZTRACER on this mesh.

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
`ForwardContext.layer_kind_timer` is **comment-only** — no `TOKENSPEED_LOG_LAYER_KIND_TIMING` implementation yet.

Add **capture-safe** sub-timers inside decode forward:
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

**Current status: fair fully warmed Graph comparisons still place TokenSpeed well below vLLM-Ascend, and the earlier 30% advantage is invalid. Pure-decode profiling identified repeated index-key gather/dequantization; per-owner key reuse passed three NPU tests and improved the same-service five-repeat production mean from 24.2318 to 27.6735 output tokens/s (1.14203x), reducing mean TPOT from 0.311667 to 0.265537 s (14.80%). This is a real TokenSpeed self-improvement, not parity with vLLM and not semantic text-quality acceptance. Repeated reloads change full greedy outputs even without the patch, so text reproducibility remains open. Twelve PD runs have no overall performance benefit; PD remains lower priority than closing the aggregate-engine gap.**
