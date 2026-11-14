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

**Torch FlatKV is pinned and OOM-safe with a mild gather win; Silu/ngram-commit already fixed; TokenSpeed eager decode is still ~4× slower than vLLM because ~⅔ of model_fwd is unattributed non-FlatKV compute — next ship fine-grained forward timers, then gate the next kernel change.**
