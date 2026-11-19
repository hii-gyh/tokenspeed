# Ascend V4.1 native decode path

## Current status (2026-11-19)

The measurements below supersede the earlier performance estimates and phase
gates in this document. The retained implementation now includes BF16-only KV
writes, the wide/Triton index and compressor-tail readers, fused mHC
Sinkhorn/pre/post chains, and INT8 MC2 dispatch feeding the existing INT8
expert GEMMs. The best valid five-round bs8 result is `80.638 tok/s` with
`85.556 ms` TPOT; pre/post request correctness passed.

The production MoE shapes are `96x5120 -> 96x4608 -> 96x2304 -> 96x5120`.
The earlier `7168/3584` figures do not describe this deployment. The next
native candidate is grouped-matmul + dequant + clamped SwiGLU + requant fusion,
but it may only be wired after the current image's API documentation confirms
its signature, dtype/layout, clamp semantics, and graph-replay behavior.
`torch_npu.npu_mhc_pre` is not a replacement for `v41_hc_pre(x, pre_mix)`:
TokenSpeed consumes the previous sublayer's pre-mix, while the stock operator
has no delayed `pre_mix` input. The rejected expert-scale Triton and
`repeat_interleave` replacements must not be restored.

Phase -1 CPU regression also fixed five baseline issues without changing the
serving algorithm: invalid cache slots now use a non-aliasing safe write address,
aligned arena fields include the typed view's absolute storage offset, compressor
quantization lookup falls back to the attention parent's fused shards, packed
MXFP4 checkpoint weights are distinguished from Ascend INT8 weights by the
quantization config rather than by `int8` storage alone, and a globally empty MoE
batch returns before querying expert backend capabilities. The remote core CPU
suite passed `60 passed, 1 skipped`; the model suite passed `150 passed, 70
skipped` with 36 accelerator-kernel-planning variants deselected. The remaining
runtime/CI behavior suite passed `600 passed, 93 skipped` with only the Ascend
paged-attention symbol test and the profiler's fresh-process Triton-import canary
deselected. Those deselected cases, accelerator-kernel-planning variants, and the
mHC NPU kernels still require real-device validation; CPU platform overrides are
not evidence of NPU correctness.

Final acceptance is not the historical TPOT targets below. It requires a fresh,
same-day, same-protocol comparison with vLLM-Ascend, at least five rounds per
engine, passing correctness, and a median TokenSpeed `output_tok_per_s` of at
least `1.03x` vLLM-Ascend. See `outputs/v41-debug/stage1-ab-20261119.md` and
`outputs/v41-debug/stage2-mhc-20261119.md` for the current A/B evidence.

## Original cache-path analysis

This document scopes the one change that can still move TokenSpeed's Ascend
decode cost: stop paying for two representations of every KV page. It was written
when the plan was to drop the BF16 mirrors and read the packed quantized pages
directly; the P0 probe below established that the packed-page attention is
arch-gated to A5, so the surviving form of the change is the mirror removal itself
(one BF16 copy, no gather), which is exactly the cache arrangement vLLM-Ascend runs
on this hardware. It is not a plan to import
`vllm_ascend/models/deepseek_v41/model.py`; that module was
already exercised inside vLLM in stage `20261117T064751Z-3874` and produced the
same `31.887 ms` TPOT, so the model structure is not the variable. The variable
is the attention backend and the KV cache read path.

## Why this, and why not "tune further"

`outputs/v41-debug/device-ledger-20261117.md` measures a steady pure-decode step
at `330.0 ms` stage with `318.7 ms` of kernel time on rank0, grouped as
data-movement/index/layout `154.9 ms (46.9%)`, elementwise `63.5 ms (19.3%)`,
MoE dispatch/combine `32.4 ms (9.8%)`, other `32.6 ms (9.9%)` and
matmul/attention `16.5 ms (5.0%)`. The routing, GEMM and expert work is therefore
not the bottleneck; the KV write/read path is.

The reason is written in the cache recipe itself:

```python
# python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py:86
def _direct_paged_bf16(self) -> bool:
    # CANN SparseFlashMla reads BF16 PA_BBND pages directly but rejects the
    # quantized uint8 fields. Only Ascend pays for the dequantized mirrors.
    return self.attn_config.device == "npu"
```

That flag turns on a second copy of every KV page in BF16, maintained by
`cache_gather` + `cache_scatter_bf16` on the write path, and consumed by
`ascend_paged_attention_from_bf16` on the read path. The comment's premise is
outdated for this image: the CANN toolkit (`cann_ops_transformer`, CANN 9.2.1)
publishes the quantized variants alongside the BF16 ones:

```
quant_sparse_flash_mla, quant_sparse_flash_mla_metadata,
mixed_quant_sparse_flash_mla, mixed_quant_sparse_flash_mla_metadata,
quant_flash_mla, quant_flash_mla_metadata,
quant_lightning_indexer, quant_lightning_indexer_metadata,
sparse_lightning_indexer, sparse_flash_mla, sparse_flash_mla_metadata
```

and vLLM-Ascend already drives them: `vllm_ascend/ops/packaged_attention.py`
registers `vllm_ascend::mixed_quant_sparse_flash_mla`, whose schema carries
`int quant_mode, float? softmax_scale=None, str layout_q="TND",
str layout_kv="PA_BBND", bool return_softmax_lse=False) -> (Tensor, Tensor)`,
and `vllm_ascend/attention/dsa_v41.py` selects between the BF16
`torch.ops._C_ascend.npu_sparse_flash_mla` (line 717) and the quantized
`mixed_quant_sparse_flash_mla_metadata` / `quant_lightning_indexer_metadata`
paths (lines 1327, 1422).

So the gap is a wiring gap, not a missing-operator gap. The full operator list
and the vLLM-Ascend reference call sites are recorded in
`outputs/v41-debug/cann-quant-op-inventory-20261117.txt`.

## P0 result (2026-11-17): L3 is arch-gated off on 910B3

The premise above was tested and is wrong on this hardware. Executed on rank0 with a
real 288 B MXFP4 page (`outputs/v41-debug/probe_p0_quant_mla.py`), the metadata builder
accepts `quant_mode=1`, but the compute kernel refuses it:

```
Invalid_Argument(EZ0026): Parameter quant_mode of MixedQuantSparseFlashMla has
incorrect value 1. Reason: quant_mode 3 is supported only on arch22 and arch 92;
quant_mode 1/2 are supported only on arch35.
```

`Ascend910B3` is arch22, so the mixed packed-page attention (`quant_mode=1`, the exact
call vLLM makes from `qsmla`) **cannot execute here**; only `quant_mode=3`
(TurboQuant C4, a different page format) is legal.

vLLM-Ascend does not use that path on this machine either:
`device_op.py:2096` maps the standard hardware family to `BaseDeviceAdaptor`, whose
`get_dsv41_packed_cache_ops()` returns `None`, while only `A5DeviceAdaptor` returns
`MixedQuantPackedCacheOps`. `dsa_v41.py:838` sets `_uses_a5_packed_cache` from that
result, so our own vLLM run `20261117T064751Z-3874` produced its `31.887 ms` TPOT on
**standard BF16 caches**.

Two consequences:

- `L3a` and `P0b` are closed for 910B3 (see `outputs/v41-debug/l3-p0-quant-mode-20261117.md`).
- The causal story shifts: TokenSpeed spends `8.25 ms` per layer where vLLM spends
  `0.80 ms` on identical hardware *without* a quantized cache. The KV page format is
  not the binding constraint; the per-layer step construction is. The mirror removal
  (`L2`) stays worth doing — it is legal without any operator and is the largest
  single structural win available here — but it is a `1.5x` lever, not the `8.4x` one.

`L1` (native indexer via `quant_lightning_indexer`) is untested on this arch and needs
the same gate check before it is planned.

## Landing points

Three edits, in dependency order. L3 must land before L2 can be deleted, and L1
is independent.

### L3 — decode attends from packed quantized pages

**Status: closed on 910B3.** The operator is arch35-only (see the P0 result above);
this section is kept as the record of the A5-only path and must not be re-planned
without an A5 target.

Current read path (`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py:2352`):

```python
out = ascend_paged_attention_from_bf16(
    q,
    swa_cache=self.cache_pool.swa_bf16(layer_id),
    global_cache=self.cache_pool.global_bf16(owner),
    ...
)
```

which reaches `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/ascend.py:375`
and eventually `npu_sparse_flash_mla(..., layout_kv="PA_BBND")` on BF16 pages.

Target: a kernel entry that takes the packed pages (`global_kv(owner)` /
`layer kv` with the `"global"` / `"global_v4"` / `"swa"` cache format) and calls
`torch.ops.cann_ops_transformer.quant_sparse_flash_mla` (or
`mixed_quant_sparse_flash_mla` when the compress ratio needs both branches),
passing a `quant_mode` that describes the page layout.

Work items:

1. `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/dsv41.py` — add
   `npu_quant_sparse_flash_mla` / `npu_mixed_quant_sparse_flash_mla` thin
   wrappers next to the existing two, so the vendor boundary stays in that
   package.
2. `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/pages.py` —
   extend the FlatKV → `PA_BBND` adapter to quantized rows: it currently builds
   a BF16 page view, and a quantized page needs the row width, the scale layout
   and the `quant_mode` value instead.
3. `ascend.py` — register the quantized solution (signature on
   `torch.uint8`/`int8` dense format, new traits carrying `quant_mode` and the
   cache format), keep the BF16 one for fallback until L2 flips.
4. `deepseek_v41.py:2352` — call the new kernel with the packed pools; this is
   the same call site, so no scheduler, metadata or graph changes are needed.

### L2 — one representation for the KV pages

Measured first (2026-11-18, `outputs/v41-debug/phase-a-attribution-20261118.md`), the
mirror is **load bearing**: `_forward_v41_ascend_decode` reads
`cache_pool.swa_bf16(layer_id)` and `global_bf16(owner)` unconditionally on `npu`, so
flipping `_direct_paged_bf16()` to false first is not a valid intermediate state — decode
would raise on a missing field, not fall back to the packed pages. Deleting the mirror
alone is therefore not a step; L2 is a choice between one of two single-representation
caches:

- **BF16 only** (what vLLM-Ascend runs): drop the packed write and let the mirror pages be
  the cache. Removes the whole ATen quantize path (`_pack_rows`, ~45–50 ms/step of
  `ReduceSum`/`Mul`/`Cast`/`Adds`/`Div`), the gather and the mirror write. Costs: 4× page
  bytes for the global group (256 B + scale → 1024 B rows) against a 0.90 gpu-mem budget
  at `max_model_len=1024`, and the prefix/L2/PD policies and the `index` group would need
  their own format decision.
- **Packed only**: keep the quantized cache and serve attention from it. That requires the
  CANN path the P0 probe closed on arch22 (`EZ0026`), so it is unavailable on 910B3.

Whichever is chosen, both write sites and the read sites must move together:
`deepseek_v41.py` global mirror, SWA decode mirror, SWA extend mirror (mirror seeds
included), and `_forward_v41_ascend_decode`'s two pool reads. `SlotStateExporter` coverage
in `test/runtime/test_slot_state.py` must still pass for whichever field remains.

**Interim step taken (2026-11-18).** The mirror write no longer round-trips through the
device: `dsv41.cache_scatter_with_mirror` packs the rows once, decodes the bytes it just
packed, and writes the mirror from that decode. The mirror is byte-identical to the
previous `cache_scatter` + `cache_gather` + `cache_scatter_bf16` sequence (host test
`test_scatter_with_mirror_equals_the_scatter_then_gather_mirror` asserts it for both
`swa` and `global`, including duplicate and invalid slots), while one gather per layer
disappears. The fused-RoPE branch (`swa_rope_scatter` / `global_rope_scatter`) still
reads its rows back, because those kernels return no packed rows.

### L1 — native indexer

Current: `ascend_dsv41_index_topk` (`ascend.py:621`) declares
`native_indexer: frozenset({False})`, `supports_index_query_input:
frozenset({False})` and runs the Torch Full/Reindex path. The registry already
has a native slot — `ops/attention/dsv41/__init__.py:796` computes `native = shaped
and (...)`, with a NVIDIA DeepGEMM/DeepSelect implementation behind
`is_native_indexer_available()`.

Work items:

1. Add an Ascend solution registering `native_indexer: True` backed by
   `torch.ops.cann_ops_transformer.quant_lightning_indexer` plus
   `quant_lightning_indexer_metadata`.
2. Relax the `shaped` gate in `__init__.py:784`, which currently requires
   `query_values.is_cuda`, so an NPU tensor can select the native branch. Keep
   the gate explicit per platform rather than removing it.
3. Map the TS index cache format onto the operator's `quant_mode`; the TS
   `index_k_format` trait values (`"mxfp4"`, `"fp8_scaled"`, `"unknown"`) are the
   natural key.

L1 is independent of L2 and can ship on its own.

## Device kernel ledger (2026-11-17): the gap is op granularity

Both engines were measured at the same batch (8 concurrent) on the same hardware and
topology, and their per-step op ledgers were opened side by side
(`outputs/v41-debug/kernel-ledger-comparison-20261117.md`). The vLLM side was captured
with `msprof --application --task-time=l0` wrapping the eager baseline; the TS side is
the existing `torch_npu.profiler` ledger from stage 415.

| Per step | TS | vLLM-Ascend |
| --- | --- | --- |
| kernel launches | **27,549** | **≈2,124** |
| kernel launches per layer | **689** | **≈53** |
| wall per layer | 8.25 ms | 0.80 ms |

Of TS's 27,549 launches, **25,651 (93.1%) are sub-10 µs kernels** (`Adds`, `ReduceSum`,
`Div`, `Cast`, `Mul`, `SelectV2`) that together only account for `75.2 ms`; a time slice
of the rank0 CSV shows them issuing as a serial `Adds → Div → ReduceSum` chain at ~10 µs
spacing, with the device 100% occupied by that chain. The fused Triton implementations
this repo uses on the CUDA path (`tokenspeed_kernel.ops.layernorm.triton.rmsnorm`,
`add_rmsnorm`, `gemma_rmsnorm`) have no Ascend counterpart, so normalization and
quantization-scale arithmetic falls back to ATen granularity. vLLM does the same math in
≈1/13 the kernels (`RmsNorm`, `aclnnDynamicQuantV2`, `SparseFlashMla`, `AivKernel`).

The BF16 mirror is real but is not the headline: `GatherV3` 69.58 ms + `ViewCopy`
31.90 ms + `ScatterNdUpdate` 20.36 ms = `121.84 ms/step` (41% of the `330 ms` stage). L2
removes it and lands at ≈200 ms, i.e. `1.65x`. Reaching vLLM's `31.9 ms` requires cutting
launches per step by ~13x, which is kernel fusion in `tokenspeed-kernel`, not a cache
format change.

Caveat recorded with the numbers: l0 instrumentation inflates absolute durations ~13x
(throughput 209 → 19.2 tok/s during capture), so only **launch counts** are compared
across engines; durations are compared within one ledger. The TS ledger's own
instrumentation cost is 1.23x (`stage_ms 330.0` vs TPOT `267.8 ms`).

## CANN indexing rules for arena views (2026-11-18, graph ledger)

The graph-mode ledger (`POST /start_profile`, NPU Graph, bs8) put device busy at
185 ms of a 213 ms step, while GEMM + attention + MoE together account for only
~20 ms. The rest is data movement, and three CANN behaviours explain most of it.
Every rule below was measured by capturing the op in an NPU Graph and replaying
it on production arena geometry, then confirmed end to end.

1. **Never assign through advanced indexing into an arena view.**
   `view[i, j, k] = x` lowers to `aclnnIndexPutImpl`, which first copies the
   *whole underlying arena* (`ViewCopy`). Writing one 512-element FP32 row into
   a 4.5 GB arena cost 5.49 ms; the actual `IndexPutV2` was 7.1 µs. Use the
   vendor ND scatter (`_ascend_codec.scatter_*`) instead: 6.8618 → 0.0320 ms per
   call, bitwise identical.
2. **Never read through advanced indexing into an arena view either.**
   `view[i, j, k]` lowers to `aclnnIndex_AsStrided`, which reconstitutes the
   arena before selecting rows (1.15 ms/call). Select from a 1-D view of the
   arena with `index_select` instead: 1.2709 → 0.0775 ms per call.
3. **Gather with the widest aligned elements, not bytes.**
   `aclnnIndexSelect_GatherV3` on a uint8 pool costs ~35 ns *per index element*
   (MTE2 was 94% of a 2.8 ms call), so a 68-byte row read is dominated by index
   count, not by bytes. Where width and origin are 8-byte aligned, gather the
   same bytes as int64 and view back: 3.9–4.2x on the swa/global fields.

Rules 1 and 2 apply to the compressor tail (`compressor_torch.py`), rule 3 to
the planar cache read (`cache_codec_torch.py::_read_planar`). The byte-column
path stays as the fallback for unaligned pools and for CPU/CUDA.

Measured end to end, graph / bs8, each row adding one fix:

| Configuration | TPOT (s) | Throughput (tok/s) |
| --- | --- | --- |
| Frozen baseline | 0.267813 | 27.6516 |
| + BF16-only mirror write | 0.226110 | 32.0282 |
| + KV paged scatter | 0.215158 | 34.4571 |
| + clamped SwiGLU | 0.207278 | 34.9003 |
| + compressor tail vendor scatter (rule 1) | 0.167119 | 42.8929 |
| + widened cache gather (rule 3) | 0.139571 | 51.1439 |

**Prioritise from the graph ledger only.** The eager ledger overestimated
`_silu_and_mul_kernel` by ~22x (12.60 ms/step eager vs 0.230 ms graph replay)
and three successive extrapolations from it failed. `kernel_details.csv`
carries `Input Shapes` even without `record_shapes`, which is what named the
`ViewCopy` arena; `operator_details.csv` has neither shapes nor a time column
and cannot be windowed.

## Phasing and gates

| Phase | Content | Gate to pass before the next phase |
| --- | --- | --- |
| P0 (done) | Probe `mixed_quant_sparse_flash_mla` with `quant_mode=1` on one real 288 B page set | **Failed on 910B3**: `EZ0026`, `quant_mode 1/2` are arch35-only. `L3` closed, `P0b` does not apply |
| P1a (done) | Device kernel ledger for both engines | Gap attributed to op granularity: 27,549 vs ≈2,124 launches/step |
| P1b (1 day) | Fuse the largest micro-op triple (`Adds`+`Div`+`ReduceSum`) into one Ascend kernel, nothing else changed | Per-layer drop ≥ 1 ms (≥ 40 ms/step, 12% of the stage) — otherwise the micro-op block is distributed and must be attacked item by item |
| P1c (2–3 days) | `L2` only: single BF16 representation, mirror write sites removed, `_direct_paged_bf16()` reports no mirror | Stage ≤ 250 ms and data-movement group ≤ 90 ms; capture compare no worse than today |
| P2 (conditional) | `L1` native indexer — arch gate already **passes statically** (prebuilt `quant_lightning_indexer` / `sparse_lightning_indexer` kernels exist for `ascend910b`) | Same capture compare as the Torch indexer, ties and masks preserved |
| P3 (1 day) | Full regression, fresh baseline, doc update | Acceptance list below |

Expected outcomes, stated before measuring so the gates stay honest: L2 alone TPOT
`267.8 → ~162 ms` (`1.65x`), i.e. the `8.4x` gap becomes `~5.1x`; L2 + L1 `≈1.9x`
(`~4.4x`). The `8.4x → 1.0x` part is the fusion work of P1b and its follow-ups, not this
document's landing points.

## Acceptance

Performance, measured on the frozen pure-decode harness rather than on a claim:

- `stage_ms` from `330.0` to `≤ 250 ms`, and the data-movement/index/layout
  group from `154.9 ms` to `≤ 90 ms` in a fresh
  `profile_v41_pure_decode.py` run;
- TPOT target is `≈175 ms` (the ledger's `1.89x` ceiling). Anything better than
  `200 ms` is acceptable; **`80 ms` is explicitly out of scope for this change**
  and must not be used as the acceptance bar.

Correctness, since the representation of the cache changes:

- same-capture comparison with `test/ci/npu/compare_v41_captures.py`
  (`reference candidate --atol --rtol --report`) against the current build on the
  aligned 52-token request — after `L2` alone the numbers should agree bit-for-bit,
  which is a cheap and strict check that only the copy disappeared;
- attention output cos-similarity per layer no worse than today's
  `min 0.3971 / median 0.9880` ladder, and top-20 index overlap no worse than
  `8/20`;
- the strict text gate (Paris / count / Chinese) must not regress; note that it
  does not pass today, so "no worse" is the criterion, not "passes".

Regression:

- `test/runtime/test_slot_state.py`, `test/runtime/test_debug_tensor_capture.py`,
  the NPU runner's graph/eager parity tests, and a prefill-path run (the
  `3100` site is the extend write, so prefill must be re-measured too);
- PD topology unchanged, no new host syncs (`_sync_ms` calls must not appear on
  the decode path).

## Tests to add

- `tokenspeed-kernel-npu/test/` (or the existing NPU test root): quantized page
  attention vs BF16-mirror attention on generated pages, tight `atol`/`rtol`,
  covering `swa`, `global` and `global_v4` formats and both compress ratios.
- `tokenspeed-kernel/test/npu/ops/test_dsv41_ascend_indexer.py`: extend to the
  native branch — same top-k as `index_topk_torch` for tie-heavy and masked
  candidates, plus the page contract test that already exists for the page
  geometry.
- A runtime behavior test that the mirror pools are absent (asserted through the
  cache recipe's declared fields, not by source inspection) once
  `_direct_paged_bf16()` is `False`.
- A bench guard that the decode step issues no `cache_gather` on the write path
  when the native path is selected.

## Risks and stop-loss

| Risk | Effect | Handling |
| --- | --- | --- |
| The packed-page operator is arch-gated (happened) | `L3` cannot run on 910B3 | `P0` closed it; do not re-plan without an A5 target |
| Something else reads the mirror pools | PD, transfer or snapshot path breaks | enumerate readers first; after `L2` the pages are a single BF16 copy, so `SlotStateExporter` coverage in `test/runtime/test_slot_state.py` must still pass |
| Per-row write volume grows | prefill write bandwidth rises | `L2` writes one BF16 row (~1 KB) instead of a `288 B` packed row plus a gathered mirror row; total written bytes go down, but prefill must be re-measured (the `3100` site is the extend write) |
| `quant_lightning_indexer` is also arch-gated | `L1` blocked | static check **passed**: prebuilt kernels exist for `ascend910b`; still confirm at runtime when wiring `L1`, the way `P0` confirmed the attention kernel |
| Graph replay changes because the page set changes | capture/replay mismatch | run the eager/graph parity tests in `P1`, not at the end |
| Micro-op fusion stalls (each triple needs its own numerics contract) | `P1b` yields far less than the `75.2 ms` the sub-10 µs block nominally costs | `P1b` gate is a single fused triple end to end; if it does not move ≥ 1 ms/layer, stop expanding it and re-rank the remaining micro ops by measured cost |

Stop-loss rule: if after `P1c` the data-movement group is still above `120 ms`, the
mirror was not the lever either; and if `P1b` plus the ranked micro-op work cannot cut
launches per step below ~10k (a `2.7x` reduction from `27,549`), then the recommendation
in `go-no-go-report-20261117.md` (production on vLLM-Ascend, TokenSpeed needs a step-level
rewrite) stands unchanged.

## What this is not

- Not a transplant of `vllm_ascend` modules into the runtime: 35 `vllm` imports
  in that model file, and the repository rule that runtime reaches kernels only
  through `tokenspeed-kernel`.
- Not a parameter search: block sizes, `max_num_seqs`, EP/TP layouts and graph
  modes were already covered by the earlier rounds and none of them touch the
  154.9 ms group.
- Not an `80 ms` plan. The ledger shows `80 ms` needs ~90% of the kernel sum
  gone, which is a different scope than this document.
