# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend DeepSeek V4.1 FlatKV attention registration.

Registers ``ascend_deepseek_v41_attention`` once an NPU is visible. Callers
adapt LCM FlatKV selections through ``dsv41.pages`` before this leaf.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch import (
    cache_gather_torch,
    cache_pack_torch,
    cache_scatter_torch,
    cache_unpack_torch,
)
from tokenspeed_kernel.ops.attention.dsv41.compressor_torch import (
    compressor_pool_torch,
    compressor_tail_scatter_torch,
)
from tokenspeed_kernel.ops.attention.dsv41.indexer_torch import index_topk_torch
from tokenspeed_kernel.ops.attention.dsv41.pages import (
    PaBbndSelection,
    adapt_flatkv_decode_selections,
    a2_cmp_topk,
    align_pa_block_for_a2,
    decode_cu_seqlens_q,
    host_seqused_max,
    pad_sparse_indices,
    prefill_workspace_selections,
    repeat_query_heads,
    selected_attention_dense,
)
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.dsv41 import (
        npu_sparse_flash_mla,
        npu_sparse_flash_mla_metadata,
    )

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _FP32_SIG = frozenset({format_signature(x=dense_tensor_format(torch.float32))})
    _BYTE_SIG = frozenset({format_signature(x=dense_tensor_format(torch.uint8))})
    _BF16_SIG = frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))})
    # The metadata kernel writes 1024 int32s. That buffer already holds the
    # full per-core schedule (36 cube records of 9 and 72 vector records of 8).
    # Zero-filling another length changes that stride.
    _SPARSE_MLA_METADATA_LEN = 1024
    # The A2 example launches 64 query heads. TP ranks may hold fewer; the
    # vector tiles are 16-wide, so repeat up to the example width and slice back.
    _SPARSE_MLA_QUERY_HEADS = 64

    def _metadata_from_kernel(metadata: torch.Tensor) -> torch.Tensor:
        if metadata.ndim != 1 or metadata.shape[0] != _SPARSE_MLA_METADATA_LEN:
            raise ValueError(
                "SparseFlashMla metadata must be the kernel's "
                f"{_SPARSE_MLA_METADATA_LEN}-int32 vector, got {tuple(metadata.shape)}"
            )
        return metadata

    @register_kernel(
        "attention",
        "dsv41_sparse_flash_mla",
        name="ascend_deepseek_v41_attention",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("q", "dense", {torch.bfloat16}),
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_deepseek_v41_attention(
        q: torch.Tensor,
        *,
        ori_kv: torch.Tensor,
        cmp_kv: torch.Tensor | None,
        ori_sparse_indices: torch.Tensor | None,
        ori_topk_length: torch.Tensor | None,
        cmp_sparse_indices: torch.Tensor | None,
        ori_block_table: torch.Tensor,
        cmp_block_table: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        seqused_ori_kv: torch.Tensor,
        seqused_cmp_kv: torch.Tensor | None,
        cmp_residual_kv: torch.Tensor | None,
        sinks: torch.Tensor | None,
        metadata: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        ori_mask_mode: int,
        cmp_mask_mode: int,
        ori_win_left: int,
        ori_win_right: int,
        layout_q: str = "TND",
        layout_kv: str = "PA_BBND",
        topk_value_mode: int = 1,
        return_softmax_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Leaf entry for V4.1 sparse flash MLA after FlatKV pages are adapted."""
        return npu_sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            ori_topk_length=ori_topk_length,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=_metadata_from_kernel(metadata),
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            topk_value_mode=topk_value_mode,
            return_softmax_lse=return_softmax_lse,
        )

    def ascend_selected_attention_from_flatkv_rows(
        q: torch.Tensor,
        *,
        swa_rows: torch.Tensor,
        swa_lens: torch.Tensor,
        global_rows: torch.Tensor | None,
        global_lens: torch.Tensor | None,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        seq_lens: torch.Tensor,
        # A2 OPP: non-zero ori_topk (DSpark / sparse SWA) requires non-negative
        # window bounds; -1 is rejected on this product line.
        ori_win_left: int = 127,
        ori_win_right: int = 0,
    ) -> torch.Tensor:
        """Run sparse MLA on BF16 rows gathered from FlatKV physical slots.

        Materializes each query's SWA/global selection as one compact PA_BBND
        block so packed LCM pages can drive ``npu_sparse_flash_mla`` without a
        second cache geometry. Decode only: one query token per request row.
        """
        ori, cmp, residual = adapt_flatkv_decode_selections(
            swa_rows=swa_rows,
            swa_lens=swa_lens,
            global_rows=global_rows,
            global_lens=global_lens,
            cmp_ratio=cmp_ratio if global_rows is not None else 0,
            seq_lens=seq_lens,
        )
        return _launch_pa_bbnd_decode(
            q,
            ori=ori,
            cmp=cmp,
            residual=residual,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio if cmp is not None else 0,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
        )

    def ascend_prefill_attention_from_workspace(
        q: torch.Tensor,
        workspace: torch.Tensor,
        indices: torch.Tensor,
        *,
        swa_width: int,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        seq_lens: torch.Tensor,
        ori_win_left: int,
        ori_win_right: int = 0,
    ) -> torch.Tensor:
        """Run the decode sparse-MLA launch on a prefill key workspace.

        The workspace holds one request's SWA rows, then its compressed rows.
        ``indices`` selects both. They become the same ori/cmp PA_BBND blocks
        decode builds from FlatKV gathers, and the same launch scores them.
        """
        ori, cmp, residual = prefill_workspace_selections(
            workspace, indices, swa_width, cmp_ratio, seq_lens
        )
        return _launch_pa_bbnd_decode(
            q,
            ori=ori,
            cmp=cmp,
            residual=residual,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            cmp_ratio=0 if cmp is None else cmp_ratio,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
        )

    def _launch_pa_bbnd_decode(
        q: torch.Tensor,
        *,
        ori: PaBbndSelection,
        cmp: PaBbndSelection | None,
        residual: torch.Tensor | None,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        ori_win_left: int,
        ori_win_right: int,
    ) -> torch.Tensor:
        tokens = q.shape[0]
        live_heads = q.shape[1]
        if live_heads < 1 or q.shape[-1] != 512:
            raise ValueError(
                f"sparse MLA query must be [tokens, heads, 512], got {tuple(q.shape)}"
            )
        q, sinks = repeat_query_heads(q, attn_sink, _SPARSE_MLA_QUERY_HEADS)
        heads = q.shape[1]
        ori = align_pa_block_for_a2(ori)
        index_width = 0
        if cmp is not None:
            cmp = align_pa_block_for_a2(cmp)
            # cmp_topk is the index width, 512 or 1024. The KV block stays the
            # compact row count. Extra index columns are -1 and are not read.
            index_width = a2_cmp_topk(int(cmp.sparse_indices.shape[-1]))
            if cmp.sparse_indices.shape[-1] != index_width:
                cmp = PaBbndSelection(
                    cmp.kv,
                    cmp.block_table,
                    torch.nn.functional.pad(
                        cmp.sparse_indices,
                        (0, index_width - cmp.sparse_indices.shape[-1]),
                        value=-1,
                    ),
                    cmp.topk_length,
                )
        block_rows = int(ori.kv.shape[1])
        cu_seqlens_q = decode_cu_seqlens_q(tokens, device=q.device)
        seqused_ori = (
            swa_lens_as_seqused(ori.topk_length).to(torch.int32).clamp(0, block_rows)
        )
        seqused_cmp = (
            swa_lens_as_seqused(cmp.topk_length)
            .to(torch.int32)
            .clamp(0, int(cmp.kv.shape[1]))
            if cmp is not None
            else None
        )
        # A2/A3 SparseFlashMla binaries only compile SWA and CSA templates.
        # ori_mask_mode 0 with a non-zero ori_topk selects ORI_SPARSE, which
        # has no A2 tiling key. Compact rows are already the selected window,
        # so the causal band (mode 4, ori_topk 0) matches vLLM's SWA path.
        # cmp_topk is the padded index width. max_seqlen_cmp_kv stays the
        # compact block length, matching vLLM's split of those two arguments.
        has_cmp = cmp is not None
        ori_mask_mode = 4
        cmp_mask_mode = 3 if has_cmp else 0
        win_left = max(int(ori_win_left), 0)
        win_right = max(int(ori_win_right), 0)
        cmp_sparse = None
        cmp_topk = 0
        max_seqlen_cmp = 0
        if has_cmp:
            max_seqlen_cmp = int(cmp.kv.shape[1])
            cmp_topk = index_width
            cmp_sparse = cmp.sparse_indices
            # cmp_mask_mode 3 with cmp_ratio other than 1 requires a residual
            # vector. Ratio-1 layers have none; zeros mean the compressed
            # block already covers the sequence.
            if residual is None:
                residual = torch.zeros(tokens, dtype=torch.int32, device=q.device)
        # Device .max() here is one stream drain per attention layer. Host
        # lengths still skip an empty launch; device lengths launch.
        ori_max = host_seqused_max(seqused_ori)
        cmp_max = host_seqused_max(seqused_cmp)
        if ori_max == 0 and cmp_max == 0:
            return q.new_zeros((tokens, live_heads, q.shape[-1]))
        metadata = npu_sparse_flash_mla_metadata(
            heads,
            1,
            q.shape[-1],
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=seqused_ori,
            seqused_cmp_kv=seqused_cmp,
            cmp_residual_kv=residual,
            batch_size=tokens,
            max_seqlen_q=1,
            max_seqlen_ori_kv=block_rows,
            max_seqlen_cmp_kv=max_seqlen_cmp,
            ori_topk=0,
            ori_topk_length=None,
            cmp_topk=cmp_topk,
            # Metadata keeps the model compress_ratio (0 SWA / 1 / 2) so
            # residual geometry stays correct; the kernel attr below may differ.
            cmp_ratio=0 if not has_cmp else cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=win_left,
            ori_win_right=win_right,
            has_ori_kv=True,
            has_cmp_kv=has_cmp,
        )
        metadata = _metadata_from_kernel(metadata)
        # Same ratio the metadata kernel just consumed. vLLM on this image
        # passes 0 without cmp_kv and the model ratio (1 or 2) with it.
        # Call the OPP wrapper directly so optional cmp tensors are not
        # dropped by kernel-dispatch wrapping.
        kernel_cmp_ratio = 0 if not has_cmp else int(cmp_ratio)

        def _attend(metadata_vector: torch.Tensor) -> torch.Tensor:
            attended, _ = npu_sparse_flash_mla(
                q,
                ori_kv=ori.kv.contiguous(),
                cmp_kv=None if not has_cmp else cmp.kv.contiguous(),
                ori_sparse_indices=None,
                ori_topk_length=None,
                cmp_sparse_indices=cmp_sparse,
                ori_block_table=ori.block_table,
                cmp_block_table=None if not has_cmp else cmp.block_table,
                cu_seqlens_q=cu_seqlens_q,
                seqused_ori_kv=seqused_ori,
                seqused_cmp_kv=seqused_cmp,
                cmp_residual_kv=residual,
                sinks=sinks,
                metadata=metadata_vector,
                softmax_scale=softmax_scale,
                cmp_ratio=kernel_cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=win_left,
                ori_win_right=win_right,
            )
            return attended

        out = _attend(metadata)
        return out[:, :live_heads]

    def swa_lens_as_seqused(topk_length: torch.Tensor) -> torch.Tensor:
        """Flatten ``[T, 1]`` topk lengths to the ``[T]`` seqused vector."""
        return topk_length.reshape(-1).contiguous()

    @register_kernel(
        "attention",
        "dsv41_compressor_pool",
        name="ascend_dsv41_compressor_pool",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_FP32_SIG,
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_compressor_pool(
        content,
        scores,
        previous,
        tail,
        tail_slots,
        active,
        out,
        norm_weight,
        norm_eps,
    ):
        """Ascend leaf: torch pool over the TokenSpeed 2-row compressor tail."""
        return compressor_pool_torch(
            content,
            scores,
            previous,
            tail,
            tail_slots,
            active,
            out,
            norm_weight,
            norm_eps,
        )

    @register_kernel(
        "attention",
        "dsv41_compressor_tail_scatter",
        name="ascend_dsv41_compressor_tail_scatter",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_FP32_SIG,
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_compressor_tail_scatter(content, scores, tail, slots):
        """Ascend leaf: write FP32 projections into the 2-row compressor tail."""
        compressor_tail_scatter_torch(content, scores, tail, slots)

    # PIN: Ascend uses torch SPECIALIZED FlatKV codecs; Triton FlatKV stays
    # nvidia/amd-only until single-NPU dtype/UB probes pass.
    @register_kernel(
        "attention",
        "dsv41_cache_gather",
        name="ascend_dsv41_cache_gather",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_BYTE_SIG,
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_cache_gather(cache, slots, cache_format, out):
        return cache_gather_torch(cache, slots, cache_format, out)

    @register_kernel(
        "attention",
        "dsv41_cache_scatter",
        name="ascend_dsv41_cache_scatter",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_BF16_SIG | _FP32_SIG,
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_cache_scatter(rows, cache, slots, cache_format):
        cache_scatter_torch(rows, cache, slots, cache_format)

    @register_kernel(
        "attention",
        "dsv41_cache_unpack",
        name="ascend_dsv41_cache_unpack",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_BYTE_SIG,
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_cache_unpack(packed, cache_format, out):
        return cache_unpack_torch(packed, cache_format, out)

    @register_kernel(
        "attention",
        "dsv41_cache_pack",
        name="ascend_dsv41_cache_pack",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_BF16_SIG | _FP32_SIG,
        traits={},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_cache_pack(rows, cache_format, out):
        return cache_pack_torch(rows, cache_format, out)

    @register_kernel(
        "attention",
        "dsv41_index_topk",
        name="ascend_dsv41_index_topk",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=_BF16_SIG,
        traits={
            "index_heads": frozenset(range(1, 65)),
            "index_k_format": frozenset({"mxfp4", "unknown"}),
            "index_shards": frozenset({1}),
            "native_indexer": frozenset({False}),
            "supports_index_query_input": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_index_topk(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
        index_metadata,
    ):
        """Gather FlatKV index pages and run torch Top-K (Ascend leaf)."""
        del query_chunk_size, score_chunk_size, index_metadata
        if process_group is not None:
            raise NotImplementedError(
                "Ascend V4.1 indexer currently requires unreplicated index heads"
            )
        from tokenspeed_kernel.ops.attention.dsv41._inputs import IndexQueryInput

        if isinstance(index_q, IndexQueryInput):
            raise NotImplementedError(
                "Ascend V4.1 indexer expects post-RoPE index_q tensors"
            )
        return index_topk_torch(
            index_q,
            weights,
            index_cache,
            page_table,
            visible_lens,
            candidate_blocks,
            topk,
            candidate_topk,
            candidate_block_size,
            out,
        )

    @register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="ascend_dsv41_selected_attention",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
        traits={"flashmla_eligible": frozenset({False})},
        priority=Priority.SPECIALIZED,
        features={"ascend", "flatkv"},
    )
    def ascend_dsv41_selected_attention(
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        softmax_scale,
        out,
        query_chunk_size,
        schedule,
        prefill_kv,
        prefill_indices,
        *,
        lse=None,
    ):
        """Ascend selected attention: dense prefill workspace or gathered pages."""
        del query_chunk_size, schedule
        if attn_sink is not None:
            attn_sink = attn_sink[: q.shape[1]]
        if prefill_kv is not None:
            if prefill_indices is None:
                raise ValueError("prefill_kv requires prefill_indices")
            return selected_attention_dense(
                q, prefill_kv, prefill_indices, attn_sink, softmax_scale, out, lse
            )
        if prefill_indices is not None:
            raise ValueError("prefill_indices requires prefill_kv")
        # Decode/paged path: gather SWA (+ global) into per-query key rows.
        swa_rows = cache_gather_torch(swa_cache, swa_slots, "swa", None)
        pieces = [swa_rows]
        if global_cache is not None:
            if global_slots is None or global_lens is None:
                raise ValueError("global cache requires slots and lengths")
            gfmt = {288: "global", 584: "global_v4"}.get(int(global_cache.shape[2]))
            if gfmt != "global":
                raise NotImplementedError(
                    "Ascend selected attention expects v41 global"
                )
            pieces.append(cache_gather_torch(global_cache, global_slots, gfmt, None))
        keys = torch.cat(pieces, dim=1)  # [T, W, 512]
        tokens, width, _ = keys.shape
        indices = (
            torch.arange(width, device=q.device, dtype=torch.int32)[None, :]
            .expand(tokens, -1)
            .contiguous()
        )
        valid = swa_slots >= 0
        if swa_lens is not None:
            cols = torch.arange(swa_slots.shape[-1], device=q.device)
            valid = valid & (cols[None, :] < swa_lens.to(torch.int64)[:, None])
        if global_cache is not None:
            g_valid = global_slots >= 0
            if global_lens is not None:
                g_cols = torch.arange(global_slots.shape[-1], device=q.device)
                g_valid = g_valid & (
                    g_cols[None, :] < global_lens.to(torch.int64)[:, None]
                )
            valid = torch.cat((valid, g_valid), dim=-1)
        indices = indices.masked_fill(~valid, -1)
        # Score against per-query keys directly (indices address the W axis).
        scores = torch.einsum("thd,twd->thw", q.float(), keys.float()) * float(
            softmax_scale
        )
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        if attn_sink is not None:
            sink = (
                attn_sink.to(torch.float32).view(1, q.shape[1], 1).expand(tokens, -1, 1)
            )
            scores = torch.cat((scores, sink), dim=-1)
        probs = torch.softmax(scores, dim=-1)
        if attn_sink is not None:
            probs = probs[..., :-1]
        probs = probs.masked_fill(~valid[:, None, :], 0)
        result = torch.einsum("thw,twd->thd", probs, keys.float()).to(q.dtype)
        if out is None:
            out = result
        else:
            out.copy_(result)
        if lse is not None:
            base = scores[..., :width] if attn_sink is not None else scores
            lse.copy_(torch.logsumexp(base, dim=-1))
        return out
