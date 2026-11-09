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

"""FlatKV LCM pages → Ascend ``npu_sparse_flash_mla`` PA_BBND ABI.

TokenSpeed FlatKV addresses history with physical slots into page-planar
fields (64 rows per page). Ascend sparse flash MLA reads PA_BBND caches
``[blocks, block_size, 1, 512]`` through INT32 block tables and logical
sparse indices.

Two bridges are provided:

* ``lcm_block_table_as_pa`` / ``as_pa_bbnd`` — identity mapping when the
  cache is already BF16 page-planar with ``block_size == 64``.
* ``materialize_bf16_rows_as_pa_bbnd`` — pack already-gathered BF16 selection
  rows into one compact block per query so packed FlatKV storage can call
  the Ascend op without a second cache layout.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

FLATKV_PAGE_ROWS = 64
SWA_WINDOW = 128
HEAD_DIM = 512


class PaBbndSelection(NamedTuple):
    """One PA_BBND cache view plus the tables/indices that address it."""

    kv: torch.Tensor
    block_table: torch.Tensor
    sparse_indices: torch.Tensor
    topk_length: torch.Tensor


def as_pa_bbnd(cache: torch.Tensor) -> torch.Tensor:
    """View BF16 page-planar cache as PA_BBND ``[blocks, rows, 1, 512]``.

    Accepts ``[P, R, 512]`` or ``[P, R, 1, 512]``. Does not copy.
    """
    if cache.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"PA_BBND view requires BF16/FP16 cache, got {cache.dtype}"
        )
    if cache.ndim == 4:
        if cache.shape[2:] != (1, HEAD_DIM):
            raise ValueError(
                f"PA_BBND cache must end in (1, {HEAD_DIM}), got {tuple(cache.shape)}"
            )
        return cache
    if cache.ndim != 3 or cache.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"page-planar cache must be [P, R, {HEAD_DIM}], got {tuple(cache.shape)}"
        )
    return cache.unsqueeze(2)


def lcm_block_table_as_pa(table: torch.Tensor) -> torch.Tensor:
    """Copy an LCM page-id table to contiguous INT32 for PA_BBND.

    Valid when each LCM page holds ``FLATKV_PAGE_ROWS`` tokens and Ascend
    ``block_size`` equals that page size, so page ids are block ids.
    """
    if table.ndim != 2:
        raise ValueError(f"block table must be rank 2, got {tuple(table.shape)}")
    if table.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"block table must be INT32/INT64, got {table.dtype}")
    return table.to(dtype=torch.int32, copy=False).contiguous()


def identity_block_table(
    num_queries: int, *, device: torch.device, dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    """``[T, 1]`` table mapping query ``i`` onto compact block ``i``."""
    if num_queries < 0:
        raise ValueError("num_queries must be non-negative")
    return torch.arange(num_queries, device=device, dtype=dtype).unsqueeze(1)


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert compact ``[T, K]`` selections into SMLA ``[T, 1, topk]`` INT32."""
    if indices.ndim != 2:
        raise ValueError(f"indices must be [T, K], got {tuple(indices.shape)}")
    if topk < 1:
        raise ValueError(f"topk must be positive, got {topk}")
    width = indices.shape[-1]
    if width > topk:
        raise ValueError(f"indices width {width} exceeds topk {topk}")
    out = indices.to(dtype=torch.int32, copy=False)
    if width < topk:
        out = torch.nn.functional.pad(out, (0, topk - width), value=-1)
    return out.unsqueeze(1).contiguous()


def compact_selection_indices(
    lengths: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense ``[T, 1, W]`` indices ``0..len-1`` for one compact PA block."""
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be rank 1, got {tuple(lengths.shape)}")
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    tokens = lengths.shape[0]
    device = lengths.device
    columns = torch.arange(width, device=device, dtype=torch.int32)
    valid = columns[None, :] < lengths.to(torch.int32)[:, None]
    indices = torch.where(valid, columns[None, :], torch.full((), -1, dtype=torch.int32, device=device))
    topk_length = lengths.to(dtype=torch.int32).clamp(0, width).view(tokens, 1)
    return indices.unsqueeze(1).contiguous(), topk_length.contiguous()


def cmp_residual_kv(seq_lens: torch.Tensor, cmp_ratio: int) -> torch.Tensor | None:
    """Return per-request residual token counts for ratio-2 compressed KV."""
    if cmp_ratio <= 1:
        return None
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be rank 1, got {tuple(seq_lens.shape)}")
    return (seq_lens.to(torch.int32) % int(cmp_ratio)).contiguous()


_A2_PA_BLOCK_ALIGN = 16
_A2_PA_BLOCK_MAX = 1024


def align_pa_block_for_a2(selection: PaBbndSelection) -> PaBbndSelection:
    """Pad one compact PA block to the A2/A3 block-size rule.

    A2 sparse MLA accepts block sizes in ``[16, 1024]`` that are multiples of
    16. Gathered selections can be shorter. Extra columns stay masked (``-1``)
    and ``topk_length`` is unchanged, so the visible rows stay the same.
    """
    if selection.kv.ndim != 4 or selection.kv.shape[2] != 1:
        raise ValueError(
            f"PA_BBND kv must be [blocks, block, 1, dim], got {tuple(selection.kv.shape)}"
        )
    width = int(selection.kv.shape[1])
    aligned = (
        (width + _A2_PA_BLOCK_ALIGN - 1) // _A2_PA_BLOCK_ALIGN
    ) * _A2_PA_BLOCK_ALIGN
    aligned = max(aligned, _A2_PA_BLOCK_ALIGN)
    if aligned > _A2_PA_BLOCK_MAX:
        raise ValueError(
            f"A2 sparse MLA block size {width} aligns to {aligned}, "
            f"above {_A2_PA_BLOCK_MAX}"
        )
    if aligned == width:
        return selection
    return pad_pa_block(selection, aligned)


_A2_CMP_TOPK_BUCKETS = (512, 1024)


def a2_cmp_topk(block_rows: int) -> int:
    """``cmp_topk`` accepted by this image when compressed KV is present.

    ``aclnnSparseFlashMlaMetadata`` only allows 0, 512, or 1024 once
    ``has_cmp_kv`` is set. 0 is the no-compressed-KV case. A present block
    uses the smallest of 512 and 1024 that covers its row count.
    """
    if block_rows < 1:
        raise ValueError(f"compressed block rows must be positive, got {block_rows}")
    for bucket in _A2_CMP_TOPK_BUCKETS:
        if block_rows <= bucket:
            return bucket
    raise ValueError(
        f"compressed block of {block_rows} rows exceeds {_A2_CMP_TOPK_BUCKETS[-1]}"
    )


def pad_pa_block(selection: PaBbndSelection, width: int) -> PaBbndSelection:
    """Pad a compact PA block on the row axis. Extra rows stay masked."""
    if width < 1:
        raise ValueError(f"block width must be positive, got {width}")
    if selection.kv.ndim != 4 or selection.kv.shape[2] != 1:
        raise ValueError(
            f"PA_BBND kv must be [blocks, block, 1, dim], got {tuple(selection.kv.shape)}"
        )
    current = int(selection.kv.shape[1])
    if current > width:
        raise ValueError(f"block width {current} exceeds target {width}")
    if current == width:
        return selection
    pad = width - current
    kv = torch.nn.functional.pad(selection.kv, (0, 0, 0, 0, 0, pad))
    indices = torch.nn.functional.pad(selection.sparse_indices, (0, pad), value=-1)
    return PaBbndSelection(kv, selection.block_table, indices, selection.topk_length)


def repeat_query_heads(
    q: torch.Tensor, sinks: torch.Tensor, target: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat query heads up to ``target`` so A2 vector tiles stay in range.

    Sparse MLA's A2 example launches 64 query heads. Tensor-parallel ranks can
    hold fewer. Repeating heads does not change the first ``live`` heads, which
    the caller slices back off.
    """
    if q.ndim != 3:
        raise ValueError(f"query must be [tokens, heads, dim], got {tuple(q.shape)}")
    live = int(q.shape[1])
    if live < 1 or target < live:
        raise ValueError(f"cannot repeat {live} query heads to {target}")
    sinks = sinks.detach().float().reshape(-1)
    if sinks.numel() < live:
        raise ValueError(
            f"sparse MLA sinks has {sinks.numel()} values for {live} query heads"
        )
    sinks = sinks[:live].contiguous()
    if live == target:
        return q, sinks
    reps = (target + live - 1) // live
    return (
        q.repeat(1, reps, 1)[:, :target].contiguous(),
        sinks.repeat(reps)[:target].contiguous(),
    )


def selected_attention_dense(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
) -> torch.Tensor:
    """Score ``q`` against selected rows of a dense BF16 key workspace.

    ``indices`` addresses the row axis of ``kv``. Negative indices contribute
    nothing. Only those rows are read, so a larger unused workspace does not
    enter the softmax.
    """
    if kv.ndim == 3:
        kv = kv.squeeze(1)
    tokens, heads, _ = q.shape
    width = indices.shape[-1]
    safe = indices.clamp_min(0).long()
    valid = indices >= 0
    gathered = kv[safe]
    gathered = gathered.masked_fill(~valid[..., None], 0)
    scores = torch.einsum("thd,twd->thw", q.float(), gathered.float()) * float(
        softmax_scale
    )
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).view(1, heads, 1).expand(tokens, -1, 1)
        scores = torch.cat((scores, sink), dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if attn_sink is not None:
        probs = probs[..., :-1]
    probs = probs.masked_fill(~valid[:, None, :], 0)
    result = torch.einsum("thw,twd->thd", probs, gathered.float()).to(q.dtype)
    if out is None:
        out = result
    else:
        out.copy_(result)
    if lse is not None:
        base = scores[..., :width] if attn_sink is not None else scores
        lse.copy_(torch.logsumexp(base, dim=-1))
    return out


def compact_selected_rows(
    kv: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather valid ``indices`` into a dense prefix, preserving column order.

    ``kv`` is ``[rows, dim]`` or ``[rows, 1, dim]``. ``indices`` is ``[T, W]``.
    Negative entries are dropped. The result width is the longest valid
    selection, so an unselected workspace row is never copied.
    """
    if kv.ndim == 3:
        kv = kv.squeeze(1)
    if kv.ndim != 2:
        raise ValueError(f"kv must be [rows, dim], got {tuple(kv.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be [tokens, width], got {tuple(indices.shape)}")
    valid = indices >= 0
    lengths = valid.sum(dim=-1).to(torch.int32)
    width = int(lengths.max()) if lengths.numel() else 0
    rows = kv.new_zeros((indices.shape[0], width, kv.shape[-1]))
    if width == 0:
        return rows, lengths
    place = valid.to(torch.int64).cumsum(dim=-1) - 1
    gathered = kv[indices.clamp_min(0).to(torch.int64)]
    token = torch.arange(indices.shape[0], device=kv.device)[:, None].expand_as(indices)
    rows[token[valid], place[valid]] = gathered[valid]
    return rows, lengths


def prefill_workspace_selections(
    kv: torch.Tensor,
    indices: torch.Tensor,
    swa_width: int,
    cmp_ratio: int,
    seq_lens: torch.Tensor,
) -> tuple[PaBbndSelection, PaBbndSelection | None, torch.Tensor | None]:
    """Split a prefill key workspace into the decode sparse-MLA blocks.

    ``indices[:, :swa_width]`` address the SWA prefix of ``kv``. Later columns
    address the compressed rows packed after it. Each query's valid keys are
    compacted in that column order, which is the set the dense softmax reads.
    An empty compressed selection returns no cmp block.
    """
    if indices.ndim != 2:
        raise ValueError(f"indices must be [tokens, width], got {tuple(indices.shape)}")
    if swa_width < 0 or swa_width > indices.shape[-1]:
        raise ValueError(
            f"swa_width must be in [0, {indices.shape[-1]}], got {swa_width}"
        )
    if seq_lens.shape != (indices.shape[0],):
        raise ValueError(
            f"seq_lens must be [T]={indices.shape[0]}, got {tuple(seq_lens.shape)}"
        )
    ori_rows, ori_lens = compact_selected_rows(kv, indices[:, :swa_width])
    if ori_rows.shape[1] == 0:
        dim = kv.shape[-1]
        ori_rows = kv.new_zeros((indices.shape[0], 1, dim))
    ori = materialize_bf16_rows_as_pa_bbnd(ori_rows, ori_lens)
    cmp_indices = indices[:, swa_width:]
    if cmp_indices.shape[-1] == 0:
        return ori, None, None
    cmp_rows, cmp_lens = compact_selected_rows(kv, cmp_indices)
    if cmp_rows.shape[1] == 0:
        return ori, None, None
    if cmp_ratio not in (1, 2):
        raise ValueError(
            f"compressed prefill rows require cmp_ratio 1 or 2, got {cmp_ratio}"
        )
    return (
        ori,
        materialize_bf16_rows_as_pa_bbnd(cmp_rows, cmp_lens),
        cmp_residual_kv(seq_lens, cmp_ratio),
    )


def materialize_bf16_rows_as_pa_bbnd(
    rows: torch.Tensor, lengths: torch.Tensor
) -> PaBbndSelection:
    """Pack gathered BF16 ``[T, K, 512]`` rows into one PA_BBND block per query.

    Each query owns compact block ``i`` of ``block_size = K``. Invalid trailing
    columns (beyond ``lengths[i]``) stay in the tensor but are masked out of
    ``sparse_indices`` / ``topk_length``.
    """
    if rows.ndim != 3 or rows.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"rows must be BF16 [T, K, {HEAD_DIM}], got {tuple(rows.shape)}"
        )
    if rows.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"rows must be BF16/FP16, got {rows.dtype}")
    if lengths.shape != (rows.shape[0],):
        raise ValueError(
            f"lengths must be [T]={rows.shape[0]}, got {tuple(lengths.shape)}"
        )
    tokens, width, _ = rows.shape
    kv = rows.unsqueeze(2).contiguous()
    block_table = identity_block_table(tokens, device=rows.device)
    sparse_indices, topk_length = compact_selection_indices(lengths, width)
    return PaBbndSelection(kv, block_table, sparse_indices, topk_length)


def decode_cu_seqlens_q(num_queries: int, *, device: torch.device) -> torch.Tensor:
    """Decode ``cu_seqlens_q`` for one query token per request: ``0..T``."""
    return torch.arange(num_queries + 1, device=device, dtype=torch.int32)


def materialize_index_history_as_pa_bbnd(
    index_rows: torch.Tensor, seq_lens: torch.Tensor, *, block_size: int | None = None
) -> PaBbndSelection:
    """Pack gathered BF16 index-K history ``[T, K, 128]`` into PA_BBND blocks.

    Ascend ``npu_lightning_indexer`` expects PA_BBND keys. TokenSpeed FlatKV
    index rows are gathered/dequantized first (128-D), then compacted here the
    same way as attention selections. ``block_size`` defaults to ``K``.
    """
    if index_rows.ndim != 3 or index_rows.shape[-1] != 128:
        raise ValueError(
            f"index rows must be BF16 [T, K, 128], got {tuple(index_rows.shape)}"
        )
    if index_rows.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"index rows must be BF16/FP16, got {index_rows.dtype}")
    if seq_lens.shape != (index_rows.shape[0],):
        raise ValueError(
            f"seq_lens must be [T]={index_rows.shape[0]}, got {tuple(seq_lens.shape)}"
        )
    width = index_rows.shape[1] if block_size is None else int(block_size)
    if width != index_rows.shape[1]:
        raise ValueError("block_size must match the gathered index width")
    tokens = index_rows.shape[0]
    kv = index_rows.unsqueeze(2).contiguous()
    block_table = identity_block_table(tokens, device=index_rows.device)
    sparse_indices, topk_length = compact_selection_indices(seq_lens, width)
    return PaBbndSelection(kv, block_table, sparse_indices, topk_length)


def adapt_flatkv_decode_selections(
    *,
    swa_rows: torch.Tensor,
    swa_lens: torch.Tensor,
    global_rows: torch.Tensor | None,
    global_lens: torch.Tensor | None,
    cmp_ratio: int,
    seq_lens: torch.Tensor,
) -> tuple[PaBbndSelection, PaBbndSelection | None, torch.Tensor | None]:
    """Adapt FlatKV decode gathers into Ascend sparse-MLA page arguments.

    ``swa_rows`` / ``global_rows`` are already dequantized BF16 selections
    (typically from ``cache_gather``). Returns ``(ori, cmp_or_None, residual)``.
    """
    ori = materialize_bf16_rows_as_pa_bbnd(swa_rows, swa_lens)
    if global_rows is None:
        if global_lens is not None:
            raise ValueError("global_lens requires global_rows")
        if cmp_ratio not in (0,):
            # SWA-only layers pass ratio 0; compressed layers must supply rows.
            if cmp_ratio in (1, 2):
                raise ValueError(
                    f"cmp_ratio={cmp_ratio} requires gathered global_rows"
                )
        return ori, None, None
    if global_lens is None:
        raise ValueError("global_rows requires global_lens")
    if cmp_ratio not in (1, 2):
        raise ValueError(f"compressed global KV requires cmp_ratio 1 or 2, got {cmp_ratio}")
    cmp = materialize_bf16_rows_as_pa_bbnd(global_rows, global_lens)
    residual = cmp_residual_kv(seq_lens, cmp_ratio)
    return ori, cmp, residual
