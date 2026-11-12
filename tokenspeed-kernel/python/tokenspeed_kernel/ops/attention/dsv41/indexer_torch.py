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

"""Torch Full/Reindex indexer over gathered FlatKV index-K rows.

Used as the Ascend ``dsv41_index_topk`` leaf until a native lightning-indexer
path consumes PA_BBND pages directly. Scores are ``sum_h weight_h * relu(q_h·k)``.
"""

from __future__ import annotations

import torch

from tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch import cache_gather_torch

_PAGE_ROWS = 64


def _index_format(index_cache: torch.Tensor) -> str:
    width = int(index_cache.shape[2])
    if width == 68:
        return "index"
    if width == 132:
        raise ValueError("Ascend torch indexer expects v41 index rows (68B), not index_v4")
    raise ValueError(f"unsupported index cache row width {width}")


def _logical_slots(
    page_table: torch.Tensor, visible_lens: torch.Tensor
) -> tuple[torch.Tensor, int]:
    tokens, pages = page_table.shape
    max_cols = pages * _PAGE_ROWS
    logical = torch.arange(max_cols, device=page_table.device, dtype=torch.int64)
    page_col = torch.div(logical, _PAGE_ROWS, rounding_mode="floor")
    row = logical % _PAGE_ROWS
    safe_col = page_col.clamp_max(max(pages - 1, 0))
    page_ids = page_table[:, safe_col].to(torch.int64)
    valid = (
        (logical[None, :] < visible_lens.to(torch.int64)[:, None])
        & (page_col[None, :] < pages)
        & (page_ids > 0)
    )
    slots = torch.where(
        valid, page_ids * _PAGE_ROWS + row[None, :], torch.full((), -1, device=page_table.device, dtype=torch.int64)
    )
    return slots, max_cols


def _pack_topk(
    scores: torch.Tensor,
    visible_lens: torch.Tensor,
    topk: int,
    candidate_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.masked_fill(
        torch.arange(scores.shape[1], device=scores.device)[None, :]
        >= visible_lens.to(torch.int64)[:, None],
        float("-inf"),
    )
    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
    width = min(topk, scores.shape[1])
    vals, idx = torch.topk(scores, k=width, dim=-1)
    valid = vals > float("-inf")
    # Stable position order among selected IDs.
    ordered = idx.masked_fill(~valid, scores.shape[1] + 1).sort(dim=-1).values
    rows = ordered.masked_fill(ordered > scores.shape[1], -1).to(torch.int32)
    if width < topk:
        rows = torch.nn.functional.pad(rows, (0, topk - width), value=-1)
    lengths = valid.sum(dim=-1, dtype=torch.int32)
    return rows, lengths


def _candidate_blocks_from_rows(
    rows: torch.Tensor,
    lengths: torch.Tensor,
    visible_lens: torch.Tensor,
    candidate_topk: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smallest unique blocks in the selected prefix, plus the newest visible block.

    ``candidate_topk`` is a fixed width. Counts stay on device so graph capture
    does not read one length per token back to the host.
    """
    tokens = rows.shape[0]
    blocks = torch.full(
        (tokens, candidate_topk), -1, dtype=torch.int32, device=rows.device
    )
    block_lens = torch.zeros(tokens, dtype=torch.int32, device=rows.device)
    if candidate_topk == 0 or tokens == 0:
        return blocks, block_lens
    width = rows.shape[1]
    vis = visible_lens.to(torch.int64)
    newest = torch.div((vis - 1).clamp(min=0), block_size, rounding_mode="floor")
    if width == 0:
        pool = newest[:, None]
        keep = vis[:, None] > 0
    else:
        cols = torch.arange(width, device=rows.device)
        take = cols[None, :] < lengths.to(torch.int64)[:, None].clamp(0, width)
        valid = take & (rows >= 0)
        block = torch.div(rows.to(torch.int64).clamp(min=0), block_size, rounding_mode="floor")
        hi = torch.maximum(block.amax(), newest.amax()) + 1
        keyed = torch.where(valid, block, hi)
        already = (valid & (block == newest[:, None])).any(dim=-1)
        extra = torch.where((vis > 0) & ~already, newest, hi)
        pool = torch.cat((keyed, extra[:, None]), dim=-1).sort(dim=-1).values
        prev = torch.cat((pool.new_full((tokens, 1), -1), pool[:, :-1]), dim=-1)
        keep = (pool < hi) & (pool != prev)
    order = keep.to(torch.int64).cumsum(dim=-1)
    chosen = keep & (order <= candidate_topk)
    rank = torch.where(chosen, order - 1, candidate_topk)
    dest = pool.new_full((tokens, candidate_topk + 1), -1)
    dest.scatter_(1, rank, torch.where(chosen, pool, 0))
    blocks.copy_(dest[:, :candidate_topk].to(torch.int32))
    block_lens.copy_(order[:, -1].clamp(max=candidate_topk).to(torch.int32))
    return blocks, block_lens


def _candidate_mask(
    candidate_blocks: torch.Tensor, max_cols: int, block_size: int
) -> torch.Tensor:
    """True for columns covered by a non-negative candidate block."""
    tokens = candidate_blocks.shape[0]
    mask = torch.zeros(tokens, max_cols, dtype=torch.bool, device=candidate_blocks.device)
    if max_cols == 0 or candidate_blocks.numel() == 0:
        return mask
    blocks = candidate_blocks.to(torch.int64)
    offset = torch.arange(block_size, device=blocks.device)
    index = blocks.unsqueeze(-1) * block_size + offset
    live = (blocks >= 0).unsqueeze(-1) & (index >= 0) & (index < max_cols)
    flat = torch.zeros(tokens, max_cols + 1, dtype=torch.bool, device=blocks.device)
    flat.scatter_(
        1,
        torch.where(live, index, max_cols).reshape(tokens, -1),
        live.reshape(tokens, -1),
    )
    mask.copy_(flat[:, :max_cols])
    return mask


def index_topk_torch(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_lens: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    topk: int,
    candidate_topk: int,
    candidate_block_size: int,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full/Reindex Top-K over FlatKV index pages (device torch path)."""
    if index_q.ndim != 3 or index_q.shape[-1] != 128:
        raise ValueError("index_q must be BF16 [T, H, 128]")
    if weights.shape != index_q.shape[:2]:
        raise ValueError("weights must be [T, H]")
    if candidate_block_size != 8:
        raise ValueError("candidate_block_size must be 8")
    tokens = index_q.shape[0]
    if out is None:
        rows = torch.empty((tokens, topk), dtype=torch.int32, device=index_q.device)
        lens = torch.empty(tokens, dtype=torch.int32, device=index_q.device)
        blocks = torch.empty(
            (tokens, candidate_topk), dtype=torch.int32, device=index_q.device
        )
        block_lens = torch.empty(tokens, dtype=torch.int32, device=index_q.device)
    else:
        rows, lens, blocks, block_lens = out
    if tokens == 0:
        return rows, lens, blocks, block_lens

    fmt = _index_format(index_cache)
    slots, max_cols = _logical_slots(page_table, visible_lens)
    keys = cache_gather_torch(index_cache, slots, fmt, None)  # [T, K, 128]
    # scores[t, k] = sum_h weight[t,h] * relu(q[t,h]·key[t,k])
    q = index_q.float()
    k = keys.float()
    # [T, H, K]
    dots = torch.einsum("thd,tkd->thk", q, k).relu()
    scores = torch.einsum("th,thk->tk", weights.float(), dots)

    cand_mask = None
    if candidate_blocks is not None:
        cand_mask = _candidate_mask(candidate_blocks, max_cols, candidate_block_size)

    sel_rows, sel_lens = _pack_topk(scores, visible_lens, topk, cand_mask)
    rows.copy_(sel_rows)
    lens.copy_(sel_lens)
    if candidate_topk > 0:
        cb, cl = _candidate_blocks_from_rows(
            rows, lens, visible_lens, candidate_topk, candidate_block_size
        )
        blocks.copy_(cb)
        block_lens.copy_(cl)
    else:
        blocks.fill_(-1)
        block_lens.zero_()
    return rows, lens, blocks, block_lens
