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

"""Device-portable torch reference for V4.1 ratio-2 compressor pool/scatter.

Matches the Triton ``dsv41_compressor_pool`` numerics (per-channel log-sum-exp
weights, BF16 round before RMSNorm). Used as the Ascend leaf and for host
tests that cannot import Triton.
"""

from __future__ import annotations

import torch


def compressor_pool_torch(
    content: torch.Tensor,
    scores: torch.Tensor,
    previous: torch.Tensor,
    tail: torch.Tensor,
    tail_slots: torch.Tensor,
    active: torch.Tensor,
    out: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Pool ratio-2 pairs from current rows and the 2-row compressor tail."""
    if (
        content.ndim != 2
        or content.shape[1] != 512
        or scores.shape != content.shape
        or content.dtype != torch.float32
        or scores.dtype != torch.float32
        or tail.ndim != 4
        or tail.shape[1:] != (2, 2, 512)
        or tail.dtype != torch.float32
    ):
        raise ValueError(
            "Compressor pooling requires FP32 [N,512] and [pages,2,2,512] tail"
        )
    count = content.shape[0]
    for value in (previous, tail_slots):
        if value.shape != (count,) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Compressor previous rows and slots must be integer [N]")
    if active.shape != (count,) or active.dtype != torch.bool:
        raise ValueError("Compressor active mask must be bool [N]")
    if any(
        x.device != content.device for x in (scores, previous, tail, tail_slots, active)
    ):
        raise ValueError("Compressor operands must share one device")
    dtype = torch.float32 if norm_weight is None else torch.bfloat16
    if norm_weight is not None and (
        norm_weight.shape != (512,)
        or norm_weight.device != content.device
        or norm_eps <= 0
    ):
        raise ValueError(
            "Compressor normalization requires a device [512] weight and positive epsilon"
        )
    if out is None:
        out = torch.empty(content.shape, device=content.device, dtype=dtype)
    elif (
        out.shape != content.shape or out.dtype != dtype or out.device != content.device
    ):
        raise ValueError("Compressor output must match [N,512] on the input device")
    if count == 0:
        return out

    pages = tail.shape[0]
    live = active
    in_current = live & (previous >= 0) & (previous < count)
    in_tail = live & (previous < 0) & (tail_slots >= 2) & (tail_slots < pages * 2)

    safe_prev = previous.clamp(0, max(count - 1, 0))
    zeros = content.new_zeros(content.shape)
    old_content = torch.where(in_current[:, None], content[safe_prev], zeros)
    old_scores = torch.where(in_current[:, None], scores[safe_prev], zeros)

    capacity = max(pages * 2 - 1, 0)
    safe_slots = tail_slots.clamp(0, capacity)
    page = torch.div(safe_slots, 2, rounding_mode="floor")
    row = safe_slots % 2
    tail_content = tail[page, row, 0]
    tail_scores = tail[page, row, 1]
    old_content = torch.where(in_tail[:, None], tail_content, old_content)
    old_scores = torch.where(in_tail[:, None], tail_scores, old_scores)

    current = torch.where(live[:, None], content, zeros)
    live_scores = torch.where(live[:, None], scores, zeros)
    maximum = torch.maximum(old_scores, live_scores)
    old_exp = torch.exp(old_scores - maximum)
    new_exp = torch.exp(live_scores - maximum)
    denominator = old_exp + new_exp
    pooled = old_content * (old_exp / denominator) + current * (new_exp / denominator)
    pooled = torch.where(live[:, None], pooled, zeros)

    if norm_weight is not None:
        normalized = pooled.to(torch.bfloat16).float()
        inv_rms = torch.rsqrt(normalized.square().mean(-1, keepdim=True) + norm_eps)
        pooled = (normalized * inv_rms * norm_weight.float()).to(torch.bfloat16)
    out.copy_(pooled)
    return out


def _winning_rows(
    key: torch.Tensor, active: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Repeat the last active row across rows that share a tail slot."""
    count = key.shape[0]
    if count <= 1:
        return values
    priority = torch.where(
        active,
        torch.arange(count, device=key.device),
        torch.full((count,), -1, device=key.device),
    )
    order = torch.argsort(
        key.to(torch.int64) * (count + 1) + (priority.to(torch.int64) + 1)
    )
    grouped = key.to(torch.int64)[order]
    is_end = torch.ones(count, dtype=torch.bool, device=key.device)
    is_end[:-1] = grouped[:-1] != grouped[1:]
    position = torch.arange(count, device=key.device)
    nearest_end = torch.cummin(
        torch.where(is_end, position, torch.full_like(position, count)).flip(0),
        dim=0,
    ).values.flip(0)
    inverse = torch.empty_like(order)
    inverse[order] = position
    return values[order[nearest_end[inverse]]]


def compressor_tail_scatter_torch(
    content: torch.Tensor,
    scores: torch.Tensor,
    tail: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Store FP32 projected compressor inputs into the 2-row tail field."""
    if (
        content.ndim != 2
        or content.shape[1] != 512
        or scores.shape != content.shape
        or content.dtype != torch.float32
        or scores.dtype != torch.float32
        or tail.ndim != 4
        or tail.shape[1:] != (2, 2, 512)
        or tail.dtype != torch.float32
    ):
        raise ValueError(
            "require FP32 content/scores [T, 512] and tail [pages, 2, 2, 512]"
        )
    if slots.shape != (content.shape[0],) or slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Compressor tail slots must be integer [T]")
    if any(x.device != content.device for x in (scores, tail, slots)):
        raise ValueError("Compressor tail scatter operands must share one device")
    count = content.shape[0]
    if count == 0:
        return
    capacity = tail.shape[0] * 2
    live = (slots >= 0) & (slots < capacity)
    safe = slots.clamp(0, max(capacity - 1, 0))
    page = torch.div(safe, 2, rounding_mode="floor")
    row = safe.remainder(2)
    # Rank-1 indices stay on device. A 0-dim index copies the coordinate back
    # to the host and aborts graph capture.
    current_content = tail[page, row, 0]
    current_scores = tail[page, row, 1]
    mask = live.unsqueeze(1)
    key = page.to(torch.int64) * 2 + row.to(torch.int64)
    tail[page, row, 0] = _winning_rows(
        key, live, torch.where(mask, content, current_content)
    )
    tail[page, row, 1] = _winning_rows(
        key, live, torch.where(mask, scores, current_scores)
    )
