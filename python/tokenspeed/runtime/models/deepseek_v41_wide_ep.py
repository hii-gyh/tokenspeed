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

"""Expert ownership and token slicing when V4.1 expert parallel is wider than attention TP.

Attention TP all-reduces a replica, so every rank in that replica holds the same
hidden rows. An expert group that spans several replicas must send each row
once. vLLM's data-parallel plus tensor-parallel expert dispatch has the same
volume: ``num_tokens * hidden * dtype_bytes`` each way, per layer.
"""

from __future__ import annotations

import torch


def local_expert_range(ep_rank: int, ep_size: int, num_experts: int) -> tuple[int, int]:
    """Return the half-open expert ids owned by one expert-parallel rank.

    Ownership is contiguous. Every expert is owned once across ``ep_size``.
    """
    if ep_size < 1 or ep_rank < 0 or ep_rank >= ep_size:
        raise ValueError(
            f"ep_rank must be in [0, {ep_size}), got ep_rank={ep_rank} ep_size={ep_size}"
        )
    if num_experts < 1 or num_experts % ep_size:
        raise ValueError(
            f"num_experts ({num_experts}) must be a positive multiple of ep_size ({ep_size})"
        )
    local = num_experts // ep_size
    start = ep_rank * local
    return start, start + local


def wide_ep_separates_attention(
    *,
    world_size: int,
    attn_tp_size: int,
    moe_tp_size: int,
    moe_ep_size: int,
) -> bool:
    """True when experts span the whole world once and attention TP is narrower.

    ``moe_tp_size`` stays 1: an expert is not tensor-sharded and expert-parallel
    at the same time. The matched layout ``attn_tp == moe_tp * moe_ep`` is not
    this case.
    """
    if world_size < 1 or attn_tp_size < 1 or moe_tp_size < 1 or moe_ep_size < 1:
        raise ValueError("parallel sizes must be positive")
    return (
        moe_tp_size == 1
        and moe_ep_size == world_size
        and attn_tp_size > 1
        and world_size % attn_tp_size == 0
        and attn_tp_size != moe_tp_size * moe_ep_size
    )


def ced_narrowing_needs_row_exchange(
    *,
    keep_rows_present: bool,
    has_global_token_counts: bool,
    owns_ep_communication: bool,
    wide_ep: bool,
) -> bool:
    """Whether CED narrowing must publish per-rank row counts before MoE.

    A non-fused expert group sizes its collectives from the attention-DP token
    table, which still describes the rows before narrowing. Fused EP learns the
    real counts inside its dispatch. Wide EP splits the replicated local rows
    and all-gathers that slice inside the attention group, so it does not read
    the table either.
    """
    return (
        keep_rows_present
        and has_global_token_counts
        and not owns_ep_communication
        and not wide_ep
    )


def mapping_uses_wide_ep(mapping) -> bool:
    """True for attention-TP narrower than a world-sized expert group.

    Dense TP stays equal to attention TP, so shared experts are not sliced
    across the expert group.
    """
    return (
        wide_ep_separates_attention(
            world_size=mapping.world_size,
            attn_tp_size=mapping.attn.tp_size,
            moe_tp_size=mapping.moe.tp_size,
            moe_ep_size=mapping.moe.ep_size,
        )
        and mapping.dense.tp_size == mapping.attn.tp_size
    )


def split_replicated_tokens(
    hidden: torch.Tensor, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Keep the rows this attention-TP rank sends into a wider expert group.

    ``hidden`` is the replicated batch, shape ``[tokens, ...]``. Row ``i`` is
    sent by rank ``i % tp_size``. ``tp_size == 1`` returns ``hidden``.
    """
    if tp_size < 1 or tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(
            f"tp_rank must be in [0, {tp_size}), got tp_rank={tp_rank} tp_size={tp_size}"
        )
    if hidden.ndim < 1:
        raise ValueError("hidden must have a token dimension")
    if tp_size == 1:
        return hidden
    if hidden.shape[0] % tp_size:
        raise ValueError(
            f"token count {hidden.shape[0]} must divide across attention TP {tp_size} "
            "before expert dispatch"
        )
    return hidden[tp_rank::tp_size].contiguous()


def pad_token_dim(hidden: torch.Tensor, tp_size: int) -> tuple[torch.Tensor, int]:
    """Pad the token axis so it divides ``tp_size``. Returns the pad count."""
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if hidden.ndim < 1:
        raise ValueError("hidden must have a token dimension")
    remainder = hidden.shape[0] % tp_size
    if remainder == 0:
        return hidden, 0
    pad = tp_size - remainder
    filler = hidden.new_zeros((pad, *hidden.shape[1:]))
    return torch.cat((hidden, filler), dim=0), pad


def unpad_token_dim(hidden: torch.Tensor, pad: int) -> torch.Tensor:
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")
    if pad == 0:
        return hidden
    if pad > hidden.shape[0]:
        raise ValueError(f"pad {pad} exceeds token count {hidden.shape[0]}")
    return hidden[:-pad]


def restore_gathered_tokens(gathered: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Invert a rank-order gather of :func:`split_replicated_tokens` slices.

    ``gathered`` is the concatenation of rank 0's slice, then rank 1's, and so
    on. That is what a token all-gather returns. The result is the original
    token order.
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if gathered.ndim < 1 or gathered.shape[0] % tp_size:
        raise ValueError(
            f"gathered token count {gathered.shape[0]} must divide across TP {tp_size}"
        )
    if tp_size == 1:
        return gathered
    rows = gathered.shape[0] // tp_size
    parts = [gathered[rank * rows : (rank + 1) * rows] for rank in range(tp_size)]
    return restore_replicated_tokens(parts)


def restore_replicated_tokens(parts: list[torch.Tensor]) -> torch.Tensor:
    """Invert :func:`split_replicated_tokens` from one slice per attention-TP rank.

    ``parts[rank]`` is that rank's slice. Every slice has the same shape.
    """
    if not parts:
        raise ValueError("restore requires one slice per attention-TP rank")
    tp_size = len(parts)
    rows = parts[0].shape[0]
    rest = parts[0].shape[1:]
    for rank, part in enumerate(parts):
        if part.shape != (rows, *rest):
            raise ValueError(
                f"attention-TP rank {rank} slice has shape {tuple(part.shape)}, "
                f"expected {(rows, *rest)}"
            )
    if tp_size == 1:
        return parts[0]
    out = torch.empty(
        (rows * tp_size, *rest), dtype=parts[0].dtype, device=parts[0].device
    )
    for rank, part in enumerate(parts):
        out[rank::tp_size] = part
    return out


def expert_dispatch_bytes(num_tokens: int, hidden: int, dtype_bytes: int) -> int:
    """One-way bytes to send each token once.

    This is the same volume vLLM moves for one expert-dispatch or combine
    across a DP+TP expert group: the attention TP width does not multiply it.
    """
    if num_tokens < 0 or hidden < 1 or dtype_bytes < 1:
        raise ValueError("token, hidden, and dtype sizes must be non-negative")
    return num_tokens * hidden * dtype_bytes


def apply_wide_ep_rowwise(
    hidden: torch.Tensor,
    tp_size: int,
    forward_rows,
) -> tuple[torch.Tensor, int]:
    """Run a row-wise MoE on one slice per attention-TP rank and restore order.

    ``forward_rows`` maps ``[rows, hidden]`` to ``[rows, hidden]`` and must not
    depend on rows it is not given. The returned byte count is one dispatch of
    the original batch, not one dispatch per attention-TP rank.
    """
    if hidden.ndim != 2:
        raise ValueError(f"MoE hidden must be [tokens, hidden], got {tuple(hidden.shape)}")
    if tp_size == 1:
        output = forward_rows(hidden)
    else:
        parts = []
        for rank in range(tp_size):
            shard = split_replicated_tokens(hidden, rank, tp_size)
            part = forward_rows(shard)
            if part.shape != shard.shape:
                raise ValueError(
                    f"row-wise MoE changed shape from {tuple(shard.shape)} to {tuple(part.shape)}"
                )
            parts.append(part)
        output = restore_replicated_tokens(parts)
    one_way = expert_dispatch_bytes(
        hidden.shape[0], hidden.shape[1], hidden.element_size()
    )
    return output, one_way


def join_residual_on_attention_rank(
    residual: torch.Tensor, moe_output: torch.Tensor
) -> torch.Tensor:
    """Add a restored MoE output to the residual held by an attention rank.

    Both tensors are the full replicated batch of one attention replica,
    ``[tokens, hidden]``. The sum is local. It is not an all-reduce over the
    expert-parallel group: that group also holds other replicas' tokens.
    """
    if residual.shape != moe_output.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match MoE output "
            f"{tuple(moe_output.shape)}"
        )
    if residual.ndim != 2:
        raise ValueError(
            f"attention residual must be [tokens, hidden], got {tuple(residual.shape)}"
        )
    return residual + moe_output
