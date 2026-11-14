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

"""Ascend BF16 and W8A8 INT8 MoE primitives."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch_npu

_MAX_A2_FULLMESH_EXPERTS = 1024


def _format_grouped_weight_nz(weight: torch.Tensor) -> torch.Tensor:
    config = torch.npu.config
    previous = getattr(config, "allow_internal_format", None)
    try:
        config.allow_internal_format = True
        weight = torch_npu.npu_format_cast(weight, 29)
        if torch_npu.get_npu_format(weight) != 29:
            raise RuntimeError("Ascend MoE weight conversion did not produce NZ")
        return weight
    finally:
        config.allow_internal_format = previous


def ascend_softmax_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Lite's selection-only biased softmax TopK."""
    del enable_pdl
    weights, ids, _ = torch_npu.npu_moe_gating_top_k(
        router_logits,
        topk,
        bias=correction_bias,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
    )
    return weights.float(), ids.to(torch.int32)


def ascend_bf16_process_moe_weights(*, plan: dict, w: torch.nn.Module) -> None:
    """Convert canonical TokenSpeed weights to the Ascend GMM layout."""
    if not getattr(w, "_ascend_bf16_moe_weights_processed", False):
        expected = (
            (w.num_local_experts, 2 * w.intermediate_size, w.hidden_size),
            (w.num_local_experts, w.hidden_size, w.intermediate_size),
        )
        if (tuple(w.w13_weight.shape), tuple(w.w2_weight.shape)) != expected:
            raise ValueError("Ascend MoE weights must use canonical W13/W2 layout")
        w.w13_weight.data = w.w13_weight.data.transpose(-1, -2).contiguous()
        w.w2_weight.data = w.w2_weight.data.transpose(-1, -2).contiguous()
        w._ascend_bf16_moe_weights_processed = True

    if plan.get("a2a_backend") != "ascend":
        return
    process_group = plan.get("ep_group") or plan.get("process_group")
    if process_group is None:
        raise ValueError("Ascend distributed MoE requires an EP process group")
    backend = process_group._get_backend(w.w13_weight.device)
    plan["ascend_hccl_group_name"] = backend.get_hccl_comm_name(int(w.ep_rank))

    num_groups = int(plan.get("num_expert_groups", 1))
    experts_per_group = plan.get("num_experts_per_group")
    experts_per_group = (
        int(w.num_experts) if experts_per_group is None else int(experts_per_group)
    )
    if num_groups <= 0 or experts_per_group <= 0:
        raise ValueError("Ascend distributed MoE group dimensions must be positive")
    if num_groups * experts_per_group != int(w.num_experts):
        raise ValueError("Ascend distributed MoE group dimensions mismatch weights")
    if experts_per_group % int(w.ep_size):
        raise ValueError("Ascend distributed MoE groups must divide across EP ranks")
    groups_per_chunk = _MAX_A2_FULLMESH_EXPERTS // experts_per_group
    if groups_per_chunk <= 0:
        raise ValueError(
            "One logical expert group exceeds the Ascend dispatch expert limit"
        )
    local_per_group = experts_per_group // int(w.ep_size)
    chunks = []
    for group_start in range(0, num_groups, groups_per_chunk):
        chunk_groups = min(groups_per_chunk, num_groups - group_start)
        local_start = group_start * local_per_group
        local_count = chunk_groups * local_per_group
        chunks.append(
            SimpleNamespace(
                group_start=group_start,
                num_groups=chunk_groups,
                num_experts=chunk_groups * experts_per_group,
                num_local_experts=local_count,
                hidden_size=int(w.hidden_size),
                intermediate_size=int(w.intermediate_size),
                ep_rank=int(w.ep_rank),
                ep_size=int(w.ep_size),
                w13_weight=w.w13_weight[local_start : local_start + local_count],
                w2_weight=w.w2_weight[local_start : local_start + local_count],
                _ascend_bf16_moe_weights_processed=True,
            )
        )
    plan["ascend_expert_chunks"] = tuple(chunks)


def ascend_int8_process_moe_weights(*, plan: dict, w: torch.nn.Module) -> None:
    """Transpose canonical INT8 expert weights and persist them in NZ."""
    if not getattr(w, "_ascend_int8_moe_weights_processed", False):
        expected = (
            (w.num_local_experts, 2 * w.intermediate_size, w.hidden_size),
            (w.num_local_experts, w.hidden_size, w.intermediate_size),
            (w.num_local_experts, 2 * w.intermediate_size),
            (w.num_local_experts, w.hidden_size),
        )
        actual = (
            tuple(w.w13_weight.shape),
            tuple(w.w2_weight.shape),
            tuple(w.w13_weight_scale.shape),
            tuple(w.w2_weight_scale.shape),
        )
        if actual != expected:
            raise ValueError("Ascend W8A8 MoE weights must use canonical layouts")
        if w.w13_weight.dtype != torch.int8 or w.w2_weight.dtype != torch.int8:
            raise TypeError("Ascend W8A8 MoE weights must be INT8")

        w.w13_weight.data = _format_grouped_weight_nz(
            w.w13_weight.data.transpose(-1, -2).contiguous()
        )
        w.w2_weight.data = _format_grouped_weight_nz(
            w.w2_weight.data.transpose(-1, -2).contiguous()
        )
        w.w13_weight_scale.data = w.w13_weight_scale.data.float().contiguous()
        w.w2_weight_scale.data = w.w2_weight_scale.data.contiguous()
        for name in ("w13_smooth_scale", "w2_smooth_scale"):
            smooth = getattr(w, name, None)
            if smooth is not None:
                data = smooth.data.float()
                if data.ndim == 1:
                    data = data.unsqueeze(0)
                # Official DSQ requires one row per group, even for a shared scale.
                # Materialize once during preparation, never in forward/capture.
                if name == "w2_smooth_scale" and data.shape[0] == 1:
                    data = data.expand(int(w.num_local_experts), -1)
                smooth.data = data.contiguous()
        w._ascend_int8_moe_weights_processed = True

    if plan.get("a2a_backend") != "ascend":
        return
    process_group = plan.get("ep_group") or plan.get("process_group")
    if process_group is None:
        raise ValueError("Ascend distributed MoE requires an EP process group")
    backend = process_group._get_backend(w.w13_weight.device)
    plan["ascend_hccl_group_name"] = backend.get_hccl_comm_name(int(w.ep_rank))

    num_groups = int(plan.get("num_expert_groups", 1))
    experts_per_group = plan.get("num_experts_per_group")
    experts_per_group = (
        int(w.num_experts) if experts_per_group is None else int(experts_per_group)
    )
    if num_groups <= 0 or experts_per_group <= 0:
        raise ValueError("Ascend distributed MoE group dimensions must be positive")
    if num_groups * experts_per_group != int(w.num_experts):
        raise ValueError("Ascend distributed MoE group dimensions mismatch weights")
    if experts_per_group % int(w.ep_size):
        raise ValueError("Ascend distributed MoE groups must divide across EP ranks")
    groups_per_chunk = _MAX_A2_FULLMESH_EXPERTS // experts_per_group
    if groups_per_chunk <= 0:
        raise ValueError(
            "One logical expert group exceeds the Ascend dispatch expert limit"
        )
    local_per_group = experts_per_group // int(w.ep_size)
    chunks = []
    for group_start in range(0, num_groups, groups_per_chunk):
        chunk_groups = min(groups_per_chunk, num_groups - group_start)
        local_start = group_start * local_per_group
        local_count = chunk_groups * local_per_group
        w13_smooth = getattr(w, "w13_smooth_scale", None)
        w2_smooth = getattr(w, "w2_smooth_scale", None)
        chunks.append(
            SimpleNamespace(
                group_start=group_start,
                num_groups=chunk_groups,
                num_experts=chunk_groups * experts_per_group,
                num_local_experts=local_count,
                hidden_size=int(w.hidden_size),
                intermediate_size=int(w.intermediate_size),
                ep_rank=int(w.ep_rank),
                ep_size=int(w.ep_size),
                w13_weight=w.w13_weight[local_start : local_start + local_count],
                w2_weight=w.w2_weight[local_start : local_start + local_count],
                w13_weight_scale=w.w13_weight_scale[
                    local_start : local_start + local_count
                ],
                w2_weight_scale=w.w2_weight_scale[
                    local_start : local_start + local_count
                ],
                w13_smooth_scale=(
                    None
                    if w13_smooth is None
                    else (
                        w13_smooth
                        if w13_smooth.shape[0] == 1
                        else w13_smooth[local_start : local_start + local_count]
                    )
                ),
                w2_smooth_scale=(
                    None
                    if w2_smooth is None
                    else w2_smooth[local_start : local_start + local_count]
                ),
                _ascend_int8_moe_weights_processed=True,
            )
        )
    plan["ascend_expert_chunks"] = tuple(chunks)


def _ascend_bf16_expert_gmm(
    expanded: torch.Tensor,
    expert_counts: torch.Tensor,
    w: torch.nn.Module,
) -> torch.Tensor:
    """Apply the two local expert GMMs to dispatch-ordered real routes."""
    hidden = torch_npu.npu_grouped_matmul(
        [expanded],
        [w.w13_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]
    hidden = torch_npu.mlp_split_swiglu(
        hidden,
        expert_counts.to(torch.int32),
        local_exp_start=0,
        local_exp_end=int(w.num_local_experts),
    )
    return torch_npu.npu_grouped_matmul(
        [hidden],
        [w.w2_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]


def ascend_bf16_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply the graph-stable Ascend local-expert chain."""
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Ascend BF16 MoE does not support deferred finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend BF16 MoE requires precomputed top-k tensors")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend BF16 MoE requires BF16 NPU hidden states")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and hidden states must share a device")
    if plan.get("activation") not in {"silu", "swiglu"}:
        raise ValueError("Ascend BF16 MoE supports standard SwiGLU only")
    if not getattr(w, "_ascend_bf16_moe_weights_processed", False):
        raise RuntimeError("Ascend BF16 MoE weights were not processed after loading")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    local_experts = int(w.num_local_experts)
    hidden_size = int(w.hidden_size)
    intermediate_size = int(w.intermediate_size)
    if tuple(w.w13_weight.shape) != (
        local_experts,
        hidden_size,
        2 * intermediate_size,
    ) or tuple(w.w2_weight.shape) != (
        local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("Ascend MoE weights have an incompatible GMM layout")
    if w.w13_weight.dtype != x.dtype or w.w2_weight.dtype != x.dtype:
        raise TypeError("hidden states and Ascend MoE weights must share a dtype")

    num_experts = int(w.num_experts)
    expert_lo = int(w.ep_rank) * local_experts
    expert_hi = expert_lo + local_experts
    ep_size = int(w.ep_size)
    if expert_hi > num_experts or ep_size <= 0:
        raise ValueError("Ascend local-expert routing requires contiguous EP ownership")
    ids = topk_ids.to(torch.int32)
    routing_num_experts = num_experts
    if ep_size == 1:
        routing_num_experts += int(plan.get("num_zero_experts", 0))
    expanded, row_indices, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids,
        scale=None,
        active_num=ids.numel(),
        expert_num=routing_num_experts,
        quant_mode=-1,
        active_expert_range=[expert_lo, expert_hi],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    hidden = _ascend_bf16_expert_gmm(expanded, expert_counts, w)
    return torch_npu.npu_moe_finalize_routing(
        hidden.unsqueeze(1),
        None,
        None,
        None,
        topk_weights,
        row_indices,
        ids,
        3,
    )


def _swiglu_clamp_limit(plan: dict, w: torch.nn.Module) -> float | None:
    """DeepSeek V4 finite SwiGLU clamp, or None when the fused op is enough."""
    limit = plan.get("swiglu_limit")
    if limit is None:
        arg = getattr(w, "swiglu_arg", None)
        limit = None if arg is None else arg.limit
    if limit is None:
        return None
    value = float(limit)
    return value if value > 0.0 else None


def expert_row_scales(
    weight_scale: torch.Tensor, expert_counts: torch.Tensor, rows: int
) -> torch.Tensor:
    """Repeat per-expert channel scales across ``rows`` without a host read.

    Rows past ``sum(expert_counts)`` are zero. ``tolist`` / ``sum().item()``
    would drain the NPU stream once per MoE layer to learn that sum.
    """
    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    if expert_counts.ndim != 1:
        raise ValueError(
            f"expert_counts must be rank 1, got {tuple(expert_counts.shape)}"
        )
    if weight_scale.shape[0] != expert_counts.shape[0]:
        raise ValueError(
            "weight_scale rows must match expert_counts, "
            f"got {weight_scale.shape[0]} and {expert_counts.shape[0]}"
        )
    channels = weight_scale.shape[-1]
    if rows == 0 or expert_counts.numel() == 0:
        return weight_scale.new_empty((rows, channels))
    ends = expert_counts.to(dtype=torch.int64).cumsum(0)
    starts = torch.cat((ends.new_zeros(1), ends[:-1]), dim=0)
    row = torch.arange(rows, device=ends.device).unsqueeze(1)
    inside = (row >= starts) & (row < ends)
    expert = inside.to(dtype=torch.int64).argmax(dim=1)
    valid = inside.any(dim=1)
    scales = weight_scale.index_select(0, expert)
    return scales * valid.to(dtype=scales.dtype).unsqueeze(1)


def _expand_expert_scales(
    weight_scale: torch.Tensor, expert_counts: torch.Tensor
) -> torch.Tensor:
    """Repeat each expert's channel scales for that expert's expanded rows."""
    counts = expert_counts.tolist()
    if not any(counts):
        return weight_scale.new_empty((0, weight_scale.shape[-1]))
    parts = [
        weight_scale[expert].expand(count, -1)
        for expert, count in enumerate(counts)
        if count
    ]
    return torch.cat(parts, dim=0)


def _ascend_int8_dequant_swiglu_quant(
    hidden_i32: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    *,
    quant_scale: torch.Tensor | None,
    limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant INT32 gate/up, SwiGLU, and per-token INT8 requant.

    When ``limit`` is set, apply DeepSeek V4 clamp ordering explicitly:
    ``gate.clamp_max(limit)``, ``up.clamp(-limit, limit)``, then
    ``silu(gate) * up``. Fused ``npu_dequant_swiglu_quant`` clamp modes do not
    match that contract (see vLLM-Ascend DeepSeek V4 SwiGLU issues).

    Ascend MC2 dispatch may size the token axis to a capacity that is larger
    than ``sum(expert_counts)``. The fused op only touches the live packed
    prefix and leaves the suffix undefined; the clamp path must do the same
    instead of broadcasting expert scales across the whole capacity buffer.
    """
    if limit is None:
        return torch_npu.npu_dequant_swiglu_quant(
            hidden_i32,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=None,
            quant_scale=quant_scale,
            quant_offset=None,
            group_index=expert_counts,
            activate_left=True,
            quant_mode=1,
        )
    if quant_scale is not None:
        raise ValueError(
            "Ascend W8A8 MoE SwiGLU clamp does not combine with w2 smooth scales"
        )
    tokens = hidden_i32.shape[0]
    width = hidden_i32.shape[-1] // 2
    # Host counts can still take the packed prefix. Device counts must not
    # be read back: sum().item() drains the stream between the two grouped
    # matmuls on every MoE layer.
    if expert_counts.device.type == "cpu":
        total = int(expert_counts.to(torch.int64).sum())
        if total > tokens:
            raise ValueError(
                "Ascend W8A8 MoE SwiGLU clamp expert counts exceed the token buffer"
            )
        if total == 0:
            return (
                hidden_i32.new_zeros((tokens, width), dtype=torch.int8),
                hidden_i32.new_ones((tokens,), dtype=torch.float32),
            )
        act = activation_scale.float().reshape(-1)
        if act.numel() < total:
            raise ValueError(
                "Ascend W8A8 MoE SwiGLU clamp activation scales are shorter than "
                "the live expert rows"
            )
        row_scale = _expand_expert_scales(weight_scale.float(), expert_counts)
        hidden_live = hidden_i32[:total]
        act_live = act[:total]
    else:
        if tokens == 0:
            return (
                hidden_i32.new_zeros((0, width), dtype=torch.int8),
                hidden_i32.new_ones((0,), dtype=torch.float32),
            )
        act = activation_scale.float().reshape(-1)
        if act.numel() < tokens:
            act = torch.nn.functional.pad(act, (0, tokens - act.numel()))
        else:
            act = act[:tokens]
        row_scale = expert_row_scales(weight_scale.float(), expert_counts, tokens)
        hidden_live = hidden_i32
        act_live = act
    gate_up = (hidden_live.float() * act_live.unsqueeze(-1) * row_scale).to(
        dtype=torch.bfloat16
    )
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp_max(limit)
    up = up.clamp(-limit, limit)
    mid = torch.nn.functional.silu(gate) * up
    mid_q, mid_s = torch_npu.npu_dynamic_quant(mid)
    if mid_q.shape[0] == tokens:
        return mid_q, mid_s
    # CPU packed prefix: restore the capacity-sized outputs the grouped
    # down-projection indexes with expert_counts.
    total = mid_q.shape[0]
    out_q = mid_q.new_zeros((tokens, mid_q.shape[-1]))
    out_s = mid_s.new_ones((tokens,))
    out_q[:total] = mid_q
    out_s[:total] = mid_s
    return out_q, out_s


def ascend_int8_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply dynamic-token/per-channel W8A8 experts with NZ weights."""
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Ascend W8A8 MoE does not support deferred finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend W8A8 MoE requires precomputed top-k tensors")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend W8A8 MoE requires BF16 NPU hidden states")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and hidden states must share a device")
    if plan.get("activation") not in {"silu", "swiglu"}:
        raise ValueError("Ascend W8A8 MoE supports standard SwiGLU only")
    if not getattr(w, "_ascend_int8_moe_weights_processed", False):
        raise RuntimeError("Ascend W8A8 MoE weights were not processed after loading")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    local_experts = int(w.num_local_experts)
    hidden_size = int(w.hidden_size)
    intermediate_size = int(w.intermediate_size)
    if tuple(w.w13_weight.shape) != (
        local_experts,
        hidden_size,
        2 * intermediate_size,
    ) or tuple(w.w2_weight.shape) != (
        local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("Ascend W8A8 MoE weights have an incompatible GMM layout")
    if w.w13_weight.dtype != torch.int8 or w.w2_weight.dtype != torch.int8:
        raise TypeError("Ascend W8A8 MoE GMM weights must be INT8")
    if (
        torch_npu.get_npu_format(w.w13_weight) != 29
        or torch_npu.get_npu_format(w.w2_weight) != 29
    ):
        raise RuntimeError("Ascend W8A8 MoE GMM weights must use NZ format")

    num_experts = int(w.num_experts)
    expert_lo = int(w.ep_rank) * local_experts
    expert_hi = expert_lo + local_experts
    if expert_hi > num_experts or int(w.ep_size) <= 0:
        raise ValueError("Ascend local-expert routing requires contiguous EP ownership")
    ids = topk_ids.to(torch.int32)
    routing_num_experts = num_experts
    if int(w.ep_size) == 1:
        routing_num_experts += int(plan.get("num_zero_experts", 0))
    smooth13 = getattr(w, "w13_smooth_scale", None)
    expanded, row_indices, expert_counts, activation_scale = (
        torch_npu.npu_moe_init_routing_v2(
            x,
            ids,
            scale=smooth13,
            offset=None,
            active_num=ids.numel(),
            expert_num=routing_num_experts,
            quant_mode=1,
            active_expert_range=[expert_lo, expert_hi],
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            drop_pad_mode=0,
        )
    )
    hidden = torch_npu.npu_grouped_matmul(
        [expanded],
        [w.w13_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.int32,
        group_type=0,
        group_list_type=1,
    )[0]
    hidden, hidden_scale = _ascend_int8_dequant_swiglu_quant(
        hidden,
        w.w13_weight_scale,
        activation_scale,
        expert_counts,
        quant_scale=getattr(w, "w2_smooth_scale", None),
        limit=_swiglu_clamp_limit(plan, w),
    )
    return _ascend_int8_gmm2_finalize(
        hidden, hidden_scale, expert_counts, row_indices, ids, topk_weights, w
    )


def _ascend_int8_gmm2_finalize(hidden, hidden_scale, counts, rows, ids, weights, w):
    """Unchanged shared tail for composed and input-fused W8A8 paths."""
    hidden = torch_npu.npu_grouped_matmul(
        [hidden],
        [w.w2_weight],
        scale=[w.w2_weight_scale],
        per_token_scale=[hidden_scale.float()],
        bias=None,
        group_list=counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]
    return torch_npu.npu_moe_finalize_routing(
        hidden.unsqueeze(1), None, None, None, weights.to(hidden.dtype), rows, ids, 3
    )


def _ascend_mc2_global_bs(
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    num_groups: int,
    chunk_groups: int,
    ep_size: int,
) -> int:
    """Shared ``global_bs`` for one dispatch and its matching combine.

    vLLM-Ascend's uneven path passes ``max_bs_per_rank * ep_world_size`` and
    does not pass ``x_active_mask``. SGLang's tiling accepts that value, or 0
    when every rank has the same local batch. ``comm_alg="fullmesh"`` allows
    those local batches to differ, including 0. ``fullmesh_v2`` rejects a
    differing batch, so the call stays on fullmesh.
    """
    chunk_global_tokens = 0
    if num_tokens_global:
        chunk_global_tokens = int(num_tokens_global) // num_groups * int(chunk_groups)
    chunk_max_tokens = 0
    if max_num_tokens_per_gpu:
        chunk_max_tokens = int(max_num_tokens_per_gpu) // num_groups * int(chunk_groups)
    if chunk_max_tokens:
        return chunk_max_tokens * int(ep_size)
    return chunk_global_tokens


def ascend_bf16_distributed_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
    low_latency: bool | None = None,
    overlap_fn=None,
) -> torch.Tensor:
    """Run global-ID Ascend MC2 dispatch, local expert GMMs, and combine.

    Real experts use contiguous global IDs in ``[0, w.num_experts)``. Identity
    experts immediately follow that range and are implemented by MC2 copy
    experts, so Combine applies their route weights directly to ``ori_x``.
    """
    del router_logits, enable_pdl, low_latency
    if overlap_fn is not None:
        raise ValueError("Ascend distributed MoE does not support overlap_fn")
    if not do_finalize:
        raise ValueError("Ascend distributed MoE requires finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend distributed MoE requires precomputed top-k")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend distributed MoE requires BF16 NPU hidden states")
    if not getattr(w, "_ascend_bf16_moe_weights_processed", False):
        raise RuntimeError("Ascend MoE weights were not processed after loading")
    group_name = plan.get("ascend_hccl_group_name")
    if not group_name:
        raise RuntimeError("Ascend distributed MoE has no HCCL communicator")

    num_experts = int(w.num_experts)
    num_zero_experts = int(plan.get("num_zero_experts", 0))
    num_copy_experts = int(plan.get("num_copy_experts", 0))
    num_groups = int(plan.get("num_expert_groups", 1))
    experts_per_group = plan.get("num_experts_per_group")
    experts_per_group = (
        num_experts if experts_per_group is None else int(experts_per_group)
    )
    chunks = plan.get("ascend_expert_chunks")
    if not chunks:
        raise RuntimeError("Ascend distributed MoE expert chunks were not prepared")
    if x.shape[0] % num_groups:
        raise ValueError("Flattened grouped MoE tokens must divide by group count")
    tokens_per_group = x.shape[0] // num_groups
    local_per_group = experts_per_group // int(w.ep_size)
    outputs = []
    for chunk in chunks:
        row_start = int(chunk.group_start) * tokens_per_group
        row_count = int(chunk.num_groups) * tokens_per_group
        chunk_x = x.narrow(0, row_start, row_count)
        chunk_weights = topk_weights.narrow(0, row_start, row_count).float()
        logical_ids = topk_ids.narrow(0, row_start, row_count).to(torch.int32)

        is_real = (logical_ids >= 0) & (logical_ids < num_experts)
        group_ids = torch.div(
            logical_ids.clamp(0, num_experts - 1),
            experts_per_group,
            rounding_mode="floor",
        )
        expert_ids = logical_ids.remainder(experts_per_group)
        owner = torch.div(expert_ids, local_per_group, rounding_mode="floor")
        local_id = expert_ids.remainder(local_per_group)
        physical_ids = (
            owner * int(chunk.num_local_experts)
            + (group_ids - int(chunk.group_start)) * local_per_group
            + local_id
        )
        special_offset = logical_ids - num_experts
        if num_zero_experts:
            special_ids = int(chunk.num_experts) + special_offset.clamp(
                0, num_zero_experts - 1
            )
        elif num_copy_experts:
            special_ids = int(chunk.num_experts) + special_offset.clamp(
                0, num_copy_experts - 1
            )
        else:
            special_ids = logical_ids.remainder(int(chunk.num_experts))
            chunk_weights = torch.where(
                is_real, chunk_weights, torch.zeros_like(chunk_weights)
            )
        physical_ids = torch.where(is_real, physical_ids, special_ids).to(torch.int32)

        global_bs = _ascend_mc2_global_bs(
            num_tokens_global,
            max_num_tokens_per_gpu,
            num_groups,
            int(chunk.num_groups),
            int(w.ep_size),
        )

        dispatch = torch_npu.npu_moe_distribute_dispatch_v2(
            x=chunk_x,
            expert_ids=physical_ids,
            expert_scales=chunk_weights,
            group_ep=group_name,
            group_tp="",
            ep_world_size=int(w.ep_size),
            tp_world_size=1,
            ep_rank_id=int(w.ep_rank),
            tp_rank_id=0,
            moe_expert_num=int(chunk.num_experts),
            shared_expert_num=0,
            shared_expert_rank_num=0,
            zero_expert_num=num_zero_experts,
            copy_expert_num=num_copy_experts,
            quant_mode=0,
            global_bs=global_bs,
            expert_token_nums_type=1,
            comm_alg="fullmesh",
        )
        hidden = _ascend_bf16_expert_gmm(dispatch[0], dispatch[3], chunk)
        outputs.append(
            torch_npu.npu_moe_distribute_combine_v2(
                expand_x=hidden,
                expert_ids=physical_ids,
                assist_info_for_combine=dispatch[2],
                ep_send_counts=dispatch[4],
                tp_send_counts=dispatch[5],
                expert_scales=chunk_weights,
                expand_scales=dispatch[6],
                ori_x=chunk_x,
                group_ep=group_name,
                group_tp="",
                ep_world_size=int(w.ep_size),
                tp_world_size=1,
                ep_rank_id=int(w.ep_rank),
                tp_rank_id=0,
                moe_expert_num=int(chunk.num_experts),
                shared_expert_num=0,
                shared_expert_rank_num=0,
                zero_expert_num=num_zero_experts,
                copy_expert_num=num_copy_experts,
                global_bs=global_bs,
                comm_quant_mode=0,
                comm_alg="fullmesh",
            )
        )
    return torch.cat(outputs, dim=0)


def _ascend_int8_expert_gmm_from_bf16(
    expanded: torch.Tensor,
    expert_counts: torch.Tensor,
    w: torch.nn.Module,
    plan: dict,
) -> torch.Tensor:
    """Dynamic-quant BF16 tokens through local INT8 NZ expert GMMs."""
    if expanded.numel() == 0:
        return expanded
    values, activation_scale = torch_npu.npu_dynamic_quant(expanded)
    smooth13 = getattr(w, "w13_smooth_scale", None)
    if smooth13 is not None:
        # Per-token scales already absorbed activation; smooth is folded in
        # the weight path for local routing. Distributed BF16 dispatch has no
        # smooth quant; keep dynamic token scales only.
        del smooth13
    hidden = torch_npu.npu_grouped_matmul(
        [values],
        [w.w13_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.int32,
        group_type=0,
        group_list_type=1,
    )[0]
    hidden, hidden_scale = _ascend_int8_dequant_swiglu_quant(
        hidden,
        w.w13_weight_scale,
        activation_scale,
        expert_counts,
        quant_scale=getattr(w, "w2_smooth_scale", None),
        limit=_swiglu_clamp_limit(plan, w),
    )
    return torch_npu.npu_grouped_matmul(
        [hidden],
        [w.w2_weight],
        scale=[w.w2_weight_scale],
        per_token_scale=[hidden_scale.float()],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]


def ascend_int8_distributed_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
    low_latency: bool | None = None,
    overlap_fn=None,
) -> torch.Tensor:
    """Ascend MC2 dispatch of BF16 tokens, then local INT8 NZ expert GMMs."""
    del router_logits, enable_pdl, low_latency
    if overlap_fn is not None:
        raise ValueError("Ascend distributed MoE does not support overlap_fn")
    if not do_finalize:
        raise ValueError("Ascend distributed MoE requires finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend distributed MoE requires precomputed top-k")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend distributed MoE requires BF16 NPU hidden states")
    if not getattr(w, "_ascend_int8_moe_weights_processed", False):
        raise RuntimeError("Ascend W8A8 MoE weights were not processed after loading")
    group_name = plan.get("ascend_hccl_group_name")
    if not group_name:
        raise RuntimeError("Ascend distributed MoE has no HCCL communicator")

    num_experts = int(w.num_experts)
    num_zero_experts = int(plan.get("num_zero_experts", 0))
    num_copy_experts = int(plan.get("num_copy_experts", 0))
    num_groups = int(plan.get("num_expert_groups", 1))
    experts_per_group = plan.get("num_experts_per_group")
    experts_per_group = (
        num_experts if experts_per_group is None else int(experts_per_group)
    )
    chunks = plan.get("ascend_expert_chunks")
    if not chunks:
        raise RuntimeError("Ascend distributed MoE expert chunks were not prepared")
    if x.shape[0] % num_groups:
        raise ValueError("Flattened grouped MoE tokens must divide by group count")
    tokens_per_group = x.shape[0] // num_groups
    local_per_group = experts_per_group // int(w.ep_size)
    outputs = []
    for chunk in chunks:
        row_start = int(chunk.group_start) * tokens_per_group
        row_count = int(chunk.num_groups) * tokens_per_group
        chunk_x = x.narrow(0, row_start, row_count)
        chunk_weights = topk_weights.narrow(0, row_start, row_count).float()
        logical_ids = topk_ids.narrow(0, row_start, row_count).to(torch.int32)

        is_real = (logical_ids >= 0) & (logical_ids < num_experts)
        group_ids = torch.div(
            logical_ids.clamp(0, num_experts - 1),
            experts_per_group,
            rounding_mode="floor",
        )
        expert_ids = logical_ids.remainder(experts_per_group)
        owner = torch.div(expert_ids, local_per_group, rounding_mode="floor")
        local_id = expert_ids.remainder(local_per_group)
        physical_ids = (
            owner * int(chunk.num_local_experts)
            + (group_ids - int(chunk.group_start)) * local_per_group
            + local_id
        )
        special_offset = logical_ids - num_experts
        if num_zero_experts:
            special_ids = int(chunk.num_experts) + special_offset.clamp(
                0, num_zero_experts - 1
            )
        elif num_copy_experts:
            special_ids = int(chunk.num_experts) + special_offset.clamp(
                0, num_copy_experts - 1
            )
        else:
            special_ids = logical_ids.remainder(int(chunk.num_experts))
            chunk_weights = torch.where(
                is_real, chunk_weights, torch.zeros_like(chunk_weights)
            )
        physical_ids = torch.where(is_real, physical_ids, special_ids).to(torch.int32)

        global_bs = _ascend_mc2_global_bs(
            num_tokens_global,
            max_num_tokens_per_gpu,
            num_groups,
            int(chunk.num_groups),
            int(w.ep_size),
        )

        dispatch = torch_npu.npu_moe_distribute_dispatch_v2(
            x=chunk_x,
            expert_ids=physical_ids,
            expert_scales=chunk_weights,
            group_ep=group_name,
            group_tp="",
            ep_world_size=int(w.ep_size),
            tp_world_size=1,
            ep_rank_id=int(w.ep_rank),
            tp_rank_id=0,
            moe_expert_num=int(chunk.num_experts),
            shared_expert_num=0,
            shared_expert_rank_num=0,
            zero_expert_num=num_zero_experts,
            copy_expert_num=num_copy_experts,
            quant_mode=0,
            global_bs=global_bs,
            expert_token_nums_type=1,
            comm_alg="fullmesh",
        )
        hidden = _ascend_int8_expert_gmm_from_bf16(
            dispatch[0], dispatch[3], chunk, plan
        )
        outputs.append(
            torch_npu.npu_moe_distribute_combine_v2(
                expand_x=hidden,
                expert_ids=physical_ids,
                assist_info_for_combine=dispatch[2],
                ep_send_counts=dispatch[4],
                tp_send_counts=dispatch[5],
                expert_scales=chunk_weights,
                expand_scales=dispatch[6],
                ori_x=chunk_x,
                group_ep=group_name,
                group_tp="",
                ep_world_size=int(w.ep_size),
                tp_world_size=1,
                ep_rank_id=int(w.ep_rank),
                tp_rank_id=0,
                moe_expert_num=int(chunk.num_experts),
                shared_expert_num=0,
                shared_expert_rank_num=0,
                zero_expert_num=num_zero_experts,
                copy_expert_num=num_copy_experts,
                global_bs=global_bs,
                comm_quant_mode=0,
                comm_alg="fullmesh",
            )
        )
    return torch.cat(outputs, dim=0)


__all__ = [
    "ascend_bf16_distributed_precomputed_moe_apply",
    "ascend_bf16_precomputed_moe_apply",
    "ascend_bf16_process_moe_weights",
    "ascend_int8_distributed_precomputed_moe_apply",
    "ascend_int8_precomputed_moe_apply",
    "ascend_int8_process_moe_weights",
    "ascend_softmax_bias_topk",
]
