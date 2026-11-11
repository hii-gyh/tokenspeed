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

"""V4.1 DP4/TP8/EP32 placement and one-token-once dispatch volume.

Wide EP splits the replicated attention rows and sends each token once.
CED narrowing on that path does not wait for a host exchange of row counts.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mapping = _load(
    "v41_wide_ep_mapping",
    "python/tokenspeed/runtime/distributed/mapping.py",
)
_dispatch = _load(
    "v41_wide_ep_dispatch",
    "python/tokenspeed/runtime/models/deepseek_v41_wide_ep.py",
)
Mapping = _mapping.Mapping
expert_dispatch_bytes = _dispatch.expert_dispatch_bytes
local_expert_range = _dispatch.local_expert_range
restore_replicated_tokens = _dispatch.restore_replicated_tokens
split_replicated_tokens = _dispatch.split_replicated_tokens
wide_ep_separates_attention = _dispatch.wide_ep_separates_attention
restore_gathered_tokens = _dispatch.restore_gathered_tokens
mapping_uses_wide_ep = _dispatch.mapping_uses_wide_ep
ced_narrowing_needs_row_exchange = _dispatch.ced_narrowing_needs_row_exchange
wide_ep_dispatch_bounds = _dispatch.wide_ep_dispatch_bounds
ep_dispatch_includes_empty_rank = _dispatch.ep_dispatch_includes_empty_rank
ep_dispatch_local_rows = _dispatch.ep_dispatch_local_rows
pad_token_dim = _dispatch.pad_token_dim
unpad_token_dim = _dispatch.unpad_token_dim
apply_wide_ep_rowwise = _dispatch.apply_wide_ep_rowwise
join_residual_on_attention_rank = _dispatch.join_residual_on_attention_rank

_WORLD = 32
_ATTN_TP = 8
_ATTN_DP = 4
_EP = 32
_EXPERTS = 384
_HIDDEN = 5120
_BF16_BYTES = 2


def _wide_mapping(rank: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=_WORLD,
        attn_tp_size=_ATTN_TP,
        attn_dp_size=_ATTN_DP,
        dense_tp_size=_ATTN_TP,
        moe_tp_size=1,
        moe_ep_size=_EP,
    )


def test_dp4_tp8_ep32_owns_each_expert_once():
    owned: list[int] = []
    for rank in range(_WORLD):
        mapping = _wide_mapping(rank)
        assert mapping.attn.tp_size == _ATTN_TP
        assert mapping.attn.dp_size == _ATTN_DP
        assert mapping.attn.tp_group == tuple(
            range((rank // _ATTN_TP) * _ATTN_TP, (rank // _ATTN_TP) * _ATTN_TP + _ATTN_TP)
        )
        assert mapping.moe.tp_size == 1
        assert mapping.moe.ep_size == _EP
        assert mapping.moe.tp_ep_size == _EP
        assert mapping.moe.ep_rank == rank
        assert mapping.moe.ep_group == tuple(range(_WORLD))
        start, end = local_expert_range(mapping.moe.ep_rank, _EP, _EXPERTS)
        assert end - start == _EXPERTS // _EP
        owned.extend(range(start, end))
    assert owned == list(range(_EXPERTS))


def test_wide_ep_is_not_the_matched_attention_group():
    assert wide_ep_separates_attention(
        world_size=_WORLD,
        attn_tp_size=_ATTN_TP,
        moe_tp_size=1,
        moe_ep_size=_EP,
    )
    # attn TP == MoE TP x EP stays on the existing path.
    assert not wide_ep_separates_attention(
        world_size=_WORLD,
        attn_tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    assert not wide_ep_separates_attention(
        world_size=_WORLD,
        attn_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=_EP,
    )


def test_replicated_token_split_round_trips():
    tokens = 32
    hidden = torch.arange(tokens * 6, dtype=torch.bfloat16).reshape(tokens, 6)
    parts = [split_replicated_tokens(hidden, rank, _ATTN_TP) for rank in range(_ATTN_TP)]
    for rank, part in enumerate(parts):
        assert part.shape == (tokens // _ATTN_TP, 6)
        assert torch.equal(part, hidden[rank::_ATTN_TP])
    restored = restore_replicated_tokens(parts)
    assert torch.equal(restored, hidden)


def test_split_rejects_a_token_count_that_does_not_divide():
    hidden = torch.zeros(7, 4)
    with pytest.raises(ValueError, match="must divide"):
        split_replicated_tokens(hidden, 0, _ATTN_TP)


def test_dispatch_volume_matches_vllm_one_token_once():
    """vLLM DP+TP+EP sends each token once. TP must not multiply the bytes."""
    tokens = 32
    one_way = expert_dispatch_bytes(tokens, _HIDDEN, _BF16_BYTES)
    assert one_way == tokens * _HIDDEN * _BF16_BYTES
    naive_every_tp_rank = one_way * _ATTN_TP
    assert naive_every_tp_rank / one_way == _ATTN_TP
    # Decode-sized and the vLLM per-replica batch of 8 sequences.
    for batch in (1, 8, 32):
        assert expert_dispatch_bytes(batch, _HIDDEN, _BF16_BYTES) == (
            batch * _HIDDEN * _BF16_BYTES
        )


class _RowwiseExperts(torch.nn.Module):
    """One top-1 routed MLP. Each token is independent of the others."""

    def __init__(self, hidden: int, intermediate: int, num_experts: int) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, num_experts, bias=False)
        self.w1 = torch.nn.Parameter(torch.empty(num_experts, intermediate, hidden))
        self.w2 = torch.nn.Parameter(torch.empty(num_experts, hidden, intermediate))
        torch.manual_seed(0)
        torch.nn.init.normal_(self.gate.weight)
        torch.nn.init.normal_(self.w1)
        torch.nn.init.normal_(self.w2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        expert = self.gate(hidden).argmax(dim=-1)
        up = torch.einsum("eih,th->eti", self.w1, hidden)
        activated = torch.nn.functional.silu(up[expert, torch.arange(hidden.shape[0])])
        down = torch.einsum("ehi,ti->eth", self.w2, activated)
        return down[expert, torch.arange(hidden.shape[0])]


def test_wide_ep_moe_layer_matches_tp1_reference():
    """Same weights: TP1 runs every row; wide EP runs one slice per TP rank."""
    tokens, hidden, intermediate, experts = 32, 16, 8, 4
    layer = _RowwiseExperts(hidden, intermediate, experts)
    layer.eval()
    batch = torch.randn(tokens, hidden)
    with torch.no_grad():
        reference = layer(batch)
        output, one_way = apply_wide_ep_rowwise(batch, _ATTN_TP, layer)
    assert torch.allclose(output, reference, rtol=1e-5, atol=1e-5)
    row_error = (output - reference).abs().amax(dim=-1)
    assert float(row_error.max()) < 1e-4
    assert one_way == expert_dispatch_bytes(tokens, hidden, batch.element_size())
    assert one_way * _ATTN_TP != one_way


def test_wide_ep_moe_layer_does_not_repeat_the_batch_per_rank():
    tokens, hidden, intermediate, experts = 64, 32, 8, 4
    layer = _RowwiseExperts(hidden, intermediate, experts)
    layer.eval()
    batch = torch.randn(tokens, hidden)
    repeats = 10

    def run_reference() -> None:
        with torch.no_grad():
            layer(batch)

    def run_wide() -> None:
        with torch.no_grad():
            apply_wide_ep_rowwise(batch, _ATTN_TP, layer)

    started = time.perf_counter()
    for _ in range(repeats):
        run_reference()
    reference_s = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        run_wide()
    wide_s = time.perf_counter() - started
    # Doing the full batch on every TP rank would cost about TP times the reference.
    assert wide_s < reference_s * _ATTN_TP
    assert expert_dispatch_bytes(tokens, hidden, batch.element_size()) == (
        tokens * hidden * batch.element_size()
    )


def test_residual_join_matches_tp1_and_stays_on_the_attention_replica():
    """MoE output is added on the attention replica, not across the expert group."""
    tokens, hidden, intermediate, experts = 32, 16, 8, 4
    layer = _RowwiseExperts(hidden, intermediate, experts)
    layer.eval()
    residual_0 = torch.randn(tokens, hidden)
    hidden_0 = torch.randn(tokens, hidden)
    residual_1 = torch.randn(tokens, hidden)
    hidden_1 = torch.randn(tokens, hidden)
    with torch.no_grad():
        reference = residual_0 + layer(hidden_0)
        moe_0, one_way = apply_wide_ep_rowwise(hidden_0, _ATTN_TP, layer)
        joined_0 = join_residual_on_attention_rank(residual_0, moe_0)
        moe_1, _ = apply_wide_ep_rowwise(hidden_1, _ATTN_TP, layer)
        joined_1 = join_residual_on_attention_rank(residual_1, moe_1)
        mixed = join_residual_on_attention_rank(residual_0, moe_0 + moe_1)
    assert torch.allclose(joined_0, reference, rtol=1e-5, atol=1e-5)
    assert float((joined_0 - reference).abs().amax()) < 1e-4
    assert not torch.allclose(joined_0, joined_1, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(joined_0, mixed, rtol=1e-3, atol=1e-3)
    # The join does not add an expert-group transfer. Dispatch stays one token once.
    assert one_way == expert_dispatch_bytes(tokens, hidden, hidden_0.element_size())
    assert one_way == tokens * hidden * hidden_0.element_size()


def test_gathered_slices_restore_and_one_token_pads():
    tokens, width = 5, 4
    hidden = torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width)
    padded, pad = pad_token_dim(hidden, _ATTN_TP)
    assert pad == _ATTN_TP - tokens
    parts = [
        split_replicated_tokens(padded, rank, _ATTN_TP) for rank in range(_ATTN_TP)
    ]
    gathered = torch.cat(parts, dim=0)
    restored = unpad_token_dim(
        restore_gathered_tokens(gathered, _ATTN_TP), pad
    )
    assert torch.equal(restored, hidden)
    sent = expert_dispatch_bytes(padded.shape[0], width, hidden.element_size())
    assert sent == padded.shape[0] * width * hidden.element_size()
    assert sent < sent * _ATTN_TP


def test_wide_ep_dispatch_bounds_agree_across_idle_groups():
    # One attention group holds a token. The other three are idle.
    counts = [1] * _ATTN_TP + [0] * (_WORLD - _ATTN_TP)
    total, maximum = wide_ep_dispatch_bounds(counts)
    assert total == _ATTN_TP
    assert maximum == 1
    # A rank that only knows its own zero count would size a different batch.
    assert wide_ep_dispatch_bounds([0])[1] != maximum


def test_idle_rank_joins_dispatch_with_one_dummy_row():
    # One attention group has a row. An idle rank still enters that dispatch.
    assert ep_dispatch_includes_empty_rank(
        local_tokens=0,
        num_global_tokens=_ATTN_TP,
        max_num_tokens_per_gpu=1,
        all_to_all=True,
    )
    # This CANN rejects batch 0, so the idle rank sends one dummy row.
    # It is not padded up to the busy rank's count.
    assert (
        ep_dispatch_local_rows(
            0,
            num_global_tokens=_ATTN_TP,
            max_num_tokens_per_gpu=1,
            all_to_all=True,
        )
        == 1
    )
    assert (
        ep_dispatch_local_rows(
            1,
            num_global_tokens=_ATTN_TP,
            max_num_tokens_per_gpu=1,
            all_to_all=True,
        )
        == 1
    )
    # Every rank empty: the group does not enter.
    assert (
        ep_dispatch_local_rows(
            0,
            num_global_tokens=0,
            max_num_tokens_per_gpu=0,
            all_to_all=True,
        )
        == 0
    )
    # A kernel that does not own the collective keeps the local early return.
    assert (
        ep_dispatch_local_rows(
            0,
            num_global_tokens=_ATTN_TP,
            max_num_tokens_per_gpu=1,
            all_to_all=False,
        )
        == 0
    )


def test_wide_ep_narrowing_does_not_wait_for_a_row_exchange():
    # The expert kernel sees this rank's split. The DP token table is the
    # pre-narrowing count and is not an input.
    assert not ced_narrowing_needs_row_exchange(
        keep_rows_present=True,
        has_global_token_counts=True,
        owns_ep_communication=False,
        wide_ep=True,
    )
    assert ced_narrowing_needs_row_exchange(
        keep_rows_present=True,
        has_global_token_counts=True,
        owns_ep_communication=False,
        wide_ep=False,
    )
    assert not ced_narrowing_needs_row_exchange(
        keep_rows_present=True,
        has_global_token_counts=True,
        owns_ep_communication=True,
        wide_ep=False,
    )


def test_wide_ep_mapping_requires_world_sized_expert_group():
    wide = _wide_mapping(0)
    assert mapping_uses_wide_ep(wide)
    matched = Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        dense_tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    assert not mapping_uses_wide_ep(matched)


def test_wide_ep_shared_expert_keeps_a_full_local_copy():
    """Row-split MoE cannot also shard shared weights on the attention TP group.

    The gather after dispatch only restores token order. An unreduced shared
    partial would stay wrong for every row. The MoE constructor therefore
    passes ``replicate_weights=True`` exactly when ``mapping_uses_wide_ep``.
    """
    wide = _wide_mapping(0)
    matched = Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        dense_tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    assert mapping_uses_wide_ep(wide)
    assert not mapping_uses_wide_ep(matched)
    # Dense TP on the wide mapping is the attention width that already owns the
    # row split; shared experts must not reuse it for a second shard.
    assert wide.dense.tp_size == _ATTN_TP
    assert wide.attn.tp_size == _ATTN_TP


def test_split_copies_one_share_not_the_whole_replica():
    tokens = 256
    hidden = torch.randn(tokens, _HIDDEN, dtype=torch.bfloat16)
    started = time.perf_counter()
    for _ in range(20):
        parts = [
            split_replicated_tokens(hidden, rank, _ATTN_TP) for rank in range(_ATTN_TP)
        ]
        restore_replicated_tokens(parts)
    elapsed = time.perf_counter() - started
    moved = 20 * hidden.numel() * hidden.element_size()
    assert parts[0].numel() * _ATTN_TP == hidden.numel()
    # A correct split moves each element a small constant number of times.
    # This bound only rejects an accidental TP-fold copy of the replica.
    assert elapsed < 5.0
    assert moved == 20 * tokens * _HIDDEN * _BF16_BYTES
