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

"""Host tests for the FlatKV → Ascend PA_BBND page adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_PAGES_PATH = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel"
    / "python"
    / "tokenspeed_kernel"
    / "ops"
    / "attention"
    / "dsv41"
    / "pages.py"
)
_spec = importlib.util.spec_from_file_location("dsv41_pages_host", _PAGES_PATH)
_pages = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pages)

FLATKV_PAGE_ROWS = _pages.FLATKV_PAGE_ROWS
HEAD_DIM = _pages.HEAD_DIM
adapt_flatkv_decode_selections = _pages.adapt_flatkv_decode_selections
as_pa_bbnd = _pages.as_pa_bbnd
cmp_residual_kv = _pages.cmp_residual_kv
compact_selection_indices = _pages.compact_selection_indices
identity_block_table = _pages.identity_block_table
lcm_block_table_as_pa = _pages.lcm_block_table_as_pa
align_pa_block_for_a2 = _pages.align_pa_block_for_a2
a2_cmp_topk = _pages.a2_cmp_topk
materialize_bf16_rows_as_pa_bbnd = _pages.materialize_bf16_rows_as_pa_bbnd
pad_sparse_indices = _pages.pad_sparse_indices


def test_cmp_topk_uses_the_metadata_buckets():
    assert a2_cmp_topk(16) == 512
    assert a2_cmp_topk(512) == 512
    assert a2_cmp_topk(513) == 1024


def test_as_pa_bbnd_views_page_planar_without_copy():
    cache = torch.zeros(3, FLATKV_PAGE_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    viewed = as_pa_bbnd(cache)
    assert viewed.shape == (3, FLATKV_PAGE_ROWS, 1, HEAD_DIM)
    assert viewed.data_ptr() == cache.data_ptr()


def test_lcm_block_table_as_pa_is_contiguous_int32():
    table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int64)
    out = lcm_block_table_as_pa(table)
    assert out.dtype == torch.int32
    assert out.is_contiguous()
    assert out.tolist() == [[1, 2], [3, 0]]


def test_materialize_builds_one_block_per_query():
    rows = torch.randn(2, 4, HEAD_DIM, dtype=torch.bfloat16)
    lengths = torch.tensor([3, 1], dtype=torch.int32)
    selection = materialize_bf16_rows_as_pa_bbnd(rows, lengths)
    assert selection.kv.shape == (2, 4, 1, HEAD_DIM)
    assert selection.block_table.tolist() == [[0], [1]]
    assert selection.sparse_indices.shape == (2, 1, 4)
    assert selection.sparse_indices[0, 0].tolist() == [0, 1, 2, -1]
    assert selection.sparse_indices[1, 0].tolist() == [0, -1, -1, -1]
    assert selection.topk_length.tolist() == [[3], [1]]


def test_a2_block_alignment_pads_short_selections_and_keeps_visible_rows():
    rows = torch.randn(2, 4, HEAD_DIM, dtype=torch.bfloat16)
    lengths = torch.tensor([3, 1], dtype=torch.int32)
    aligned = align_pa_block_for_a2(materialize_bf16_rows_as_pa_bbnd(rows, lengths))
    assert aligned.kv.shape == (2, 16, 1, HEAD_DIM)
    assert aligned.kv[:, :4].equal(rows.unsqueeze(2))
    assert aligned.sparse_indices.shape == (2, 1, 16)
    assert aligned.sparse_indices[0, 0, :4].tolist() == [0, 1, 2, -1]
    assert aligned.sparse_indices[0, 0, 4:].eq(-1).all()
    assert aligned.topk_length.tolist() == [[3], [1]]
    wide = materialize_bf16_rows_as_pa_bbnd(
        torch.randn(1, 128, HEAD_DIM, dtype=torch.bfloat16),
        torch.tensor([128], dtype=torch.int32),
    )
    assert align_pa_block_for_a2(wide).kv.shape[1] == 128


def test_dense_selected_attention_ignores_unselected_rows():
    torch.manual_seed(0)
    q = torch.randn(2, 2, 4, dtype=torch.bfloat16)
    kv = torch.randn(6, 4, dtype=torch.bfloat16)
    indices = torch.tensor([[0, -1], [2, 1]], dtype=torch.int32)
    sink = torch.tensor([0.0, 0.0])
    got = _pages.selected_attention_dense(q, kv, indices, sink, 1.0, None, None)
    assert got.shape == q.shape
    poisoned = kv.clone()
    poisoned[4:] = 9
    again = _pages.selected_attention_dense(q, poisoned, indices, sink, 1.0, None, None)
    assert torch.equal(got, again)
    # One valid key and no sink: the output is that key.
    q1 = torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16)
    kv1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    out = _pages.selected_attention_dense(
        q1, kv1, torch.tensor([[0, -1]]), None, 1.0, None, None
    )
    assert torch.allclose(out.float(), torch.tensor([[[1.0, 0.0]]]), atol=1e-2)


def test_repeat_query_heads_keeps_the_live_heads():
    q = torch.arange(2 * 2 * 4, dtype=torch.bfloat16).reshape(2, 2, 4)
    sinks = torch.tensor([0.25, 0.5])
    padded_q, padded_sinks = _pages.repeat_query_heads(q, sinks, 8)
    assert padded_q.shape == (2, 8, 4)
    assert padded_sinks.shape == (8,)
    assert padded_q[:, :2].equal(q)
    assert padded_sinks[:2].tolist() == [0.25, 0.5]
    same_q, same_sinks = _pages.repeat_query_heads(q, sinks, 2)
    assert same_q.shape == q.shape
    assert same_sinks.tolist() == [0.25, 0.5]


def test_adapt_flatkv_decode_selections_ratio2_residual():
    swa = torch.randn(2, 8, HEAD_DIM, dtype=torch.bfloat16)
    swa_lens = torch.tensor([8, 5], dtype=torch.int32)
    global_rows = torch.randn(2, 4, HEAD_DIM, dtype=torch.bfloat16)
    global_lens = torch.tensor([4, 2], dtype=torch.int32)
    seq_lens = torch.tensor([9, 4], dtype=torch.int32)
    ori, cmp, residual = adapt_flatkv_decode_selections(
        swa_rows=swa,
        swa_lens=swa_lens,
        global_rows=global_rows,
        global_lens=global_lens,
        cmp_ratio=2,
        seq_lens=seq_lens,
    )
    assert ori.kv.shape[0] == 2
    assert cmp is not None and cmp.kv.shape[0] == 2
    assert residual is not None
    assert residual.tolist() == [1, 0]


def test_adapt_swa_only_skips_compressed():
    swa = torch.randn(1, 4, HEAD_DIM, dtype=torch.bfloat16)
    ori, cmp, residual = adapt_flatkv_decode_selections(
        swa_rows=swa,
        swa_lens=torch.tensor([4], dtype=torch.int32),
        global_rows=None,
        global_lens=None,
        cmp_ratio=0,
        seq_lens=torch.tensor([4], dtype=torch.int32),
    )
    assert cmp is None and residual is None
    assert ori.block_table.tolist() == [[0]]


def test_pad_and_compact_helpers():
    padded = pad_sparse_indices(torch.tensor([[0, 2], [1, -1]]), topk=4)
    assert padded.shape == (2, 1, 4)
    assert padded[0, 0].tolist() == [0, 2, -1, -1]
    indices, lengths = compact_selection_indices(
        torch.tensor([2, 0], dtype=torch.int32), 3
    )
    assert indices[0, 0].tolist() == [0, 1, -1]
    assert lengths.tolist() == [[2], [0]]
    assert identity_block_table(3, device=torch.device("cpu")).tolist() == [
        [0],
        [1],
        [2],
    ]
    assert cmp_residual_kv(torch.tensor([5, 4]), 2).tolist() == [1, 0]
    assert cmp_residual_kv(torch.tensor([5, 4]), 1) is None


def test_prefill_workspace_matches_the_dense_key_set():
    kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16)
    indices = torch.tensor(
        [[0, -1, 2, 4, -1], [1, -1, -1, -1, -1]], dtype=torch.int32
    )
    seq_lens = torch.tensor([8, 9], dtype=torch.int32)
    ori, cmp, residual = _pages.prefill_workspace_selections(
        kv, indices, 3, 2, seq_lens
    )
    assert residual.tolist() == [0, 1]
    assert ori.topk_length.view(-1).tolist() == [2, 1]
    assert cmp is not None
    assert cmp.topk_length.view(-1).tolist() == [1, 0]
    assert torch.equal(ori.kv[0, :2, 0], kv[[0, 2]])
    assert torch.equal(ori.kv[1, :1, 0], kv[[1]])
    assert torch.equal(cmp.kv[0, :1, 0], kv[[4]])
    # The compact width is the longest selected run, not the workspace.
    assert ori.kv.shape[1] == 2
    assert ori.kv.shape[1] < kv.shape[0]
    poisoned = kv.clone()
    poisoned[3] = 9
    poisoned[5] = 9
    ori_poison, cmp_poison, _ = _pages.prefill_workspace_selections(
        poisoned, indices, 3, 2, seq_lens
    )
    assert torch.equal(ori.kv, ori_poison.kv)
    assert torch.equal(cmp.kv, cmp_poison.kv)
    q = torch.randn(2, 1, HEAD_DIM, dtype=torch.bfloat16)
    dense = _pages.selected_attention_dense(q, kv, indices, None, 1.0, None, None)
    token0 = _pages.selected_attention_dense(
        q[:1],
        torch.stack((kv[0], kv[2], kv[4])),
        torch.tensor([[0, 1, 2]]),
        None,
        1.0,
        None,
        None,
    )
    token1 = _pages.selected_attention_dense(
        q[1:], kv[1:2], torch.tensor([[0]]), None, 1.0, None, None
    )
    assert torch.allclose(dense[:1].float(), token0.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(dense[1:].float(), token1.float(), atol=1e-2, rtol=1e-2)
    aligned = align_pa_block_for_a2(ori)
    assert aligned.kv.shape[1] % 16 == 0
    assert aligned.sparse_indices.shape[-1] == aligned.kv.shape[1]
    swa_only, no_cmp, no_residual = _pages.prefill_workspace_selections(
        kv, indices[:, :3], 3, 0, seq_lens
    )
    assert no_cmp is None and no_residual is None
    assert swa_only.topk_length.view(-1).tolist() == [2, 1]


def test_host_seqused_max_does_not_read_device_tensors():
    assert _pages.host_seqused_max(None) == 0
    assert _pages.host_seqused_max(torch.empty(0, dtype=torch.int32)) == 0
    lengths = torch.tensor([0, 4, 2], dtype=torch.int32)
    assert _pages.host_seqused_max(lengths) == 4
    assert _pages.host_seqused_max(torch.zeros(3, dtype=torch.int32)) == 0
    # A non-CPU tensor must not be reduced. meta is enough to prove the branch.
    meta = torch.empty(3, dtype=torch.int32, device="meta")
    assert _pages.host_seqused_max(meta) is None
