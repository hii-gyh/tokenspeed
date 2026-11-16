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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Ascend torch indexer tests for invocation-shared key gathers."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.dsv41._metadata import IndexMetadata
from tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch import cache_scatter_torch
from tokenspeed_kernel.ops.attention.dsv41.indexer_torch import index_topk_torch

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


def _inputs():
    device = torch.device("npu:0")
    keys = torch.zeros(8, 128, dtype=torch.bfloat16, device=device)
    for row in range(8):
        keys[row, row] = row + 1
    cache = torch.zeros(2, 64, 68, dtype=torch.uint8, device=device)
    slots = torch.arange(64, 72, dtype=torch.int64, device=device)
    cache_scatter_torch(keys, cache, slots, "index")
    query = torch.ones(1, 1, 128, dtype=torch.bfloat16, device=device)
    weights = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    table = torch.ones(1, 1, dtype=torch.int32, device=device)
    visible = torch.tensor([8], dtype=torch.int32, device=device)
    return query, weights, cache, table, visible


def _run(query, weights, cache, table, visible, metadata=None):
    return index_topk_torch(
        query,
        weights,
        cache,
        table,
        visible,
        None,
        4,
        0,
        8,
        None,
        metadata,
    )


def _assert_selection(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_shared_metadata_reuses_keys_without_changing_selection():
    query, weights, cache, table, visible = _inputs()
    expected = _run(query, weights, cache, table, visible)
    metadata = IndexMetadata(table, visible, None, None)
    first = _run(query, weights, cache, table, visible, metadata)
    _assert_selection(first, expected)

    # Invalidate the source only to detect an erroneous second gather. Runtime
    # keeps cache/table/length inputs unchanged for the metadata's lifetime.
    cache.zero_()
    second = _run(query, weights, cache, table, visible, metadata)
    _assert_selection(second, expected)
    uncached = _run(query, weights, cache, table, visible)
    assert not torch.equal(uncached[0], expected[0])


def test_shared_key_gather_replays_with_refreshed_cache():
    query, weights, cache, table, visible = _inputs()
    expected = _run(query, weights, cache, table, visible)
    populated = cache.clone()
    metadata = IndexMetadata(table, visible, None, None)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        first = _run(query, weights, cache, table, visible, metadata)
        second = _run(query, weights, cache, table, visible, metadata)

    cache.zero_()
    zero_expected = _run(query, weights, cache, table, visible)
    graph.replay()
    _assert_selection(first, zero_expected)
    _assert_selection(second, zero_expected)

    cache.copy_(populated)
    graph.replay()
    _assert_selection(first, expected)
    _assert_selection(second, expected)


def test_dense_prefill_metadata_does_not_retain_keys():
    query, weights, cache, table, visible = _inputs()
    expected = _run(query, weights, cache, table, visible)
    metadata = IndexMetadata(table, visible, None, (1,))
    first = _run(query, weights, cache, table, visible, metadata)
    _assert_selection(first, expected)

    cache.zero_()
    zero_expected = _run(query, weights, cache, table, visible)
    second = _run(query, weights, cache, table, visible, metadata)
    _assert_selection(second, zero_expected)
