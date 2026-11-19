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

"""Ascend FlatKV scatter tests for payloads larger than one ND update."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch import (
    _cache_flat1d,
    _column_indexes,
    _read_planar,
    _transfer_selected,
    cache_pack_torch,
    cache_scatter_bf16_torch,
    cache_scatter_torch,
    cache_unpack_torch,
)
from tokenspeed_kernel.ops.attention.dsv41.compressor_torch import (
    _tail_rows,
    compressor_tail_scatter_torch,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize(
    ("cache_format", "num_rows"),
    (("swa", 19), ("global", 19), ("index", 65)),
)
def test_scatter_large_payload_matches_pack_in_eager_and_graph(
    cache_format: str, num_rows: int
):
    dim = {"swa": 512, "global": 512, "index": 128}[cache_format]
    row_bytes = {"swa": 528, "global": 288, "index": 68}[cache_format]
    device = torch.device("npu:0")
    torch.manual_seed(19)
    rows = torch.randn(num_rows, dim, dtype=torch.bfloat16, device=device)
    slots = torch.arange(num_rows, dtype=torch.int64, device=device)
    cache = torch.zeros(2, 64, row_bytes, dtype=torch.uint8, device=device)
    expected = cache_pack_torch(rows, cache_format, None)

    cache_scatter_torch(rows, cache, slots, cache_format)
    torch.testing.assert_close(
        _read_planar(cache, slots, cache_format), expected, rtol=0, atol=0
    )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        cache_scatter_torch(rows, cache, slots, cache_format)
        _read_planar(cache, slots, cache_format)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    cache.zero_()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cache_scatter_torch(rows, cache, slots, cache_format)
        observed = _read_planar(cache, slots, cache_format)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_rows", (55, 512, 2048))
def test_bf16_mirror_strided_graph_tracks_quantized_roundtrip(num_rows: int):
    device = torch.device("npu:0")
    torch.manual_seed(23)
    rows = torch.randn(num_rows, 512, dtype=torch.bfloat16, device=device)
    rows = cache_unpack_torch(cache_pack_torch(rows, "swa", None), "swa", None)
    slots = (torch.arange(num_rows, dtype=torch.int64, device=device) * 7 + 3) % (
        2 * num_rows
    )
    slots[[3, 17]] = -1
    arena = torch.randn(
        num_rows // 32 + 2, 96, 512, dtype=torch.bfloat16, device=device
    )
    cache = arena[:, 32:96, :]

    def expected_cache(source: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        expected = baseline.clone()
        valid = slots >= 0
        expected.view(-1, 512)[slots[valid]] = source[valid]
        return expected

    initial = cache.clone()
    cache_scatter_bf16_torch(rows, cache, slots)
    torch.npu.synchronize()
    torch.testing.assert_close(cache, expected_cache(rows, initial), rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cache_scatter_bf16_torch(rows, cache, slots)
    cache.zero_()
    blank = cache.clone()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(cache, expected_cache(rows, blank), rtol=0, atol=0)

    updated = torch.randn_like(rows)
    updated = cache_unpack_torch(cache_pack_torch(updated, "swa", None), "swa", None)
    rows.copy_(updated)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(cache, expected_cache(updated, blank), rtol=0, atol=0)


def test_bf16_mirror_duplicate_and_inactive_slots_match_last_active_write():
    device = torch.device("npu:0")
    rows = torch.randn(4, 512, dtype=torch.bfloat16, device=device)
    slots = torch.tensor([2, 2, -1, 0], dtype=torch.int64, device=device)
    cache = torch.randn(2, 64, 512, dtype=torch.bfloat16, device=device)
    initial = cache.clone()
    expected = initial.clone()
    expected.view(-1, 512)[2] = rows[1]
    expected.view(-1, 512)[0] = rows[3]

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cache_scatter_bf16_torch(rows, cache, slots)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)

    updated = torch.randn_like(rows)
    rows.copy_(updated)
    expected.view(-1, 512)[2] = updated[1]
    expected.view(-1, 512)[0] = updated[3]
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("swa", "global", "index"))
@pytest.mark.parametrize("storage_offset", (8, 4))
def test_read_planar_padded_arena_matches_byte_columns(cache_format, storage_offset):
    """The widened index gather must return the same bytes as byte columns.

    A padded arena keeps the flat view non-contiguous, and an offset that is
    not a multiple of eight forces the byte-wide fallback.
    """
    row_bytes = {"swa": 528, "global": 288, "index": 68}[cache_format]
    value_bytes = {"swa": 512, "global": 256, "index": 64}[cache_format]
    scale_bytes = row_bytes - value_bytes
    device = torch.device("npu:0")
    torch.manual_seed(31)
    pages = 40
    page_stride = 64 * row_bytes + 64
    storage = torch.randint(
        0,
        255,
        (pages * page_stride + 64,),
        dtype=torch.uint8,
        device=device,
    )
    cache = storage.as_strided(
        (pages, 64, row_bytes), (page_stride, row_bytes, 1), storage_offset
    )
    slots = torch.randint(0, pages * 64, (300,), dtype=torch.int64, device=device)
    slots[7] = slots[3]

    flat1d, stride = _cache_flat1d(cache, row_bytes)
    columns = torch.cat(
        (
            _column_indexes(slots, value_bytes, 0),
            _column_indexes(slots, scale_bytes, 64 * value_bytes),
        ),
        dim=1,
    )
    reference = cache.new_empty(slots.shape[0], row_bytes)
    _transfer_selected(flat1d, stride, slots, columns, reference, write=False)
    torch.testing.assert_close(
        _read_planar(cache, slots, cache_format), reference, rtol=0, atol=0
    )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        observed = _read_planar(cache, slots, cache_format)
    storage.random_()
    reference = cache.new_empty(slots.shape[0], row_bytes)
    _transfer_selected(flat1d, stride, slots, columns, reference, write=False)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_compressor_tail_scatter_padded_arena_matches_cpu_reference():
    """The vendor tail scatter must store what advanced indexing stores."""
    device = torch.device("npu:0")
    torch.manual_seed(37)
    pages = 37
    row = 2 * 512
    page_stride = 2 * row + 32
    shape = (pages, 2, 2, 512)
    strides = (page_stride, row, 512, 1)
    host_storage = torch.randn(pages * page_stride + 64, dtype=torch.float32)
    device_storage = host_storage.to(device)
    host_tail = host_storage.as_strided(shape, strides, 16)
    device_tail = device_storage.as_strided(shape, strides, 16)
    content = torch.randn(12, 512, dtype=torch.float32)
    scores = torch.randn(12, 512, dtype=torch.float32)
    values = [5, 17, 4096, 73, 1, 5, -1, 999999, 0, 2, 2, 71]
    slots = torch.tensor(values, dtype=torch.int64)

    compressor_tail_scatter_torch(content, scores, host_tail, slots)
    device_content = content.to(device)
    device_scores = scores.to(device)
    device_slots = slots.to(device)
    compressor_tail_scatter_torch(
        device_content, device_scores, device_tail, device_slots
    )
    torch.npu.synchronize()
    torch.testing.assert_close(device_storage.cpu(), host_storage, rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        compressor_tail_scatter_torch(
            device_content, device_scores, device_tail, device_slots
        )
    content.normal_()
    scores.normal_()
    compressor_tail_scatter_torch(content, scores, host_tail, slots)
    device_storage.copy_(host_storage)
    device_content.copy_(content)
    device_scores.copy_(scores)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(device_storage.cpu(), host_storage, rtol=0, atol=0)


@pytest.mark.parametrize("half", (0, 1))
@pytest.mark.parametrize("pad", (32, 31))
def test_tail_rows_padded_arena_matches_advanced_indexing(half: int, pad: int):
    """The flat-view tail read must return exactly what advanced indexing returns.

    An odd page stride leaves the storage length odd, which forces the
    unpaired fallback, so both branches are covered.
    """
    device = torch.device("npu:0")
    torch.manual_seed(41)
    pages = 37
    row = 2 * 512
    page_stride = 2 * row + pad
    storage = torch.randn(pages * page_stride + 64, dtype=torch.float32, device=device)
    tail = storage.as_strided((pages, 2, 2, 512), (page_stride, row, 512, 1), 16)
    slots = torch.tensor([5, 17, 40, 73, 1, 5, 0, 71], dtype=torch.int64, device=device)
    page = torch.div(slots, 2, rounding_mode="floor")
    row_index = slots.remainder(2)

    torch.testing.assert_close(
        _tail_rows(tail, page, row_index, half),
        tail[page, row_index, half],
        rtol=0,
        atol=0,
    )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        observed = _tail_rows(tail, page, row_index, half)
    storage.normal_()
    expected = tail[page, row_index, half].clone()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
