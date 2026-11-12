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

"""Host round-trips for the Ascend FlatKV torch cache codec."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel"
    / "python"
    / "tokenspeed_kernel"
    / "ops"
    / "attention"
    / "dsv41"
    / "cache_codec_torch.py"
)
_spec = importlib.util.spec_from_file_location("dsv41_cache_codec_host", _PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


@pytest.mark.parametrize("fmt,dim,row_bytes", [("index", 128, 68), ("global", 512, 288), ("swa", 512, 528)])
def test_pack_unpack_roundtrip_close(fmt, dim, row_bytes):
    torch.manual_seed(0)
    rows = torch.randn(5, dim, dtype=torch.bfloat16)
    packed = _mod.cache_pack_torch(rows, fmt, None)
    assert packed.shape == (5, row_bytes)
    decoded = _mod.cache_unpack_torch(packed, fmt, None)
    assert decoded.dtype == torch.bfloat16
    # Quantization is lossy; require the same sign pattern on large values.
    assert torch.isfinite(decoded).all()
    corr = torch.corrcoef(torch.stack((rows.float().flatten(), decoded.float().flatten())))[
        0, 1
    ]
    assert corr > 0.95


@pytest.mark.parametrize(
    "fmt,dim,row_bytes",
    [("index", 128, 68), ("global", 512, 288), ("swa", 512, 528)],
)
def test_scatter_gather_matches_pack_unpack(fmt, dim, row_bytes):
    torch.manual_seed(1)
    rows = torch.randn(3, dim, dtype=torch.bfloat16)
    cache = torch.zeros(2, 64, row_bytes, dtype=torch.uint8)
    slots = torch.tensor([1, 65, 2], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots, fmt, None)
    packed = _mod.cache_pack_torch(rows, fmt, None)
    expect = _mod.cache_unpack_torch(packed, fmt, None)
    assert torch.equal(got, expect)


def test_gather_from_a_large_pool_touches_only_selected_slots():
    fmt, dim, row_bytes = "global", 512, 288
    pages = 4096
    torch.manual_seed(2)
    rows = torch.randn(3, dim, dtype=torch.bfloat16)
    cache = torch.full((pages, 64, row_bytes), 0x5A, dtype=torch.uint8)
    slots = torch.tensor([1, 64 + 7, pages * 64 - 1], dtype=torch.int64)
    before = cache.clone()
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots, fmt, None)
    expect = _mod.cache_unpack_torch(_mod.cache_pack_torch(rows, fmt, None), fmt, None)
    assert torch.equal(got, expect)
    flat = _mod._flat_pages(cache, row_bytes)
    value_offsets = _mod._selected_byte_offsets(
        slots, _mod._column_indexes(slots, 256, 0), flat.stride(0)
    )
    scale_offsets = _mod._selected_byte_offsets(
        slots, _mod._column_indexes(slots, 32, 64 * 256), flat.stride(0)
    )
    assert value_offsets.numel() + scale_offsets.numel() == slots.numel() * row_bytes
    assert value_offsets.numel() < cache.numel()
    changed = cache != before
    assert int(changed.sum()) <= slots.numel() * row_bytes


def test_scatter_leaves_invalid_slots_unread():
    fmt, dim, row_bytes = "index", 128, 68
    torch.manual_seed(3)
    rows = torch.randn(3, dim, dtype=torch.bfloat16)
    cache = torch.full((2, 64, row_bytes), 0x11, dtype=torch.uint8)
    before = cache.clone()
    slots = torch.tensor([-1, -64, 10_000], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    assert torch.equal(cache, before)


def test_invalid_slot_does_not_clobber_the_row_it_aliases():
    fmt, dim, row_bytes = "index", 128, 68
    torch.manual_seed(4)
    # slot -1 addresses page 0 row 63, the same row as slot 63.
    rows = torch.randn(2, dim, dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, row_bytes, dtype=torch.uint8)
    slots = torch.tensor([63, -1], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots[:1], fmt, None)
    expect = _mod.cache_unpack_torch(_mod.cache_pack_torch(rows[:1], fmt, None), fmt, None)
    assert torch.equal(got, expect)

    cache.zero_()
    slots = torch.tensor([-1, 63], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots[1:], fmt, None)
    expect = _mod.cache_unpack_torch(_mod.cache_pack_torch(rows[1:], fmt, None), fmt, None)
    assert torch.equal(got, expect)


def test_duplicate_slot_keeps_the_later_row():
    fmt, dim, row_bytes = "swa", 512, 528
    torch.manual_seed(5)
    rows = torch.randn(2, dim, dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, row_bytes, dtype=torch.uint8)
    slots = torch.tensor([4, 4], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots[:1], fmt, None)
    expect = _mod.cache_unpack_torch(_mod.cache_pack_torch(rows[1:], fmt, None), fmt, None)
    assert torch.equal(got, expect)


def test_chunk_pages_keep_the_byte_offset_inside_int32():
    stride = 64 * 528
    pages = _mod._chunk_pages(stride)
    assert pages * stride <= 2**31 - 1
    assert (pages + 1) * stride > 2**31 - 1


def test_selected_byte_offsets_stay_int64_past_2gb():
    page = 100_000
    page_stride = 64 * 528
    slots = torch.tensor([page * 64 + 3], dtype=torch.int64)
    cols = torch.arange(4, dtype=torch.int64).view(1, 4)
    offsets = _mod._selected_byte_offsets(slots, cols, page_stride)
    assert offsets.dtype == torch.int64
    assert offsets.shape == (1, 4)
    base = page * page_stride
    assert int(offsets[0, 0]) == base
    assert int(offsets[0, 3]) == base + 3
    assert base > 2**31
