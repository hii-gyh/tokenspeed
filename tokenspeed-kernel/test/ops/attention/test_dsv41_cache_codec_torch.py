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

"""CPU tests for the Ascend torch FlatKV gather/unpack codecs."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch import (
    _e2m1_pair_table,
    _e4m3_decode_table,
    _e4m3_to_float,
    _e2m1_decode,
    cache_gather_torch,
    cache_pack_torch,
    cache_scatter_torch,
    cache_unpack_torch,
)

_FORMATS = ("swa", "global", "index")
_PAGE_ROWS = 64


@pytest.mark.parametrize("fmt", _FORMATS)
def test_cache_gather_torch_matches_pack_unpack_roundtrip(fmt: str):
    torch.manual_seed(5)
    dim = {"swa": 512, "global": 512, "index": 128}[fmt]
    pages = 8
    row_bytes = {"swa": 528, "global": 288, "index": 68}[fmt]
    cache = torch.zeros(pages, _PAGE_ROWS, row_bytes, dtype=torch.uint8)
    rows = torch.randn(48, dim, dtype=torch.bfloat16)
    slots = torch.arange(48, dtype=torch.int32)
    cache_scatter_torch(rows, cache, slots, fmt)

    gathered = cache_gather_torch(cache, slots.view(3, 16), fmt, None)
    packed = cache_pack_torch(rows, fmt, None)
    unpacked = cache_unpack_torch(packed, fmt, None)
    torch.testing.assert_close(gathered.reshape(48, dim), unpacked, rtol=0, atol=0)

    # Invalid slots read as zeros without disturbing valid neighbors.
    mixed = torch.tensor([0, -1, 47, 10_000], dtype=torch.int32)
    out = cache_gather_torch(cache, mixed, fmt, None)
    torch.testing.assert_close(out[0], unpacked[0], rtol=0, atol=0)
    torch.testing.assert_close(out[2], unpacked[47], rtol=0, atol=0)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[3]) == 0


def test_decode_tables_match_scalar_paths():
    device = torch.device("cpu")
    codes = torch.arange(256, dtype=torch.int32)
    e4 = _e4m3_decode_table(device)
    # E4M3FN NaN codes (0x7F / 0xFF) must match, including NaN bits.
    torch.testing.assert_close(
        e4, _e4m3_to_float(codes.to(torch.uint8)), rtol=0, atol=0, equal_nan=True
    )

    pair = _e2m1_pair_table(device)
    expected = torch.stack(
        (_e2m1_decode(codes & 15), _e2m1_decode((codes >> 4) & 15)),
        dim=-1,
    )
    torch.testing.assert_close(pair, expected, rtol=0, atol=0)


def test_cache_gather_torch_chunked_matches_single_pass():
    """Gather past ``_GATHER_ROW_CHUNK`` must match a single decode window."""
    from tokenspeed_kernel.ops.attention.dsv41.cache_codec_torch import (
        _GATHER_ROW_CHUNK,
    )

    torch.manual_seed(7)
    fmt = "global"
    dim, row_bytes = 512, 288
    pages = 64
    cache = torch.zeros(pages, _PAGE_ROWS, row_bytes, dtype=torch.uint8)
    n_fill = pages * _PAGE_ROWS
    rows = torch.randn(n_fill, dim, dtype=torch.bfloat16)
    cache_scatter_torch(rows, cache, torch.arange(n_fill, dtype=torch.int32), fmt)
    n = _GATHER_ROW_CHUNK + 1024
    slots = torch.randint(0, n_fill, (n,), dtype=torch.int32)
    got = cache_gather_torch(cache, slots, fmt, None)
    # Reference: gather in two explicit halves with the public API shape.
    mid = n // 2
    ref = torch.empty(n, dim, dtype=torch.bfloat16)
    ref[:mid] = cache_gather_torch(cache, slots[:mid], fmt, None)
    ref[mid:] = cache_gather_torch(cache, slots[mid:], fmt, None)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_cache_scatter_torch_duplicate_slots_keep_last(fmt: str):
    """Duplicate destinations store the last active row's payload."""
    torch.manual_seed(11)
    dim = {"swa": 512, "global": 512, "index": 128}[fmt]
    row_bytes = {"swa": 528, "global": 288, "index": 68}[fmt]
    cache = torch.zeros(4, _PAGE_ROWS, row_bytes, dtype=torch.uint8)
    rows = torch.randn(3, dim, dtype=torch.bfloat16)
    slots = torch.tensor([5, 5, 5], dtype=torch.int32)
    cache_scatter_torch(rows, cache, slots, fmt)
    got = cache_gather_torch(cache, torch.tensor([5], dtype=torch.int32), fmt, None)
    expect = cache_unpack_torch(cache_pack_torch(rows[2:3], fmt, None), fmt, None)
    torch.testing.assert_close(got, expect, rtol=0, atol=0)

    # Fully inactive slots must leave prior contents unchanged.
    cache.zero_()
    cache_scatter_torch(rows[:1], cache, torch.tensor([7], dtype=torch.int32), fmt)
    before = cache_gather_torch(cache, torch.tensor([7], dtype=torch.int32), fmt, None)
    cache_scatter_torch(
        rows,
        cache,
        torch.tensor([-1, -1, -1], dtype=torch.int32),
        fmt,
    )
    after = cache_gather_torch(cache, torch.tensor([7], dtype=torch.int32), fmt, None)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
