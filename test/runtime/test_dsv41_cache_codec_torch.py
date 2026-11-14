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


@pytest.mark.parametrize(
    "fmt,dim,row_bytes", [("index", 128, 68), ("global", 512, 288), ("swa", 512, 528)]
)
def test_pack_unpack_roundtrip_close(fmt, dim, row_bytes):
    torch.manual_seed(0)
    rows = torch.randn(5, dim, dtype=torch.bfloat16)
    packed = _mod.cache_pack_torch(rows, fmt, None)
    assert packed.shape == (5, row_bytes)
    decoded = _mod.cache_unpack_torch(packed, fmt, None)
    assert decoded.dtype == torch.bfloat16
    # Quantization is lossy; require the same sign pattern on large values.
    assert torch.isfinite(decoded).all()
    corr = torch.corrcoef(
        torch.stack((rows.float().flatten(), decoded.float().flatten()))
    )[0, 1]
    assert corr > 0.95


def test_float_to_e4m3_roundtrip_corr_on_cpu():
    """Bit-level E4M3 encode must stay close enough for SWA KV cache use."""
    torch.manual_seed(7)
    values = torch.randn(1024, dtype=torch.float32) * 50
    values = values.clamp(-448, 448)
    codes = _mod._float_to_e4m3(values)
    decoded = _mod._e4m3_to_float(codes)
    corr = torch.corrcoef(torch.stack((values, decoded)))[0, 1]
    assert corr > 0.98
    assert codes.dtype == torch.uint8


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
    expect = _mod.cache_unpack_torch(
        _mod.cache_pack_torch(rows[:1], fmt, None), fmt, None
    )
    assert torch.equal(got, expect)

    cache.zero_()
    slots = torch.tensor([-1, 63], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots[1:], fmt, None)
    expect = _mod.cache_unpack_torch(
        _mod.cache_pack_torch(rows[1:], fmt, None), fmt, None
    )
    assert torch.equal(got, expect)


def test_duplicate_slot_keeps_the_later_row():
    fmt, dim, row_bytes = "swa", 512, 528
    torch.manual_seed(5)
    rows = torch.randn(2, dim, dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, row_bytes, dtype=torch.uint8)
    slots = torch.tensor([4, 4], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots[:1], fmt, None)
    expect = _mod.cache_unpack_torch(
        _mod.cache_pack_torch(rows[1:], fmt, None), fmt, None
    )
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


def test_as_flat1d_is_view_over_padded_page_stride():
    """Ascend OOM'd when reshape(-1) densified a padded page-flat view."""
    row_bytes = 288
    pages = 8
    page_bytes = 64 * row_bytes
    page_stride = page_bytes + 4096
    storage = torch.zeros(pages * page_stride, dtype=torch.uint8)
    cache = storage.as_strided((pages, 64, row_bytes), (page_stride, row_bytes, 1))
    flat = _mod._flat_pages(cache, row_bytes)
    flat1d = _mod._as_flat1d(flat)
    assert flat1d.data_ptr() == flat.data_ptr()
    assert flat1d.numel() == (pages - 1) * page_stride + page_bytes
    assert flat1d.stride(0) == 1
    # Dense reshape would be pages * page_bytes and break padded offsets.
    assert flat1d.numel() != pages * page_bytes


def test_cache_flat1d_dense_uses_contiguous_view():
    """Dense pools must use cache.view(-1), not the as_strided chain."""
    row_bytes = 288
    cache = torch.zeros(4, 64, row_bytes, dtype=torch.uint8)
    flat1d, page_stride = _mod._cache_flat1d(cache, row_bytes)
    assert page_stride == 64 * row_bytes
    assert flat1d.data_ptr() == cache.data_ptr()
    assert flat1d.is_contiguous()
    assert flat1d.numel() == cache.numel()
    assert flat1d.stride(0) == 1


def test_cache_flat1d_padded_does_not_densify():
    row_bytes = 288
    pages = 8
    page_bytes = 64 * row_bytes
    page_stride = page_bytes + 4096
    storage = torch.zeros(pages * page_stride, dtype=torch.uint8)
    cache = storage.as_strided((pages, 64, row_bytes), (page_stride, row_bytes, 1))
    flat1d, stride = _mod._cache_flat1d(cache, row_bytes)
    assert stride == page_stride
    assert flat1d.data_ptr() == cache.data_ptr()
    assert flat1d.numel() == (pages - 1) * page_stride + page_bytes


def test_scatter_gather_on_padded_page_stride_roundtrips():
    fmt, dim, row_bytes = "global", 512, 288
    pages = 8
    page_bytes = 64 * row_bytes
    page_stride = page_bytes + 2048
    torch.manual_seed(8)
    rows = torch.randn(3, dim, dtype=torch.bfloat16)
    storage = torch.full((pages * page_stride,), 0x5A, dtype=torch.uint8)
    cache = storage.as_strided((pages, 64, row_bytes), (page_stride, row_bytes, 1))
    slots = torch.tensor([1, 64 + 7, pages * 64 - 2], dtype=torch.int64)
    _mod.cache_scatter_torch(rows, cache, slots, fmt)
    got = _mod.cache_gather_torch(cache, slots, fmt, None)
    expect = _mod.cache_unpack_torch(_mod.cache_pack_torch(rows, fmt, None), fmt, None)
    assert torch.equal(got, expect)


def _soft_f32_to_e4m3fn_u8(x: torch.Tensor) -> torch.Tensor:
    """Mirror of Triton ``_f32_to_e4m3fn_u8`` (Ascend TruncF workaround)."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=448.0, neginf=-448.0)
    x = x.clamp(-448.0, 448.0)
    bits = x.view(torch.int32)
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF
    abs_bits = torch.minimum(
        abs_bits, torch.full((), 0x43E00000, dtype=torch.int32, device=x.device)
    )
    exp = (abs_bits >> 23) & 0xFF
    mant = abs_bits & 0x7FFFFF
    mant3 = mant >> 20
    round_bit = (mant >> 19) & 1
    sticky = (mant & ((1 << 19) - 1)) != 0
    incr = (round_bit != 0) & (sticky | ((mant3 & 1) != 0))
    mant3 = mant3 + incr.to(torch.int32)
    exp_adj = (mant3 > 7).to(torch.int32)
    mant3 = mant3 & 7
    e4_exp = (exp - 120) + exp_adj
    a = abs_bits.view(torch.float32)
    scaled = a * (2.0**9)
    floor = torch.floor(scaled)
    frac = scaled - floor
    odd = (floor.to(torch.int32) & 1) != 0
    rounded = floor + ((frac > 0.5) | ((frac == 0.5) & odd)).to(torch.float32)
    den_code = torch.where(
        rounded >= 8,
        torch.full((), 8, dtype=torch.int32, device=x.device),
        rounded.to(torch.int32).clamp(0, 7),
    )
    is_den = e4_exp <= 0
    normal_code = ((e4_exp.clamp(1, 15) << 3) | mant3).clamp(0, 0x7E)
    code = torch.where(is_den, den_code, normal_code)
    code = torch.where(abs_bits == 0, torch.zeros_like(code), code)
    code = torch.where(code == 0x7F, torch.full_like(code, 0x7E), code)
    return ((sign << 7) | code).to(torch.uint8)


def test_soft_e4m3_encode_matches_torch_float8_cast():
    torch.manual_seed(6)
    vals = torch.cat(
        [
            torch.randn(8000) * 100,
            torch.linspace(-500, 500, 2001),
            torch.tensor([0.0, 448.0, -448.0, 2**-9, 2**-6, 240.0, 256.0, 288.0]),
        ]
    )
    expect = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    got = _soft_f32_to_e4m3fn_u8(vals)
    assert torch.equal(got, expect)
