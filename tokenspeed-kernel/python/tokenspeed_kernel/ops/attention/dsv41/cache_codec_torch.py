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

"""Device-portable torch codecs for V4.1 page-planar FlatKV fields.

Covers the checkpoint ``v41`` layouts (``swa`` / ``global`` / ``index``) that
Ascend A2 uses with TokenSpeed FlatKV. Packs contiguous rows the same way as
the numerics reference, then places value bytes before scale bytes in each
page so gather matches the Triton planar reader.
"""

from __future__ import annotations

import torch

# dim, group, value_bytes, row_bytes
_LAYOUTS = {
    "swa": (512, 32, 512, 528),
    "global": (512, 16, 256, 288),
    "index": (128, 32, 64, 68),
}
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
# E4M3FN NaN encodings are 0x7F and 0xFF. The finite codes are a constant.
_E4M3_FINITE = tuple(code for code in range(256) if (code & 0x7F) != 0x7F)
_PAGE_ROWS = 64
# Bound gather scratch (int64 column indexes + LUT expands). Capture can pad
# slots far past decode N; processing this many rows at a time keeps peak
# scratch on the order of tens of MiB instead of GiB.
_GATHER_ROW_CHUNK = 8192
_DEVICE_CONSTS: dict[tuple[str, str], torch.Tensor] = {}


def _device_const(name: str, device: torch.device, build) -> torch.Tensor:
    """Reuse a tiny device tensor built on the first call.

    Graph capture cannot boolean-index or copy these tables from the host.
    Warmup builds them; capture and replay read the same storage.
    """
    key = (name, str(device))
    found = _DEVICE_CONSTS.get(key)
    if found is None:
        found = build(device)
        _DEVICE_CONSTS[key] = found
    return found


def _layout(cache_format: str):
    if cache_format not in _LAYOUTS:
        raise ValueError(
            f"Ascend torch codec supports {sorted(_LAYOUTS)}, got {cache_format!r}"
        )
    return _LAYOUTS[cache_format]


def _validate_cache_pages(cache: torch.Tensor, row_bytes: int) -> None:
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1] != _PAGE_ROWS
        or cache.shape[2] != row_bytes
    ):
        raise ValueError(
            f"cache must be uint8 [pages, {_PAGE_ROWS}, {row_bytes}], got {tuple(cache.shape)}"
        )
    if cache.stride(2) != 1 or cache.stride(1) != row_bytes:
        raise ValueError("cache pages must have contiguous row bytes")


def _flat_pages(cache: torch.Tensor, row_bytes: int) -> torch.Tensor:
    _validate_cache_pages(cache, row_bytes)
    return cache.as_strided(
        (cache.shape[0], _PAGE_ROWS * row_bytes), (cache.stride(0), 1)
    )


def _as_flat1d(flat: torch.Tensor) -> torch.Tensor:
    """1-D storage view covering ``page * stride(0) + col`` without densifying.

    Ascend ``reshape(-1)`` on the page-flat ``as_strided`` view copies the whole
    pool (~16 GiB) and OOMs under post-load headroom (~11 GiB free). Keep a view
    so scatter/gather only touch selected bytes; offsets still use page stride.
    """
    pages = int(flat.shape[0])
    page_bytes = int(flat.shape[1])
    page_stride = int(flat.stride(0))
    if pages == 0:
        return flat.as_strided((0,), (1,))
    storage_len = (pages - 1) * page_stride + page_bytes
    return flat.as_strided((storage_len,), (1,))


def _cache_flat1d(cache: torch.Tensor, row_bytes: int) -> tuple[torch.Tensor, int]:
    """Byte-addressable 1-D view of ``cache`` plus its page stride.

    Dense contiguous pools use ``cache.view(-1)`` so Ascend index-gathers hit the
    fast contiguous path (the as_strided chain after ``_flat_pages`` was ~2.5×
    slower on gather_global vs the 5a baseline). Padded arenas keep
    ``_as_flat1d`` — never ``reshape(-1)`` densify or ``flat[page]``.
    """
    _validate_cache_pages(cache, row_bytes)
    page_stride = int(cache.stride(0))
    page_bytes = _PAGE_ROWS * row_bytes
    if cache.is_contiguous() and page_stride == page_bytes:
        return cache.view(-1), page_stride
    flat = cache.as_strided((cache.shape[0], page_bytes), (page_stride, 1))
    return _as_flat1d(flat), page_stride


def _e2m1_decode(nibbles: torch.Tensor) -> torch.Tensor:
    levels = _device_const(
        "e2m1_levels",
        nibbles.device,
        lambda device: torch.tensor(_E2M1_LEVELS, device=device, dtype=torch.float32),
    )
    mag = levels[(nibbles & 7).long()]
    return torch.where((nibbles & 8) != 0, -mag, mag)


def _e2m1_pair_table(device: torch.device) -> torch.Tensor:
    """Map each packed E2M1 byte to float32 ``[lo, hi]`` levels.

    Unpack used to nibble-mask, table-gather, and ``stack`` twice per row; one
    256-entry LUT replaces that on the gather hot path.
    """

    def build(dev: torch.device) -> torch.Tensor:
        codes = torch.arange(256, device=dev, dtype=torch.int32)
        return torch.stack(
            (_e2m1_decode(codes & 15), _e2m1_decode((codes >> 4) & 15)),
            dim=-1,
        )

    return _device_const("e2m1_pair_table", device, build)


def _e4m3_decode_table(device: torch.device) -> torch.Tensor:
    """Map each E4M3FN code byte to float32.

    Built once per device from ``_e4m3_to_float`` so CUDA keeps the float8 cast
    path while Ascend pays the bit decode only at warmup.
    """

    def build(dev: torch.device) -> torch.Tensor:
        # Ascend rejects ``arange(..., dtype=uint8)``; build as int32 then cast.
        codes = torch.arange(256, device=dev, dtype=torch.int32).to(torch.uint8)
        return _e4m3_to_float(codes)

    return _device_const("e4m3_decode_table", device, build)


def _e2m1_encode(values: torch.Tensor) -> torch.Tensor:
    # Eight levels: avoid a broadcast argmin on NPU (same cost class as E4M3).
    levels = _device_const(
        "e2m1_levels",
        values.device,
        lambda device: torch.tensor(_E2M1_LEVELS, device=device, dtype=torch.float32),
    )
    ax = values.abs().reshape(-1)
    # searchsorted midpoints → nearest level index in natural order 0..7.
    # Midpoints between consecutive levels.
    mids = _device_const(
        "e2m1_mids",
        values.device,
        lambda device: 0.5
        * (
            torch.tensor(_E2M1_LEVELS, device=device, dtype=torch.float32)[:-1]
            + torch.tensor(_E2M1_LEVELS, device=device, dtype=torch.float32)[1:]
        ),
    )
    idx = torch.searchsorted(mids, ax, right=False).clamp(0, 7)
    negative = torch.signbit(values).reshape(-1)
    return (idx | (negative.int() << 3)).to(torch.uint8).reshape(values.shape)


def _e4m3_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN bytes. CUDA uses the float8 cast; NPU has no such copy."""
    if codes.is_cuda:
        return codes.view(torch.float8_e4m3fn).float()
    bits = codes.to(torch.int32)
    sign = (bits >> 7) & 1
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    subnormal = mantissa.to(torch.float32) * (2.0**-9)
    normal = (1.0 + mantissa.to(torch.float32) / 8.0) * torch.exp2(
        (exponent - 7).to(torch.float32)
    )
    value = torch.where(exponent == 0, subnormal, normal)
    nan = (exponent == 0xF) & (mantissa == 0x7)
    value = torch.where(nan, torch.full((), float("nan"), device=codes.device), value)
    return torch.where(sign == 1, -value, value)


def _e4m3_finite(device: torch.device) -> torch.Tensor:
    return _device_const(
        "e4m3_finite",
        device,
        lambda dev: torch.tensor(_E4M3_FINITE, dtype=torch.int32, device=dev),
    )


def _float_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Encode finite floats as E4M3FN bytes. CUDA uses the float8 cast.

    Non-CUDA (Ascend) uses a bit-level rebias instead of a 240-wide nearest
    table. Decode SWA scatter was launching that broadcast every layer and
    dominated attention wall time on NPU.
    """
    if values.is_cuda:
        return values.to(torch.float8_e4m3fn).view(torch.uint8)
    x = values.float()
    sign = torch.signbit(x).to(torch.int32) << 7
    ax = x.abs()
    # E4M3FN max finite is 448; saturate rather than emit NaN encodings.
    ax = ax.clamp(max=448.0)
    bits = ax.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    # min normal = 2^(1-7) = 2^-6; below that, encode as subnormal.
    min_normal_exp = 1 - 7
    is_sub = exp < min_normal_exp
    # Normal: stored_exp = exp + 7, take top 3 mantissa bits with round-nearest.
    stored_exp = (exp + 7).clamp(0, 15)
    # Round: look at bit 19 (0.5 ULP of 3-bit mantissa in the 23-bit field).
    round_bit = (mant >> 19) & 1
    sticky = (mant & ((1 << 19) - 1)) != 0
    mant3 = mant >> 20
    mant3 = mant3 + torch.where(
        (round_bit != 0) & (sticky | ((mant3 & 1) != 0)),
        torch.ones_like(mant3),
        torch.zeros_like(mant3),
    )
    # Mantissa overflow into exponent.
    exp_carry = mant3 > 7
    mant3 = torch.where(exp_carry, torch.zeros_like(mant3), mant3)
    stored_exp = stored_exp + exp_carry.to(stored_exp.dtype)
    # Subnormals: value = mant_field/8 * 2^-6, round into 3-bit field.
    # ax / 2^-9 = ax * 512 gives the 3-bit subnormal payload with 0.5 ULPs at
    # bit weight 2^-9.
    sub_payload = (ax * (2.0**9)).round().clamp(0, 7).to(torch.int32)
    out_exp = torch.where(is_sub, torch.zeros_like(stored_exp), stored_exp)
    out_mant = torch.where(is_sub, sub_payload, mant3.clamp(0, 7))
    # Clamp exp overflow from rounding into the finite max code (0x7E).
    overflow = out_exp >= 15
    out_exp = torch.where(overflow, torch.full_like(out_exp, 14), out_exp)
    out_mant = torch.where(overflow, torch.full_like(out_mant, 7), out_mant)
    zero = ax == 0
    code = (out_exp << 3) | out_mant
    code = torch.where(zero, torch.zeros_like(code), code)
    return (sign | code).to(torch.uint8)


def _pack_rows(rows: torch.Tensor, cache_format: str) -> torch.Tensor:
    """Return contiguous uint8 [N, row_bytes] = values then scale bytes."""
    dim, group, value_bytes, row_bytes = _layout(cache_format)
    x = rows.float().reshape(-1, dim // group, group)
    amax = x.abs().amax(dim=-1)
    if cache_format == "global":
        scale = _float_to_e4m3(amax.clamp_min(6 * 2.0**-9) / 6)
        scale_bytes = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
        scale_f = _e4m3_decode_table(rows.device)[scale_bytes.long()]
    else:
        swa = cache_format == "swa"
        scaled = amax.clamp_min(1e-4 if swa else 6 * 2.0**-126)
        scaled = scaled * (1 / (448 if swa else 6))
        bits = scaled.view(torch.int32)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
        scale_f = torch.exp2(exponent.float())
        scale_bytes = (exponent + 127).to(torch.uint8)
    normalized = x / scale_f.unsqueeze(-1)
    if cache_format == "swa":
        quant = _float_to_e4m3(normalized.clamp(-448, 448))
        values = quant.flatten(1)
    else:
        codes = _e2m1_encode(normalized.reshape(-1, dim)).reshape(-1, dim // 2, 2)
        values = codes[..., 0] | (codes[..., 1] << 4)
    packed = torch.empty(
        rows.shape[0], row_bytes, dtype=torch.uint8, device=rows.device
    )
    packed[:, :value_bytes] = values
    packed[:, value_bytes:] = scale_bytes.flatten(1)
    return packed


def _unpack_rows(
    packed: torch.Tensor, cache_format: str, dtype: torch.dtype
) -> torch.Tensor:
    dim, group, value_bytes, row_bytes = _layout(cache_format)
    if packed.shape[-1] != row_bytes:
        raise ValueError(f"packed rows must be {row_bytes} bytes")
    values = packed[..., :value_bytes]
    scale_u8 = packed[..., value_bytes:]
    if cache_format == "global":
        scale = _e4m3_decode_table(packed.device)[scale_u8.long()]
    else:
        scale = torch.where(
            scale_u8 == 0,
            torch.full((), 2.0**-127, device=packed.device),
            (scale_u8.to(torch.int32) << 23).view(torch.float32),
        )
    scale = scale.reshape(*packed.shape[:-1], dim // group)
    if cache_format == "swa":
        decoded = _e4m3_decode_table(packed.device)[values.long()].reshape(
            *scale.shape, group
        )
    else:
        # One LUT gather yields both E2M1 nibbles; avoids mask/stack on NPU.
        decoded = _e2m1_pair_table(packed.device)[values.long()].reshape(
            *scale.shape, group
        )
    return (decoded * scale.unsqueeze(-1)).reshape(*packed.shape[:-1], dim).to(dtype)


def _selected_byte_offsets(
    slots: torch.Tensor, cols: torch.Tensor, page_stride: int
) -> torch.Tensor:
    """Byte offsets of selected in-page columns. One int64 per copied byte.

    ``page * page_stride`` passes 2^31 once the pool is larger than 2 GiB, so
    the multiply stays in int64. The result has one entry per selected byte,
    never one per page of the pool.
    """
    page = torch.div(slots, _PAGE_ROWS, rounding_mode="floor").to(torch.int64)
    return page.unsqueeze(1) * int(page_stride) + cols.to(torch.int64)


def _column_indexes(slots: torch.Tensor, width: int, origin: int) -> torch.Tensor:
    page_row = (slots % _PAGE_ROWS).to(torch.int64).unsqueeze(1)
    return (
        origin
        + page_row * width
        + torch.arange(width, device=slots.device, dtype=torch.int64)
    )


def _winning_values(
    key: torch.Tensor, active: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Repeat the last active row's values across rows that share ``key``.

    The index write stores that one payload at every duplicate, so the stored
    bytes do not depend on which duplicate a device applies last.
    """
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


def _transfer_selected(
    flat1d: torch.Tensor,
    page_stride: int,
    slots: torch.Tensor,
    cols: torch.Tensor,
    packed_cols: torch.Tensor,
    *,
    write: bool,
    live: torch.Tensor | None = None,
) -> None:
    """Copy selected columns through int64 byte offsets into the page-flat view.

    Ascend ``view[page, cols]`` materializes ``[N, page_stride]`` (multi-GiB at
    capture-padded N). Indexing a 1-D view by ``page * stride + col`` only
    touches the selected bytes. Slot rows are chunked so offset scratch stays
    bounded; iteration count follows the static slot shape for graph capture.
    """
    if slots.numel() == 0:
        return
    page = torch.div(slots, _PAGE_ROWS, rounding_mode="floor").to(torch.int64)
    page_row = torch.remainder(slots, _PAGE_ROWS).to(torch.int64)
    n_rows = slots.shape[0]
    for start in range(0, n_rows, _GATHER_ROW_CHUNK):
        stop = min(start + _GATHER_ROW_CHUNK, n_rows)
        page_c = page[start:stop]
        page_row_c = page_row[start:stop]
        cols_c = cols[start:stop]
        packed_c = packed_cols[start:stop]
        offsets = page_c.unsqueeze(1) * int(page_stride) + cols_c
        flat_off = offsets.reshape(-1)
        if not write:
            packed_c.copy_(flat1d[flat_off].reshape_as(packed_c))
            continue
        taken = flat1d[flat_off].reshape_as(packed_c)
        active = (
            live[start:stop]
            if live is not None
            else torch.ones(stop - start, dtype=torch.bool, device=slots.device)
        )
        payload = torch.where(active.unsqueeze(1), packed_c, taken)
        key = page_c * _PAGE_ROWS + page_row_c
        flat1d[flat_off] = _winning_values(key, active, payload).reshape(-1)


def _write_planar(
    cache: torch.Tensor,
    slots: torch.Tensor,
    packed: torch.Tensor,
    cache_format: str,
    live: torch.Tensor,
) -> None:
    _, _, value_bytes, row_bytes = _layout(cache_format)
    scale_bytes = row_bytes - value_bytes
    flat1d, page_stride = _cache_flat1d(cache, row_bytes)
    _transfer_selected(
        flat1d,
        page_stride,
        slots,
        _column_indexes(slots, value_bytes, 0),
        packed[:, :value_bytes],
        write=True,
        live=live,
    )
    _transfer_selected(
        flat1d,
        page_stride,
        slots,
        _column_indexes(slots, scale_bytes, _PAGE_ROWS * value_bytes),
        packed[:, value_bytes:],
        write=True,
        live=live,
    )


def _read_planar(
    cache: torch.Tensor, slots: torch.Tensor, cache_format: str
) -> torch.Tensor:
    """Gather planar value/scale bytes for ``slots``.

    One fused index-gather for value+scale columns (planar layout is not a
    contiguous row). Do not replace this with whole-pool page gathering that
    materializes ``[N, page_stride]`` (~16 GiB at capture-padded N on Ascend).
    """
    _, _, value_bytes, row_bytes = _layout(cache_format)
    scale_bytes = row_bytes - value_bytes
    flat1d, page_stride = _cache_flat1d(cache, row_bytes)
    packed = cache.new_empty(slots.shape[0], row_bytes)
    # Fuse value+scale into one gather: half the Ascend index launches vs two
    # _transfer_selected reads (gather_global hot path).
    cols = torch.cat(
        (
            _column_indexes(slots, value_bytes, 0),
            _column_indexes(slots, scale_bytes, _PAGE_ROWS * value_bytes),
        ),
        dim=1,
    )
    _transfer_selected(
        flat1d,
        page_stride,
        slots,
        cols,
        packed,
        write=False,
    )
    return packed


def cache_gather_torch(
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_format: str,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Dequantize selected physical slots from a page-planar uint8 field."""
    dim, _, _, _ = _layout(cache_format)
    shape = (*slots.shape, dim)
    dtype = torch.bfloat16 if out is None else out.dtype
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("dequantized output must be BF16 or FP32")
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=cache.device)
    elif out.shape != shape or out.dtype != dtype or out.device != cache.device:
        raise ValueError("out must match gather shape/dtype/device")
    flat_slots = slots.reshape(-1).contiguous()
    if flat_slots.numel() == 0:
        return out
    capacity = cache.shape[0] * _PAGE_ROWS
    capacity_last = max(capacity - 1, 0)
    out_flat = out.view(-1, dim)
    # Chunk so E2M1/E4M3 LUT expands stay bounded at capture-padded slot counts.
    n_rows = flat_slots.shape[0]
    for start in range(0, n_rows, _GATHER_ROW_CHUNK):
        stop = min(start + _GATHER_ROW_CHUNK, n_rows)
        chunk = flat_slots[start:stop]
        valid = (chunk >= 0) & (chunk < capacity)
        packed = _read_planar(cache, chunk.clamp(0, capacity_last), cache_format)
        decoded = _unpack_rows(packed, cache_format, dtype)
        out_flat[start:stop] = decoded.masked_fill(~valid[:, None], 0)
    return out


def cache_scatter_torch(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_format: str,
) -> None:
    """Quantize BF16/FP32 rows into page-planar uint8 slots."""
    dim, _, _, _ = _layout(cache_format)
    if (
        rows.ndim != 2
        or rows.shape[1] != dim
        or rows.dtype
        not in (
            torch.bfloat16,
            torch.float32,
        )
    ):
        raise ValueError(f"rows must be BF16/FP32 [N, {dim}]")
    if slots.shape != (rows.shape[0],) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must be integer [N]")
    if rows.shape[0] == 0:
        return
    capacity = cache.shape[0] * _PAGE_ROWS
    live = (slots >= 0) & (slots < capacity)
    packed = _pack_rows(rows, cache_format)
    _write_planar(cache, slots, packed, cache_format, live)


def cache_unpack_torch(
    packed: torch.Tensor, cache_format: str, out: torch.Tensor | None
) -> torch.Tensor:
    """Decode contiguous uint8 ``[N, row_bytes]`` rows of ``cache_format``."""
    dim, _, _, row_bytes = _layout(cache_format)
    if packed.ndim != 2 or packed.dtype != torch.uint8 or packed.shape[1] != row_bytes:
        raise ValueError("packed must be a uint8 row matrix of the specified format")
    dtype = torch.bfloat16 if out is None else out.dtype
    decoded = _unpack_rows(packed, cache_format, dtype)
    if out is None:
        return decoded
    out.copy_(decoded)
    return out


def cache_pack_torch(
    rows: torch.Tensor, cache_format: str, out: torch.Tensor | None
) -> torch.Tensor:
    """Quantize rows into contiguous uint8 ``[N, row_bytes]`` packed rows."""
    packed = _pack_rows(rows, cache_format)
    if out is None:
        return packed
    out.copy_(packed)
    return out
