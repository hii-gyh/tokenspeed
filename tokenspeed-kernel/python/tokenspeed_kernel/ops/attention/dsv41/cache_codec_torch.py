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
_PAGE_ROWS = 64


def _layout(cache_format: str):
    if cache_format not in _LAYOUTS:
        raise ValueError(
            f"Ascend torch codec supports {sorted(_LAYOUTS)}, got {cache_format!r}"
        )
    return _LAYOUTS[cache_format]


def _flat_pages(cache: torch.Tensor, row_bytes: int) -> torch.Tensor:
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
    return cache.as_strided(
        (cache.shape[0], _PAGE_ROWS * row_bytes), (cache.stride(0), 1)
    )


def _e2m1_decode(nibbles: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(_E2M1_LEVELS, device=nibbles.device, dtype=torch.float32)
    mag = levels[(nibbles & 7).long()]
    return torch.where((nibbles & 8) != 0, -mag, mag)


def _e2m1_encode(values: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(_E2M1_LEVELS, device=values.device, dtype=torch.float32)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=values.device)
    distance = (values.abs().unsqueeze(-1) - levels[order]).abs()
    code = order[distance.argmin(dim=-1)]
    negative = torch.signbit(values)
    return (code | (negative.int() << 3)).to(torch.uint8)


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


def _float_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Encode finite floats as E4M3FN bytes. CUDA uses the float8 cast."""
    if values.is_cuda:
        return values.to(torch.float8_e4m3fn).view(torch.uint8)
    finite = torch.arange(256, device=values.device, dtype=torch.int32)
    finite = finite[(finite & 0x7F) != 0x7F]
    table = _e4m3_to_float(finite.to(torch.uint8))
    distance = (values.float().unsqueeze(-1) - table).abs()
    # Exact midpoints round to the even trailing mantissa bit, matching torch.
    odd = ((finite & 1) != 0).to(torch.float32)
    nearest = (distance + odd * 1e-6).argmin(dim=-1)
    return finite[nearest].to(torch.uint8)


def _pack_rows(rows: torch.Tensor, cache_format: str) -> torch.Tensor:
    """Return contiguous uint8 [N, row_bytes] = values then scale bytes."""
    dim, group, value_bytes, row_bytes = _layout(cache_format)
    x = rows.float().reshape(-1, dim // group, group)
    amax = x.abs().amax(dim=-1)
    if cache_format == "global":
        scale = _float_to_e4m3(amax.clamp_min(6 * 2.0**-9) / 6)
        scale_bytes = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
        scale_f = scale.float()
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
    packed = torch.empty(rows.shape[0], row_bytes, dtype=torch.uint8, device=rows.device)
    packed[:, :value_bytes] = values
    packed[:, value_bytes:] = scale_bytes.flatten(1)
    return packed


def _unpack_rows(packed: torch.Tensor, cache_format: str, dtype: torch.dtype) -> torch.Tensor:
    dim, group, value_bytes, row_bytes = _layout(cache_format)
    if packed.shape[-1] != row_bytes:
        raise ValueError(f"packed rows must be {row_bytes} bytes")
    values = packed[..., :value_bytes]
    scale_u8 = packed[..., value_bytes:]
    if cache_format == "global":
        scale = _e4m3_to_float(scale_u8)
    else:
        scale = torch.where(
            scale_u8 == 0,
            torch.full((), 2.0**-127, device=packed.device),
            (scale_u8.to(torch.int32) << 23).view(torch.float32),
        )
    scale = scale.reshape(*packed.shape[:-1], dim // group)
    if cache_format == "swa":
        decoded = _e4m3_to_float(values).reshape(*scale.shape, group)
    else:
        lo = _e2m1_decode(values & 15)
        hi = _e2m1_decode((values >> 4) & 15)
        decoded = torch.stack((lo, hi), dim=-1).reshape(*scale.shape, group)
    return (decoded * scale.unsqueeze(-1)).reshape(*packed.shape[:-1], dim).to(dtype)


def _write_planar(cache: torch.Tensor, slots: torch.Tensor, packed: torch.Tensor, cache_format: str) -> None:
    _, _, value_bytes, row_bytes = _layout(cache_format)
    scale_bytes = row_bytes - value_bytes
    flat = _flat_pages(cache, row_bytes)
    page = torch.div(slots, _PAGE_ROWS, rounding_mode="floor")[:, None]
    page_row = (slots % _PAGE_ROWS)[:, None]
    value_cols = page_row * value_bytes + torch.arange(value_bytes, device=cache.device)
    scale_cols = (
        _PAGE_ROWS * value_bytes
        + page_row * scale_bytes
        + torch.arange(scale_bytes, device=cache.device)
    )
    flat[page.expand_as(value_cols), value_cols] = packed[:, :value_bytes]
    flat[page.expand_as(scale_cols), scale_cols] = packed[:, value_bytes:]


def _read_planar(cache: torch.Tensor, slots: torch.Tensor, cache_format: str) -> torch.Tensor:
    _, _, value_bytes, row_bytes = _layout(cache_format)
    scale_bytes = row_bytes - value_bytes
    flat = _flat_pages(cache, row_bytes)
    page = torch.div(slots, _PAGE_ROWS, rounding_mode="floor")[:, None]
    page_row = (slots % _PAGE_ROWS)[:, None]
    value_cols = page_row * value_bytes + torch.arange(value_bytes, device=cache.device)
    scale_cols = (
        _PAGE_ROWS * value_bytes
        + page_row * scale_bytes
        + torch.arange(scale_bytes, device=cache.device)
    )
    packed = cache.new_empty(slots.shape[0], row_bytes)
    packed[:, :value_bytes] = flat[page.expand_as(value_cols), value_cols]
    packed[:, value_bytes:] = flat[page.expand_as(scale_cols), scale_cols]
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
    valid = (flat_slots >= 0) & (flat_slots < capacity)
    safe = flat_slots.clamp(0, max(capacity - 1, 0))
    packed = _read_planar(cache, safe, cache_format)
    decoded = _unpack_rows(packed, cache_format, dtype)
    decoded = decoded.masked_fill(~valid[:, None], 0)
    out.view(-1, dim).copy_(decoded)
    return out


def cache_scatter_torch(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_format: str,
) -> None:
    """Quantize BF16/FP32 rows into page-planar uint8 slots."""
    dim, _, _, _ = _layout(cache_format)
    if rows.ndim != 2 or rows.shape[1] != dim or rows.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(f"rows must be BF16/FP32 [N, {dim}]")
    if slots.shape != (rows.shape[0],) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must be integer [N]")
    capacity = cache.shape[0] * _PAGE_ROWS
    live = (slots >= 0) & (slots < capacity)
    if not bool(live.any()):
        return
    packed = _pack_rows(rows[live], cache_format)
    _write_planar(cache, slots[live], packed, cache_format)


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
