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

"""Ascend-native byte scatter for the FlatKV torch codec."""

import torch


def scatter_nd_update_(
    flat1d: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """Update a flat uint8 tensor at int64 indices without allocating its output.

    CANN's ND scatter supports NPU Graph replay on large external cache tensors.
    The int8 views only reinterpret the bytes; they do not copy or change values.
    CANN 9.2.1 can silently corrupt ND updates larger than 4096 bytes. Keep
    each launch within that bound; decode-sized writes still use one launch.
    """
    if flat1d.dtype != torch.uint8 or flat1d.ndim != 1:
        raise ValueError("flat1d must be a one-dimensional uint8 tensor")
    if indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("indices must be a one-dimensional int64 tensor")
    if updates.dtype != torch.uint8 or updates.shape != indices.shape:
        raise ValueError("updates must be uint8 with the same shape as indices")
    for start in range(0, indices.numel(), 4096):
        end = min(start + 4096, indices.numel())
        torch.ops.npu.npu_scatter_nd_update_(
            flat1d.view(torch.int8),
            indices[start:end].unsqueeze(1),
            updates[start:end].view(torch.int8),
        )


def scatter_bf16_nd_update_(
    cache: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """Update BF16 ``[page,row]`` rows without expanding byte-column indices.

    This three-dimensional BF16 operator path is distinct from the flat uint8
    path above. Real CANN 9.2.1 probes match advanced-index writes through 2048
    rows, so the 4096-byte flat-payload workaround does not apply here.
    """
    if cache.dtype != torch.bfloat16 or cache.ndim != 3 or cache.shape[1:] != (64, 512):
        raise ValueError("cache must be BF16 [pages,64,512]")
    if indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype != torch.int64:
        raise ValueError("indices must be int64 [N,2]")
    if updates.dtype != torch.bfloat16 or updates.shape != (indices.shape[0], 512):
        raise ValueError("updates must be BF16 [N,512]")
    torch.ops.npu.npu_scatter_nd_update_(cache, indices, updates)


def scatter_compressor_tail_(
    tail: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """Write FP32 ``[N,512]`` rows at ``[N,3]`` (page, row, half) coordinates.

    Advanced-index assignment on the tail's strided arena view makes CANN copy
    the whole underlying arena per call (``IndexPutImpl``'s ``ViewCopy``), which
    costs milliseconds regardless of how few rows are written. The vendor ND
    scatter addresses the same rows in place.
    """
    if tail.dtype != torch.float32 or tail.ndim != 4 or tail.shape[1:] != (2, 2, 512):
        raise ValueError("tail must be FP32 [pages,2,2,512]")
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype != torch.int64:
        raise ValueError("indices must be int64 [N,3]")
    if updates.dtype != torch.float32 or updates.shape != (indices.shape[0], 512):
        raise ValueError("updates must be FP32 [N,512]")
    torch.ops.npu.npu_scatter_nd_update_(tail, indices, updates)


def scatter_bf16_paged_(
    cache: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Write BF16 rows into a paged field with one CANN paged-cache scatter.

    ``npu_scatter_pa_kv_cache`` skips a slot of -1 itself, so the caller neither
    reads the old row back nor builds byte-column indexes. It applies duplicate
    slots in an unspecified order, so the caller resolves them first. Slots at
    or past the field's capacity must arrive as -1: the kernel writes them
    literally and would fault the device.

    The cache is passed as both key and value destination. MLA keeps one latent
    field, and writing the same rows twice into it is idempotent.
    """
    if cache.dtype != torch.bfloat16 or cache.ndim != 3 or cache.shape[1:] != (64, 512):
        raise ValueError("cache must be BF16 [pages,64,512]")
    if rows.dtype != torch.bfloat16 or rows.ndim != 2 or rows.shape[1] != 512:
        raise ValueError("rows must be BF16 [N,512]")
    if slots.ndim != 1 or slots.shape[0] != rows.shape[0]:
        raise ValueError("slots must be [N]")
    paged = cache.unsqueeze(2)
    payload = rows.unsqueeze(1)
    torch.ops.npu.npu_scatter_pa_kv_cache(
        payload,
        payload,
        paged,
        paged,
        slots.to(torch.int32),
        cache_mode="Norm",
    )
