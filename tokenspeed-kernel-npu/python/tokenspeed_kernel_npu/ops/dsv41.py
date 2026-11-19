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

"""Ascend SparseFlashMla operators supplied by the matching CANN toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

# Loading the toolkit package registers its PyTorch operators and any required
# JIT extensions. Keep this import in the NPU kernel package so the vendor-neutral
# runtime continues to cross one kernel boundary.
import_module("cann_ops_transformer")


def npu_sparse_flash_mla(q: torch.Tensor, **kwargs: Any):
    """Run SparseFlashMla with the CANN native operator contract."""
    return torch.ops.cann_ops_transformer.sparse_flash_mla(q, **kwargs)


def npu_sparse_flash_mla_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Build the device metadata consumed by SparseFlashMla."""
    return torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        **kwargs,
    )


__all__ = ["npu_sparse_flash_mla", "npu_sparse_flash_mla_metadata"]
