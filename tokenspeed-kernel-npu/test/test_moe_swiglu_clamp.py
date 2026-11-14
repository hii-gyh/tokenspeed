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

from __future__ import annotations

import pytest
import torch
import torch_npu

from tokenspeed_kernel_npu.ops.moe import (
    _ascend_int8_dequant_swiglu_quant,
    _expand_expert_scales,
)


@pytest.fixture(autouse=True, scope="module")
def _requires_npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)


def test_expand_expert_scales_skips_empty_experts() -> None:
    weight_scale = torch.arange(4 * 6, dtype=torch.float32).view(4, 6)
    counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    out = _expand_expert_scales(weight_scale, counts)
    assert out.shape == (6, 6)
    assert torch.equal(out[0:2], weight_scale[0].expand(2, -1))
    assert torch.equal(out[2:5], weight_scale[2].expand(3, -1))
    assert torch.equal(out[5], weight_scale[3])


def test_swiglu_clamp_accepts_capacity_padded_token_buffer() -> None:
    """MC2 dispatch capacity can exceed sum(expert_counts); clamp must not broadcast."""
    device = torch.device("npu:0")
    experts, channels, capacity, live = 4, 32, 16, 7
    hidden = torch.randint(
        -32, 32, (capacity, channels), dtype=torch.int32, device=device
    )
    weight_scale = torch.full(
        (experts, channels), 0.02, dtype=torch.float32, device=device
    )
    activation_scale = torch.ones(capacity, dtype=torch.float32, device=device)
    counts = torch.tensor([2, 0, 3, 2], dtype=torch.int64, device=device)
    assert int(counts.sum().item()) == live

    values, scales = _ascend_int8_dequant_swiglu_quant(
        hidden,
        weight_scale,
        activation_scale,
        counts,
        quant_scale=None,
        limit=10.0,
    )
    torch.npu.synchronize()
    assert values.shape == (capacity, channels // 2)
    assert scales.shape == (capacity,)
    assert values.dtype == torch.int8
    assert torch.isfinite(scales[:live]).all()
