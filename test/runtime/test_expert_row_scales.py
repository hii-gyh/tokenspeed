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

"""Host checks for MoE scale expansion that must not read counts back."""

from __future__ import annotations

import types
from pathlib import Path

import torch

_MOE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel-npu"
    / "python"
    / "tokenspeed_kernel_npu"
    / "ops"
    / "moe.py"
)
_source = _MOE_PATH.read_text(encoding="utf-8").replace(
    "import torch_npu\n", "torch_npu = None  # host test\n", 1
)
_moe = types.ModuleType("moe_row_scales_host")
_moe.__file__ = str(_MOE_PATH)
exec(compile(_source, str(_MOE_PATH), "exec"), _moe.__dict__)


def test_expert_row_scales_match_the_packed_prefix():
    weight_scale = torch.arange(4 * 6, dtype=torch.float32).view(4, 6)
    counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    packed = _moe._expand_expert_scales(weight_scale, counts)
    full = _moe.expert_row_scales(weight_scale, counts, rows=10)
    assert full.shape == (10, 6)
    assert torch.equal(full[:6], packed)
    assert torch.equal(full[6:], torch.zeros(4, 6))


def test_expert_row_scales_empty_rows():
    weight_scale = torch.ones(3, 4)
    counts = torch.zeros(3, dtype=torch.int64)
    assert _moe.expert_row_scales(weight_scale, counts, rows=0).shape == (0, 4)
    zeros = _moe.expert_row_scales(weight_scale, counts, rows=5)
    assert torch.equal(zeros, torch.zeros(5, 4))
