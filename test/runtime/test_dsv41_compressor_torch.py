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

"""Host tests for the Ascend/torch V4.1 ratio-2 compressor pool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_PATH = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel"
    / "python"
    / "tokenspeed_kernel"
    / "ops"
    / "attention"
    / "dsv41"
    / "compressor_torch.py"
)
_spec = importlib.util.spec_from_file_location("dsv41_compressor_torch_host", _PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def test_pool_reads_odd_prefix_from_tail_then_scatters():
    # One completed pair: previous=-1 reads tail slot 2 (page 1, row 0).
    content = torch.zeros(2, 512, dtype=torch.float32)
    scores = torch.zeros(2, 512, dtype=torch.float32)
    content[1] = 4.0
    # Match the tail score so log-sum-exp weights are equal.
    scores[1] = 0.0
    tail = torch.zeros(4, 2, 2, 512, dtype=torch.float32)
    tail[1, 0, 0] = 2.0
    tail[1, 0, 1] = 0.0
    previous = torch.tensor([-1, -1], dtype=torch.int64)
    slots = torch.tensor([2, 2], dtype=torch.int64)
    active = torch.tensor([False, True])
    out = _mod.compressor_pool_torch(
        content, scores, previous, tail, slots, active, None, None, 1e-6
    )
    assert out[0].abs().sum() == 0
    # equal scores → 0.5/0.5 mix of 2 and 4
    assert torch.allclose(out[1], torch.full((512,), 3.0))

    write = torch.tensor([-1, 3], dtype=torch.int64)
    _mod.compressor_tail_scatter_torch(content, scores, tail, write)
    assert torch.equal(tail[1, 1, 0], content[1])
    assert torch.equal(tail[1, 1, 1], scores[1])


def test_tail_scatter_invalid_slot_keeps_an_earlier_write():
    content = torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512)
    scores = content + 1
    tail = torch.full((2, 2, 2, 512), 7.0)
    # -1 clamps onto slot 0, which the first row has just written.
    _mod.compressor_tail_scatter_torch(
        content, scores, tail, torch.tensor([0, -1], dtype=torch.int64)
    )
    assert torch.equal(tail[0, 0, 0], content[0])
    assert torch.equal(tail[0, 0, 1], scores[0])
    assert torch.equal(tail[0, 1], torch.full((2, 512), 7.0))


def test_tail_scatter_all_invalid_leaves_the_tail():
    content = torch.ones(2, 512)
    scores = torch.ones(2, 512)
    tail = torch.full((1, 2, 2, 512), 4.0)
    before = tail.clone()
    _mod.compressor_tail_scatter_torch(
        content, scores, tail, torch.tensor([-1, -2], dtype=torch.int64)
    )
    assert torch.equal(tail, before)


def test_pool_with_norm_rounds_to_bf16_before_rmsnorm():
    content = torch.randn(1, 512, dtype=torch.float32)
    scores = torch.zeros(1, 512, dtype=torch.float32)
    previous = torch.tensor([0], dtype=torch.int64)
    # previous=0 is in-current: pools content[0] with itself → identity before norm
    slots = torch.tensor([-1], dtype=torch.int64)
    active = torch.tensor([True])
    tail = torch.zeros(2, 2, 2, 512, dtype=torch.float32)
    weight = torch.ones(512, dtype=torch.bfloat16)
    out = _mod.compressor_pool_torch(
        content, scores, previous, tail, slots, active, None, weight, 1e-6
    )
    assert out.dtype == torch.bfloat16
    bf16 = content.to(torch.bfloat16).float()
    expected = (
        bf16 * torch.rsqrt(bf16.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    ).to(torch.bfloat16)
    assert torch.equal(out, expected)
