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

"""The fused mHC Sinkhorn loop must match the torch loop it replaces."""

import pytest
import torch
from tokenspeed_kernel.ops.residual import triton as residual
from utils import assert_no_triton_compile

DEVICE = "cuda" if torch.cuda.is_available() else "npu"


def _reference(comb: torch.Tensor, hc_eps: float, iters: int) -> torch.Tensor:
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return comb


@pytest.mark.parametrize("tokens", [1, 4, 17, 257])
@pytest.mark.parametrize("iters", [1, 3, 20])
def test_sinkhorn_matches_torch_loop(tokens, iters):
    torch.manual_seed(7)
    hc_eps = 1e-6
    comb = (
        torch.randn(tokens, 4, 4, device=DEVICE, dtype=torch.float32).softmax(-1)
        + hc_eps
    )
    expected = _reference(comb, hc_eps, iters)
    residual.mhc_sinkhorn_hc4(comb, hc_eps=hc_eps, sinkhorn_iters=iters)
    torch.testing.assert_close(comb, expected, rtol=1e-5, atol=1e-6)


def test_sinkhorn_token_count_does_not_recompile():
    hc_eps, iters = 1e-6, 20

    def run(tokens):
        comb = (
            torch.randn(tokens, 4, 4, device=DEVICE, dtype=torch.float32).softmax(-1)
            + hc_eps
        )
        expected = _reference(comb, hc_eps, iters)
        residual.mhc_sinkhorn_hc4(comb, hc_eps=hc_eps, sinkhorn_iters=iters)
        torch.testing.assert_close(comb, expected, rtol=1e-5, atol=1e-6)

    torch.manual_seed(11)
    for tokens in (16, 17):
        run(tokens)
    with assert_no_triton_compile(residual._mhc_sinkhorn_hc4_kernel):
        for tokens in (1, 4, 8, 32, 33, 257):
            run(tokens)
