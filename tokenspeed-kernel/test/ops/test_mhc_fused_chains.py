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

"""The fused mHC collapse/combine and engram gate must match their torch forms."""

import pytest
import torch
from tokenspeed_kernel.ops.residual import triton as residual
from utils import assert_no_triton_compile

DEVICE = "cuda" if torch.cuda.is_available() else "npu"
HIDDEN = 5120


def _streams(tokens: int, hidden: int = HIDDEN) -> torch.Tensor:
    return torch.randn(tokens, 4, hidden, device=DEVICE, dtype=torch.bfloat16)


def _collapse_reference(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def _combine_reference(
    x: torch.Tensor,
    residual_streams: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    mixed = (comb.unsqueeze(-1) * residual_streams.float().unsqueeze(-2)).sum(-3)
    return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)


def _gate_reference(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    weight = query_weight * key_weight
    rstd = torch.rsqrt(hidden.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dim = hidden.shape[-1]
    dot = (hidden * weight * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(dot < 0, -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 4, 17, 257])
def test_collapse_pre_matches_torch(tokens):
    torch.manual_seed(3)
    x = _streams(tokens)
    pre = torch.rand(tokens, 4, device=DEVICE, dtype=torch.float32)
    expected = _collapse_reference(x, pre)
    actual = residual.mhc_collapse_pre_hc4(x, pre)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tokens", [1, 4, 17, 257])
def test_combine_post_matches_torch(tokens):
    torch.manual_seed(5)
    streams = _streams(tokens)
    x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    post = torch.rand(tokens, 4, device=DEVICE, dtype=torch.float32)
    comb = torch.randn(tokens, 4, 4, device=DEVICE, dtype=torch.float32).softmax(-1)
    expected = _combine_reference(x, streams, post, comb)
    actual = residual.mhc_combine_post_hc4(x, streams, post, comb)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_collapse_pre_handles_a_hidden_size_that_is_not_a_block_multiple():
    torch.manual_seed(13)
    hidden = 1025
    x = _streams(3, hidden)
    pre = torch.rand(3, 4, device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(
        residual.mhc_collapse_pre_hc4(x, pre),
        _collapse_reference(x, pre),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("tokens", [1, 4, 17])
def test_engram_gate_matches_torch(tokens):
    torch.manual_seed(17)
    hidden = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    key = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    value = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.float32)
    query_weight = torch.randn(4, HIDDEN, device=DEVICE, dtype=torch.float32)
    key_weight = torch.randn(4, HIDDEN, device=DEVICE, dtype=torch.float32)
    token_mask = torch.rand(tokens, device=DEVICE) > 0.3
    eps = 1e-6
    expected = _gate_reference(
        hidden, key, value, query_weight, key_weight, token_mask, eps
    )
    actual = residual.engram_gate_hc4(
        hidden, hidden, key, value, query_weight, key_weight, token_mask, eps
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_engram_gate_zeroes_masked_tokens():
    torch.manual_seed(19)
    tokens = 4
    hidden = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    key = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    value = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(4, HIDDEN, device=DEVICE, dtype=torch.float32)
    token_mask = torch.tensor([True, False, True, False], device=DEVICE)
    actual = residual.engram_gate_hc4(
        hidden, hidden, key, value, weight, weight, token_mask, 1e-6
    )
    torch.testing.assert_close(
        actual[~token_mask], hidden[~token_mask].to(torch.bfloat16)
    )


def test_engram_gate_accepts_an_undone_rotation():
    """The gate normalizes ``original`` but adds to ``hidden``; they may differ."""
    torch.manual_seed(23)
    tokens = 4
    hidden = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    original = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    key = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
    value = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(4, HIDDEN, device=DEVICE, dtype=torch.float32)
    token_mask = torch.ones(tokens, dtype=torch.bool, device=DEVICE)
    eps = 1e-6
    weight_pair = weight * weight
    rstd = torch.rsqrt(original.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (original * weight_pair * key).sum(-1) * rstd * HIDDEN**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(dot < 0, -magnitude, magnitude))
    expected = (hidden + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(torch.bfloat16)
    actual = residual.engram_gate_hc4(
        hidden, original, key, value, weight, weight, token_mask, eps
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_token_count_does_not_recompile():
    torch.manual_seed(29)

    def run(tokens):
        streams = _streams(tokens)
        x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        pre = torch.rand(tokens, 4, device=DEVICE, dtype=torch.float32)
        comb = torch.randn(tokens, 4, 4, device=DEVICE, dtype=torch.float32).softmax(-1)
        torch.testing.assert_close(
            residual.mhc_collapse_pre_hc4(streams, pre),
            _collapse_reference(streams, pre),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            residual.mhc_combine_post_hc4(x, streams, pre, comb),
            _combine_reference(x, streams, pre, comb),
            rtol=1e-2,
            atol=1e-2,
        )
        hidden = streams.float()
        key = torch.randn(tokens, 4, HIDDEN, device=DEVICE, dtype=torch.float32)
        value = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(4, HIDDEN, device=DEVICE, dtype=torch.float32)
        mask = torch.ones(tokens, dtype=torch.bool, device=DEVICE)
        torch.testing.assert_close(
            residual.engram_gate_hc4(
                hidden, hidden, key, value, weight, weight, mask, 1e-6
            ),
            _gate_reference(hidden, key, value, weight, weight, mask, 1e-6),
            rtol=2e-2,
            atol=2e-2,
        )

    for tokens in (16, 17):
        run(tokens)
    with assert_no_triton_compile(residual._mhc_collapse_pre_hc4_kernel):
        with assert_no_triton_compile(residual._mhc_combine_post_hc4_kernel):
            with assert_no_triton_compile(residual._engram_gate_hc4_kernel):
                for tokens in (1, 4, 8, 32, 33, 257):
                    run(tokens)
