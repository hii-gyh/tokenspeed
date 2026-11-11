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

"""Trait tables for Ascend MoE registration.

The registration in ``ascend.py`` runs only when an NPU is visible, because
it imports ``tokenspeed_kernel_npu``. These tables are the selection contract
and stay importable on a host that has no NPU.
"""

from __future__ import annotations

ASCEND_ALL2ALL_BACKEND = "ascend"


def distributed_bf16_apply_traits() -> dict[str, frozenset]:
    """Traits for ``ascend_bf16_distributed_precomputed_moe_apply``.

    ``a2a_backend`` is declared so a DeepEP or Petit plan cannot select this
    kernel. An undeclared trait would match every backend.
    """
    return {
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({True}),
        "ispp_alignment": frozenset({32}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
        "a2a_backend": frozenset({ASCEND_ALL2ALL_BACKEND}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False, True}),
    }


def distributed_int8_apply_traits() -> dict[str, frozenset]:
    """Traits for ``ascend_int8_distributed_precomputed_moe_apply``."""
    return {
        "weight_dtype": frozenset({"int8"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({True}),
        "ispp_alignment": frozenset({32}),
        "internal_activation_dtype": frozenset({"int8"}),
        "supports_bias": frozenset({False}),
        "a2a_backend": frozenset({ASCEND_ALL2ALL_BACKEND}),
        "swiglu_form": frozenset({"standard"}),
        # True: DeepSeek V4 ``swiglu_limit`` uses an explicit clamp before
        # requant; False: fused ``npu_dequant_swiglu_quant`` without a limit.
        "activation_clamped": frozenset({False, True}),
    }
