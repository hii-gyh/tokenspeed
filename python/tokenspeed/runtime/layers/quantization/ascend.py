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

"""Ascend modelslim W8A8, the format ``--quantization ascend`` selects.

``FLOAT`` entries stay on the unquantized linear path. ``W8A8`` entries store
int8 weights with a per-output-channel scale and zero-point, matching the
tensors vLLM-Ascend reads from ``quant_model_description.json``.
"""

from __future__ import annotations

from typing import Any

import torch

from tokenspeed.runtime.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from tokenspeed.runtime.layers.quantization.ascend_format import (
    DESCRIPTION_FILENAME,
    is_w8a8,
    lookup_kind,
    tensor_kinds,
)
from tokenspeed.runtime.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)


def _as_channel_column(loaded_weight: torch.Tensor) -> torch.Tensor:
    """Per-output scale or offset as ``[N, 1]``."""
    if loaded_weight.ndim == 0:
        return loaded_weight.reshape(1, 1)
    if loaded_weight.ndim == 1:
        return loaded_weight.unsqueeze(-1)
    if loaded_weight.ndim == 2 and loaded_weight.shape[-1] != 1:
        return loaded_weight.reshape(-1, 1)
    return loaded_weight


class AscendW8A8LinearMethod(LinearMethodBase):
    """Int8 weight, per-channel scale and zero-point."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader=None,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, 1, dtype=torch.float32),
            output_dim=0,
            weight_loader=_channel_loader(weight_loader),
        )
        offset = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, 1, dtype=torch.int32),
            output_dim=0,
            weight_loader=_channel_loader(weight_loader),
        )
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", scale)
        layer.register_parameter("weight_offset", offset)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from tokenspeed_kernel.platform import current_platform

        if not current_platform().is_npu:
            return
        if layer.weight_offset is not None and bool(
            torch.any(layer.weight_offset != 0).item()
        ):
            return
        from tokenspeed_kernel.ops.gemm.int8 import prepare_int8_linear

        layer._int8_plan = prepare_int8_linear(
            layer.weight,
            layer.weight_scale,
            output_dtype=layer.params_dtype,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        plan = getattr(layer, "_int8_plan", None)
        if plan is not None:
            return plan(x, bias=bias)
        scale = layer.weight_scale.to(dtype=torch.float32)
        offset = layer.weight_offset.to(dtype=torch.float32)
        weight = (layer.weight.to(dtype=torch.float32) - offset) * scale
        return torch.nn.functional.linear(x, weight.to(dtype=x.dtype), bias)


def _channel_loader(weight_loader):
    """Adapt a linear ``weight_loader`` for per-channel scale/offset tensors.

    ``RowParallelLinear.weight_loader`` takes only ``(param, loaded_weight)``.
    Merged column loaders also accept ``shard_id`` / ``begin_size``. Pass the
    optional slots only when the caller supplied them.
    """

    def load(param, loaded_weight, shard_id=None, begin_size=None):
        loaded_weight = _as_channel_column(loaded_weight).to(dtype=param.dtype)
        if begin_size is not None:
            weight_loader(param, loaded_weight, shard_id, begin_size)
        elif shard_id is not None:
            weight_loader(param, loaded_weight, shard_id)
        else:
            weight_loader(param, loaded_weight)

    return load


class AscendQuantConfig(QuantizationConfig):
    """Per-tensor modelslim labels for one checkpoint."""

    def __init__(self, kinds: dict[str, str]) -> None:
        super().__init__()
        self.kinds = kinds

    def get_name(self) -> str:
        return "ascend"

    def moe_weight_dtype(self, prefix: str = "") -> str:
        """Modelslim W8A8 experts select the Ascend INT8 MoE kernels."""
        del prefix
        return "int8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [DESCRIPTION_FILENAME]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AscendQuantConfig:
        return cls(tensor_kinds(config))

    def get_scaled_act_names(self) -> list[str]:
        return []

    def is_quantized_layer(self, name: str) -> bool:
        """Whether ``name`` is a W8A8 checkpoint module.

        ``FLOAT`` and a missing entry are unquantized. Any other label is
        refused so a W4 pack is not loaded as dense BF16.
        """
        kind = lookup_kind(self.kinds, name)
        if kind is None or kind == "FLOAT":
            return False
        if is_w8a8(kind):
            return True
        raise ValueError(
            f"Ascend quant type {kind!r} for {name} is not supported; "
            "expected FLOAT or W8A8"
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod

        if self.is_quantized_layer(prefix):
            return AscendW8A8LinearMethod()
        return UnquantizedLinearMethod()
