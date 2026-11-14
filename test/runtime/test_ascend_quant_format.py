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

"""Modelslim description lookup and W8A8 linear loading."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_FORMAT_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "quantization"
    / "ascend_format.py"
)


def _format():
    spec = importlib.util.spec_from_file_location("ascend_format_host", _FORMAT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_description_ignores_bookkeeping_and_matches_language_model_names():
    fmt = _format()
    kinds = fmt.tensor_kinds(
        {
            "group_size": 0,
            "metadata": {},
            "quant_method": "ascend",
            "model_quant_type": "W8A8_DYNAMIC",
            "model.language_model.layers.0.mlp.down_proj.weight": "W8A8",
            "model.language_model.embed_tokens.weight": "FLOAT",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {
                "dtype": "W8A8_DYNAMIC"
            },
        }
    )
    assert "quant_method" not in kinds
    assert "model_quant_type" not in kinds
    assert fmt.lookup_kind(kinds, "model.layers.0.mlp.down_proj") == "W8A8"
    assert fmt.is_w8a8("W8A8_DYNAMIC")
    assert not fmt.is_w8a8("FLOAT")
    assert (
        fmt.lookup_kind(kinds, "model.layers.0.linear_attn.in_proj_qkv")
        == "W8A8_DYNAMIC"
    )
    assert fmt.lookup_kind(kinds, "model.language_model.embed_tokens") == "FLOAT"


def test_fused_projection_lookup_follows_checkpoint_shards():
    fmt = _format()
    kinds = fmt.tensor_kinds(
        {
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight": "FLOAT",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight": "FLOAT",
            "model.language_model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.3.q_proj.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.3.k_proj.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.3.v_proj.weight": "W8A8_DYNAMIC",
        }
    )
    assert (
        fmt.lookup_kind(kinds, "model.layers.0.linear_attn.in_proj_qkvz")
        == "W8A8_DYNAMIC"
    )
    assert fmt.lookup_kind(kinds, "model.layers.0.linear_attn.in_proj_ba") == "FLOAT"
    assert fmt.lookup_kind(kinds, "model.layers.0.mlp.gate_up_proj") == "W8A8_DYNAMIC"
    assert fmt.lookup_kind(kinds, "model.layers.3.qkv_proj") == "W8A8_DYNAMIC"


def test_fused_projection_mixed_labels_are_refused():
    fmt = _format()
    kinds = fmt.tensor_kinds(
        {
            "model.layers.0.linear_attn.in_proj_qkv.weight": "W8A8",
            "model.layers.0.linear_attn.in_proj_z.weight": "FLOAT",
        }
    )
    with pytest.raises(ValueError, match="Mixed Ascend quant labels"):
        fmt.lookup_kind(kinds, "model.layers.0.linear_attn.in_proj_qkvz")


def test_v41_shared_expert_lookup_uses_w1_w3_w2():
    """Shared-expert MLP is gate_up_proj/down_proj; modelslim names them w1/w3/w2."""
    fmt = _format()
    kinds = fmt.tensor_kinds(
        {
            "layers.0.ffn.shared_experts.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.shared_experts.w3.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.shared_experts.w2.weight": "W8A8_DYNAMIC",
        }
    )
    assert (
        fmt.lookup_kind(kinds, "model.layers.0.ffn.shared_experts.gate_up_proj")
        == "W8A8_DYNAMIC"
    )
    assert (
        fmt.lookup_kind(kinds, "model.layers.0.ffn.shared_experts.down_proj")
        == "W8A8_DYNAMIC"
    )


def test_v41_fused_attn_lookup_uses_bare_layers_shards():
    """DeepSeek V4.1 modelslim packs name shards as ``layers.*`` without ``model.``."""
    fmt = _format()
    kinds = fmt.tensor_kinds(
        {
            "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wkv.weight": "W8A8_DYNAMIC",
            "layers.2.attn.wkv.weight": "W8A8_DYNAMIC",
            "layers.2.attn.wgate.weight": "W8A8_DYNAMIC",
            "layers.1.engram.wkv.weight": "FLOAT",
        }
    )
    assert (
        fmt.lookup_kind(kinds, "model.layers.0.attn.wq_a_wkv") == "W8A8_DYNAMIC"
    )
    assert (
        fmt.lookup_kind(kinds, "model.layers.2.attn.compressor.wkv_wgate")
        == "W8A8_DYNAMIC"
    )
    assert fmt.lookup_kind(kinds, "model.layers.1.engram.wkv") == "FLOAT"


def test_weights_index_name_follows_the_modelslim_file(tmp_path):
    fmt = _format()
    assert fmt.weights_index_filename(str(tmp_path)) is None
    (tmp_path / fmt.WEIGHTS_INDEX_NAME).write_text("{}", encoding="utf-8")
    assert fmt.weights_index_filename(str(tmp_path)) == fmt.WEIGHTS_INDEX_NAME


def test_w8a8_linear_dequant_matches_the_scale_and_offset():
    pytest.importorskip("tokenspeed_triton")
    from tokenspeed.runtime.layers.linear import ReplicatedLinear
    from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig

    config = AscendQuantConfig.from_config(
        {"model.layers.0.mlp.down_proj.weight": "W8A8"}
    )
    layer = ReplicatedLinear(
        2,
        2,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.0.mlp.down_proj",
    )
    assert layer.weight.dtype == torch.int8
    layer.weight_loader(layer.weight, torch.tensor([[2, 0], [0, 4]], dtype=torch.int8))
    layer.weight_loader(layer.weight_scale, torch.tensor([0.5, 0.25]))
    layer.weight_loader(layer.weight_offset, torch.tensor([1, 0], dtype=torch.int8))
    output = layer(torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16))
    # row 0: (2-1)*0.5=0.5, (0-1)*0.5=-0.5; row 1: 0, 4*0.25=1
    # [1, 1] dotted with those rows is 0 and 1
    torch.testing.assert_close(output.float(), torch.tensor([[0.0, 1.0]]))


def test_fused_in_proj_qkvz_selects_w8a8_from_shard_description():
    pytest.importorskip("tokenspeed_triton")
    from tokenspeed.runtime.layers.linear import MergedColumnParallelLinear
    from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig

    config = AscendQuantConfig.from_config(
        {
            "model.layers.0.linear_attn.in_proj_qkv.weight": "W8A8_DYNAMIC",
            "model.layers.0.linear_attn.in_proj_z.weight": "W8A8_DYNAMIC",
        }
    )
    assert config.is_quantized_layer("model.layers.0.linear_attn.in_proj_qkvz")
    layer = MergedColumnParallelLinear(
        4,
        [2, 2, 2, 2],
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        tp_rank=0,
        tp_size=1,
        prefix="model.layers.0.linear_attn.in_proj_qkvz",
    )
    assert layer.weight.dtype == torch.int8
    assert hasattr(layer, "weight_scale")
    assert hasattr(layer, "weight_offset")


def test_float_entry_stays_an_unquantized_linear():
    pytest.importorskip("tokenspeed_triton")
    from tokenspeed.runtime.layers.linear import ReplicatedLinear
    from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig

    config = AscendQuantConfig.from_config(
        {"model.layers.0.mlp.down_proj.weight": "FLOAT"}
    )
    layer = ReplicatedLinear(
        2,
        2,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.0.mlp.down_proj",
    )
    assert layer.weight.dtype == torch.bfloat16


def test_unsupported_ascend_quant_type_is_refused():
    pytest.importorskip("tokenspeed_triton")
    from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig

    config = AscendQuantConfig.from_config(
        {"model.layers.0.mlp.down_proj.weight": "W4A8"}
    )
    with pytest.raises(ValueError, match="W4A8"):
        config.is_quantized_layer("model.layers.0.mlp.down_proj")


def test_channel_loader_matches_row_and_merged_weight_loader_arity():
    """RowParallel takes (param, weight); merged columns also take shard_id."""
    from tokenspeed.runtime.layers.quantization.ascend import _channel_loader

    two_arg_calls: list[tuple[torch.Tensor, tuple[int, ...]]] = []

    def two_arg(param, loaded_weight):
        two_arg_calls.append((param, tuple(loaded_weight.shape)))

    load = _channel_loader(two_arg)
    param = torch.nn.Parameter(torch.zeros(2, 1))
    load(param, torch.tensor([1.0, 2.0]))
    assert two_arg_calls == [(param, (2, 1))]

    three_arg_calls: list[object] = []

    def three_arg(param, loaded_weight, shard_id):
        three_arg_calls.append(shard_id)

    load = _channel_loader(three_arg)
    load(param, torch.tensor([1.0, 2.0]), shard_id="down")
    assert three_arg_calls == ["down"]


def test_v41_dense_quant_passes_through_ascend_w8a8():
    pytest.importorskip("tokenspeed_triton")
    from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig
    from tokenspeed.runtime.models.deepseek_v41 import v41_dense_quant

    config = AscendQuantConfig.from_config(
        {"model.layers.0.attn.wq_b.weight": "W8A8"}
    )
    assert v41_dense_quant(config) is config
    assert v41_dense_quant(None) is None


def test_ascend_quant_moe_weight_dtype_is_int8():
    from tokenspeed.runtime.layers.quantization.ascend import AscendQuantConfig

    config = AscendQuantConfig.from_config({})
    assert config.moe_weight_dtype() == "int8"
    assert config.moe_weight_dtype("model.layers.0.ffn") == "int8"
