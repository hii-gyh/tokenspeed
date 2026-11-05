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

"""MindSpeed / modelslim checkpoint layout used by vLLM-Ascend.

The pack is ``quant_model_description.json`` plus
``quant_model_weights*.safetensors``. Each description entry names one
checkpoint tensor as ``FLOAT`` or a ``W8A8`` variant. Token names in the
description use ``model.language_model.``; runtime modules drop that segment.

Runtime fused linears (``qkv_proj``, ``gate_up_proj``, ``in_proj_qkvz``, …)
are not named in the description; their shard members are. Lookups for a
fused module must resolve through those shards.
"""

from __future__ import annotations

import os

DESCRIPTION_FILENAME = "quant_model_description.json"
WEIGHTS_INDEX_NAME = "quant_model_weights.safetensors.index.json"

_NON_TENSOR_KEYS = frozenset(
    {
        "group_size",
        "metadata",
        "version",
        "quant_type",
        "model_quant_type",
        "quant_method",
    }
)

# Runtime fused leaf -> checkpoint shard groups. The first group with any
# description hit wins; members that agree must share one quant label.
# DeepSeek V4.1 shared experts name the gate/up pair w1/w3 and down w2.
_FUSED_PROJECTION_SHARDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "qkv_proj": (("q_proj", "k_proj", "v_proj"),),
    "gate_up_proj": (("gate_proj", "up_proj"), ("w1", "w3")),
    "down_proj": (("w2",),),
    "in_proj_qkvz": (("in_proj_qkv", "in_proj_z"),),
    "in_proj_ba": (("in_proj_b", "in_proj_a"),),
    "in_proj_qkvzba": (("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),),
    "wq_a_wkv": (("wq_a", "wkv"),),
    "wkv_wgate": (("wkv", "wgate"),),
}


def weights_index_filename(folder: str) -> str | None:
    """Return the modelslim index name when that file is in ``folder``."""
    if os.path.isfile(os.path.join(folder, WEIGHTS_INDEX_NAME)):
        return WEIGHTS_INDEX_NAME
    return None


def tensor_kinds(description: dict) -> dict[str, str]:
    """Tensor name to quant label. Non-tensor bookkeeping keys are dropped."""
    kinds: dict[str, str] = {}
    for key, value in description.items():
        if key in _NON_TENSOR_KEYS:
            continue
        if isinstance(value, str):
            kinds[key] = value
            continue
        if isinstance(value, dict):
            label = value.get("dtype") or value.get("quant_type")
            if isinstance(label, str):
                kinds[key] = label
    return kinds


def weight_name_variants(name: str) -> tuple[str, ...]:
    """Checkpoint weight names that can refer to runtime module ``name``."""
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    weight = f"{base}.weight"
    names = [weight]
    if ".language_model." in weight:
        names.append(weight.replace(".language_model.", ".", 1))
    if weight.startswith("model."):
        # Modelslim packs may use either ``model.layers…`` or bare ``layers…``.
        names.append("model.language_model." + weight[len("model.") :])
        names.append(weight[len("model.") :])
    return tuple(dict.fromkeys(names))


def _direct_kind(kinds: dict[str, str], name: str) -> str | None:
    """Quant label for an exact module name, ignoring fused-projection aliases."""
    for candidate in weight_name_variants(name):
        kind = kinds.get(candidate)
        if kind is not None:
            return kind
    return None


def lookup_kind(kinds: dict[str, str], name: str) -> str | None:
    """Quant label for ``name``, or None when the description has no entry.

    Direct hits win. Fused runtime modules fall back to their checkpoint
    shard members; every present shard must carry the same label.
    """
    kind = _direct_kind(kinds, name)
    if kind is not None:
        return kind

    leaf = name.rsplit(".", 1)[-1]
    groups = _FUSED_PROJECTION_SHARDS.get(leaf)
    if groups is None:
        return None
    base = name.rsplit(".", 1)[0]
    for shards in groups:
        found: set[str] = set()
        for shard in shards:
            shard_kind = _direct_kind(kinds, f"{base}.{shard}")
            if shard_kind is not None:
                found.add(shard_kind)
        if not found:
            continue
        if len(found) > 1:
            raise ValueError(
                f"Mixed Ascend quant labels within fused layer {name}: "
                f"{sorted(found)}. All members must use the same quantization."
            )
        return found.pop()
    return None


def is_w8a8(kind: str) -> bool:
    """True for modelslim W8A8 labels, including dynamic and mix suffixes."""
    return kind == "W8A8" or kind.startswith("W8A8_")
