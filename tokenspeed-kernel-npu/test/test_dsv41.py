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

"""Contracts at the optional CANN SparseFlashMla boundary."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import tokenspeed_kernel_npu.ops


def test_sparse_mla_adapters_preserve_native_calls(monkeypatch):
    extension = ModuleType("cann_ops_transformer")
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", extension)

    calls = []

    def attention(*args, **kwargs):
        calls.append(("attention", args, kwargs))
        return "attention-output", "lse-output"

    def metadata(*args, **kwargs):
        calls.append(("metadata", args, kwargs))
        return "metadata-output"

    native_ops = ModuleType("torch.ops.cann_ops_transformer")
    native_ops.sparse_flash_mla = attention
    native_ops.sparse_flash_mla_metadata = metadata
    ops = ModuleType("torch.ops")
    ops.cann_ops_transformer = native_ops
    fake_torch = ModuleType("torch")
    fake_torch.ops = ops
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    module_path = Path(tokenspeed_kernel_npu.ops.__file__).with_name("dsv41.py")
    adapter = runpy.run_path(str(module_path))

    query = object()
    sinks = object()
    result = adapter["npu_sparse_flash_mla"](
        query,
        sinks=sinks,
        layout_q="TND",
        layout_kv="PA_BBND",
        return_softmax_lse=False,
    )
    metadata_result = adapter["npu_sparse_flash_mla_metadata"](
        64,
        1,
        512,
        batch_size=2,
        max_seqlen_q=1,
        has_ori_kv=True,
        has_cmp_kv=False,
    )

    assert result == ("attention-output", "lse-output")
    assert metadata_result == "metadata-output"
    assert calls == [
        (
            "attention",
            (query,),
            {
                "sinks": sinks,
                "layout_q": "TND",
                "layout_kv": "PA_BBND",
                "return_softmax_lse": False,
            },
        ),
        (
            "metadata",
            (64, 1, 512),
            {
                "batch_size": 2,
                "max_seqlen_q": 1,
                "has_ori_kv": True,
                "has_cmp_kv": False,
            },
        ),
    ]
