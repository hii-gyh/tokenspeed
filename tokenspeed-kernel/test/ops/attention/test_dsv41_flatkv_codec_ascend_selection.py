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

"""Ascend FlatKV codec selection: Triton FlatKV stays nvidia/amd-only."""

from __future__ import annotations

import pytest

from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41_triton


def test_flatkv_triton_capability_excludes_ascend():
    """Pin: Ascend uses torch SPECIALIZED; Triton FlatKV is nvidia/amd-only."""
    vendors = dsv41_triton._FLATKV_CODEC_CAPABILITY.vendors
    assert "ascend" not in vendors
    assert vendors == frozenset({"nvidia", "amd"})


def test_same_device_accepts_npu_device_type():
    class _Dev:
        def __init__(self, type_name: str):
            self.type = type_name

        def __eq__(self, other):
            return isinstance(other, _Dev) and other.type == self.type

    class _T:
        def __init__(self, type_name: str):
            self.device = _Dev(type_name)

    dsv41_triton._same_device(_T("npu"), (_T("npu"), None))
    with pytest.raises(ValueError):
        dsv41_triton._same_device(_T("cpu"), (_T("cpu"),))
