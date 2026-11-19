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

import pytest

from tokenspeed.runtime.accelerators import graph_class_for


class CudaGraph:
    pass


class NpuGraph:
    pass


class DeviceModule:
    CUDAGraph = CudaGraph
    NPUGraph = NpuGraph


@pytest.mark.parametrize(
    ("device", "expected"),
    (("cuda", CudaGraph), ("npu", NpuGraph)),
)
def test_graph_class_for_selects_real_device_attribute(device, expected):
    assert graph_class_for(DeviceModule, device) is expected


def test_graph_class_for_rejects_unknown_device():
    with pytest.raises(ValueError, match="No decode-graph contract"):
        graph_class_for(DeviceModule, "unknown")


def test_graph_class_for_rejects_missing_device_graph():
    class IncompleteDeviceModule:
        pass

    with pytest.raises(ValueError, match="does not provide NPUGraph"):
        graph_class_for(IncompleteDeviceModule, "npu")
