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

"""Device-specific execution contracts used by the shared forward path."""

from __future__ import annotations

from typing import Any


def graph_class_for(device_module: Any, device: str) -> type:
    """Return the decode-graph class exposed by the selected device module."""
    graph_name = {"cuda": "CUDAGraph", "npu": "NPUGraph"}.get(device)
    if graph_name is None:
        raise ValueError(f"No decode-graph contract for device {device!r}")
    try:
        return getattr(device_module, graph_name)
    except AttributeError:
        raise ValueError(
            f"Device module does not provide {graph_name} for {device!r}"
        ) from None


__all__ = ["graph_class_for"]
