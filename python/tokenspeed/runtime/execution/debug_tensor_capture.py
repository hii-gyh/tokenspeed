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

"""Explicit eager-only tensor capture for offline V4.1 numerics debugging."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext


def _is_capturing(device: torch.device) -> bool:
    if device.type == "cpu":
        return False
    from tokenspeed.runtime.utils.device_stream import is_current_stream_capturing

    return is_current_stream_capturing(device)


class DebugTensorSink(Protocol):
    """Behavior-only capture consumer attached to one eager forward."""

    def capture(self, name: str, tensor: torch.Tensor, *, layer_id: int | None) -> None:
        """Copy one tensor before its producer reuses the storage."""


def capture_debug_tensor(
    ctx: ForwardContext | None,
    name: str,
    tensor: torch.Tensor | None,
    *,
    layer_id: int | None = None,
) -> None:
    """Publish a tensor to the explicitly attached sink, if one is active."""
    if ctx is None or tensor is None or ctx.debug_tensor_sink is None:
        return
    if _is_capturing(tensor.device):
        raise RuntimeError("Debug tensor capture is forbidden during graph capture")
    ctx.debug_tensor_sink.capture(name, tensor, layer_id=layer_id)


class DebugTensorCapture:
    """One rank's bounded, synchronous CPU copies for a single eager step."""

    def __init__(self, directory: Path, *, step: int, mode: str) -> None:
        self.directory = directory
        self.step = step
        self.mode = mode
        self._ordinal = 0
        self._occurrences: dict[tuple[int | None, str], int] = {}
        self._records: list[dict[str, object]] = []
        self.directory.mkdir(parents=True, exist_ok=False)

    def capture(self, name: str, tensor: torch.Tensor, *, layer_id: int | None) -> None:
        if _is_capturing(tensor.device):
            raise RuntimeError("Debug tensor capture is forbidden during graph capture")
        occurrence_key = (layer_id, name)
        occurrence = self._occurrences.get(occurrence_key, 0)
        self._occurrences[occurrence_key] = occurrence + 1
        layer = "none" if layer_id is None else f"{layer_id:03d}"
        capture_key = f"layer_{layer}:{name}:{occurrence}"
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
        filename = f"{self._ordinal:04d}_layer_{layer}_{safe_name}_{occurrence}.pt"
        copied = tensor.detach().to(device="cpu", copy=True)
        torch.save(copied, self.directory / filename)
        self._records.append(
            {
                "ordinal": self._ordinal,
                "capture_key": capture_key,
                "name": name,
                "layer_id": layer_id,
                "occurrence": occurrence,
                "file": filename,
                "shape": list(copied.shape),
                "dtype": str(copied.dtype).removeprefix("torch."),
            }
        )
        self._ordinal += 1

    def finish(self) -> None:
        metadata = {
            "schema_version": 1,
            "step": self.step,
            "mode": self.mode,
            "captures": self._records,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


class DebugTensorCaptureRecorder:
    """Create at most the configured number of rank-local eager captures."""

    def __init__(self, *, rank: int) -> None:
        self.rank = rank
        self._next_step = 0
        self._arm_token: tuple[int, int] | None = None
        self._skip_remaining = 0

    def _is_armed(self, arm_file: str) -> bool:
        if not arm_file:
            return True
        try:
            stat = os.stat(arm_file)
            token = (stat.st_dev, stat.st_ino)
            if token != self._arm_token:
                self._arm_token = token
                self._skip_remaining = int(
                    Path(arm_file).read_text(encoding="utf-8").strip() or "0"
                )
        except (FileNotFoundError, ValueError):
            return False
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return False
        return True

    def start(self, device: torch.device, *, mode: str) -> DebugTensorCapture | None:
        root = os.environ.get("TOKENSPEED_V41_DEBUG_CAPTURE_DIR", "")
        arm_file = os.environ.get("TOKENSPEED_V41_DEBUG_CAPTURE_ARM_FILE", "")
        max_steps = int(os.environ.get("TOKENSPEED_V41_DEBUG_CAPTURE_STEPS", "1"))
        step = self._next_step
        if (
            not root
            or not self._is_armed(arm_file)
            or self.rank != 0
            or step >= max_steps
            or _is_capturing(device)
        ):
            return None
        self._next_step += 1
        directory = Path(root) / f"step_{step:06d}_{mode.lower()}"
        return DebugTensorCapture(directory, step=step, mode=mode)

    def finish(self, capture: DebugTensorCapture | None) -> None:
        if capture is not None:
            capture.finish()
