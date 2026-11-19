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

import json

import pytest
import torch

from tokenspeed.runtime.execution import debug_tensor_capture
from tokenspeed.runtime.execution.debug_tensor_capture import (
    DebugTensorCapture,
    DebugTensorCaptureRecorder,
)


def test_capture_clones_cpu_tensor_and_records_stable_keys(tmp_path):
    capture = DebugTensorCapture(tmp_path / "step", step=0, mode="DECODE")
    source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    capture.capture("layer_output", source, layer_id=3)
    source[0] = 9.0
    capture.capture("layer_output", source, layer_id=3)
    capture.finish()

    manifest = json.loads((tmp_path / "step" / "manifest.json").read_text())
    first = torch.load(tmp_path / "step" / manifest["captures"][0]["file"])
    assert manifest["captures"][0]["capture_key"] == "layer_003:layer_output:0"
    assert manifest["captures"][1]["capture_key"] == "layer_003:layer_output:1"
    assert first.dtype == torch.bfloat16
    assert first.tolist() == [1.0, 2.0]


def test_capture_rejects_graph_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_tensor_capture, "_is_capturing", lambda _: True)
    capture = DebugTensorCapture(tmp_path / "step", step=0, mode="DECODE")

    with pytest.raises(RuntimeError, match="forbidden during graph capture"):
        capture.capture("logits", torch.ones(1), layer_id=None)


def test_recorder_is_bounded_and_disabled_outside_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_tensor_capture, "_is_capturing", lambda _: False)
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_STEPS", "1")
    recorder = DebugTensorCaptureRecorder(rank=0)

    first = recorder.start(torch.device("cpu"), mode="DECODE")
    second = recorder.start(torch.device("cpu"), mode="DECODE")

    assert first is not None
    assert second is None
    assert recorder.start(torch.device("cpu"), mode="DECODE") is None
    other_rank = DebugTensorCaptureRecorder(rank=1)
    assert other_rank.start(torch.device("cpu"), mode="DECODE") is None


def test_disarmed_forward_does_not_consume_step_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_tensor_capture, "_is_capturing", lambda _: False)
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_DIR", str(tmp_path))
    arm = tmp_path / "capture.arm"
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_ARM_FILE", str(arm))
    recorder = DebugTensorCaptureRecorder(rank=0)

    assert recorder.start(torch.device("cpu"), mode="EAGER") is None
    arm.touch()
    capture = recorder.start(torch.device("cpu"), mode="EAGER")

    assert capture is not None
    assert capture.step == 0


def test_arm_file_can_skip_prefill_before_decode_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_tensor_capture, "_is_capturing", lambda _: False)
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_DIR", str(tmp_path))
    arm = tmp_path / "capture.arm"
    arm.write_text("1", encoding="utf-8")
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_ARM_FILE", str(arm))
    recorder = DebugTensorCaptureRecorder(rank=0)

    assert recorder.start(torch.device("cpu"), mode="PREFILL") is None
    capture = recorder.start(torch.device("cpu"), mode="DECODE")

    assert capture is not None
    assert capture.mode == "DECODE"
    assert capture.step == 0


def test_recorder_skips_graph_capture_without_creating_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_V41_DEBUG_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(debug_tensor_capture, "_is_capturing", lambda _: True)
    recorder = DebugTensorCaptureRecorder(rank=0)

    assert recorder.start(torch.device("cpu"), mode="DECODE") is None
    assert list(tmp_path.iterdir()) == []
