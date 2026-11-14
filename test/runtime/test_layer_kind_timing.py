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

from tokenspeed.runtime.execution import layer_kind_timing as timing
from tokenspeed.runtime.execution.layer_kind_timing import (
    LayerKindTimer,
    LayerKindTimingRecorder,
)


class FakeEvent:
    """Completion-controlled timing event; deliberately has no synchronize API."""

    def __init__(self, owner):
        self.owner = owner
        self.ready = False
        self.stamp = None
        self.elapsed_calls = 0

    def record(self):
        self.stamp = self.owner.clock
        self.owner.clock += 10

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        self.elapsed_calls += 1
        if not self.ready or not end.ready:
            raise RuntimeError("elapsed_time read before event completion")
        return end.stamp - self.stamp


class FakeDeviceModule:
    def __init__(self):
        self.clock = 0
        self.events = []

    def Event(self, *, enable_timing):
        if not enable_timing:
            raise ValueError("layer timing requires timed events")
        event = FakeEvent(self)
        self.events.append(event)
        return event

    def complete_all(self):
        for event in self.events:
            event.ready = True


def _enable(monkeypatch, module):
    monkeypatch.setenv("TOKENSPEED_LOG_LAYER_KIND_TIMING", "1")
    monkeypatch.setattr(timing.torch, "get_device_module", lambda device: module)
    monkeypatch.setattr(timing, "is_current_stream_capturing", lambda device: False)


def _logged_samples(lines):
    return [json.loads(line.split("v41_layer_timing ", 1)[1]) for line in lines]


def test_disabled_timer_does_not_touch_device(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_LOG_LAYER_KIND_TIMING", "0")

    def unexpected(device):
        pytest.fail("disabled layer timing touched the device")

    monkeypatch.setattr(timing.torch, "get_device_module", unexpected)
    monkeypatch.setattr(timing, "is_current_stream_capturing", unexpected)
    recorder = LayerKindTimingRecorder(rank=0)

    assert recorder.start(torch.device("cuda"), mode="DECODE", bs=1) is None


def test_graph_capture_does_not_allocate_events(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_LOG_LAYER_KIND_TIMING", "1")
    monkeypatch.setattr(timing, "is_current_stream_capturing", lambda device: True)

    def unexpected(device):
        pytest.fail("capture timing allocated a device module or event")

    monkeypatch.setattr(timing.torch, "get_device_module", unexpected)
    recorder = LayerKindTimingRecorder(rank=0)

    assert recorder.start(torch.device("npu"), mode="DECODE", bs=1) is None


def test_incomplete_events_are_not_read_or_synchronized(monkeypatch):
    module = FakeDeviceModule()
    _enable(monkeypatch, module)
    lines = []
    monkeypatch.setattr(timing.logger, "info", lines.append)
    recorder = LayerKindTimingRecorder(rank=0)
    timer = recorder.start(torch.device("npu"), mode="DECODE", bs=1)
    assert timer is not None

    with timer.scope("layer", 0):
        with timer.scope("attention", 0):
            pass
        with timer.scope("ffn", 0):
            pass
    recorder.finish(timer)
    recorder.poll()

    assert lines == []
    assert all(event.elapsed_calls == 0 for event in module.events)

    module.complete_all()
    recorder.poll()
    samples = _logged_samples(lines)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["forward_ms"] == 70.0
    assert sample["kinds_ms"] == {
        "attention": 10.0,
        "ffn": 10.0,
        "layer": 50.0,
    }
    assert sample["layers_ms"]["0"] == {
        "attention": 10.0,
        "ffn": 10.0,
        "layer": 50.0,
    }

    recorder.poll()
    assert len(lines) == 1


def test_exception_still_closes_scope():
    module = FakeDeviceModule()
    timer = LayerKindTimer(
        torch.device("npu"),
        mode="DECODE",
        bs=1,
        step=0,
        device_module=module,
    )

    with pytest.raises(ValueError, match="original failure"):
        with timer.scope("engram", 7):
            raise ValueError("original failure")
    timer.finish()
    module.complete_all()

    sample = timer.poll()
    assert sample is not None
    assert sample["kinds_ms"] == {"engram": 10.0}
    assert sample["layers_ms"]["7"] == {"engram": 10.0}


def test_unsampled_forward_allocates_no_events(monkeypatch):
    module = FakeDeviceModule()
    _enable(monkeypatch, module)
    lines = []
    monkeypatch.setattr(timing.logger, "info", lines.append)
    recorder = LayerKindTimingRecorder(rank=0)
    for _ in range(4):
        timer = recorder.start(torch.device("npu"), mode="DECODE", bs=1)
        assert timer is not None
        recorder.finish(timer)
    before = len(module.events)

    assert recorder.start(torch.device("npu"), mode="DECODE", bs=1) is None
    assert len(module.events) == before

    module.complete_all()
    recorder.poll()
    assert len(_logged_samples(lines)) == 4
