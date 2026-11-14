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

"""Non-blocking eager-forward attribution with accelerator events.

Timing is sampled on one rank and disabled during graph capture. Completed
samples are polled on later forwards: the forward thread never waits for an
event just to produce a timing log.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from tokenspeed.runtime.utils.device_stream import is_current_stream_capturing
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TimedScope:
    kind: str
    layer_id: int
    start: Any
    end: Any


class LayerKindTimer:
    """Event pairs for one sampled eager forward.

    ``poll`` returns no result until the final event completes. It never
    synchronizes an event or stream. Scope and whole-forward durations are
    device elapsed times; nested scopes must not be added together as if they
    were disjoint wall-clock intervals.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        mode: str,
        bs: int,
        step: int,
        device_module: Any,
    ) -> None:
        self.device = device
        self.mode = mode
        self.bs = bs
        self.step = step
        self._device_module = device_module
        self._scopes: list[_TimedScope] = []
        self._finished = False
        self._start = self._event()
        self._end: Any | None = None
        self._start.record()

    def _event(self) -> Any:
        return self._device_module.Event(enable_timing=True)

    @contextmanager
    def scope(self, kind: str, layer_id: int):
        """Record one nested model scope without reading device state."""
        if self._finished:
            raise RuntimeError("cannot add a scope to a finished layer timer")
        start = self._event()
        start.record()
        try:
            yield
        finally:
            end = self._event()
            end.record()
            self._scopes.append(_TimedScope(kind, layer_id, start, end))

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("layer timer was finished twice")
        self._end = self._event()
        self._end.record()
        self._finished = True

    def poll(self) -> dict[str, Any] | None:
        """Return a completed sample, or None while device work is pending."""
        if not self._finished or self._end is None:
            raise RuntimeError("layer timer must be finished before polling")
        if not self._end.query():
            return None
        kinds: dict[str, float] = defaultdict(float)
        layers: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for scope in self._scopes:
            elapsed = float(scope.start.elapsed_time(scope.end))
            kinds[scope.kind] += elapsed
            layers[scope.layer_id][scope.kind] += elapsed
        return {
            "step": self.step,
            "mode": self.mode,
            "bs": self.bs,
            "forward_ms": round(float(self._start.elapsed_time(self._end)), 3),
            "kinds_ms": {
                kind: round(value, 3) for kind, value in sorted(kinds.items())
            },
            "layers_ms": {
                str(layer_id): {
                    kind: round(value, 3) for kind, value in sorted(values.items())
                }
                for layer_id, values in sorted(layers.items())
            },
        }


class LayerKindTimingRecorder:
    """Own sampled timers and release their logs without blocking serving."""

    def __init__(self, *, rank: int, max_pending: int = 4) -> None:
        self.rank = rank
        self.max_pending = max_pending
        self._next_step = 0
        self._pending: deque[LayerKindTimer] = deque()
        self._dropped = 0

    def start(
        self, device: torch.device, *, mode: str, bs: int
    ) -> LayerKindTimer | None:
        """Poll older samples and optionally start this forward's sample."""
        self.poll()
        step = self._next_step
        self._next_step += 1
        sampled = step < 4 or step % 100 == 0
        if (
            self.rank != 0
            or not sampled
            or device.type not in ("cuda", "npu")
            or not envs.TOKENSPEED_LOG_LAYER_KIND_TIMING.get()
            or is_current_stream_capturing(device)
        ):
            return None
        return LayerKindTimer(
            device,
            mode=mode,
            bs=bs,
            step=step,
            device_module=torch.get_device_module(device),
        )

    def finish(self, timer: LayerKindTimer) -> None:
        timer.finish()
        if len(self._pending) == self.max_pending:
            self._pending.popleft()
            self._dropped += 1
        self._pending.append(timer)

    def poll(self) -> None:
        """Log every newly completed sample; leave pending samples queued."""
        while self._pending:
            timer = self._pending[0]
            sample = timer.poll()
            if sample is None:
                break
            self._pending.popleft()
            if self._dropped:
                sample["dropped_before_this"] = self._dropped
                self._dropped = 0
            logger.info(f"v41_layer_timing {json.dumps(sample, sort_keys=True)}")


@contextmanager
def layer_kind_scope(ctx: ForwardContext, kind: str, layer_id: int):
    """Time a model scope when this forward carries an active timer."""
    timer = ctx.layer_kind_timer
    if timer is None:
        yield
        return
    with timer.scope(kind, layer_id):
        yield
