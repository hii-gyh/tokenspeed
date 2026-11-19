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

"""Smoke-test the NPU event API used by the serving attribution recorder.

Run inside one agent container with an explicit visible device. This standalone
test waits for completion; the serving recorder must only poll on later forwards.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    device = f"npu:{args.device}"
    torch_npu.npu.set_device(args.device)
    value = torch.ones((1024, 1024), dtype=torch.float16, device=device)
    weight = torch.eye(1024, dtype=torch.float16, device=device)
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(8):
        value = value @ weight
    end.record()

    deadline = time.monotonic() + args.timeout
    polls = 0
    while not end.query():
        polls += 1
        if time.monotonic() >= deadline:
            raise TimeoutError("NPU event did not complete before the timeout")
        time.sleep(0.001)
    elapsed_ms = start.elapsed_time(end)
    if elapsed_ms <= 0:
        raise RuntimeError(f"NPU event returned a non-positive duration: {elapsed_ms}")
    print(
        json.dumps(
            {
                "device": device,
                "elapsed_ms": elapsed_ms,
                "polls_after_completion_check": polls,
                "result": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
