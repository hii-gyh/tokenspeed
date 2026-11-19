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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "outputs" / "v41-debug" / "negative_control_v41_capture.py"
SPEC = importlib.util.spec_from_file_location("negative_control_v41_capture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_real_capture_negative_control_workflow(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    torch.save(tensor, source / "value.pt")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captures": [
                    {
                        "capture_key": "layer_none/logits/0",
                        "name": "logits",
                        "layer_id": None,
                        "file": "value.pt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = MODULE.run_negative_control(source, tmp_path / "mutated")

    assert summary["status"] == "pass"
    assert summary["capture_key"] == "layer_none/logits/0"
    assert summary["negative_report"]["first_divergence"]["mismatch_count"] == 1
