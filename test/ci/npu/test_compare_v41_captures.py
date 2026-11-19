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
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_v41_captures import compare_capture_directories  # noqa: E402


def _write_capture(directory: Path, tensors: list[tuple[str, torch.Tensor]]) -> None:
    directory.mkdir()
    records = []
    for ordinal, (name, tensor) in enumerate(tensors):
        filename = f"{ordinal:04d}.pt"
        torch.save(tensor, directory / filename)
        records.append(
            {
                "ordinal": ordinal,
                "capture_key": f"layer_000:{name}:0",
                "name": name,
                "layer_id": 0,
                "occurrence": 0,
                "file": filename,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "captures": records}),
        encoding="utf-8",
    )


def test_identical_captures_pass(tmp_path):
    tensors = [
        ("input_ids", torch.tensor([[3, 5, 8]], dtype=torch.int64)),
        ("layer_output", torch.tensor([1.0, -2.0], dtype=torch.bfloat16)),
    ]
    _write_capture(tmp_path / "reference", tensors)
    _write_capture(tmp_path / "candidate", tensors)

    report = compare_capture_directories(
        tmp_path / "reference", tmp_path / "candidate", atol=0.02, rtol=0.02
    )

    assert report["status"] == "pass"
    assert report["compared_count"] == 2
    assert report["first_divergence"] is None


def test_single_bf16_element_negative_control_reports_first_divergence(tmp_path):
    reference = torch.zeros((2, 3), dtype=torch.bfloat16)
    changed = reference.clone()
    changed[1, 2] = 0.25
    _write_capture(tmp_path / "reference", [("layer_output", reference)])
    _write_capture(tmp_path / "candidate", [("layer_output", changed)])

    report = compare_capture_directories(
        tmp_path / "reference", tmp_path / "candidate", atol=0.02, rtol=0.02
    )

    assert report["status"] == "fail"
    divergence = report["first_divergence"]
    assert divergence["capture_key"] == "layer_000:layer_output:0"
    assert divergence["mismatch_count"] == 1
    assert divergence["first_mismatch_flat_index"] == 5
    assert divergence["max_abs_diff"] == 0.25


def test_cross_engine_coverage_uses_common_keys_and_reports_missing(tmp_path):
    _write_capture(
        tmp_path / "reference",
        [("layer_input", torch.tensor([1.0])), ("ts_only", torch.tensor([2.0]))],
    )
    _write_capture(
        tmp_path / "candidate",
        [("layer_input", torch.tensor([1.0])), ("vllm_only", torch.tensor([3.0]))],
    )

    report = compare_capture_directories(
        tmp_path / "reference", tmp_path / "candidate", atol=0.0, rtol=0.0
    )

    assert report["status"] == "pass"
    assert report["coverage"]["common_capture_count"] == 1
    assert report["coverage"]["missing_from_candidate"] == ["layer_000:ts_only:0"]
    assert report["coverage"]["missing_from_reference"] == ["layer_000:vllm_only:0"]


def test_integer_indices_require_exact_match_even_with_large_tolerance(tmp_path):
    _write_capture(tmp_path / "reference", [("selection_slots", torch.tensor([4, 7]))])
    _write_capture(tmp_path / "candidate", [("selection_slots", torch.tensor([4, 8]))])

    report = compare_capture_directories(
        tmp_path / "reference", tmp_path / "candidate", atol=100.0, rtol=100.0
    )

    assert report["status"] == "fail"
    assert report["first_divergence"]["comparison"] == "exact_integer"
    assert report["first_divergence"]["mismatch_count"] == 1


def test_logits_report_includes_top1_change(tmp_path):
    reference = torch.tensor([[0.0, 2.0, 1.0]])
    candidate = torch.tensor([[0.0, 1.0, 2.0]])
    _write_capture(tmp_path / "reference", [("logits", reference)])
    _write_capture(tmp_path / "candidate", [("logits", candidate)])

    report = compare_capture_directories(
        tmp_path / "reference", tmp_path / "candidate", atol=0.01, rtol=0.01
    )

    assert report["status"] == "fail"
    assert report["first_divergence"]["logits_top1_match"] is False
