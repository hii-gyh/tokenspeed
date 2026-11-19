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

"""Compare two explicit eager V4.1 tensor captures in capture order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_manifest(directory: Path) -> dict[str, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported capture schema in {directory}")
    records = manifest["captures"]
    keys = [record["capture_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate capture keys in {directory}")
    return {record["capture_key"]: record for record in records}


def _tensor_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict:
    result = {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype).removeprefix("torch."),
        "numel": reference.numel(),
    }
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        result.update(
            {
                "match": False,
                "reason": "shape_or_dtype_mismatch",
                "candidate_shape": list(candidate.shape),
                "candidate_dtype": str(candidate.dtype).removeprefix("torch."),
            }
        )
        return result
    if not (reference.is_floating_point() or reference.is_complex()):
        mismatch = reference != candidate
        mismatch_count = int(mismatch.sum().item())
        result.update(
            {
                "match": mismatch_count == 0,
                "comparison": "exact_integer",
                "mismatch_count": mismatch_count,
            }
        )
        if mismatch_count:
            result["first_mismatch_flat_index"] = int(
                mismatch.reshape(-1).nonzero()[0, 0].item()
            )
        return result

    left = reference.float()
    right = candidate.float()
    difference = (left - right).abs()
    allowed = atol + rtol * right.abs()
    equal = (left == right) | (torch.isnan(left) & torch.isnan(right))
    within = equal | (difference <= allowed)
    mismatch = ~within
    mismatch_count = int(mismatch.sum().item())
    finite_difference = torch.where(torch.isfinite(difference), difference, 0.0)
    result.update(
        {
            "match": mismatch_count == 0,
            "comparison": "floating_tolerance",
            "atol": atol,
            "rtol": rtol,
            "mismatch_count": mismatch_count,
            "max_abs_diff": (
                float(finite_difference.max().item()) if reference.numel() else 0.0
            ),
            "nan_counts": [int(torch.isnan(left).sum()), int(torch.isnan(right).sum())],
            "inf_counts": [int(torch.isinf(left).sum()), int(torch.isinf(right).sum())],
        }
    )
    if mismatch_count:
        flat = mismatch.reshape(-1).nonzero()[0, 0].item()
        result["first_mismatch_flat_index"] = int(flat)
        result["first_mismatch_values"] = [
            float(left.reshape(-1)[flat].item()),
            float(right.reshape(-1)[flat].item()),
        ]
    return result


def compare_capture_directories(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    atol: float,
    rtol: float,
) -> dict:
    """Return ordered comparisons and stop at the first divergent capture."""
    reference = _load_manifest(reference_dir)
    candidate = _load_manifest(candidate_dir)
    common_keys = [key for key in reference if key in candidate]
    coverage = {
        "reference_capture_count": len(reference),
        "candidate_capture_count": len(candidate),
        "common_capture_count": len(common_keys),
        "missing_from_candidate": [key for key in reference if key not in candidate],
        "missing_from_reference": [key for key in candidate if key not in reference],
    }
    if not common_keys:
        return {
            "status": "fail",
            "reason": "no_common_capture_keys",
            "coverage": coverage,
        }
    candidate_common_keys = [key for key in candidate if key in reference]
    if common_keys != candidate_common_keys:
        return {
            "status": "fail",
            "reason": "common_capture_sequence_mismatch",
            "coverage": coverage,
            "reference_common_keys": common_keys,
            "candidate_common_keys": candidate_common_keys,
        }

    comparisons = []
    first_divergence = None
    for key in common_keys:
        reference_record = reference[key]
        candidate_record = candidate[key]
        left = torch.load(reference_dir / reference_record["file"], weights_only=True)
        right = torch.load(candidate_dir / candidate_record["file"], weights_only=True)
        comparison = _tensor_comparison(left, right, atol=atol, rtol=rtol)
        comparison.update(
            {
                "capture_key": key,
                "name": reference_record["name"],
                "layer_id": reference_record["layer_id"],
            }
        )
        if reference_record["name"] == "logits" and left.ndim >= 1:
            comparison["logits_top1_match"] = bool(
                torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))
            )
        comparisons.append(comparison)
        if not comparison["match"]:
            first_divergence = comparison
            break
    return {
        "status": "pass" if first_divergence is None else "fail",
        "atol": atol,
        "rtol": rtol,
        "coverage": coverage,
        "compared_count": len(comparisons),
        "first_divergence": first_divergence,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = compare_capture_directories(
        args.reference, args.candidate, atol=args.atol, rtol=args.rtol
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
