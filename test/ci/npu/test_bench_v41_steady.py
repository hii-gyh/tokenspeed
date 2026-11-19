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

"""Behavior tests for fixed-load streaming and latency accounting."""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_v41_steady  # noqa: E402
from bench_v41_steady import (  # noqa: E402
    aggregate_stream_metrics,
    post_stream,
    request_prompt,
    runtime_manifest,
    source_manifest,
    stream_metrics,
    weight_manifest,
)


def test_prompt_tag_prevents_cross_measurement_prefix_reuse():
    first = request_prompt(0, 3, "20261114T120000Z-100")
    second = request_prompt(0, 3, "20261114T120100Z-101")

    assert first != second
    assert len(first) == len(second)
    assert "tag=20261114T120000Z-100" in first


def test_correctness_prompt_also_isolates_measurement_cache(monkeypatch):
    payloads = []

    def fake_post(api, payload, timeout):
        payloads.append(payload)
        return {"choices": [{"text": " Paris."}], "usage": {}}

    monkeypatch.setattr(bench_v41_steady, "post", fake_post)
    result = bench_v41_steady.paris_correctness(
        "http://127.0.0.1:8000", "deepseek-v41", 1.0, "index-ab"
    )

    assert result["ok"] is True
    assert payloads[0]["prompt"].startswith("Request ")
    assert payloads[0]["prompt"].endswith(": The capital of France is")
    assert "index-ab" not in payloads[0]["prompt"]

    bench_v41_steady.paris_correctness(
        "http://127.0.0.1:8000", "deepseek-v41", 1.0, "another-ab"
    )
    assert payloads[1]["prompt"] != payloads[0]["prompt"]


def test_stream_metrics_normalize_tpot_by_server_token_usage():
    metrics = stream_metrics(
        0.5,
        [(1.0, "4 5"), (1.2, "6 7"), (1.5, "8 9")],
        completion_tokens=6,
        elapsed_s=2.0,
    )
    assert metrics["ttft_s"] == 0.5
    assert metrics["tpot_mean_s"] == 0.3
    assert metrics["itl_p50_s"] == 0.19999999999999996
    assert metrics["itl_p90_s"] == 0.30000000000000004
    assert metrics["streamed_chunks"] == 3
    assert stream_metrics(0.5, [], 0, 0.0)["ttft_s"] is None


def test_stream_aggregates_request_ttft_and_tpot():
    rows = [
        {"ttft_s": 0.4, "tpot_mean_s": 0.2},
        {"ttft_s": 0.6, "tpot_mean_s": 0.3},
    ]
    metrics = aggregate_stream_metrics(rows)
    assert metrics["ttft_p50_s"] == 0.4
    assert metrics["ttft_p90_s"] == 0.6
    assert metrics["tpot_mean_s"] == 0.25
    assert metrics["tpot_p90_s"] == 0.3


def test_sse_reader_preserves_text_and_final_usage(monkeypatch):
    events = [
        {"choices": [{"text": " Paris"}], "usage": None},
        {"choices": [{"text": " is"}], "usage": None},
        {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 3}},
    ]
    payload = (
        b"".join([f"data: {json.dumps(event)}\n\n".encode() for event in events])
        + b"data: [DONE]\n\n"
    )
    monkeypatch.setattr(
        "bench_v41_steady.urllib.request.urlopen",
        lambda request, timeout: io.BytesIO(payload),
    )
    chunks, usage = post_stream("http://127.0.0.1:8000", {}, 1.0)
    assert "".join(text for _, text in chunks) == " Paris is"
    assert usage == {"completion_tokens": 2, "prompt_tokens": 3}


def test_concurrent_warmup_is_separate_from_measured_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_v41_steady.py",
            "--engine",
            "tokenspeed",
            "--protocol",
            "stream",
            "--topology",
            "test",
            "--prompt-tag",
            "test",
            "--warmup-requests",
            "8",
            "--requests",
            "16",
            "--max-tokens",
            "128",
        ],
    )
    monkeypatch.setattr(
        bench_v41_steady,
        "post",
        lambda *args: {
            "choices": [{"text": " Paris"}],
            "usage": {"completion_tokens": 1},
        },
    )

    def fake_measured_run(api, model, run, args):
        completion_tokens = args.requests * args.max_tokens
        return {
            "run": run,
            "ok": True,
            "wall_s": 1.0,
            "completion_tokens": completion_tokens,
            "expected_completion_tokens": completion_tokens,
            "errors": 0,
            "fill_ratio": 1.0,
            "output_tok_per_s": float(completion_tokens),
            "rows": [{"prompt_tokens": 55, "ttft_s": 1.0, "tpot_mean_s": 0.1}],
        }

    monkeypatch.setattr(bench_v41_steady, "measured_run", fake_measured_run)
    assert bench_v41_steady.main() == 0
    document = json.loads(capsys.readouterr().out)
    assert document["warmup"]["requests"] == 8
    assert document["warmup"]["batch"]["run"] == -1
    assert document["warmup"]["batch"]["completion_tokens"] == 1024
    assert document["load"]["warmup_max_tokens"] == 128
    assert len(document["runs"]) == 1
    assert document["runs"][0]["completion_tokens"] == 2048
    assert document["correctness_post"]["ok"] is True


def test_post_measurement_correctness_failure_rejects_run(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_v41_steady.py",
            "--engine",
            "vllm-ascend",
            "--protocol",
            "stream",
            "--topology",
            "test",
            "--prompt-tag",
            "test",
        ],
    )
    monkeypatch.setattr(bench_v41_steady, "post", lambda *args: {"usage": {}})
    checks = iter(
        [
            {"ok": True, "text": " Paris"},
            {"ok": False, "text": " unrelated text"},
        ]
    )
    monkeypatch.setattr(
        bench_v41_steady, "paris_correctness", lambda *args: next(checks)
    )
    monkeypatch.setattr(
        bench_v41_steady,
        "measured_run",
        lambda api, model, run, args: {
            "run": run,
            "ok": True,
            "wall_s": 1.0,
            "completion_tokens": args.requests * args.max_tokens,
            "expected_completion_tokens": args.requests * args.max_tokens,
            "errors": 0,
            "fill_ratio": 1.0,
            "output_tok_per_s": 1.0,
            "rows": [],
        },
    )

    assert bench_v41_steady.main() == 1
    document = json.loads(capsys.readouterr().out)
    assert len(document["runs"]) == 1
    assert document["correctness"]["ok"] is True
    assert document["correctness_post"]["ok"] is False
    assert document["ok"] is False


def test_source_manifest_preserves_full_diff_and_untracked_hashes(
    monkeypatch, tmp_path
):
    payload = b"value = 1" + os.linesep.encode()
    (tmp_path / "new.py").write_bytes(payload)
    outputs = {
        "rev-parse": "abc123\n",
        "--porcelain=v1": " M changed.py\n?? new.py\n",
        "--stat": " changed.py | 2 +-\n",
        "--binary": "diff --git a/changed.py b/changed.py\n+new line\n",
        "--others": "new.py\n",
    }

    def fake_run(command, timeout=30.0):
        stdout = next(value for flag, value in outputs.items() if flag in command)
        return {"command": command, "rc": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(bench_v41_steady, "run_command", fake_run)
    manifest = source_manifest(tmp_path)

    assert manifest["head"] == "abc123"
    assert manifest["diff"] == outputs["--binary"]
    assert (
        manifest["diff_sha256"]
        == hashlib.sha256(outputs["--binary"].encode()).hexdigest()
    )
    assert manifest["untracked"][0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_runtime_manifest_redacts_secrets_and_records_versions(monkeypatch):
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("TOKENSPEED_HF_TOKEN", "do-not-serialize")
    monkeypatch.setattr(
        bench_v41_steady.importlib.metadata,
        "version",
        lambda name: "1.2.3" if name == "torch" else "4.5.6",
    )

    manifest = runtime_manifest()

    assert manifest["packages"]["torch-npu"] == "4.5.6"
    assert manifest["environment"]["HCCL_DETERMINISTIC"] == "true"
    assert manifest["environment"]["TOKENSPEED_HF_TOKEN"] == "<redacted>"
    assert "do-not-serialize" not in json.dumps(manifest)


def test_weight_manifest_hashes_checkpoint_configs(tmp_path):
    payload = b'{"model_type": "deepseek_v41"}\n'
    (tmp_path / "config.json").write_bytes(payload)

    manifest = weight_manifest(tmp_path, "test-weight-revision")

    assert manifest["status"] == "ok"
    assert manifest["revision"] == "test-weight-revision"
    assert (
        manifest["configs"]["config.json"]["sha256"]
        == hashlib.sha256(payload).hexdigest()
    )
    assert manifest["configs"]["generation_config.json"] is None
