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

import io
import tarfile

import pytest
from run_v41_cluster import Node
from run_v41_pd_cluster import (
    compare_pd_to_reference,
    gateway_start,
    parse_containers,
    render_pd_entry,
    restore_original_workloads,
    running_workloads,
    stop_original_workload,
    validate_runtime_archive,
)


def test_prefill_gateway_and_engine_entry() -> None:
    script = render_pd_entry(
        "prefill",
        Node(rank=0, host="public", private_ip="192.168.11.2"),
        "graph",
        "/dev/shm/test",
        16,
    )
    assert "--pd-disaggregation --backend tokenspeed" in script
    assert "--prefill tcp://192.168.11.2:31000" in script
    assert "--decode tcp://192.168.11.2:31001" in script
    assert "--disaggregation-mode prefill" in script
    assert "--disaggregation-bootstrap-port 8998" in script
    assert "--enable-prefix-caching" in script
    assert "--max-model-len 1024 --max-num-seqs 16" in script
    assert "--data-parallel-rpc-port 31000" in script
    assert (
        "TOKENSPEED_MOONCAKE_LIBRARY=/ts/mooncake/libtokenspeed_transfer_engine.so"
        in script
    )
    assert "SMG_TOKENSPEED_SERVICER_SILENCE_TIMEOUT_SECS=300" in script
    assert "--enforce-eager" not in script


def test_gateway_allows_cold_prefill_beyond_default_wedge_window() -> None:
    script = gateway_start()
    assert "--worker-wedge-secs 300" in script
    assert "--request-timeout-secs 1800" in script
    assert "--dp-aware" in script
    assert "--prefill-policy power_of_two" in script
    assert "--decode-policy power_of_two" in script
    assert "--power-of-two-load-metric requests" in script
    assert "--prefill-max-inflight-requests-per-worker 4" in script
    assert "--dp-minimum-tokens-scheduler" not in script


def test_decode_entry_has_independent_cluster_and_no_gateway() -> None:
    script = render_pd_entry(
        "decode",
        Node(rank=1, host="public", private_ip="192.168.11.5"),
        "eager",
        "/dev/shm/test",
        8,
    )
    assert "-m smg launch" not in script
    assert "--dist-init-addr 192.168.11.10:25000" in script
    assert "--disaggregation-mode decode" in script
    assert "--disaggregation-bootstrap-port" not in script
    assert "--disable-prefix-caching" in script
    assert (
        "--data-parallel-address 192.168.11.2 --data-parallel-rpc-port 31001" in script
    )
    assert "--enforce-eager" in script


def test_workload_plan_preserves_exited_and_unrelated_containers() -> None:
    rows = parse_containers(
        "pd-pat-decode-0|Up 2 hours\n"
        "vllm-ascend-68368|Up 2 hours\n"
        "deepseek-v41-dp|Exited (0) 1 hour ago\n"
        "unrelated|Up 2 hours\n"
    )
    selected = running_workloads(rows)
    assert [row["name"] for row in selected] == ["vllm-ascend-68368", "pd-pat-decode-0"]


def test_runtime_archive_validation(tmp_path) -> None:
    package = tmp_path / "runtime.tar.gz"
    payload = b"native library"
    with tarfile.open(package, "w:gz") as archive:
        entry = tarfile.TarInfo("mooncake/libtokenspeed_transfer_engine.so")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    assert len(validate_runtime_archive(package)) == 64

    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, "w:gz") as archive:
        entry = tarfile.TarInfo("mooncake/../escape")
        entry.size = 0
        archive.addfile(entry, io.BytesIO())
    with pytest.raises(RuntimeError, match="unsafe Mooncake runtime member"):
        validate_runtime_archive(unsafe)


def benchmark_document(
    prompt_tokens: int = 55, throughput: float = 23.0, tpot: float = 0.32
) -> dict:
    row = {"prompt_tokens": prompt_tokens, "ttft_s": 3.7}
    run = {
        "ok": True,
        "completion_tokens": 128,
        "expected_completion_tokens": 128,
        "rows": [row],
    }
    benchmark = {
        "engine": "tokenspeed",
        "ok": True,
        "correctness": {"ok": True},
        "topology": "test",
        "load": {
            "model": "deepseek-v41",
            "protocol": "stream",
            "concurrency": 8,
            "requests": 16,
            "max_tokens": 128,
            "repeat": 2,
            "ignore_eos": True,
        },
        "runs": [run],
        "summary": {
            "output_tok_per_s_mean": throughput,
            "ttft_p50_s": 3.0,
            "ttft_p90_s": 5.0,
            "tpot_mean_s": tpot,
            "tpot_p50_s": tpot,
            "tpot_p90_s": tpot,
        },
    }
    return {"status": "passed", "benchmark": benchmark}


def test_pd_comparison_enforces_contract_and_reports_benefit() -> None:
    reference = benchmark_document()
    pd = benchmark_document(throughput=24.0, tpot=0.30)["benchmark"]
    result = compare_pd_to_reference(pd, reference)
    assert result["performance_benefit"] is True
    assert result["metrics"]["output_tok_per_s_mean"][
        "pd_over_non_pd"
    ] == pytest.approx(24.0 / 23.0)
    assert result["ttft_all_requests_mean_s"]["pd_over_non_pd"] == pytest.approx(1.0)

    mismatched = benchmark_document(prompt_tokens=56)["benchmark"]
    with pytest.raises(RuntimeError, match="prompt-token mismatch"):
        compare_pd_to_reference(mismatched, reference)


def test_restore_attempts_all_workloads_and_reports_failures() -> None:
    stopped = {
        "decode-rank1": [
            {"name": "router", "status": "Up"},
            {"name": "worker", "status": "Up"},
        ]
    }
    attempts = []

    def start(key: str, name: str) -> dict:
        attempts.append((key, name))
        return {"rc": 1 if name == "worker" else 0, "stderr": ""}

    restores, failures = restore_original_workloads(stopped, start)
    assert attempts == [("decode-rank1", "worker"), ("decode-rank1", "router")]
    assert failures == ["decode-rank1/worker"]
    assert [row["rc"] for row in restores["decode-rank1"]] == [1, 0]


def test_original_stop_preserves_container_and_accepts_disappearance() -> None:
    stopped = stop_original_workload(
        "service",
        lambda name: {"rc": 0, "stderr": ""},
        lambda name: "exited",
    )
    assert stopped["gone"] is True
    assert stopped["status"] == "exited"

    disappeared = stop_original_workload(
        "service",
        lambda name: {"rc": 1, "stderr": "no such container"},
        lambda name: "missing",
    )
    assert disappeared["gone"] is True

    with pytest.raises(RuntimeError, match="could not stop service"):
        stop_original_workload(
            "service",
            lambda name: {"rc": 1, "stderr": "permission denied"},
            lambda name: "running",
        )
