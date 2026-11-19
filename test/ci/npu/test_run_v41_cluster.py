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

"""Behavior tests for the reversible two-node serving stage."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_cluster import (  # noqa: E402
    NODES,
    render_cluster_entry,
    stop_ts_container,
    with_debug_capture,
)


def test_rank_one_uses_rank_zero_discovery_and_explicit_ep16_topology():
    script = render_cluster_entry(NODES[1], "eager", "/dev/shm/test-engram")
    assert "--nnodes 2 --node-rank 1" in script
    assert "--world-size 16" in script
    assert "--data-parallel-size 2 --ep-size 16" in script
    assert "--dist-init-addr 192.168.11.2:25000" in script
    assert "HCCL_IF_IP=192.168.11.3" in script
    assert "--enable-expert-parallel" in script
    assert "--enforce-eager" in script


def test_debug_capture_is_rank_zero_only_and_requires_explicit_arm():
    rank_zero = render_cluster_entry(NODES[0], "eager", "/dev/shm/test-engram")
    rank_one = render_cluster_entry(NODES[1], "eager", "/dev/shm/test-engram")

    probed = with_debug_capture(rank_zero, NODES[0])

    assert "TOKENSPEED_V41_DEBUG_CAPTURE_STEPS=2" in probed
    assert "TOKENSPEED_V41_DEBUG_CAPTURE_ARM_FILE=/logs/capture.arm" in probed
    assert with_debug_capture(rank_one, NODES[1]) == rank_one


def test_cleanup_stops_before_removing_and_reports_gone(monkeypatch):
    state = {"status": "running"}

    def status(client, name):
        return state["status"]

    def command(client, command, timeout):
        if command.startswith("nerdctl stop"):
            state["status"] = "exited"
            return {"rc": 0, "stdout": "", "stderr": ""}
        if command.startswith("nerdctl rm "):
            state["status"] = "missing"
            return {"rc": 0, "stdout": "", "stderr": ""}
        raise AssertionError(command)

    monkeypatch.setattr("run_v41_cluster.container_status", status)
    monkeypatch.setattr("run_v41_cluster.run_command", command)
    result = stop_ts_container(object(), "tokenspeed-test")
    assert result["gone"] is True
    assert result["stop_rc"] == 0
    assert result["remove_rc"] == 0


def test_cleanup_never_force_removes_a_container_that_did_not_stop(monkeypatch):
    monkeypatch.setattr(
        "run_v41_cluster.container_status", lambda client, name: "running"
    )
    monkeypatch.setattr(
        "run_v41_cluster.run_command",
        lambda client, command, timeout: {
            "rc": 1,
            "stdout": "",
            "stderr": "stop timed out",
        },
    )
    result = stop_ts_container(object(), "tokenspeed-test")
    assert result["gone"] is False
    assert result["remove_rc"] == 1
    assert result["stderr"] == "stop timed out"
