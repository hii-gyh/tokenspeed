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

"""Behavior tests for the isolated same-load vLLM comparison runner."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_cluster import NODES  # noqa: E402
from run_v41_vllm_cluster import (  # noqa: E402
    container_command,
    health_status,
    render_vllm_entry,
    with_debug_capture,
)


def test_rank_one_entry_preserves_same_load_topology_and_eager_path():
    script = render_vllm_entry(NODES[1], "eager")
    assert "--data-parallel-size 2 --data-parallel-size-local 1" in script
    assert "--data-parallel-rank 1 --data-parallel-address 192.168.11.2" in script
    assert "--tensor-parallel-size 8 --enable-expert-parallel" in script
    assert (
        "--max-model-len 1024 --max-num-seqs 8 --max-num-batched-tokens 1024" in script
    )
    assert "--enable-prefix-caching --async-scheduling" in script
    assert "--enforce-eager --seed 1024" in script
    assert "HCCL_IF_IP=192.168.11.3" in script
    assert "speculative" not in script
    assert "Mooncake" not in script
    assert "dspark" not in script.lower()
    assert "@" not in script


def test_debug_capture_loads_staged_adapter_and_explicit_arm():
    script = with_debug_capture(render_vllm_entry(NODES[0], "eager"))

    assert "PYTHONPATH=/logs/vllm-debug:${PYTHONPATH:-}" in script
    assert "VLLM_V41_DEBUG_CAPTURE_STEPS=2" in script
    assert "VLLM_V41_DEBUG_CAPTURE_ARM_FILE=/logs/capture.arm" in script
    assert "--enforce-eager --seed 1024" in script


def test_graph_entry_explicitly_enables_graph_execution():
    script = render_vllm_entry(NODES[0], "graph")
    assert "--no-enforce-eager --seed 1024" in script
    assert "--enforce-eager " not in script
    with pytest.raises(ValueError, match="unsupported execution mode"):
        render_vllm_entry(NODES[0], "auto")


def test_profiler_config_reaches_serve_and_stays_absent_when_unset():
    config = '{"profiler": "torch", "torch_profiler_dir": "/logs/vllm-prof", "max_iterations": 10}'
    script = render_vllm_entry(NODES[0], "graph", config)
    assert f"--profiler-config '{config}'" in script
    assert "--profiler-config" not in render_vllm_entry(NODES[0], "graph")
    assert "@" not in script


def test_container_is_unique_read_only_and_maps_all_eight_devices():
    command = container_command("image:test", "/root/stage/run", "vllm-test")
    assert "--name vllm-test" in command
    assert "--device /dev/davinci0" in command
    assert "--device /dev/davinci7" in command
    assert "/root/stage/run:/probe:ro" in command
    assert "DeepSeek-V4.1-Flash-w8a8:ro" in command
    assert "bash /probe/vllm_entry.sh" in command


def test_vllm_health_uses_vllm_health_endpoint(monkeypatch):
    commands = []

    def fake_run_command(client, command, timeout):
        commands.append(command)
        return {"rc": 0, "stdout": "200\n", "stderr": ""}

    monkeypatch.setattr("run_v41_vllm_cluster.run_command", fake_run_command)
    assert health_status(object())
    assert "http://127.0.0.1:8000/health" in commands[0]
    assert "/readiness" not in commands[0]
