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

import importlib.util
import json
import sys
import types
from pathlib import Path

try:
    import paramiko
except ImportError:
    # Remote SSH is the genuinely missing third-party capability in CPU tests.
    paramiko = types.ModuleType("paramiko")
    paramiko.SSHClient = object
    paramiko.AutoAddPolicy = type("AutoAddPolicy", (), {})
    paramiko.RejectPolicy = type("RejectPolicy", (), {})
    sys.modules["paramiko"] = paramiko

SPEC = importlib.util.spec_from_file_location(
    "collect_v41_inventory", Path(__file__).with_name("collect_v41_inventory.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the inventory collector")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class FakeStream:
    def __init__(self, text, *, rc=0):
        self.data = text.encode()
        self.channel = self
        self.rc = rc

    def recv_exit_status(self):
        return self.rc

    def read(self):
        return self.data


class FakeClient:
    def __init__(self):
        self.policy = None
        self.connection = None
        self.commands = []
        self.closed = False

    def load_system_host_keys(self):
        pass

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, host, **kwargs):
        self.connection = (host, kwargs)

    def exec_command(self, command, *, timeout):
        self.commands.append((command, timeout))
        rc = 3 if command == "npu-smi info" else 0
        return None, FakeStream(f"out:{command}", rc=rc), FakeStream("diagnostic")

    def close(self):
        self.closed = True


def _collect(monkeypatch, **overrides):
    client = FakeClient()
    monkeypatch.setattr(inventory.paramiko, "SSHClient", lambda: client)
    kwargs = {
        "user": "root",
        "password": "not-serialized",
        "port": 22,
        "container": "serving",
        "deploy_tree": "/work/tree with space",
        "container_tree": "/ts",
        "model_path": "/weights/model",
        "accept_new_host_key": False,
    }
    kwargs.update(overrides)
    result = inventory.collect_host("192.0.2.1", **kwargs)
    return client, result


def test_run_command_preserves_caller_timeout():
    client = FakeClient()
    result = inventory.run_command(client, "true", timeout=15)
    assert result["rc"] == 0
    assert client.commands == [("true", 15)]


def test_inventory_is_read_only_and_omits_password(monkeypatch):
    client, result = _collect(monkeypatch)

    assert client.closed
    assert set(result["queries"]) == {
        "identity",
        "containers",
        "npu",
        "npu_map",
        "npu_arch_evidence",
        "stage1_prerequisites",
        "health",
        "models",
        "git_head",
        "git_status",
        "git_diff_stat",
        "container_process",
        "versions",
        "container_git_head",
        "container_git_status",
        "model_config",
    }
    assert result["queries"]["npu"]["rc"] == 3
    assert result["queries"]["npu"]["stderr"] == "diagnostic"
    assert "git -C '/work/tree with space' rev-parse HEAD" in [
        command for command, _ in client.commands
    ]
    assert "not-serialized" not in json.dumps(result)
    assert not any("nerdctl rm" in command for command, _ in client.commands)
    assert not any("nerdctl run" in command for command, _ in client.commands)
    assert not any("pip install" in command for command, _ in client.commands)
    assert not any("npu-smi reset" in command for command, _ in client.commands)
    assert "stage1_prerequisites" in result["queries"]


def test_unknown_host_keys_require_explicit_opt_in(monkeypatch):
    rejecting_client, _ = _collect(monkeypatch)
    accepting_client, _ = _collect(monkeypatch, accept_new_host_key=True)

    assert isinstance(rejecting_client.policy, paramiko.RejectPolicy)
    assert isinstance(accepting_client.policy, paramiko.AutoAddPolicy)


def test_optional_probes_are_omitted(monkeypatch):
    client, result = _collect(
        monkeypatch,
        container=None,
        deploy_tree=None,
        container_tree=None,
        model_path=None,
    )

    assert set(result["queries"]) == {
        "identity",
        "containers",
        "npu",
        "npu_map",
        "npu_arch_evidence",
        "stage1_prerequisites",
        "health",
        "models",
    }
    assert all("nerdctl exec" not in command for command, _ in client.commands)


def test_container_discovery_prefers_tokenspeed_and_existing_v41_containers():
    listing = (
        "deepseek-v41-dp|quay.io/ascend/vllm-ascend:v0.30|Up|\n"
        "tokenspeed-v41|private/tokenspeed:ascend|Up|\n"
    )
    assert inventory.select_container(listing, "dsv41-ts-dp") == "tokenspeed-v41"
    assert inventory.select_container(listing, "deepseek-v41-dp") == "deepseek-v41-dp"


def test_container_discovery_falls_back_to_vllm_without_claiming_tokenspeed():
    vllm_listing = (
        "deepseek-v41-proxy|quay.io/ascend/vllm-ascend:v0.30|Exited (0)|\n"
        "deepseek-v41-dp|quay.io/ascend/vllm-ascend:v0.30|Up|\n"
    )
    assert inventory.select_container(vllm_listing, "dsv41-ts-dp") == "deepseek-v41-dp"
    pd_listing = (
        "pd-pat-prefill-0|quay.io/ascend/vllm-ascend:v0.30|Up|\n"
        "pd-pat-decode-0|quay.io/ascend/vllm-ascend:v0.30|Up|\n"
    )
    assert inventory.select_container(pd_listing, "dsv41-ts-dp") == "pd-pat-decode-0"
    assert inventory.select_container("", "dsv41-ts-dp") == "dsv41-ts-dp"
