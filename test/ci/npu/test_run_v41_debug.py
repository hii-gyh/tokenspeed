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

import sys
import types
from pathlib import Path

try:
    import paramiko  # noqa: F401
except ImportError:
    paramiko = types.ModuleType("paramiko")
    paramiko.SSHClient = object
    paramiko.AutoAddPolicy = type("AutoAddPolicy", (), {})
    paramiko.RejectPolicy = type("RejectPolicy", (), {})
    sys.modules["paramiko"] = paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bench_v41_steady import percentile  # noqa: E402
from run_v41_debug import (  # noqa: E402
    healthy_hosts,
    verified_heads,
    verified_worktrees,
)


def _host_result(health_rc=0, health_text="HTTP:200", models_rc=0, head="abc"):
    return {
        "queries": {
            "health": {"rc": health_rc, "stdout": health_text},
            "models": {"rc": models_rc, "stdout": "{}"},
            "container_git_head": {"rc": 0, "stdout": f"{head}\n"},
            "container_git_status": {"rc": 0, "stdout": ""},
            "git_diff_stat": {"rc": 0, "stdout": ""},
        }
    }


def test_only_healthy_and_model_serving_hosts_are_selected():
    inventory = {
        "healthy": _host_result(),
        "bad_health": _host_result(health_rc=1),
        "bad_http": _host_result(health_text="HTTP:503"),
        "bad_models": _host_result(models_rc=2),
        "connection_error": {"error": "unreachable"},
    }
    assert healthy_hosts(inventory) == ["healthy"]


def test_git_heads_must_be_successful_and_nonempty():
    inventory = {
        "one": _host_result(head="abc"),
        "two": _host_result(head=""),
        "three": {"queries": {"container_git_head": {"rc": 1, "stdout": "abc"}}},
    }
    assert verified_heads(inventory) == {"one": "abc"}


def test_worktree_fingerprint_includes_local_changes():
    clean = _host_result()
    changed = _host_result()
    changed["queries"]["container_git_status"]["stdout"] = " M model.py\n"
    inventory = {"clean": clean, "changed": changed}

    worktrees = verified_worktrees(inventory)
    assert worktrees["clean"]["status"] == ""
    assert worktrees["changed"]["status"] == " M model.py\n"
    assert worktrees["clean"]["head"] == worktrees["changed"]["head"]
    assert worktrees["clean"] != worktrees["changed"]


def test_latency_percentile_stays_inside_observed_values():
    values = [1.0, 2.0, 4.0, 8.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 4.0
    assert percentile(values, 90) == 8.0
    assert percentile(values, 100) == 8.0
