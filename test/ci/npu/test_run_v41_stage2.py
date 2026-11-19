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

"""Regression tests for the isolated single-node serving stage."""

import io
import shutil
import subprocess
import sys
import tarfile
import types
from pathlib import Path

try:
    import paramiko  # noqa: F401
except ImportError:
    paramiko = types.ModuleType("paramiko")
    paramiko.SSHClient = object
    sys.modules["paramiko"] = paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_stage1 import SOURCE_DIRS  # noqa: E402
from run_v41_stage2 import (  # noqa: E402
    ENGRAM_DIR,
    MODEL,
    TOPOLOGY,
    build_stage2_archive,
    container_command,
    entry_script,
    read_service_log,
)


def render_entry(execution_mode: str) -> str:
    return entry_script(
        "192.168.11.2",
        "bond1",
        execution_mode,
        dist_init_ip="192.168.11.2",
        nnodes=1,
        node_rank=0,
        world_size=8,
        dp_size=1,
        ep_size=8,
        engram_dir=ENGRAM_DIR,
        topology=TOPOLOGY,
    )


def test_entry_uses_unbiased_timing_and_explicit_eager_mode():
    script = render_entry("eager")
    assert "--nnodes 1" in script
    assert "--world-size 8" in script
    assert "--ep-size 8" in script
    assert "--enforce-eager" in script
    assert "--allow-reload-forward" in script
    assert "--disable-kvstore --enable-prefix-caching" in script
    assert "TOKENSPEED_LOG_LAYER_KIND_TIMING=1" in script
    assert "TOKENSPEED_LOG_MM_TIMING" not in script
    assert "TOKENSPEED_LOG_V41_ATTN" not in script
    assert TOPOLOGY in script
    assert ENGRAM_DIR in script


def test_graph_mode_omits_only_eager_flag():
    script = render_entry("graph")
    assert "--enforce-eager" not in script
    assert "--disable-prefill-graph" in script


def test_entry_uses_only_a_hash_verified_cached_engine_wheel():
    script = render_entry("eager")
    assert "ENGINE_PACKAGE=tokenspeed-engine==0.1.0.post20261112" in script
    assert 'if [ -f "$ENGINE_WHEEL" ]' in script
    assert "cace479db91ea665ae61bb86df0275fd628833ceef96e5e549409970fd8f3033" in script
    assert "sha256sum -c -" in script
    assert '"$ENGINE_PACKAGE"' in script


def test_entry_has_valid_bash_syntax():
    if shutil.which("bash") is None:
        return
    result = subprocess.run(
        ["bash", "-n"],
        input=render_entry("eager"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_archive_contains_source_and_executable_entry(tmp_path):
    for source_dir in SOURCE_DIRS:
        package = tmp_path / source_dir
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    archive, manifest = build_stage2_archive(tmp_path, "#!/bin/bash\nexit 0\n")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        entry = handle.getmember("stage2_entry.sh")
        assert entry.mode == 0o755
        assert handle.extractfile(entry).read() == b"#!/bin/bash\nexit 0\n"
    assert len(manifest) == len(SOURCE_DIRS) + 1


def test_serving_container_has_read_only_weights_and_all_eight_devices():
    command = container_command(
        "example/ascend:image", "/root/stage2/run", "stage2-test"
    )
    assert f"-v {MODEL}:{MODEL}:ro" in command
    assert "-v /root/stage2/run:/ts:ro" in command
    assert "--network=host" in command
    assert "--pull=never" in command
    assert "npu-smi reset" not in command
    assert "deepseek-v41-dp" not in command
    for index in range(8):
        assert f"--device /dev/davinci{index}" in command


def test_preflight_failure_preserves_container_logs(monkeypatch):
    def remote_result(client, command, timeout):
        if command.startswith("tail -c"):
            return {"stdout": "", "stderr": ""}
        return {
            "stdout": "ModuleNotFoundError: native adapter\n",
            "stderr": "",
        }

    monkeypatch.setattr("run_v41_stage2.run_command", remote_result)
    log = read_service_log(object(), "/root/stage2/run", "stage2-test")
    assert log == "ModuleNotFoundError: native adapter\n"
