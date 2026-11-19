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

import io
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

from run_v41_stage1 import (  # noqa: E402
    INNER_SCRIPT,
    SOURCE_DIRS,
    build_source_archive,
    busy_devices,
    container_command,
    runtime_script,
)


def test_busy_devices_reads_only_npu_process_rows():
    npu_info = """
| 0     910B3               | OK             | 54.8     | 3458              |
| 0                         | 0000:81:00.0   | 1        | 0                 |
| 0       0                 | 1899337        | python   | 3320              |
| 1       0                 | No running processes found in NPU 1               |
| 2       1                 | 2000000        | vllm     | 5000              |
"""
    assert busy_devices(npu_info) == {0, 2}


def test_source_archive_contains_only_python_package_payload(tmp_path):
    for index, source_dir in enumerate(SOURCE_DIRS):
        package = tmp_path / source_dir
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
        cache = package / "__pycache__"
        cache.mkdir()
        (cache / "ignored.pyc").write_bytes(b"not source")
    archive, manifest = build_source_archive(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        names = handle.getnames()
    assert set(names) == {f"{source_dir}/__init__.py" for source_dir in SOURCE_DIRS}
    assert len(manifest) == len(SOURCE_DIRS)
    assert all(item["bytes"] > 0 and len(item["sha256"]) == 64 for item in manifest)


def test_runtime_container_is_one_shot_and_excludes_weights():
    image = "example/ascend:image"
    command = container_command(image, "/root/stage/run", "stage1-test", 0, False)
    assert "--rm" in command
    assert "--pull=never" in command
    assert "--device /dev/davinci0" in command
    assert "-v /root/stage/run:/ts:ro" in command
    assert "weights" not in command
    assert "pip install" not in command
    assert "npu-smi reset" not in command
    assert "deepseek-v41-dp" not in command
    assert image in command
    assert "--network=none" in command
    assert "TS_NATIVE_SMOKE=0" in command


def test_native_container_enables_downloads_without_mounting_weights():
    command = container_command(
        "example/ascend:image", "/root/stage/run", "native", 0, True
    )
    script = runtime_script(True)
    assert "--network=host" in command
    assert "TS_NATIVE_SMOKE=1" in command
    assert "weights" not in command
    assert "tokenspeed-engine==0.1.0.post20261112" in script
    assert "/tmp/ts-native/bin/python -B" in script
    assert "ts-host dynamic library check failed" in script
    assert runtime_script(False) == INNER_SCRIPT


def test_smoke_marks_native_engine_as_unverified():
    assert 'native_status = "not_exercised"' in INNER_SCRIPT
    assert '"model_weights": "not_loaded"' in INNER_SCRIPT
    assert "LayerKindTimingRecorder" in INNER_SCRIPT
    assert "pip install" not in INNER_SCRIPT
