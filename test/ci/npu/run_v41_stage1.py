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

"""Run an isolated TokenSpeed Python/NPU smoke test on one idle Ascend node.

This stage uploads only Python source to a unique directory. It does not pull an
image, install packages, load weights, or stop existing serving containers. The
short-lived container is removed automatically. Native engine/scheduler wheels
are deliberately outside this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko
from collect_v41_inventory import run_command
from ssh_auth import DEFAULT_KEY_PATH, connect_client

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = "quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc"
SOURCE_DIRS = (
    "python/tokenspeed",
    "tokenspeed-kernel/python/tokenspeed_kernel",
    "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu",
)
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024

INNER_SCRIPT = r"""set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/ts/python:/ts/tokenspeed-kernel/python:/ts/tokenspeed-kernel-npu/python
export TOKENSPEED_LOG_LAYER_KIND_TIMING=1
if [ -f /usr/local/Ascend/cann-9.2.1/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.1/bin/setenv.bash
elif [ -f /usr/local/Ascend/cann-9.2.0/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.0/bin/setenv.bash
fi
python3 -B - <<'PY'
import importlib
import importlib.metadata
import json
import os
import subprocess
import time

import torch
import torch_npu

imports = {}
for name in ("tokenspeed", "tokenspeed_kernel", "tokenspeed_kernel_npu"):
    module = importlib.import_module(name)
    imports[name] = module.__file__
    if not module.__file__ or not module.__file__.startswith("/ts/"):
        raise RuntimeError(f"unexpected import path for {name}: {module.__file__}")

native_status = "not_exercised"
if os.environ.get("TS_NATIVE_SMOKE") == "1":
    native_modules = {}
    for name in (
        "tokenspeed_engine",
        "tokenspeed_scheduler",
        "smg",
        "smg_grpc_proto",
        "smg_grpc_servicer",
    ):
        module = importlib.import_module(name)
        native_modules[name] = module.__file__
    import tokenspeed_engine

    host_binary = str(tokenspeed_engine.host_binary())
    linked = subprocess.run(
        ["ldd", host_binary], capture_output=True, text=True, timeout=30, check=False
    )
    linked_output = linked.stdout + linked.stderr
    if linked.returncode != 0 or "not found" in linked_output:
        raise RuntimeError(f"ts-host dynamic library check failed: {linked_output}")
    host_probe = subprocess.run(
        [host_binary], capture_output=True, text=True, timeout=30, check=False
    )
    probe_output = host_probe.stdout + host_probe.stderr
    if "usage: ts-host" not in probe_output:
        raise RuntimeError(f"ts-host process probe failed: {probe_output}")
    package_names = (
        "tokenspeed-engine",
        "tokenspeed-scheduler",
        "tokenspeed-smg",
        "tokenspeed-smg-grpc-proto",
        "tokenspeed-smg-grpc-servicer",
    )
    native_status = {
        "status": "passed",
        "modules": native_modules,
        "versions": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "host_binary": host_binary,
        "host_probe_rc": host_probe.returncode,
        "host_ldd_rc": linked.returncode,
    }

from tokenspeed.runtime.execution.layer_kind_timing import (
    LayerKindTimingRecorder,
)

torch_npu.npu.set_device(0)
device = torch.device("npu:0")
recorder = LayerKindTimingRecorder(rank=0)
timer = recorder.start(device, mode="eager", bs=1)
if timer is None:
    raise RuntimeError("the real recorder did not start an NPU event sample")
value = torch.ones((1024, 1024), dtype=torch.float16, device=device)
weight = torch.eye(1024, dtype=torch.float16, device=device)
with timer.scope("matmul", 0):
    for _ in range(8):
        value = value @ weight
recorder.finish(timer)

polls = 0
sample = None
deadline = time.monotonic() + 30.0
while sample is None:
    sample = timer.poll()
    if sample is not None:
        break
    polls += 1
    if time.monotonic() >= deadline:
        raise TimeoutError("NPU event did not complete before the timeout")
    time.sleep(0.001)
if sample["forward_ms"] <= 0 or sample["kinds_ms"].get("matmul", 0) <= 0:
    raise RuntimeError(f"invalid event durations: {sample}")
print("STAGE1_RESULT=" + json.dumps({
    "imports": imports,
    "device": "npu:0",
    "sample": sample,
    "polls_after_completion_check": polls,
    "native_engine_scheduler": native_status,
    "model_weights": "not_loaded",
}, sort_keys=True))
PY
"""

NATIVE_INSTALL_SCRIPT = r"""python3 -m venv --system-site-packages /tmp/ts-native
/tmp/ts-native/bin/python -m pip install --disable-pip-version-check --no-deps \
  --index-url https://pypi.org/simple \
  --extra-index-url https://lightseek.org/whl/nightly \
  tokenspeed-engine==0.1.0.post20261112 \
  tokenspeed-scheduler==0.1.54 \
  tokenspeed-smg==1.15.0.post20261110 \
  tokenspeed-smg-grpc-proto==0.4.28.post20261109 \
  tokenspeed-smg-grpc-servicer==0.15.0.post20261110
"""


def runtime_script(native: bool) -> str:
    """Select the pure-Python gate or the explicit native dependency gate."""
    if not native:
        return INNER_SCRIPT
    return INNER_SCRIPT.replace(
        "python3 -B - <<'PY'",
        f"{NATIVE_INSTALL_SCRIPT}/tmp/ts-native/bin/python -B - <<'PY'",
        1,
    )


def busy_devices(npu_info: str) -> set[int]:
    """Parse numeric NPU process rows, excluding device-information rows."""
    busy: set[int] = set()
    pattern = re.compile(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s+[^|]*\|", re.MULTILINE)
    for match in pattern.finditer(npu_info):
        busy.add(int(match.group(1)))
    return busy


def build_source_archive(root: Path) -> tuple[bytes, list[dict[str, Any]]]:
    """Build a deterministic-source manifest and a compressed upload archive."""
    buffer = io.BytesIO()
    manifest: list[dict[str, Any]] = []
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for source_dir in SOURCE_DIRS:
            source = root / source_dir
            if not source.is_dir():
                raise RuntimeError(f"missing local source directory: {source}")
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(root)
                if any(part == "__pycache__" for part in relative.parts):
                    continue
                if path.suffix in {".pyc", ".pyo"} or not path.is_file():
                    continue
                payload = path.read_bytes()
                manifest.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    payload = buffer.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"source archive is unexpectedly large: {len(payload)} bytes"
        )
    return payload, manifest


def container_command(
    image: str, remote_stage: str, name: str, device: int, native: bool
) -> str:
    """Create a one-shot container command scoped to one NPU and the source tree."""
    quoted_image = shlex.quote(image)
    quoted_stage = shlex.quote(remote_stage)
    return " ".join(
        [
            "nerdctl run --rm -i --pull=never",
            f"--name {shlex.quote(name)}",
            f"--network={'host' if native else 'none'} --ipc=host --privileged --shm-size=8g",
            f"--device /dev/davinci{device}",
            "--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc",
            "-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
            "-v /usr/local/dcmi:/usr/local/dcmi:ro",
            "-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
            "-v /etc/ascend_install.info:/etc/ascend_install.info:ro",
            "-v /etc/hccn.conf:/etc/hccn.conf:ro",
            f"-v {quoted_stage}:/ts:ro",
            f"-e ASCEND_RT_VISIBLE_DEVICES={device}",
            "-e PYTHONDONTWRITEBYTECODE=1",
            f"-e TS_NATIVE_SMOKE={1 if native else 0}",
            quoted_image,
            "bash -s",
        ]
    )


def run_with_input(
    client: paramiko.SSHClient, command: str, payload: bytes | str, timeout: int
) -> dict[str, Any]:
    """Run a remote stdin command and retain bounded output and its exit status."""
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    stdin.write(payload)
    stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return {
        "command": command,
        "rc": stdout.channel.recv_exit_status(),
        "stdout": out[-48_000:],
        "stderr": err[-24_000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="39.156.164.107")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--stage-root", default="/root/gyh_dev/tokenspeed_stage1")
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "v41-debug"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("TS_SSH_PW")
    key_path = args.key_path.expanduser()
    if not password and not key_path.is_file():
        print("neither TS_SSH_PW nor the dedicated SSH key is available", flush=True)
        return 2
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    outdir = args.output_root / f"{run_id}-stage1"
    outdir.mkdir(parents=True, exist_ok=False)
    (args.output_root / "latest-stage1-run.txt").write_text(
        f"{outdir}\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "run_id": run_id,
        "host": args.host,
        "device": args.device,
        "status": "running",
        "native_engine_scheduler": "not_exercised",
        "model_weights": "not_loaded",
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Stage 1 output: {outdir}", flush=True)

    remote_stage = f"{args.stage_root.rstrip('/')}/{run_id}"
    container_name = f"dsv41-ts-stage1-{run_id.lower()}"
    container_started = False
    client: paramiko.SSHClient | None = None
    try:
        client = connect_client(
            args.host,
            user=args.user,
            password=password,
            key_path=key_path,
            port=args.port,
            accept_new_host_key=args.accept_new_host_key,
        )
        preflight = {
            "npu": run_command(client, "npu-smi info"),
            "images": run_command(
                client, "nerdctl images --format '{{.Repository}}:{{.Tag}}'"
            ),
            "capacity": run_command(client, "df -h /root; free -h"),
        }
        (outdir / "preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
        )
        if preflight["npu"]["rc"] != 0:
            raise RuntimeError("npu-smi preflight failed")
        if args.device in busy_devices(preflight["npu"]["stdout"]):
            raise RuntimeError(f"NPU {args.device} already has running processes")
        images = set(preflight["images"]["stdout"].splitlines())
        if args.image not in images:
            raise RuntimeError(f"required local image is missing: {args.image}")
        print("[PASSED] isolated preflight", flush=True)

        archive, manifest = build_source_archive(ROOT)
        (outdir / "source-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        quoted_stage = shlex.quote(remote_stage)
        upload_command = (
            f"if [ -e {quoted_stage} ]; then printf 'stage already exists\\n' >&2; "
            f"exit 73; fi; mkdir -p {quoted_stage} && tar -xzf - -C {quoted_stage}"
        )
        upload = run_with_input(client, upload_command, archive, timeout=180)
        (outdir / "upload.json").write_text(
            json.dumps(upload, indent=2) + "\n", encoding="utf-8"
        )
        if upload["rc"] != 0:
            raise RuntimeError(f"source upload failed with rc={upload['rc']}")
        summary["remote_stage"] = remote_stage
        summary["source_files"] = len(manifest)
        print(f"[PASSED] source upload ({len(manifest)} files)", flush=True)

        container_started = True
        result = run_with_input(
            client,
            container_command(
                args.image, remote_stage, container_name, args.device, args.native
            ),
            runtime_script(args.native),
            timeout=600 if args.native else 240,
        )
        container_started = result["rc"] != 0
        (outdir / "runtime.log").write_text(
            result["stdout"] + result["stderr"], encoding="utf-8"
        )
        (outdir / "runtime.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        if result["rc"] != 0:
            raise RuntimeError(f"isolated runtime failed with rc={result['rc']}")
        marker = next(
            (
                line
                for line in result["stdout"].splitlines()
                if line.startswith("STAGE1_RESULT=")
            ),
            None,
        )
        if marker is None:
            raise RuntimeError("runtime succeeded without a structured result")
        parsed = json.loads(marker.removeprefix("STAGE1_RESULT="))
        summary.update(parsed)
        summary["status"] = "passed"
        print("[PASSED] TokenSpeed imports and NPU event recorder", flush=True)
        print(json.dumps(parsed, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        message = f"{type(exc).__name__}: {exc}"
        summary["error"] = (
            message.replace(password, "<redacted>") if password else message
        )
        print(f"[FAILED] {summary['error']}", flush=True)
        return 1
    finally:
        if container_started and client is not None:
            cleanup = run_command(
                client, f"nerdctl rm -f {shlex.quote(container_name)}"
            )
            (outdir / "cleanup.json").write_text(
                json.dumps(cleanup, indent=2) + "\n", encoding="utf-8"
            )
        if client is not None:
            client.close()
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results: {outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
