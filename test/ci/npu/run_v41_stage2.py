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

"""Start and validate an isolated single-node DeepSeek V4.1 TokenSpeed service."""

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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_stage1 import (  # noqa: E402
    DEFAULT_IMAGE,
    ROOT,
    build_source_archive,
    busy_devices,
    run_command,
    run_with_input,
)
from ssh_auth import DEFAULT_KEY_PATH, connect_client  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "v41-debug"

MODEL = "/mnt/weights/DeepSeek-V4.1-Flash-w8a8"
STAGE_ROOT = "/root/gyh_dev/tokenspeed_stage2"
PIP_CACHE_HOST = "/root/gyh_dev/tokenspeed_pip_cache"
ENGRAM_DIR = "/dev/shm/tokenspeed_v41_stage2_engram"

# Experiment knobs forwarded from the runner's environment into the container;
# empty values are dropped so the default path stays byte-identical.
PASS_ENV = {
    name: os.environ.get(name, "")
    for name in (
        "TOKENSPEED_V41_BF16_ONLY_WRITE",
        "TOKENSPEED_LOG_V41_ATTN",
    )
}
TOPOLOGY = "DP1 attn/dense TP8 MoE TP1 EP8 wide-EP short-ctx"

ENTRY_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
export PATH="/tmp/ts-native/bin:/usr/local/python3.12.16/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/ts/python:/ts/tokenspeed-kernel/python:/ts/tokenspeed-kernel-npu/python
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HCCL_IF_IP=@PRIVATE_IP@
export TOKENSPEED_HOST_IP=@PRIVATE_IP@
export HOST_IP=@PRIVATE_IP@
export GLOO_SOCKET_IFNAME=@NIC@
export TP_SOCKET_IFNAME=@NIC@
export HCCL_SOCKET_IFNAME=@NIC@
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENSPEED_LOG_LAYER_KIND_TIMING=1
if [ -f /usr/local/Ascend/cann-9.2.1/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.1/bin/setenv.bash
elif [ -f /usr/local/Ascend/cann-9.2.0/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.0/bin/setenv.bash
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
python3 -m venv --system-site-packages /tmp/ts-native
ENGINE_PACKAGE=tokenspeed-engine==0.1.0.post20261112
ENGINE_WHEEL=/root/.cache/pip/tokenspeed_engine-0.1.0.post20261112-cp312-cp312-manylinux_2_28_aarch64.whl
if [ -f "$ENGINE_WHEEL" ]; then
  echo "cace479db91ea665ae61bb86df0275fd628833ceef96e5e549409970fd8f3033  $ENGINE_WHEEL" | sha256sum -c -
  ENGINE_PACKAGE="$ENGINE_WHEEL"
fi
/tmp/ts-native/bin/python -m pip install --no-deps \
  --index-url https://pypi.org/simple \
  --extra-index-url https://lightseek.org/whl/nightly \
  "$ENGINE_PACKAGE" \
  tokenspeed-scheduler==0.1.54 \
  tokenspeed-smg==1.15.0.post20261110 \
  tokenspeed-smg-grpc-proto==0.4.28.post20261109 \
  tokenspeed-smg-grpc-servicer==0.15.0.post20261110
/tmp/ts-native/bin/python -m pip install \
  grpcio protobuf aiohttp dill einops fastapi msgspec 'openai>=2.24.1' \
  orjson packaging partial-json-parser pillow prometheus-client psutil \
  pydantic pyzmq requests setproctitle tiktoken sentencepiece uvicorn \
  python-multipart compressed-tensors peft openai-harmony ninja tqdm \
  'xgrammar==0.2.8' hf_transfer huggingface_hub modelscope pybind11 \
  'transformers==5.12.0' viztracer uv py-spy
/tmp/ts-native/bin/python - <<'PY'
import tokenspeed
import tokenspeed_engine
import tokenspeed_scheduler
import tokenspeed_kernel
import tokenspeed_kernel_npu
import torch
from tokenspeed_kernel_npu.ops.dsv41 import (
    npu_sparse_flash_mla,
    npu_sparse_flash_mla_metadata,
)
assert tokenspeed.__file__.startswith("/ts/")
assert tokenspeed_kernel.__file__.startswith("/ts/")
assert tokenspeed_kernel_npu.__file__.startswith("/ts/")
assert hasattr(tokenspeed_scheduler.Forward.Batch, "block_table_windows")
print(f"[runtime] tokenspeed={tokenspeed.__file__}", flush=True)
print(f"[runtime] engine={tokenspeed_engine.__file__}", flush=True)
print(f"[runtime] scheduler={tokenspeed_scheduler.__file__}", flush=True)
print(f"[runtime] sparse_mla={npu_sparse_flash_mla.__module__}", flush=True)
print(f"[runtime] sparse_mla_schema={torch.ops.cann_ops_transformer.sparse_flash_mla.default._schema}", flush=True)
print(f"[runtime] sparse_mla_metadata={npu_sparse_flash_mla_metadata.__module__}", flush=True)
print(f"[runtime] sparse_mla_metadata_schema={torch.ops.cann_ops_transformer.sparse_flash_mla_metadata.default._schema}", flush=True)
PY
mkdir -p @ENGRAM_DIR@
echo "[serve] topology=@TOPOLOGY@ mode=@EXECUTION_MODE@" 2>&1 | tee /logs/serve.log
exec /tmp/ts-native/bin/python -m tokenspeed.cli serve @MODEL@ \
  --served-model-name deepseek-v41 --device npu --trust-remote-code --quantization ascend \
  --nnodes @NNODES@ --node-rank @NODE_RANK@ --nprocs-per-node 8 --world-size @WORLD_SIZE@ \
  --dist-init-addr @DIST_INIT_IP@:25000 \
  --attn-tp-size 8 --dense-tp-size 8 --moe-tp-size 1 \
  --data-parallel-size @DP_SIZE@ --ep-size @EP_SIZE@ \
  --enable-expert-parallel --all2all-backend ascend \
  --engram-host-table --engram-host-table-dir @ENGRAM_DIR@ \
  --engram-host-table-layout shared \
  --language-model-only \
  --attention-backend deepseek_v41 --dtype bfloat16 --kv-cache-dtype auto \
  @EAGER_FLAG@--disable-prefill-graph --disable-pdl --sampling-backend greedy \
  --yarn-ramp-mask-device cpu \
  --disable-kvstore --enable-prefix-caching --retraction-snapshot-ratio 0 \
  --max-model-len 1024 --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --chunked-prefill-size 1024 \
  --allow-reload-forward \
  --host 0.0.0.0 --port 8000 2>&1 | tee -a /logs/serve.log
"""


def entry_script(
    private_ip: str,
    nic: str,
    execution_mode: str,
    *,
    dist_init_ip: str,
    nnodes: int,
    node_rank: int,
    world_size: int,
    dp_size: int,
    ep_size: int,
    engram_dir: str,
    topology: str,
) -> str:
    """Render the reviewed serving entry with explicit topology and execution mode."""
    replacements = {
        "@PRIVATE_IP@": private_ip,
        "@DIST_INIT_IP@": dist_init_ip,
        "@NIC@": nic,
        "@ENGRAM_DIR@": engram_dir,
        "@MODEL@": MODEL,
        "@TOPOLOGY@": topology,
        "@EXECUTION_MODE@": execution_mode,
        "@NNODES@": str(nnodes),
        "@NODE_RANK@": str(node_rank),
        "@WORLD_SIZE@": str(world_size),
        "@DP_SIZE@": str(dp_size),
        "@EP_SIZE@": str(ep_size),
        "@EAGER_FLAG@": "--enforce-eager \\\n  " if execution_mode == "eager" else "",
    }
    script = ENTRY_SCRIPT
    for marker, value in replacements.items():
        script = script.replace(marker, value)
    if re.search(r"@[A-Z_]+@", script):
        raise RuntimeError("unresolved entry-script marker")
    return script


def build_stage2_archive(root: Path, script: str) -> tuple[bytes, list[dict[str, Any]]]:
    """Append the generated entry to the immutable Python source archive."""
    source, manifest = build_source_archive(root)
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as src, tarfile.open(
        fileobj=output, mode="w:gz"
    ) as dst:
        for member in src.getmembers():
            payload = src.extractfile(member).read() if member.isfile() else None
            dst.addfile(member, io.BytesIO(payload) if payload is not None else None)
        payload = script.encode("utf-8")
        entry = tarfile.TarInfo("stage2_entry.sh")
        entry.size = len(payload)
        entry.mode = 0o755
        dst.addfile(entry, io.BytesIO(payload))
    manifest = manifest + [
        {
            "path": "stage2_entry.sh",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    return output.getvalue(), manifest


def container_command(image: str, remote_stage: str, name: str) -> str:
    """Create the isolated eight-NPU serving container with read-only weights."""
    devices = " ".join(f"--device /dev/davinci{index}" for index in range(8))
    return " ".join(
        [
            "nerdctl run -d --pull=never",
            f"--name {shlex.quote(name)}",
            "--network=host --ipc=host --privileged --shm-size=64g",
            devices,
            "--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc",
            "-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
            "-v /usr/local/dcmi:/usr/local/dcmi:ro",
            "-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
            "-v /etc/ascend_install.info:/etc/ascend_install.info:ro",
            "-v /etc/hccn.conf:/etc/hccn.conf:ro",
            f"-v {shlex.quote(remote_stage)}:/ts:ro",
            f"-v {shlex.quote(remote_stage + '/logs')}:/logs:rw",
            f"-v {shlex.quote(PIP_CACHE_HOST)}:/root/.cache/pip:rw",
            f"-v {MODEL}:{MODEL}:ro",
            "-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
            "-e PYTHONDONTWRITEBYTECODE=1",
            "-e PIP_CACHE_DIR=/root/.cache/pip",
            "-e PIP_DEFAULT_TIMEOUT=60",
            "-e PIP_RETRIES=10",
            *[
                f"-e {name}={shlex.quote(value)}"
                for name, value in PASS_ENV.items()
                if value
            ],
            shlex.quote(image),
            "bash /ts/stage2_entry.sh",
        ]
    )


def container_status(client: paramiko.SSHClient, name: str) -> str:
    result = run_command(
        client,
        f"nerdctl inspect --format '{{{{.State.Status}}}}' {shlex.quote(name)}",
        timeout=30,
    )
    return result["stdout"].strip() if result["rc"] == 0 else "missing"


def health_status(client: paramiko.SSHClient) -> bool:
    command = (
        "python3 -c 'import urllib.request; "
        'print(urllib.request.urlopen("http://127.0.0.1:8000/readiness", timeout=3).status)\''
    )
    result = run_command(client, command, timeout=10)
    return result["rc"] == 0 and result["stdout"].strip() == "200"


def read_service_log(client: paramiko.SSHClient, remote_stage: str, name: str) -> str:
    path = shlex.quote(remote_stage + "/logs/serve.log")
    result = run_command(
        client, f"tail -c 240000 {path} 2>/dev/null || true", timeout=30
    )
    if result["stdout"].strip():
        return result["stdout"]
    # Import, dependency, and native-op failures happen before the entry script
    # creates serve.log. Preserve those container logs before automatic cleanup.
    container = run_command(
        client, f"nerdctl logs --tail 2000 {shlex.quote(name)} 2>&1", timeout=30
    )
    return container["stdout"][-240000:] + container["stderr"][-240000:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="39.156.164.107")
    parser.add_argument("--private-ip", default="192.168.11.2")
    parser.add_argument("--nic", default="bond1")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=1800)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    return parser.parse_args()


def timestamp_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def main() -> int:
    args = parse_args()
    run_id = timestamp_id()
    outdir = OUTPUT_ROOT / f"{run_id}-stage2"
    outdir.mkdir(parents=True, exist_ok=False)
    remote_stage = f"{STAGE_ROOT}/{run_id}"
    container_name = f"tokenspeed-v41-stage2-{run_id.lower()}"
    password = os.environ.get("TS_SSH_PW")
    client = None
    container_started = False
    healthy = False
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "host": args.host,
        "execution_mode": args.execution_mode,
        "topology": TOPOLOGY,
        "model_weights": "read_only_mount",
        "remote_stage": remote_stage,
        "container": container_name,
        "image": args.image,
        "cleanup": "not_started",
    }
    print(f"Stage 2 output: {outdir}", flush=True)
    try:
        client = connect_client(
            args.host,
            user=args.user,
            password=password,
            key_path=args.key_path,
            port=args.port,
            accept_new_host_key=args.accept_new_host_key,
        )
        preflight = run_command(
            client,
            "npu-smi info; nerdctl images --format '{{.Repository}}:{{.Tag}}'; "
            f"test -r {MODEL}/config.json; "
            "python3 -c 'import socket; s=socket.socket(); "
            's.bind(("0.0.0.0", 8000)); s.close()\'; '
            f"test ! -e {ENGRAM_DIR}",
            timeout=60,
        )
        (outdir / "preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
        )
        if preflight["rc"] != 0:
            raise RuntimeError("stage2 preflight command failed")
        busy = busy_devices(
            preflight["npu_smi"] if "npu_smi" in preflight else preflight["stdout"]
        )
        if busy:
            raise RuntimeError(f"NPUs already have running processes: {sorted(busy)}")
        if args.image not in set(
            preflight["images"].splitlines()
            if "images" in preflight
            else preflight["stdout"].splitlines()
        ):
            raise RuntimeError(f"required local image is missing: {args.image}")
        print("[PASSED] isolated eight-NPU and weight preflight", flush=True)

        script = entry_script(
            args.private_ip,
            args.nic,
            args.execution_mode,
            dist_init_ip=args.private_ip,
            nnodes=1,
            node_rank=0,
            world_size=8,
            dp_size=1,
            ep_size=8,
            engram_dir=ENGRAM_DIR,
            topology=TOPOLOGY,
        )
        archive, manifest = build_stage2_archive(ROOT, script)
        (outdir / "source-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        quoted_stage = shlex.quote(remote_stage)
        upload_command = (
            f"if [ -e {quoted_stage} ]; then exit 73; fi; "
            f"mkdir -p {quoted_stage}/logs && tar -xzf - -C {quoted_stage}"
        )
        upload = run_with_input(client, upload_command, archive, timeout=180)
        if upload["rc"] != 0:
            raise RuntimeError(f"source upload failed: {upload['stderr'][-1000:]}")
        print(f"[PASSED] source and entry upload ({len(manifest)} files)", flush=True)

        cache = run_command(
            client, f"mkdir -p -- {shlex.quote(PIP_CACHE_HOST)}", timeout=30
        )
        if cache["rc"] != 0:
            raise RuntimeError(
                f"persistent pip cache setup failed: {cache['stderr'][-1000:]}"
            )

        launch = run_command(
            client,
            container_command(args.image, remote_stage, container_name),
            timeout=60,
        )
        (outdir / "launch.json").write_text(
            json.dumps(launch, indent=2) + "\n", encoding="utf-8"
        )
        if launch["rc"] != 0:
            raise RuntimeError(f"container launch failed: {launch['stderr'][-1000:]}")
        container_started = True
        print("[RUN] model load and health wait", flush=True)
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            status = container_status(client, container_name)
            if status != "running":
                summary["container_status"] = status
                raise RuntimeError(f"serving container exited during startup: {status}")
            if health_status(client):
                healthy = True
                break
            time.sleep(10)
        summary["startup_s"] = time.monotonic() - started
        if not healthy:
            summary["cleanup"] = "deferred_while_model_may_be_loading"
            raise RuntimeError(
                f"service did not become healthy within {args.startup_timeout}s"
            )
        print(f"[PASSED] serving health ({summary['startup_s']:.1f}s)", flush=True)

        if not args.skip_benchmark:
            benchmark = (SCRIPT_DIR / "bench_v41_steady.py").read_bytes()
            command = (
                "python3 - --api http://127.0.0.1:8000 --model deepseek-v41 "
                "--engine tokenspeed --protocol non-stream "
                f"--topology {shlex.quote(TOPOLOGY)} --prompt-tag {shlex.quote(run_id)} "
                "--concurrency 8 --requests 16 --max-tokens 128"
            )
            result = run_with_input(
                client, command, benchmark, timeout=args.benchmark_timeout
            )
            (outdir / "benchmark.log").write_text(
                result["stdout"] + result["stderr"], encoding="utf-8"
            )
            (outdir / "benchmark-result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            lines = [
                line for line in result["stdout"].splitlines() if line.startswith("{")
            ]
            if not lines:
                raise RuntimeError("benchmark returned no JSON result")
            benchmark_json = json.loads(lines[-1])
            summary["benchmark"] = benchmark_json
            if result["rc"] != 0 or not benchmark_json.get("ok"):
                raise RuntimeError("serving correctness or fixed-load benchmark failed")
            print("[PASSED] correctness and fixed-load benchmark", flush=True)
        summary["status"] = "passed"
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
        if client is not None:
            if container_started:
                summary["service_log"] = read_service_log(
                    client, remote_stage, container_name
                )
                status = container_status(client, container_name)
                summary["container_status"] = status
                if healthy or status != "running":
                    cleanup = run_command(
                        client,
                        f"nerdctl rm -f {shlex.quote(container_name)}",
                        timeout=60,
                    )
                    engram_cleanup = run_command(
                        client, f"rm -rf -- {ENGRAM_DIR}", timeout=60
                    )
                    summary["cleanup"] = {
                        "container_rc": cleanup["rc"],
                        "engram_rc": engram_cleanup["rc"],
                    }
                else:
                    summary["cleanup"] = "deferred_while_model_may_be_loading"
            client.close()
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (outdir / "serve.log").write_text(
            summary.get("service_log", ""), encoding="utf-8"
        )
        print(f"Results: {outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
