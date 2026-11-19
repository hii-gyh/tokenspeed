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

"""Run the same-load two-node vLLM-Ascend comparison after TokenSpeed is stopped."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_cluster import NODES, Node, original_state, stop_ts_container  # noqa: E402
from run_v41_stage1 import (  # noqa: E402
    DEFAULT_IMAGE,
    busy_devices,
    run_command,
    run_with_input,
)
from run_v41_stage2 import (  # noqa: E402
    MODEL,
    OUTPUT_ROOT,
    container_status,
    read_service_log,
    timestamp_id,
)
from ssh_auth import DEFAULT_KEY_PATH, connect_client  # noqa: E402

TOPOLOGY = "DP2 attn/dense TP8 MoE TP1 EP16 wide-EP short-ctx"
STAGE_ROOT = "/root/gyh_dev/vllm_cluster"

ENTRY_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP=@PRIVATE_IP@
export VLLM_HOST_IP=@PRIVATE_IP@
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
export HCCL_SOCKET_IFNAME=bond1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=3600
if [ -f /usr/local/Ascend/cann-9.2.1/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.1/bin/setenv.bash
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
echo "[serve] engine=vllm-ascend topology=@TOPOLOGY@ mode=@EXECUTION_MODE@" 2>&1 | tee /logs/serve.log
exec vllm serve @MODEL@ \
  --served-model-name deepseek-v41 --trust-remote-code --quantization ascend \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-rank @NODE_RANK@ --data-parallel-address 192.168.11.2 \
  --data-parallel-rpc-port 25001 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 1024 --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --async-scheduling \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  @PROFILER_FLAG@\@EAGER_FLAG@--seed 1024 2>&1 | tee -a /logs/serve.log
"""


def health_status(client: paramiko.SSHClient) -> bool:
    command = (
        "python3 -c 'import urllib.request; "
        'print(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).status)\''
    )
    result = run_command(client, command, timeout=10)
    return result["rc"] == 0 and result["stdout"].strip() == "200"


def render_vllm_entry(
    node: Node, execution_mode: str, profiler_config: str | None = None
) -> str:
    if execution_mode not in {"eager", "graph"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    eager_flag = (
        "--enforce-eager " if execution_mode == "eager" else "--no-enforce-eager "
    )
    profiler_flag = (
        f"--profiler-config {shlex.quote(profiler_config)} \\\n  "
        if profiler_config
        else ""
    )
    return (
        ENTRY_SCRIPT.replace("@PRIVATE_IP@", node.private_ip)
        .replace("@NODE_RANK@", str(node.rank))
        .replace("@EXECUTION_MODE@", execution_mode)
        .replace("@EAGER_FLAG@", eager_flag)
        .replace("@PROFILER_FLAG@", profiler_flag)
        .replace("@MODEL@", MODEL)
        .replace("@TOPOLOGY@", TOPOLOGY)
    )


def with_debug_capture(script: str) -> str:
    """Load the staged adapter without modifying the image's installed source."""
    exports = (
        "export PYTHONPATH=/logs/vllm-debug:${PYTHONPATH:-}\n"
        "export VLLM_V41_DEBUG_CAPTURE_DIR=/logs/vllm-capture\n"
        "export VLLM_V41_DEBUG_CAPTURE_STEPS=2\n"
        "export VLLM_V41_DEBUG_CAPTURE_ARM_FILE=/logs/capture.arm\n"
    )
    marker = "export PYTHONUNBUFFERED=1\n"
    if marker not in script:
        raise RuntimeError("vLLM entry lacks capture environment marker")
    return script.replace(marker, marker + exports, 1)


def with_npu_profile(script: str) -> str:
    """Load the staged adapter and window a torch_npu.profiler over decode."""
    exports = (
        "export PYTHONPATH=/logs/vllm-debug:${PYTHONPATH:-}\n"
        "export VLLM_V41_PROFILE_DIR=/logs/vllm-profile\n"
        "export VLLM_V41_PROFILE_SKIP_FORWARDS=24\n"
        "export VLLM_V41_PROFILE_STEPS=4\n"
    )
    marker = "export PYTHONUNBUFFERED=1\n"
    if marker not in script:
        raise RuntimeError("vLLM entry lacks capture environment marker")
    return script.replace(marker, marker + exports, 1)


def upload_debug_adapter(
    client: paramiko.SSHClient, remote_stage: str
) -> dict[str, Any]:
    source = OUTPUT_ROOT / "deepseek_v41"
    names = ("model.py", "indexer.py", "debug_capture.py", "sitecustomize.py")
    archive = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in names:
            payload = (source / name).read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            member = tarfile.TarInfo(f"vllm-debug/{name}")
            member.size = len(payload)
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(payload))
    result = run_with_input(
        client,
        f"tar -xzf - -C {shlex.quote(remote_stage + '/logs')}",
        archive.getvalue(),
        timeout=60,
    )
    if result["rc"] != 0:
        raise RuntimeError(
            f"vLLM debug adapter upload failed: {result['stderr'][-1000:]}"
        )
    return {"path": "logs/vllm-debug", "sha256": hashes}


def container_command(image: str, remote_stage: str, name: str) -> str:
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
            f"-v {shlex.quote(remote_stage)}:/probe:ro",
            f"-v {shlex.quote(remote_stage + '/logs')}:/logs:rw",
            f"-v {MODEL}:{MODEL}:ro",
            "-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
            "-e PYTHONDONTWRITEBYTECODE=1",
            shlex.quote(image),
            "bash /probe/vllm_entry.sh",
        ]
    )


def preflight_node(
    client: paramiko.SSHClient, node: Node, image: str
) -> dict[str, Any]:
    port_check = ""
    if node.rank == 0:
        port_check = (
            "python3 -c 'import socket; s=socket.socket(); "
            's.bind(("0.0.0.0", 8000)); s.close()\'; '
        )
    result = run_command(
        client,
        "npu-smi info; nerdctl images --format '{{.Repository}}:{{.Tag}}'; "
        f"test -r {MODEL}/config.json; {port_check}",
        timeout=60,
    )
    if result["rc"] != 0:
        raise RuntimeError(
            f"rank {node.rank} preflight failed: {result['stderr'][-1000:]}"
        )
    if busy_devices(result["stdout"]):
        raise RuntimeError(
            f"rank {node.rank} NPUs are busy: {sorted(busy_devices(result['stdout']))}"
        )
    images = set(result["stdout"].splitlines())
    if image not in images:
        raise RuntimeError(f"rank {node.rank} lacks required local image: {image}")
    return result


def upload_entry(
    client: paramiko.SSHClient, remote_stage: str, script: str
) -> dict[str, Any]:
    payload = script.encode()
    result = run_with_input(
        client,
        f"if [ -e {shlex.quote(remote_stage)} ]; then exit 73; fi; "
        f"mkdir -p {shlex.quote(remote_stage + '/logs')} && "
        f"cat > {shlex.quote(remote_stage + '/vllm_entry.sh')}",
        payload,
        timeout=60,
    )
    if result["rc"] != 0:
        raise RuntimeError(f"entry upload failed: {result['stderr'][-1000:]}")
    return {
        "path": "vllm_entry.sh",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--success-action", choices=("keep-vllm", "stop-vllm"), required=True
    )
    parser.add_argument(
        "--bench-protocol", choices=("non-stream", "stream"), required=True
    )
    parser.add_argument("--bench-repeat", type=int, default=1)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--profile-npu",
        action="store_true",
        help="Load the staged adapter and window a torch_npu.profiler over decode steps",
    )
    parser.add_argument(
        "--debug-capture",
        action="store_true",
        help="Enable the staged eager-only tensor probe, armed by /logs/capture.arm",
    )
    parser.add_argument(
        "--profiler-config",
        default=None,
        help="Verbatim JSON passed to vllm serve --profiler-config, such as a bounded torch profile window",
    )
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--benchmark-timeout", type=int, default=5400)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bench_repeat <= 0:
        raise ValueError("--bench-repeat must be positive")
    if args.debug_capture and args.execution_mode != "eager":
        raise ValueError("--debug-capture requires --execution-mode eager")
    if args.profile_npu and args.execution_mode != "eager":
        raise ValueError("--profile-npu requires --execution-mode eager")
    run_id = timestamp_id()
    outdir = OUTPUT_ROOT / f"{run_id}-vllm-cluster2"
    outdir.mkdir(parents=True, exist_ok=False)
    remote_stage = f"{STAGE_ROOT}/{run_id}"
    container_name = f"vllm-v41-cluster2-{run_id.lower()}"
    clients: dict[int, paramiko.SSHClient] = {}
    launched: set[int] = set()
    healthy = False
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "engine": "vllm-ascend",
        "execution_mode": args.execution_mode,
        "success_action": args.success_action,
        "debug_capture": args.debug_capture,
        "topology": TOPOLOGY,
        "hosts": [node.host for node in NODES],
        "private_ips": [node.private_ip for node in NODES],
        "remote_stage": remote_stage,
        "container": container_name,
        "image": args.image,
        "model": MODEL,
        "comparison_config": {
            "max_model_len": 1024,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 1024,
            "gpu_memory_utilization": 0.90,
            "prefix_caching": True,
            "async_scheduling": True,
            "speculation": False,
            "kv_transfer": False,
            "engram": {"cpu_offload": True, "dp_shared_memory": True},
        },
    }
    print(f"vLLM cluster output: {outdir}", flush=True)
    try:
        manifests = {}
        for node in NODES:
            clients[node.rank] = connect_client(
                node.host,
                user="root",
                password=os.environ.get("TS_SSH_PW"),
                key_path=args.key_path,
                port=22,
                accept_new_host_key=args.accept_new_host_key,
            )
            result = preflight_node(clients[node.rank], node, args.image)
            (outdir / f"rank{node.rank}-preflight.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            entry = render_vllm_entry(node, args.execution_mode, args.profiler_config)
            if args.debug_capture:
                entry = with_debug_capture(entry)
            if args.profile_npu:
                entry = with_npu_profile(entry)
            manifests[str(node.rank)] = upload_entry(
                clients[node.rank], remote_stage, entry
            )
            if args.debug_capture or args.profile_npu:
                manifests[str(node.rank)]["debug_adapter"] = upload_debug_adapter(
                    clients[node.rank], remote_stage
                )
        (outdir / "entry-manifest.json").write_text(
            json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
        )
        summary["original_vllm"] = original_state(clients[1])
        print(
            "[PASSED] two-node idle-NPU, image, weight, and upload preflight",
            flush=True,
        )

        for node in reversed(NODES):
            launch = run_command(
                clients[node.rank],
                container_command(args.image, remote_stage, container_name),
                timeout=60,
            )
            (outdir / f"rank{node.rank}-launch.json").write_text(
                json.dumps(launch, indent=2) + "\n", encoding="utf-8"
            )
            if launch["rc"] != 0:
                raise RuntimeError(
                    f"rank {node.rank} launch failed: {launch['stderr'][-1000:]}"
                )
            launched.add(node.rank)
            print(f"[RUN] vLLM rank {node.rank} launched", flush=True)

        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            statuses = {
                rank: container_status(clients[rank], container_name)
                for rank in launched
            }
            summary["container_status"] = statuses
            exited = {
                rank: status for rank, status in statuses.items() if status != "running"
            }
            if exited:
                raise RuntimeError(f"vLLM container exited during startup: {exited}")
            if health_status(clients[0]):
                healthy = True
                break
            time.sleep(10)
        summary["startup_s"] = time.monotonic() - started
        if not healthy:
            summary["cleanup_policy"] = "deferred_while_model_may_be_loading"
            raise RuntimeError(
                f"vLLM did not become ready within {args.startup_timeout}s"
            )
        print(
            f"[PASSED] two-node vLLM readiness ({summary['startup_s']:.1f}s)",
            flush=True,
        )

        if not args.skip_benchmark:
            benchmark = (SCRIPT_DIR / "bench_v41_steady.py").read_bytes()
            command = (
                "python3 - --api http://127.0.0.1:8000 --model deepseek-v41 "
                "--engine vllm-ascend "
                f"--protocol {args.bench_protocol} "
                f"--topology {shlex.quote(TOPOLOGY)} --prompt-tag {shlex.quote(run_id)} "
                "--concurrency 8 --requests 16 "
                f"--max-tokens 128 --repeat {args.bench_repeat}"
            )
            result = run_with_input(
                clients[0], command, benchmark, timeout=args.benchmark_timeout
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
                raise RuntimeError("vLLM benchmark returned no JSON result")
            benchmark_json = json.loads(lines[-1])
            summary["benchmark"] = benchmark_json
            if result["rc"] != 0 or not benchmark_json.get("ok"):
                raise RuntimeError("vLLM correctness or fixed-load benchmark failed")
            print("[PASSED] vLLM correctness and fixed-load benchmark", flush=True)
        summary["status"] = "passed"
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILED] {summary['error']}", flush=True)
        return 1
    finally:
        should_cleanup = (
            summary["status"] == "failed" or args.success_action == "stop-vllm"
        )
        if summary.get("cleanup_policy") == "deferred_while_model_may_be_loading":
            should_cleanup = False
        cleanup_results: dict[str, Any] = {}
        for rank in sorted(launched):
            log = read_service_log(clients[rank], remote_stage, container_name)
            (outdir / f"rank{rank}-serve.log").write_text(log, encoding="utf-8")
            if should_cleanup:
                cleanup_results[str(rank)] = stop_ts_container(
                    clients[rank], container_name
                )
            else:
                cleanup_results[str(rank)] = {
                    "status": container_status(clients[rank], container_name)
                }
        summary["vllm_cleanup"] = cleanup_results
        for client in clients.values():
            client.close()
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results: {outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
