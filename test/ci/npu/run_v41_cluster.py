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

"""Validate DeepSeek V4.1 on two Ascend nodes with reversible vLLM replacement."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_stage1 import (  # noqa: E402
    DEFAULT_IMAGE,
    ROOT,
    busy_devices,
    run_command,
    run_with_input,
)
from run_v41_stage2 import (  # noqa: E402
    MODEL,
    OUTPUT_ROOT,
    PIP_CACHE_HOST,
    build_stage2_archive,
    container_command,
    container_status,
    entry_script,
    health_status,
    read_service_log,
    timestamp_id,
)
from ssh_auth import DEFAULT_KEY_PATH, connect_client  # noqa: E402

TOPOLOGY = "DP2 attn/dense TP8 MoE TP1 EP16 wide-EP short-ctx"
STAGE_ROOT = "/root/gyh_dev/tokenspeed_cluster"
ORIGINAL_VLLM = "deepseek-v41-dp"


@dataclass(frozen=True)
class Node:
    rank: int
    host: str
    private_ip: str


NODES = (
    Node(rank=0, host="39.156.164.107", private_ip="192.168.11.2"),
    Node(rank=1, host="39.156.164.108", private_ip="192.168.11.3"),
)


def render_cluster_entry(node: Node, execution_mode: str, engram_dir: str) -> str:
    return entry_script(
        node.private_ip,
        "bond1",
        execution_mode,
        dist_init_ip=NODES[0].private_ip,
        nnodes=2,
        node_rank=node.rank,
        world_size=16,
        dp_size=2,
        ep_size=16,
        engram_dir=engram_dir,
        topology=TOPOLOGY,
    )


def with_debug_capture(script: str, node: Node) -> str:
    """Arm bounded rank-0 capture only after an explicit request driver opts in."""
    if node.rank != 0:
        return script
    exports = (
        "export TOKENSPEED_V41_DEBUG_CAPTURE_DIR=/logs/ts-capture\n"
        "export TOKENSPEED_V41_DEBUG_CAPTURE_STEPS=2\n"
        "export TOKENSPEED_V41_DEBUG_CAPTURE_ARM_FILE=/logs/capture.arm\n"
    )
    marker = "export PYTHONUNBUFFERED=1\n"
    if marker not in script:
        raise RuntimeError("TokenSpeed entry lacks capture environment marker")
    return script.replace(marker, marker + exports, 1)


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
            f"rank {node.rank} preflight command failed: {result['stderr'][-1000:]}"
        )
    if node.rank == 0 and busy_devices(result["stdout"]):
        raise RuntimeError(
            f"rank 0 NPUs are busy: {sorted(busy_devices(result['stdout']))}"
        )
    images = (
        set(result["images"].splitlines())
        if "images" in result
        else set(result["stdout"].splitlines())
    )
    if image not in images:
        raise RuntimeError(
            f"rank {node.rank} does not have required local image: {image}"
        )
    return result


def upload_node(
    client: paramiko.SSHClient,
    node: Node,
    remote_stage: str,
    archive: bytes,
) -> None:
    quoted_stage = shlex.quote(remote_stage)
    result = run_with_input(
        client,
        f"if [ -e {quoted_stage} ]; then exit 73; fi; "
        f"mkdir -p {quoted_stage}/logs && tar -xzf - -C {quoted_stage}",
        archive,
        timeout=180,
    )
    if result["rc"] != 0:
        raise RuntimeError(
            f"rank {node.rank} source upload failed: {result['stderr'][-1000:]}"
        )
    cache = run_command(
        client, f"mkdir -p -- {shlex.quote(PIP_CACHE_HOST)}", timeout=30
    )
    if cache["rc"] != 0:
        raise RuntimeError(
            f"rank {node.rank} pip cache setup failed: {cache['stderr'][-1000:]}"
        )


def stop_ts_container(client: paramiko.SSHClient, name: str) -> dict[str, Any]:
    status = container_status(client, name)
    if status == "missing":
        return {"status": status, "gone": True}
    stopped = {"rc": 0, "stdout": "", "stderr": ""}
    if status == "running":
        stopped = run_command(
            client, f"nerdctl stop -t 60 {shlex.quote(name)}", timeout=90
        )
    status_after_stop = container_status(client, name)
    removed = {"rc": 1, "stdout": "", "stderr": ""}
    if status_after_stop != "running":
        removed = run_command(client, f"nerdctl rm {shlex.quote(name)}", timeout=60)
    final_status = container_status(client, name)
    return {
        "initial_status": status,
        "stop_rc": stopped["rc"],
        "remove_rc": removed["rc"],
        "final_status": final_status,
        "gone": final_status == "missing",
        "stderr": (stopped["stderr"] + removed["stderr"])[-1000:],
    }


def original_state(client: paramiko.SSHClient) -> dict[str, str]:
    result = run_command(
        client,
        f"nerdctl inspect --format '{{{{.State.Status}}}} {{{{.Config.Image}}}}' {ORIGINAL_VLLM}",
        timeout=30,
    )
    if result["rc"] != 0:
        return {"name": ORIGINAL_VLLM, "state": "missing", "image": ""}
    state, _, image = result["stdout"].strip().partition(" ")
    return {"name": ORIGINAL_VLLM, "state": state, "image": image}


def restore_vllm(client: paramiko.SSHClient) -> dict[str, Any]:
    result = run_command(client, f"nerdctl start {ORIGINAL_VLLM}", timeout=120)
    return {"rc": result["rc"], "stderr": result["stderr"][-1000:]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--success-action", choices=("keep-tokenspeed", "restore-vllm"), required=True
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--benchmark-timeout", type=int, default=5400)
    parser.add_argument(
        "--bench-protocol", choices=("non-stream", "stream"), default="non-stream"
    )
    parser.add_argument("--bench-repeat", type=int, default=1)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--debug-capture",
        action="store_true",
        help="Enable the eager-only rank-0 tensor probe, armed by /logs/capture.arm",
    )
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug_capture and args.execution_mode != "eager":
        raise ValueError("--debug-capture requires --execution-mode eager")
    run_id = timestamp_id()
    outdir = OUTPUT_ROOT / f"{run_id}-cluster2"
    outdir.mkdir(parents=True, exist_ok=False)
    remote_stage = f"{STAGE_ROOT}/{run_id}"
    container_name = f"tokenspeed-v41-cluster2-{run_id.lower()}"
    engram_dir = f"/dev/shm/tokenspeed_v41_cluster_{run_id.lower()}"
    clients: dict[int, paramiko.SSHClient] = {}
    launched: set[int] = set()
    original_stopped = False
    healthy = False
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "execution_mode": args.execution_mode,
        "success_action": args.success_action,
        "debug_capture": args.debug_capture,
        "topology": TOPOLOGY,
        "hosts": [node.host for node in NODES],
        "private_ips": [node.private_ip for node in NODES],
        "remote_stage": remote_stage,
        "container": container_name,
        "image": args.image,
        "model_weights": "read_only_mount",
        "untouched_hosts": ["39.156.164.109", "39.156.164.110"],
    }
    print(f"Cluster stage output: {outdir}", flush=True)
    try:
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
        print(
            "[PASSED] two-node image, weight, and rank-0 isolation preflight",
            flush=True,
        )

        original = original_state(clients[1])
        summary["original_vllm"] = original
        if original["state"] not in {"running", "exited", "missing"}:
            raise RuntimeError(f"unexpected original vLLM state: {original['state']}")
        if original["state"] != "running":
            rank1_npu = run_command(clients[1], "npu-smi info", timeout=30)
            if rank1_npu["rc"] != 0 or busy_devices(rank1_npu["stdout"]):
                raise RuntimeError(
                    "rank 1 NPUs are busy while the original vLLM is not running"
                )

        for node in NODES:
            script = render_cluster_entry(node, args.execution_mode, engram_dir)
            if args.debug_capture:
                script = with_debug_capture(script, node)
            archive, manifest = build_stage2_archive(ROOT, script)
            upload_node(clients[node.rank], node, remote_stage, archive)
        (outdir / "source-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[PASSED] source and entries uploaded to both ranks ({len(manifest)} files each)",
            flush=True,
        )

        engram_absent = []
        for node in NODES:
            result = run_command(
                clients[node.rank], f"test ! -e {shlex.quote(engram_dir)}", timeout=30
            )
            engram_absent.append(result["rc"] == 0)
        if not all(engram_absent):
            raise RuntimeError("run-specific Engram directory already exists")

        if original["state"] == "running":
            stopped = run_command(
                clients[1], f"nerdctl stop -t 120 {ORIGINAL_VLLM}", timeout=150
            )
            summary["original_vllm_stop_rc"] = stopped["rc"]
            if stopped["rc"] != 0:
                raise RuntimeError(
                    f"could not stop original vLLM: {stopped['stderr'][-1000:]}"
                )
            original_stopped = True
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                npu = run_command(clients[1], "npu-smi info", timeout=30)
                if npu["rc"] == 0 and not busy_devices(npu["stdout"]):
                    break
                time.sleep(5)
            else:
                raise RuntimeError("rank 1 NPUs did not become idle after vLLM stopped")
        if original_stopped:
            print(
                "[PASSED] rank-1 vLLM stopped reversibly; both NPU sets isolated",
                flush=True,
            )
        else:
            print(
                f"[PASSED] rank-1 original vLLM state={original['state']} preserved; "
                "both NPU sets isolated",
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
                    f"rank {node.rank} container launch failed: {launch['stderr'][-1000:]}"
                )
            launched.add(node.rank)
            print(f"[RUN] TokenSpeed rank {node.rank} launched", flush=True)

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
                raise RuntimeError(f"serving container exited during startup: {exited}")
            if health_status(clients[0]):
                healthy = True
                break
            time.sleep(10)
        summary["startup_s"] = time.monotonic() - started
        if not healthy:
            summary["cleanup_policy"] = "deferred_while_model_may_be_loading"
            raise RuntimeError(
                f"service did not become ready within {args.startup_timeout}s"
            )
        print(
            f"[PASSED] two-node serving readiness ({summary['startup_s']:.1f}s)",
            flush=True,
        )

        if not args.skip_benchmark:
            benchmark = (SCRIPT_DIR / "bench_v41_steady.py").read_bytes()
            command = (
                "python3 - --api http://127.0.0.1:8000 --model deepseek-v41 "
                "--engine tokenspeed "
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
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILED] {summary['error']}", flush=True)
        return 1
    finally:
        should_cleanup = (
            summary["status"] == "failed" or args.success_action == "restore-vllm"
        )
        if summary.get("cleanup_policy") == "deferred_while_model_may_be_loading":
            should_cleanup = False
        cleanup_results: dict[str, Any] = {}
        all_gone = True
        for rank in sorted(launched):
            log = read_service_log(clients[rank], remote_stage, container_name)
            (outdir / f"rank{rank}-serve.log").write_text(log, encoding="utf-8")
            if should_cleanup:
                cleanup_results[str(rank)] = stop_ts_container(
                    clients[rank], container_name
                )
                all_gone = all_gone and cleanup_results[str(rank)]["gone"]
            else:
                cleanup_results[str(rank)] = {
                    "status": container_status(clients[rank], container_name)
                }
        summary["tokenspeed_cleanup"] = cleanup_results
        if original_stopped and (
            summary["status"] == "failed" or args.success_action == "restore-vllm"
        ):
            if all_gone:
                summary["vllm_restore"] = restore_vllm(clients[1])
            else:
                summary["vllm_restore"] = "deferred_tokenspeed_process_still_present"
        for client in clients.values():
            client.close()
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results: {outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
