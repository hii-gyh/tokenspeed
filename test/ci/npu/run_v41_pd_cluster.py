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

"""Prepare or execute a four-node DeepSeek V4.1 Ascend 1P1D deployment.

Execution is deliberately opt-in: ``--allow-workload-replacement`` is required
because the decode nodes may have unrelated running vLLM containers. Running
containers are stopped normally, never removed (except our own TokenSpeed run
containers), and ``--success-action restore-original`` restores their original
running state after acceptance. ``--prepare-only`` performs no SSH operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import sys
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_cluster import Node, stop_ts_container  # noqa: E402
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
    health_status,
    read_service_log,
    timestamp_id,
)
from ssh_auth import DEFAULT_KEY_PATH, connect_client  # noqa: E402

TOPOLOGY = "PD 1P1D; each role DP2 attn/dense TP8 MoE TP1 EP16"
STAGE_ROOT = "/root/gyh_dev/tokenspeed_pd"
GATEWAY_IP = "192.168.11.2"
RUNTIME_PACKAGE = OUTPUT_ROOT / "tokenspeed-mooncake-ascend-runtime.tar.gz"
NON_PD_REFERENCE = OUTPUT_ROOT / "20261114T214831Z-48475-strict" / "summary.json"
TRACKED_PREFIXES = ("tokenspeed-v41-", "deepseek-v41-dp", "pd-pat-", "vllm-ascend-")

PREFILL_NODES = (
    Node(rank=0, host="39.156.164.107", private_ip="192.168.11.2"),
    Node(rank=1, host="39.156.164.108", private_ip="192.168.11.3"),
)
DECODE_NODES = (
    Node(rank=0, host="39.156.164.109", private_ip="192.168.11.10"),
    Node(rank=1, host="39.156.164.110", private_ip="192.168.11.5"),
)

PD_ENTRY_SCRIPT = r"""#!/usr/bin/env bash
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
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
export HCCL_SOCKET_IFNAME=bond1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENSPEED_LOG_LAYER_KIND_TIMING=1
# Keep servicer-side engines patient too; the headless gateway has its own
# --worker-wedge-secs setting below.
export SMG_TOKENSPEED_SERVICER_SILENCE_TIMEOUT_SECS=300
export MC_LOG_LEVEL=INFO
export NO_PROXY='*'
export no_proxy='*'
if [ -f /usr/local/Ascend/cann-9.2.1/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.1/bin/setenv.bash
elif [ -f /usr/local/Ascend/cann-9.2.0/bin/setenv.bash ]; then
  source /usr/local/Ascend/cann-9.2.0/bin/setenv.bash
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export TOKENSPEED_MOONCAKE_LIBRARY=/ts/mooncake/libtokenspeed_transfer_engine.so
export LD_LIBRARY_PATH=/ts/mooncake:${LD_LIBRARY_PATH:-}
test -r "$TOKENSPEED_MOONCAKE_LIBRARY"
if ldd "$TOKENSPEED_MOONCAKE_LIBRARY" | grep -q 'not found'; then
  ldd "$TOKENSPEED_MOONCAKE_LIBRARY" >&2
  exit 1
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
from tokenspeed.runtime.pd.host_peer import mooncake_library, mooncake_protocol
assert tokenspeed.__file__.startswith("/ts/")
assert tokenspeed_kernel.__file__.startswith("/ts/")
assert tokenspeed_kernel_npu.__file__.startswith("/ts/")
assert mooncake_library() == "/ts/mooncake/libtokenspeed_transfer_engine.so"
assert mooncake_protocol() == "ascend_direct"
print(f"[runtime] engine={tokenspeed_engine.__file__}", flush=True)
print(f"[runtime] scheduler={tokenspeed_scheduler.__file__}", flush=True)
print(f"[runtime] mooncake={mooncake_library()} protocol={mooncake_protocol()}", flush=True)
PY
mkdir -p @ENGRAM_DIR@
echo "[serve] topology=@TOPOLOGY@ role=@ROLE@ mode=@EXECUTION_MODE@" 2>&1 | tee /logs/serve.log
@GATEWAY_START@
exec /tmp/ts-native/bin/python -m tokenspeed.cli serve @MODEL@ \
  --headless \
  --served-model-name deepseek-v41 --device npu --trust-remote-code --quantization ascend \
  --nnodes 2 --node-rank @NODE_RANK@ --nprocs-per-node 8 --world-size 16 \
  --dist-init-addr @DIST_INIT_IP@:25000 \
  --attn-tp-size 8 --dense-tp-size 8 --moe-tp-size 1 \
  --data-parallel-size 2 --ep-size 16 \
  --enable-expert-parallel --all2all-backend ascend \
  --engram-host-table --engram-host-table-dir @ENGRAM_DIR@ \
  --engram-host-table-layout shared \
  --language-model-only \
  --attention-backend deepseek_v41 --dtype bfloat16 --kv-cache-dtype auto \
  @EAGER_FLAG@--disable-prefill-graph --disable-pdl --sampling-backend greedy \
  --yarn-ramp-mask-device cpu \
  --disable-kvstore @PREFIX_CACHE_FLAG@ --retraction-snapshot-ratio 0 \
  --max-model-len 1024 --max-num-seqs @MAX_NUM_SEQS@ \
  --gpu-memory-utilization 0.90 \
  --chunked-prefill-size 1024 \
  --disaggregation-mode @ROLE@ --disaggregation-transfer-backend mooncake \
  --disaggregation-layerwise-interval 0 \
  @BOOTSTRAP_FLAG@--data-parallel-address @GATEWAY_IP@ --data-parallel-rpc-port @HANDSHAKE_PORT@ \
  --zmq-engine-index 0 \
  --host @PRIVATE_IP@ --port @ENGINE_PORT@ 2>&1 | tee -a /logs/serve.log
"""


def gateway_start() -> str:
    return f"""/tmp/ts-native/bin/python -m smg launch \\
  --pd-disaggregation --backend tokenspeed \\
  --prefill tcp://{GATEWAY_IP}:31000 --decode tcp://{GATEWAY_IP}:31001 \\
  --model-path {MODEL} --tokenizer-path {MODEL} \\
  --model-alias deepseek-v41={MODEL} \\
  --prefill-policy power_of_two --decode-policy power_of_two \
  --power-of-two-load-metric requests \
  --prefill-max-inflight-requests-per-worker 4 \
  --prefill-queue-size 128 --prefill-queue-timeout-secs 300 \\
  --dp-aware \\
  --max-concurrent-requests 16 --queue-size 128 \\
  --queue-timeout-secs 1800 --request-timeout-secs 1800 \\
  --worker-wedge-secs 300 \\
  --log-level info --disable-retries --disable-circuit-breaker \\
  --prometheus-port 18422 --host 0.0.0.0 --port 8000 > /logs/gateway.log 2>&1 &
GATEWAY_PID=$!
trap 'kill \"$GATEWAY_PID\" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/liveness >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 \"$GATEWAY_PID\" 2>/dev/null; then
    tail -n 100 /logs/gateway.log >&2
    exit 1
  fi
  sleep 1
done
curl -fsS --max-time 3 http://127.0.0.1:8000/liveness >/dev/null"""


def render_pd_entry(
    role: str, node: Node, execution_mode: str, engram_dir: str, max_num_seqs: int
) -> str:
    """Render one role's complete entry with explicit topology and mode."""
    if role not in {"prefill", "decode"}:
        raise ValueError(f"unsupported PD role: {role}")
    is_prefill = role == "prefill"
    replacements = {
        "@PRIVATE_IP@": node.private_ip,
        "@DIST_INIT_IP@": (PREFILL_NODES if is_prefill else DECODE_NODES)[0].private_ip,
        "@ENGRAM_DIR@": engram_dir,
        "@MODEL@": MODEL,
        "@TOPOLOGY@": TOPOLOGY,
        "@MAX_NUM_SEQS@": str(max_num_seqs),
        "@ROLE@": role,
        "@EXECUTION_MODE@": execution_mode,
        "@NODE_RANK@": str(node.rank),
        "@EAGER_FLAG@": "--enforce-eager \\\n  " if execution_mode == "eager" else "",
        "@PREFIX_CACHE_FLAG@": (
            "--enable-prefix-caching" if is_prefill else "--disable-prefix-caching"
        ),
        "@BOOTSTRAP_FLAG@": (
            "--disaggregation-bootstrap-port 8998 \\\n  " if is_prefill else ""
        ),
        # SMG binds both role handshakes on the gateway host; every engine
        # dials that gateway rather than binding its own worker address.
        "@GATEWAY_IP@": GATEWAY_IP,
        "@HANDSHAKE_PORT@": "31000" if is_prefill else "31001",
        "@ENGINE_PORT@": "18346" if is_prefill else "18347",
        "@GATEWAY_START@": gateway_start() if is_prefill and node.rank == 0 else "",
    }
    script = PD_ENTRY_SCRIPT
    for marker, value in replacements.items():
        script = script.replace(marker, value)
    if re.search(r"@[A-Z_]+@", script):
        raise RuntimeError("unresolved PD entry-script marker")
    return script


def parse_containers(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        name, separator, status = line.partition("|")
        if separator and name:
            rows.append({"name": name, "status": status})
    return rows


def running_workloads(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only known replaceable serving containers, router first."""
    selected = [
        row
        for row in rows
        if row["status"].startswith("Up") and row["name"].startswith(TRACKED_PREFIXES)
    ]
    return sorted(
        selected,
        key=lambda row: (not row["name"].startswith("vllm-ascend-"), row["name"]),
    )


def validate_runtime_archive(path: Path) -> str:
    """Check the self-contained runtime and reject unsafe tar member names."""
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            normalized = Path(member.name)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or normalized.parts[0] != "mooncake"
            ):
                raise RuntimeError(f"unsafe Mooncake runtime member: {member.name}")
        if "mooncake/libtokenspeed_transfer_engine.so" not in archive.getnames():
            raise RuntimeError("Mooncake runtime lacks the TokenSpeed ABI library")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_original_workload(
    name: str,
    stop: Callable[[str], dict[str, Any]],
    status: Callable[[str], str],
) -> dict[str, Any]:
    """Stop an original workload without deleting its container or writable state."""
    result = stop(name)
    remaining = status(name)
    if result["rc"] != 0 and remaining == "running":
        raise RuntimeError(f"could not stop {name}: {result['stderr'][-1000:]}")
    return {"rc": result["rc"], "status": remaining, "gone": remaining != "running"}


def compare_pd_to_reference(
    pd_benchmark: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Compare PD with non-PD without hiding a changed load or token contract."""
    if reference.get("status") != "passed":
        raise RuntimeError("PD comparison requires an accepted non-PD reference")
    reference_benchmark = reference["benchmark"]
    load_keys = (
        "model",
        "protocol",
        "concurrency",
        "requests",
        "max_tokens",
        "repeat",
        "ignore_eos",
    )
    for candidate in (pd_benchmark, reference_benchmark):
        if candidate.get("engine") != "tokenspeed" or not candidate.get("ok"):
            raise RuntimeError("PD comparison requires passing TokenSpeed benchmarks")
        if not candidate.get("correctness", {}).get("ok"):
            raise RuntimeError("PD comparison requires passing correctness")
        for run in candidate["runs"]:
            if (
                not run.get("ok")
                or run["completion_tokens"] != run["expected_completion_tokens"]
            ):
                raise RuntimeError(
                    "PD comparison requires complete measured generations"
                )
    for key in load_keys:
        if pd_benchmark["load"].get(key) != reference_benchmark["load"].get(key):
            raise RuntimeError(f"PD/non-PD load mismatch: {key}")
    pd_prompt_tokens = {
        row["prompt_tokens"] for run in pd_benchmark["runs"] for row in run["rows"]
    }
    reference_prompt_tokens = {
        row["prompt_tokens"]
        for run in reference_benchmark["runs"]
        for row in run["rows"]
    }
    if pd_prompt_tokens != reference_prompt_tokens or len(pd_prompt_tokens) != 1:
        raise RuntimeError(
            f"PD/non-PD prompt-token mismatch: {pd_prompt_tokens} vs {reference_prompt_tokens}"
        )

    def ttft_mean(benchmark: dict[str, Any]) -> float:
        return statistics.mean(
            row["ttft_s"] for run in benchmark["runs"] for row in run["rows"]
        )

    metrics = {}
    for name in (
        "output_tok_per_s_mean",
        "ttft_p50_s",
        "ttft_p90_s",
        "tpot_mean_s",
        "tpot_p50_s",
        "tpot_p90_s",
    ):
        pd_value = pd_benchmark["summary"][name]
        reference_value = reference_benchmark["summary"][name]
        metrics[name] = {
            "pd": pd_value,
            "non_pd": reference_value,
            "pd_over_non_pd": pd_value / reference_value,
        }
    pd_ttft_mean = ttft_mean(pd_benchmark)
    reference_ttft_mean = ttft_mean(reference_benchmark)
    performance_benefit = (
        metrics["output_tok_per_s_mean"]["pd_over_non_pd"] > 1
        and metrics["tpot_mean_s"]["pd_over_non_pd"] < 1
    )
    return {
        "ok": True,
        "load": {key: pd_benchmark["load"][key] for key in load_keys},
        "prompt_tokens_per_request": next(iter(pd_prompt_tokens)),
        "pd_topology": pd_benchmark["topology"],
        "non_pd_topology": reference_benchmark["topology"],
        "metrics": metrics,
        "ttft_all_requests_mean_s": {
            "pd": pd_ttft_mean,
            "non_pd": reference_ttft_mean,
            "pd_over_non_pd": pd_ttft_mean / reference_ttft_mean,
        },
        "performance_benefit": performance_benefit,
        "performance_benefit_definition": "higher output throughput and lower mean TPOT than non-PD",
    }


def restore_original_workloads(
    stopped_original: dict[str, list[dict[str, str]]],
    start: Callable[[str, str], dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Attempt every restore and report all failures instead of losing partial evidence."""
    restores: dict[str, list[dict[str, Any]]] = {}
    failures: list[str] = []
    for key, rows in stopped_original.items():
        restores[key] = []
        for row in reversed(rows):
            result = start(key, row["name"])
            restores[key].append(
                {
                    "name": row["name"],
                    "rc": result["rc"],
                    "stderr": result["stderr"][-1000:],
                }
            )
            if result["rc"] != 0:
                failures.append(f"{key}/{row['name']}")
    return restores, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--success-action",
        choices=("keep-tokenspeed", "restore-original"),
        required=True,
    )
    parser.add_argument("--allow-workload-replacement", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--runtime-package", type=Path, default=RUNTIME_PACKAGE)
    parser.add_argument("--reference-summary", type=Path, default=NON_PD_REFERENCE)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--benchmark-timeout", type=int, default=5400)
    parser.add_argument("--benchmark-repeat", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and not args.allow_workload_replacement:
        parser.error(
            "execution requires --allow-workload-replacement; use --prepare-only for no SSH changes"
        )
    if args.max_num_seqs <= 0 or args.max_num_seqs % 2 != 0:
        parser.error(
            "--max-num-seqs must be positive and divisible by attention DP size 2"
        )
    return args


def all_roles() -> tuple[tuple[str, Node], ...]:
    return tuple(("prefill", node) for node in PREFILL_NODES) + tuple(
        ("decode", node) for node in DECODE_NODES
    )


def main() -> int:
    args = parse_args()
    run_id = timestamp_id()
    suffix = "pd1p1d-prepare" if args.prepare_only else "pd1p1d"
    outdir = OUTPUT_ROOT / f"{run_id}-{suffix}"
    outdir.mkdir(parents=True, exist_ok=False)
    runtime_sha256 = validate_runtime_archive(args.runtime_package)
    entries = {
        f"{role}-rank{node.rank}": render_pd_entry(
            role,
            node,
            args.execution_mode,
            f"/dev/shm/tokenspeed_v41_pd_{run_id.lower()}",
            args.max_num_seqs,
        )
        for role, node in all_roles()
    }
    for name, script in entries.items():
        (outdir / f"{name}-entry.sh").write_text(script, encoding="utf-8")
    summary: dict[str, Any] = {
        "status": "prepared" if args.prepare_only else "running",
        "run_id": run_id,
        "execution_mode": args.execution_mode,
        "max_num_seqs": args.max_num_seqs,
        "success_action": args.success_action,
        "topology": TOPOLOGY,
        "runtime_package": str(args.runtime_package),
        "runtime_sha256": runtime_sha256,
        "non_pd_reference": str(args.reference_summary),
        "roles": {
            role: [
                {"host": node.host, "private_ip": node.private_ip, "rank": node.rank}
                for node in nodes
            ]
            for role, nodes in (("prefill", PREFILL_NODES), ("decode", DECODE_NODES))
        },
        "workload_replacement_authorized": args.allow_workload_replacement,
    }
    print(f"PD stage output: {outdir}", flush=True)
    if args.prepare_only:
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[PASSED] local-only PD entries and runtime package validation; no SSH operations",
            flush=True,
        )
        return 0

    remote_stage = f"{STAGE_ROOT}/{run_id}"
    container_name = f"tokenspeed-v41-pd-{run_id.lower()}"
    engram_dir = f"/dev/shm/tokenspeed_v41_pd_{run_id.lower()}"
    clients: dict[str, paramiko.SSHClient] = {}
    launched: set[str] = set()
    original_running: dict[str, list[dict[str, str]]] = {}
    stopped_original: dict[str, list[dict[str, str]]] = {}
    healthy = False
    started = time.monotonic()
    summary.update(
        {"remote_stage": remote_stage, "container": container_name, "image": args.image}
    )
    exit_code = 0
    try:
        for role, node in all_roles():
            key = f"{role}-rank{node.rank}"
            client = connect_client(
                node.host,
                user="root",
                password=os.environ.get("TS_SSH_PW"),
                key_path=args.key_path,
                port=22,
                accept_new_host_key=args.accept_new_host_key,
            )
            clients[key] = client
            result = run_command(
                client,
                "npu-smi info; nerdctl images --format '{{.Repository}}:{{.Tag}}'; "
                "nerdctl ps -a --format '{{.Names}}|{{.Status}}'; "
                f"test -r {MODEL}/config.json",
                timeout=60,
            )
            (outdir / f"{key}-preflight.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            if result["rc"] != 0:
                raise RuntimeError(
                    f"{key} preflight failed: {result['stderr'][-1000:]}"
                )
            rows = parse_containers(result.get("containers", ""))
            if not rows:
                lines = [
                    line
                    for line in result["stdout"].splitlines()
                    if "|Up" in line or "|Exited" in line or "|Created" in line
                ]
                rows = parse_containers("\n".join(lines))
            original_running[key] = running_workloads(rows)
            if original_running[key] and not args.allow_workload_replacement:
                raise RuntimeError(
                    f"{key} has running workloads and replacement was not authorized"
                )
            if args.image not in set(result["stdout"].splitlines()):
                # The combined query also contains NPU text; inspect the explicit image list below.
                image_check = run_command(
                    client,
                    "nerdctl images --format '{{.Repository}}:{{.Tag}}'",
                    timeout=30,
                )
                if args.image not in set(image_check["stdout"].splitlines()):
                    raise RuntimeError(f"{key} lacks required image: {args.image}")

        for role, node in all_roles():
            key = f"{role}-rank{node.rank}"
            script = entries[key]
            archive, manifest = build_stage2_archive(ROOT, script)
            quoted_stage = shlex.quote(remote_stage)
            upload = run_with_input(
                clients[key],
                f"if [ -e {quoted_stage} ]; then exit 73; fi; "
                f"mkdir -p {quoted_stage}/logs && tar -xzf - -C {quoted_stage}",
                archive,
                timeout=180,
            )
            if upload["rc"] != 0:
                raise RuntimeError(
                    f"{key} source upload failed: {upload['stderr'][-1000:]}"
                )
            runtime_upload = run_with_input(
                clients[key],
                f"tar -xzf - -C {quoted_stage}",
                args.runtime_package.read_bytes(),
                timeout=180,
            )
            if runtime_upload["rc"] != 0:
                raise RuntimeError(
                    f"{key} runtime upload failed: {runtime_upload['stderr'][-1000:]}"
                )
            cache = run_command(
                clients[key], f"mkdir -p -- {shlex.quote(PIP_CACHE_HOST)}", timeout=30
            )
            if cache["rc"] != 0:
                raise RuntimeError(f"{key} pip cache setup failed")
            (outdir / f"{key}-source-manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        print(
            "[PASSED] source and verified Ascend Mooncake runtime uploaded to four nodes",
            flush=True,
        )

        (outdir / "original-running.json").write_text(
            json.dumps(original_running, indent=2) + "\n", encoding="utf-8"
        )
        summary["original_running"] = original_running
        for role, node in all_roles():
            key = f"{role}-rank{node.rank}"
            for row in original_running[key]:
                name = row["name"]
                stopped = stop_original_workload(
                    name,
                    lambda container: run_command(
                        clients[key],
                        f"nerdctl stop -t 60 {shlex.quote(container)}",
                        timeout=90,
                    ),
                    lambda container: container_status(clients[key], container),
                )
                if not stopped["gone"]:
                    raise RuntimeError(
                        f"{key} original container is still running: {name}"
                    )
                stopped_original.setdefault(key, []).append(row)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                npu = run_command(clients[key], "npu-smi info", timeout=30)
                if npu["rc"] == 0 and not busy_devices(npu["stdout"]):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"{key} NPUs did not become idle after workload stop"
                )
        summary["original_running"] = original_running
        print(
            "[PASSED] authorized workloads stopped normally; all four NPU sets isolated",
            flush=True,
        )

        launch_order = (
            (("prefill", PREFILL_NODES[0]),)
            + tuple(("decode", node) for node in reversed(DECODE_NODES))
            + (("prefill", PREFILL_NODES[1]),)
        )
        for role, node in launch_order:
            key = f"{role}-rank{node.rank}"
            launch = run_command(
                clients[key],
                container_command(args.image, remote_stage, container_name),
                timeout=60,
            )
            (outdir / f"{key}-launch.json").write_text(
                json.dumps(launch, indent=2) + "\n", encoding="utf-8"
            )
            if launch["rc"] != 0:
                raise RuntimeError(f"{key} launch failed: {launch['stderr'][-1000:]}")
            launched.add(key)
            print(f"[RUN] {key} launched", flush=True)

        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            statuses = {
                key: container_status(clients[key], container_name) for key in launched
            }
            summary["container_status"] = statuses
            exited = {
                key: status for key, status in statuses.items() if status != "running"
            }
            if exited:
                raise RuntimeError(
                    f"PD serving containers exited during startup: {exited}"
                )
            if health_status(clients["prefill-rank0"]):
                healthy = True
                break
            time.sleep(10)
        summary["startup_s"] = time.monotonic() - started
        if not healthy:
            summary["cleanup_policy"] = "deferred_while_model_may_be_loading"
            raise RuntimeError(
                f"PD service did not become ready within {args.startup_timeout}s"
            )
        print(
            f"[PASSED] four-node PD readiness ({summary['startup_s']:.1f}s)", flush=True
        )

        if not args.skip_benchmark:
            benchmark = (SCRIPT_DIR / "bench_v41_steady.py").read_bytes()
            command = (
                "python3 - --api http://127.0.0.1:8000 --model deepseek-v41 "
                "--engine tokenspeed --protocol stream "
                f"--topology {shlex.quote(TOPOLOGY)} --prompt-tag {shlex.quote(run_id)} "
                f"--concurrency 8 --requests 16 --max-tokens 128 --repeat {args.benchmark_repeat} "
                "--warmup-requests 16"
            )
            result = run_with_input(
                clients["prefill-rank0"],
                command,
                benchmark,
                timeout=args.benchmark_timeout,
            )
            (outdir / "benchmark.log").write_text(
                result["stdout"] + result["stderr"], encoding="utf-8"
            )
            lines = [
                line for line in result["stdout"].splitlines() if line.startswith("{")
            ]
            if not lines:
                raise RuntimeError("PD benchmark returned no JSON result")
            benchmark_json = json.loads(lines[-1])
            summary["benchmark"] = benchmark_json
            if result["rc"] != 0 or not benchmark_json.get("ok"):
                raise RuntimeError("PD correctness or fixed-load benchmark failed")
            reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
            comparison = compare_pd_to_reference(benchmark_json, reference)
            comparison["pd_source"] = str(outdir / "summary.json")
            comparison["non_pd_source"] = str(args.reference_summary)
            (outdir / "pd-vs-nonpd.json").write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
            summary["non_pd_comparison"] = comparison
            print(
                f"[PASSED] request-level PD correctness and fixed-load benchmark; "
                f"performance_benefit={comparison['performance_benefit']}",
                flush=True,
            )
        summary["status"] = "passed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILED] {summary['error']}", flush=True)
        exit_code = 1
    finally:
        should_cleanup = (
            summary["status"] == "failed" or args.success_action == "restore-original"
        )
        if summary.get("cleanup_policy") == "deferred_while_model_may_be_loading":
            should_cleanup = False
        cleanup_results: dict[str, Any] = {}
        all_gone = True
        for key in sorted(launched):
            log = read_service_log(clients[key], remote_stage, container_name)
            (outdir / f"{key}-serve.log").write_text(log, encoding="utf-8")
            if key == "prefill-rank0":
                gateway = run_command(
                    clients[key],
                    f"tail -c 240000 {shlex.quote(remote_stage + '/logs/gateway.log')} 2>/dev/null || true",
                    timeout=30,
                )
                (outdir / "gateway.log").write_text(
                    gateway["stdout"] + gateway["stderr"], encoding="utf-8"
                )
            if should_cleanup:
                cleanup_results[key] = stop_ts_container(clients[key], container_name)
                all_gone = all_gone and cleanup_results[key]["gone"]
            else:
                cleanup_results[key] = {
                    "status": container_status(clients[key], container_name)
                }
        summary["tokenspeed_cleanup"] = cleanup_results
        restores: dict[str, Any] = {}
        restore_failures: list[str] = []
        if should_cleanup and all_gone:
            restores, restore_failures = restore_original_workloads(
                stopped_original,
                lambda key, name: run_command(
                    clients[key], f"nerdctl start {shlex.quote(name)}", timeout=120
                ),
            )
        elif stopped_original:
            restores["status"] = "deferred_tokenspeed_process_still_present"
        if restore_failures:
            summary["status"] = "failed"
            restore_error = (
                f"original workload restore failed: {', '.join(restore_failures)}"
            )
            summary["restore_error"] = restore_error
            if "error" in summary:
                summary["error"] = f"{summary['error']}; {restore_error}"
            else:
                summary["error"] = restore_error
            exit_code = 1
        summary["original_restore"] = restores
        for client in clients.values():
            client.close()
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results: {outdir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
