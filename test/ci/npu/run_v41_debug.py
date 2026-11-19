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

"""One-command local and remote V4.1 debug pipeline.

Local tests and read-only remote inventory always run when credentials are
available. NPU and load tests are gated by earlier checks. No remote files are
created, no code is deployed, and no serving process is restarted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko
from collect_v41_inventory import collect_host
from ssh_auth import DEFAULT_KEY_PATH, connect_client

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HOSTS = [
    "39.156.164.107",
    "39.156.164.108",
    "39.156.164.109",
    "39.156.164.110",
]


class DebugRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.password = os.environ.get("TS_SSH_PW")
        self.key_path = DEFAULT_KEY_PATH
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.outdir = args.output_root / f"{stamp}-{os.getpid()}"
        self.outdir.mkdir(parents=True, exist_ok=False)
        (args.output_root / "latest-run.txt").write_text(
            f"{self.outdir}\n", encoding="utf-8"
        )
        self.summary: dict[str, Any] = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.outdir),
            "stages": [],
        }

    def redact(self, value: str) -> str:
        if self.password:
            return value.replace(self.password, "<redacted>")
        return value

    def write_json(self, name: str, value: Any) -> None:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
        (self.outdir / name).write_text(self.redact(text), encoding="utf-8")

    def record(self, name: str, status: str, **details: Any) -> None:
        entry = {"name": name, "status": status, **details}
        self.summary["stages"].append(entry)
        self.write_json("summary.json", self.summary)
        print(f"[{status.upper()}] {name}", flush=True)

    def local(self, name: str, command: list[str], timeout: int = 1800) -> bool:
        print(f"[RUN] {name}", flush=True)
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = result.stdout
            rc = result.returncode
        except Exception as exc:
            output = f"{type(exc).__name__}: {exc}"
            rc = 1
        (self.outdir / f"{name}.log").write_text(self.redact(output), encoding="utf-8")
        self.record(name, "passed" if rc == 0 else "failed", rc=rc)
        return rc == 0

    def client(self, host: str) -> paramiko.SSHClient:
        return connect_client(
            host,
            user="root",
            password=self.password,
            key_path=self.key_path,
            accept_new_host_key=self.args.accept_new_host_key,
        )

    def remote_stdin(
        self, host: str, command: str, source: str, timeout: int
    ) -> dict[str, Any]:
        print(f"[RUN] remote stage on {host}", flush=True)
        client = self.client(host)
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            stdin.write(source)
            stdin.channel.shutdown_write()
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            rc = stdout.channel.recv_exit_status()
            return {"rc": rc, "stdout": out, "stderr": err}
        finally:
            client.close()


def healthy_hosts(inventory: dict[str, Any]) -> list[str]:
    healthy = []
    for host, result in inventory.items():
        queries = result.get("queries", {})
        health = queries.get("health", {})
        models = queries.get("models", {})
        if (
            health.get("rc") == 0
            and "HTTP:200" in health.get("stdout", "")
            and models.get("rc") == 0
        ):
            healthy.append(host)
    return healthy


def verified_heads(inventory: dict[str, Any]) -> dict[str, str]:
    heads = {}
    for host, result in inventory.items():
        queries = result.get("queries", {})
        query = queries.get("container_git_head") or queries.get("git_head") or {}
        head = query.get("stdout", "").strip()
        if query.get("rc") == 0 and head:
            heads[host] = head
    return heads


def verified_worktrees(inventory: dict[str, Any]) -> dict[str, dict[str, str]]:
    worktrees = {}
    for host, result in inventory.items():
        queries = result.get("queries", {})
        head_query = queries.get("container_git_head") or queries.get("git_head") or {}
        status_query = (
            queries.get("container_git_status") or queries.get("git_status") or {}
        )
        diff_query = queries.get("git_diff_stat") or {}
        head = head_query.get("stdout", "").strip()
        if head_query.get("rc") == 0 and head and status_query.get("rc") == 0:
            worktrees[host] = {
                "head": head,
                "status": status_query.get("stdout", ""),
                "diff_stat": diff_query.get("stdout", ""),
            }
    return worktrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=DEFAULT_HOSTS)
    parser.add_argument("--container", default="dsv41-ts-dp")
    parser.add_argument("--deploy-tree", default="/root/gyh_dev/deploy_tokenspeed_full")
    parser.add_argument("--container-tree", default="/ts")
    parser.add_argument("--model-path", default="/mnt/weights/DeepSeek-V4.1-Flash-w8a8")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "v41-debug"
    )
    parser.add_argument("--accept-new-host-key", action="store_true")
    parser.add_argument(
        "--full", action="store_true", help="Full regression and 3 benchmark runs"
    )
    parser.add_argument("--no-benchmark", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = DebugRun(args)
    print(f"Debug output: {run.outdir}", flush=True)
    local_ok = run.local(
        "environment",
        [
            sys.executable,
            "-c",
            (
                "import json,sys,pytest,paramiko; "
                "print(json.dumps({'python':sys.version,"
                "'pytest':pytest.__version__,'paramiko':paramiko.__version__},"
                "indent=2))"
            ),
        ],
        timeout=120,
    )
    full_env_ok = True
    if args.full:
        full_env_ok = run.local(
            "full_runtime_environment",
            [
                sys.executable,
                "-c",
                (
                    "import json,torch,tokenspeed,tokenspeed_kernel; "
                    "print(json.dumps({'torch':torch.__version__,"
                    "'tokenspeed':tokenspeed.__file__,"
                    "'tokenspeed_kernel':tokenspeed_kernel.__file__},indent=2))"
                ),
            ],
            timeout=120,
        )
    compile_ok = run.local(
        "compile",
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "python/tokenspeed/runtime/execution/layer_kind_timing.py",
            "python/tokenspeed/runtime/models/deepseek_v41.py",
            "python/tokenspeed/runtime/models/deepseek_v41_moe.py",
            "test/ci/npu/collect_v41_inventory.py",
            "test/ci/npu/check_npu_event_timer.py",
            "test/ci/npu/bench_v41_steady.py",
            "test/ci/npu/run_v41_debug.py",
        ],
        timeout=120,
    )
    behavior_tests = [
        "test/ci/npu/test_collect_v41_inventory.py",
        "test/ci/npu/test_run_v41_debug.py",
        "test/ci/npu/test_run_v41_stage1.py",
        "test/ci/npu/test_ssh_auth.py",
    ]
    if args.full and full_env_ok:
        behavior_tests.insert(0, "test/runtime/test_layer_kind_timing.py")
    elif args.full:
        run.record(
            "timer_behavior_tests",
            "skipped",
            reason="full runtime environment is unavailable",
        )
    behavior_ok = run.local(
        "behavior_tests",
        [sys.executable, "-m", "pytest", "-x", "-q", *behavior_tests],
    )
    regression_ok = True
    if args.full and behavior_ok and full_env_ok:
        regression_ok = run.local(
            "v41_regression_tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                "test/runtime/test_deepseek_v41_config.py",
                "test/runtime/test_deepseek_v41_wide_ep.py",
                "test/runtime/test_deepseek_v41_engram.py",
                "test/runtime/test_deepseek_v41_model.py",
            ],
            timeout=3600,
        )
    elif args.full:
        run.record(
            "v41_regression_tests",
            "skipped",
            reason="full runtime or behavior tests failed",
        )
    local_ok = local_ok and compile_ok and behavior_ok and full_env_ok and regression_ok

    if not run.password and not run.key_path.is_file():
        run.record(
            "remote_inventory",
            "failed",
            reason="neither TS_SSH_PW nor the dedicated SSH key is available",
        )
        run.summary["overall"] = "failed"
        run.write_json("summary.json", run.summary)
        return 1

    def collect(host: str) -> tuple[str, dict[str, Any]]:
        try:
            return host, collect_host(
                host,
                user="root",
                password=run.password,
                port=22,
                container=args.container,
                deploy_tree=args.deploy_tree,
                container_tree=args.container_tree,
                model_path=args.model_path,
                accept_new_host_key=args.accept_new_host_key,
            )
        except Exception as exc:
            return host, {"error": f"{type(exc).__name__}: {exc}"}

    inventory: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        for host, result in pool.map(collect, args.hosts):
            inventory[host] = result
    run.write_json("inventory.json", inventory)
    inventory_errors = [host for host, result in inventory.items() if "error" in result]
    run.record(
        "remote_inventory",
        "passed" if not inventory_errors else "failed",
        failed_hosts=inventory_errors,
    )

    healthy = healthy_hosts(inventory)
    heads = verified_heads(inventory)
    worktrees = verified_worktrees(inventory)
    fingerprints = {
        json.dumps(worktree, sort_keys=True) for worktree in worktrees.values()
    }
    same_code = len(worktrees) == len(args.hosts) and len(fingerprints) == 1
    baseline = {
        "healthy_hosts": healthy,
        "git_heads": heads,
        "worktrees": worktrees,
        "same_code_on_all_hosts": same_code,
        "benchmark_host": healthy[0] if healthy else None,
    }
    run.write_json("baseline.json", baseline)
    run.record(
        "baseline_validation",
        "passed" if healthy and same_code else "failed",
        healthy_hosts=healthy,
        same_code=same_code,
    )

    smoke_ok = False
    if not local_ok:
        run.record("npu_event_smoke", "skipped", reason="local validation failed")
    elif not healthy:
        run.record("npu_event_smoke", "skipped", reason="no healthy serving host")
    else:
        host = healthy[0]
        command = (
            f"nerdctl exec -i {shlex.quote(args.container)} " "python3 -B - --device 0"
        )
        try:
            result = run.remote_stdin(
                host,
                command,
                (SCRIPT_DIR / "check_npu_event_timer.py").read_text(encoding="utf-8"),
                timeout=120,
            )
        except Exception as exc:
            result = {"rc": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
        run.write_json("npu_event_smoke.json", {"host": host, **result})
        smoke_ok = result["rc"] == 0
        run.record("npu_event_smoke", "passed" if smoke_ok else "failed", host=host)

    if args.no_benchmark:
        run.record("fixed_load_benchmark", "skipped", reason="--no-benchmark")
    elif not local_ok or not healthy or not same_code or not smoke_ok:
        run.record(
            "fixed_load_benchmark",
            "skipped",
            reason="local, health, code-consistency, or NPU smoke gate failed",
        )
    else:
        host = healthy[0]
        repeat = 3 if args.full else 1
        command = (
            "python3 -B - --engine tokenspeed --protocol non-stream "
            "--topology 'debug single endpoint' "
            f"--prompt-tag {shlex.quote(run.outdir.name)} --repeat {repeat}"
        )
        try:
            result = run.remote_stdin(
                host,
                command,
                (SCRIPT_DIR / "bench_v41_steady.py").read_text(encoding="utf-8"),
                timeout=1800 * repeat,
            )
        except Exception as exc:
            result = {"rc": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
        run.write_json("fixed_load_benchmark_raw.json", {"host": host, **result})
        lines = [line for line in result["stdout"].splitlines() if line.strip()]
        if lines:
            try:
                benchmark = json.loads(lines[-1])
                run.write_json("fixed_load_benchmark.json", benchmark)
            except json.JSONDecodeError:
                run.record("benchmark_json", "failed", reason="invalid final JSON")
        run.record(
            "fixed_load_benchmark",
            "passed" if result["rc"] == 0 else "failed",
            host=host,
            repeat=repeat,
        )

    failed = [stage for stage in run.summary["stages"] if stage["status"] == "failed"]
    run.summary["overall"] = "failed" if failed else "passed"
    run.summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    run.write_json("summary.json", run.summary)
    print(f"Overall: {run.summary['overall']}", flush=True)
    print(f"Results: {run.outdir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
