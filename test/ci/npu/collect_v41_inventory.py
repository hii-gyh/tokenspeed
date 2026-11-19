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

"""Collect a read-only Ascend serving inventory over direct SSH connections.

Run from WSL with Paramiko installed. Hosts are explicit arguments; credentials
come from TS_SSH_PW or the dedicated automation key and are never serialized or
requested interactively. By default, OpenSSH known_hosts verification is
enforced.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import paramiko
from ssh_auth import connect_client

OUTPUT_LIMIT = 24_000

VERSION_PROBE = r"""
import importlib
import json
import os
import shutil
result = {}
for name in ("torch", "torch_npu", "tokenspeed", "tokenspeed_kernel", "tokenspeed_kernel_npu"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        result[name] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        result[name] = {
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
    result[key] = os.environ.get(key)
result["tools"] = {
    name: shutil.which(name) for name in ("msopgen", "msprof", "atc", "asys")
}
print(json.dumps(result))
"""


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[-OUTPUT_LIMIT:]


def run_command(
    client: paramiko.SSHClient, command: str, timeout: int = 60
) -> dict[str, Any]:
    """Run one query and retain its status and bounded text output."""
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    return {
        "command": command,
        "rc": stdout.channel.recv_exit_status(),
        "stdout": _clip(stdout.read().decode("utf-8", "replace")),
        "stderr": _clip(stderr.read().decode("utf-8", "replace")),
    }


def select_container(listing: str, preferred: str | None) -> str | None:
    """Select an existing serving container without treating it as TokenSpeed."""
    rows = []
    for line in listing.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) == 4:
            rows.append({"name": parts[0], "image": parts[1], "status": parts[2]})
    if not rows:
        return preferred
    if preferred is not None and any(row["name"] == preferred for row in rows):
        return preferred

    def candidates(
        matcher: Callable[[dict[str, str]], bool], *, running_only: bool
    ) -> list[str]:
        return sorted(
            row["name"]
            for row in rows
            if matcher(row) and (not running_only or row["status"].startswith("Up"))
        )

    def is_tokenspeed(row: dict[str, str]) -> bool:
        return "tokenspeed" in f"{row['name']} {row['image']}".lower()

    def is_v41_dp(row: dict[str, str]) -> bool:
        return row["name"] == "deepseek-v41-dp" or "dsv41-ts" in row["name"]

    def is_pd_decode(row: dict[str, str]) -> bool:
        return row["name"].startswith("pd-pat-decode-")

    for matcher, running_only in (
        (is_tokenspeed, True),
        (is_v41_dp, True),
        (is_pd_decode, True),
        (is_v41_dp, False),
        (is_tokenspeed, False),
    ):
        selected = candidates(matcher, running_only=running_only)
        if selected:
            return selected[0]
    return preferred


def collect_host(
    host: str,
    *,
    user: str,
    password: str | None,
    port: int,
    container: str | None,
    deploy_tree: str | None,
    container_tree: str | None,
    model_path: str | None,
    accept_new_host_key: bool,
) -> dict[str, Any]:
    """Collect one host without writing files or changing service state."""
    queries: dict[str, str] = {
        "identity": "hostname; ip -4 -o addr show; ip -4 route show",
        "containers": (
            "nerdctl ps -a --format " "'{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}'"
        ),
        "npu": "npu-smi info",
        "npu_map": "npu-smi info -m",
        "npu_arch_evidence": (
            "if command -v asys >/dev/null 2>&1; "
            "then asys info -r=hardware; "
            "else printf 'asys unavailable; architecture probe required\\n'; fi"
        ),
        "stage1_prerequisites": (
            "printf '%s\\n' '=== paths ==='; "
            "for path in /root/gyh_dev/token_speed_dsv41flash "
            "/root/gyh_dev/deploy_tokenspeed_full "
            "/root/gyh_dev/deploy_dsv41flash_dp; do "
            'if [ -e "$path" ]; then ls -ld "$path"; '
            "else printf 'missing %s\\n' \"$path\"; fi; done; "
            "printf '%s\\n' '=== ABI and pinned wheels ==='; "
            "ls -lh /root/gyh_dev/token_speed_dsv41flash/abi_libs/libstdc++.so.6 "
            "/root/gyh_dev/token_speed_dsv41flash/wheels/"
            "tokenspeed_engine-0.1.0-*.whl "
            "/root/gyh_dev/token_speed_dsv41flash/wheels/"
            "tokenspeed_scheduler-0.1.54-*.whl "
            "/root/gyh_dev/token_speed_dsv41flash/wheels/smg-1.15.0-*.whl "
            "/root/gyh_dev/token_speed_dsv41flash/wheels/"
            "smg_grpc_proto-0.4.28-*.whl "
            "/root/gyh_dev/token_speed_dsv41flash/wheels/"
            "smg_grpc_servicer-0.15.0-*.whl 2>&1; "
            "printf '%s\\n' '=== images ==='; "
            "nerdctl images --format '{{.Repository}}:{{.Tag}}'; "
            "printf '%s\\n' '=== capacity ==='; "
            "uname -m; df -h /root /dev/shm; free -h"
        ),
        "health": (
            "curl -fsS --max-time 5 -w '\\nHTTP:%{http_code}\\n' "
            "http://127.0.0.1:8000/health"
        ),
        "models": (
            "curl -fsS --max-time 5 -w '\\nHTTP:%{http_code}\\n' "
            "http://127.0.0.1:8000/v1/models"
        ),
    }
    if deploy_tree is not None:
        quoted_tree = shlex.quote(deploy_tree)
        queries["git_head"] = f"git -C {quoted_tree} rev-parse HEAD"
        queries["git_status"] = (
            f"git --no-optional-locks -C {quoted_tree} status --porcelain=v1"
        )
        queries["git_diff_stat"] = f"git -C {quoted_tree} diff --stat"
    if model_path is not None:
        quoted_model = shlex.quote(model_path)
        queries["model_config"] = f"cat {quoted_model}/config.json"
    client = connect_client(
        host,
        user=user,
        password=password,
        port=port,
        accept_new_host_key=accept_new_host_key,
    )
    try:
        results = {
            name: run_command(client, command) for name, command in queries.items()
        }
        selected_container = select_container(
            results["containers"]["stdout"], container
        )
        if selected_container is not None:
            quoted_container = shlex.quote(selected_container)
            container_queries = {
                "container_process": (
                    f"nerdctl inspect --format "
                    f"'{{{{.Config.Image}}}}|{{{{json .Config.Entrypoint}}}}|"
                    f"{{{{json .Config.Cmd}}}}|{{{{json .Mounts}}}}|"
                    f"{{{{.State.Status}}}}|{{{{.HostConfig.NetworkMode}}}}|"
                    f"{{{{json .NetworkSettings.Ports}}}}|{{{{json .Args}}}}' "
                    f"{quoted_container}"
                ),
                "versions": (
                    f"nerdctl exec {quoted_container} python3 -B -c "
                    f"{shlex.quote(VERSION_PROBE)}"
                ),
            }
            if container_tree is not None:
                quoted_container_tree = shlex.quote(container_tree)
                container_queries["container_git_head"] = (
                    f"nerdctl exec {quoted_container} "
                    f"git -C {quoted_container_tree} rev-parse HEAD"
                )
                container_queries["container_git_status"] = (
                    f"nerdctl exec {quoted_container} "
                    f"git --no-optional-locks -C {quoted_container_tree} "
                    f"status --porcelain=v1"
                )
            results.update(
                {
                    name: run_command(client, command)
                    for name, command in container_queries.items()
                }
            )
        return {
            "host": host,
            "selected_container": selected_container,
            "queries": results,
        }
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hosts", nargs="+", help="Directly reachable SSH hosts")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--container", help="Optional nerdctl container name")
    parser.add_argument("--deploy-tree", help="Optional host Git worktree path")
    parser.add_argument("--container-tree", help="Optional container Git worktree path")
    parser.add_argument("--model-path", help="Optional checkpoint directory")
    parser.add_argument(
        "--accept-new-host-key",
        action="store_true",
        help="Accept unknown SSH host keys for this lab inventory run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "tokenspeed_ascend_inventory.json",
        help="Local JSON output; defaults outside the repository",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("TS_SSH_PW")
    results: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        futures = {
            pool.submit(
                collect_host,
                host,
                user=args.user,
                password=password,
                port=args.port,
                container=args.container,
                deploy_tree=args.deploy_tree,
                container_tree=args.container_tree,
                model_path=args.model_path,
                accept_new_host_key=args.accept_new_host_key,
            ): host
            for host in args.hosts
        }
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except Exception as exc:
                results[host] = {"error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    return 0 if all("error" not in result for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
