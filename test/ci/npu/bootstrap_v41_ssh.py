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

"""Install and verify a dedicated V4.1 automation key on the four lab hosts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collect_v41_inventory import run_command
from ssh_auth import DEFAULT_KEY_PATH, connect_client

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOSTS = [
    "39.156.164.107",
    "39.156.164.108",
    "39.156.164.109",
    "39.156.164.110",
]
KEY_COMMENT = "tokenspeed-v41-automation"


def ensure_key(key_path: Path) -> tuple[str, str]:
    """Create the dedicated key if absent, without reading the private key."""
    key_path = key_path.expanduser()
    public_path = key_path.with_suffix(key_path.suffix + ".pub")
    if key_path.exists() and public_path.exists():
        key_path.chmod(0o600)
    elif key_path.exists() or public_path.exists():
        raise RuntimeError(
            "only half of the dedicated keypair exists; refusing to overwrite it"
        )
    else:
        key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                KEY_COMMENT,
                "-f",
                str(key_path),
            ],
            check=True,
            timeout=30,
        )
        key_path.chmod(0o600)
    public_key = public_path.read_text(encoding="utf-8").strip()
    if not public_key.startswith("ssh-ed25519 ") or not public_key.endswith(
        KEY_COMMENT
    ):
        raise RuntimeError("dedicated public key has an unexpected format")
    fingerprint = subprocess.run(
        ["ssh-keygen", "-lf", str(public_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    return public_key, fingerprint


def install_command(public_key: str) -> str:
    """Append the public key once and verify its presence without printing it."""
    quoted = shlex.quote(public_key)
    return (
        'umask 077; mkdir -p "$HOME/.ssh"; touch "$HOME/.ssh/authorized_keys"; '
        'chmod 700 "$HOME/.ssh"; chmod 600 "$HOME/.ssh/authorized_keys"; '
        f'if ! grep -qxF -- {quoted} "$HOME/.ssh/authorized_keys"; then '
        f"printf '%s\\n' {quoted} >> \"$HOME/.ssh/authorized_keys\"; fi; "
        f'grep -qxF -- {quoted} "$HOME/.ssh/authorized_keys"'
    )


def install_and_verify(
    host: str,
    *,
    user: str,
    password: str,
    key_path: Path,
    public_key: str,
    port: int,
    accept_new_host_key: bool,
) -> dict[str, Any]:
    """Install through the existing password, then authenticate with key only."""
    client = connect_client(
        host,
        user=user,
        password=password,
        key_path=key_path,
        port=port,
        accept_new_host_key=accept_new_host_key,
    )
    try:
        installed = run_command(client, install_command(public_key), timeout=30)
    finally:
        client.close()
    if installed["rc"] != 0:
        return {"host": host, "status": "install_failed", "rc": installed["rc"]}
    client = connect_client(
        host,
        user=user,
        password=None,
        key_path=key_path,
        port=port,
        accept_new_host_key=accept_new_host_key,
    )
    try:
        verified = run_command(client, "true", timeout=15)
    finally:
        client.close()
    return {
        "host": host,
        "status": "passed" if verified["rc"] == 0 else "verification_failed",
        "install_rc": installed["rc"],
        "key_only_rc": verified["rc"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=DEFAULT_HOSTS)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "v41-debug"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("TS_SSH_PW")
    if not password:
        print("TS_SSH_PW is required only for this one-time bootstrap", flush=True)
        return 2
    key_path = args.key_path.expanduser()
    public_key, fingerprint = ensure_key(key_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.output_root / f"{stamp}-{os.getpid()}-ssh-bootstrap"
    outdir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "key_path": str(key_path),
        "key_fingerprint": fingerprint,
        "hosts": {},
        "status": "running",
    }
    print(f"SSH bootstrap output: {outdir}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        futures = {
            pool.submit(
                install_and_verify,
                host,
                user=args.user,
                password=password,
                key_path=key_path,
                public_key=public_key,
                port=args.port,
                accept_new_host_key=args.accept_new_host_key,
            ): host
            for host in args.hosts
        }
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "host": host,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}".replace(
                        password, "<redacted>"
                    ),
                }
            summary["hosts"][host] = result
            print(f"[{result['status'].upper()}] {host}", flush=True)
    summary["status"] = (
        "passed"
        if all(result["status"] == "passed" for result in summary["hosts"].values())
        else "failed"
    )
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SSH bootstrap: {summary['status']}", flush=True)
    print(f"Results: {outdir}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
