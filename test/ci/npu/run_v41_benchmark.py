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

"""Benchmark an already healthy V4.1 serving endpoint without redeploying it."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_v41_stage1 import run_with_input  # noqa: E402
from run_v41_stage2 import OUTPUT_ROOT, timestamp_id  # noqa: E402
from ssh_auth import DEFAULT_KEY_PATH, connect_client  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument(
        "--engine", choices=("tokenspeed", "vllm-ascend"), required=True
    )
    parser.add_argument("--protocol", choices=("non-stream", "stream"), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--model", default="deepseek-v41")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--benchmark-timeout", type=int, default=5400)
    parser.add_argument("--source-tree")
    parser.add_argument("--model-path")
    parser.add_argument("--weight-revision")
    parser.add_argument("--server-info-path", default="/get_server_info")
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--accept-new-host-key", action="store_true")
    args = parser.parse_args(argv)
    if min(args.concurrency, args.requests, args.max_tokens, args.repeat) <= 0:
        parser.error("concurrency, requests, max-tokens, and repeat must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    return args


def benchmark_command(args: argparse.Namespace, prompt_tag: str) -> str:
    options = [
        "--api",
        f"http://127.0.0.1:{args.port}",
        "--model",
        args.model,
        "--engine",
        args.engine,
        "--protocol",
        args.protocol,
        "--topology",
        args.topology,
        "--prompt-tag",
        prompt_tag,
        "--concurrency",
        str(args.concurrency),
        "--requests",
        str(args.requests),
        "--max-tokens",
        str(args.max_tokens),
        "--repeat",
        str(args.repeat),
        "--timeout",
        str(args.timeout),
    ]
    for flag, value in (
        ("--source-tree", args.source_tree),
        ("--model-path", args.model_path),
        ("--weight-revision", args.weight_revision),
        ("--server-info-path", args.server_info_path),
    ):
        if value is not None:
            options.extend((flag, value))
    return "python3 - " + " ".join(shlex.quote(option) for option in options)


def final_json(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith(("{", "["))]
    if not lines:
        raise RuntimeError("benchmark returned no JSON result")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("benchmark JSON result is not an object")
    return result


def main() -> int:
    args = parse_args()
    run_id = timestamp_id()
    outdir = OUTPUT_ROOT / f"{run_id}-benchmark-{args.engine}-{args.protocol}"
    outdir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "host": args.host,
        "engine": args.engine,
        "protocol": args.protocol,
        "topology": args.topology,
        "redeployed": False,
    }
    client = None
    print(f"Benchmark output: {outdir}", flush=True)
    try:
        client = connect_client(
            args.host,
            user="root",
            password=os.environ.get("TS_SSH_PW"),
            key_path=args.key_path,
            port=22,
            accept_new_host_key=args.accept_new_host_key,
        )
        command = benchmark_command(args, run_id)
        result = run_with_input(
            client,
            command,
            (SCRIPT_DIR / "bench_v41_steady.py").read_bytes(),
            timeout=args.benchmark_timeout,
        )
        (outdir / "benchmark.log").write_text(
            result["stdout"] + result["stderr"], encoding="utf-8"
        )
        (outdir / "benchmark-result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        benchmark = final_json(result["stdout"])
        summary["benchmark"] = benchmark
        if result["rc"] != 0 or not benchmark.get("ok"):
            raise RuntimeError("serving correctness or fixed-load benchmark failed")
        summary["status"] = "passed"
        print("[PASSED] correctness and fixed-load benchmark", flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILED] {summary['error']}", flush=True)
        return 1
    finally:
        if client is not None:
            client.close()
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results: {outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
