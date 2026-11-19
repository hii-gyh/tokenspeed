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

"""Run the fixed TokenSpeed V4.1 correctness and decode benchmark.

The script uses only the standard library and writes one JSON document to
stdout, so the caller can execute it over SSH without creating a remote file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROMPT = (
    "Continue counting integers in order starting from 1, one number per line, "
    "do not stop until you reach many hundreds. Begin now:\n1\n2\n3\n"
)

_PACKAGE_NAMES = (
    "torch",
    "torch-npu",
    "tokenspeed",
    "tokenspeed-kernel",
    "tokenspeed-engine",
    "tokenspeed-scheduler",
    "vllm",
    "vllm-ascend",
)
_ENV_PREFIXES = ("ASCEND_", "CANN_", "HCCL_", "TOKENSPEED_")
_SENSITIVE_ENV_PARTS = ("PASSWORD", "TOKEN", "SECRET", "KEY", "COOKIE")


def run_command(command: list[str], timeout: float = 30.0) -> dict:
    """Run a read-only metadata query without invoking a shell."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "rc": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "rc": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def source_manifest(source_tree: Path | None) -> dict:
    """Identify the exact source, including the complete tracked diff."""
    if source_tree is None:
        return {"status": "not_provided"}
    root = source_tree.resolve()
    if not root.is_dir():
        return {"status": "missing", "path": str(root)}
    commands = {
        "head": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "status": [
            "git",
            "--no-optional-locks",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
        ],
        "diff_stat": ["git", "-C", str(root), "diff", "--stat", "HEAD"],
        "diff": ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        "untracked": [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
        ],
    }
    results = {name: run_command(command) for name, command in commands.items()}
    if results["head"]["rc"] != 0:
        return {"status": "unavailable", "path": str(root), "queries": results}
    untracked = []
    if results["untracked"]["rc"] == 0:
        for relative in results["untracked"]["stdout"].splitlines():
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                untracked.append({"path": relative, "status": "not_readable_in_tree"})
                continue
            payload = candidate.read_bytes()
            untracked.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return {
        "status": "ok",
        "path": str(root),
        "head": results["head"]["stdout"].strip(),
        "worktree_status": results["status"]["stdout"],
        "diff_stat": results["diff_stat"]["stdout"],
        "diff": results["diff"]["stdout"],
        "diff_sha256": hashlib.sha256(
            results["diff"]["stdout"].encode("utf-8")
        ).hexdigest(),
        "untracked": untracked,
        "queries": results,
    }


def runtime_manifest() -> dict:
    """Record benchmark-process packages and non-secret serving variables."""
    packages = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
        except Exception as exc:
            packages[name] = f"{type(exc).__name__}: {exc}"
    environment = {}
    for name, value in sorted(os.environ.items()):
        if not name.startswith(_ENV_PREFIXES):
            continue
        environment[name] = (
            "<redacted>"
            if any(part in name.upper() for part in _SENSITIVE_ENV_PARTS)
            else value
        )
    return {
        "scope": "benchmark_process",
        "packages": packages,
        "environment": environment,
    }


def weight_manifest(model_path: Path | None, revision: str | None) -> dict:
    """Hash small checkpoint configs without reading multi-GB weight shards."""
    if model_path is None and revision is None:
        return {"status": "not_provided"}
    result = {
        "status": "ok",
        "path": str(model_path) if model_path else None,
        "revision": revision,
    }
    if model_path is None:
        return result
    root = model_path.resolve()
    if not root.is_dir():
        return {**result, "status": "missing"}
    configs = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        path = root / name
        if not path.is_file():
            configs[name] = None
            continue
        payload = path.read_bytes()
        configs[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    result["configs"] = configs
    return result


def server_info(api: str, path: str | None, timeout: float) -> dict:
    if path is None:
        return {"status": "not_requested"}
    url = f"{api.rstrip('/')}/{path.lstrip('/')}"
    try:
        with urllib.request.urlopen(url, timeout=min(timeout, 30.0)) as response:
            text = response.read().decode()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        return {"status": "ok", "url": url, "info": payload}
    except Exception as exc:
        return {"status": "error", "url": url, "error": f"{type(exc).__name__}: {exc}"}


def build_run_manifest(args: argparse.Namespace) -> dict:
    return {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "source": source_manifest(args.source_tree),
        "runtime": runtime_manifest(),
        "weights": weight_manifest(args.model_path, args.weight_revision),
        "server": server_info(args.api, args.server_info_path, args.timeout),
    }


def post(api: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{api}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post_stream(
    api: str, payload: dict, timeout: float
) -> tuple[list[tuple[float, str]], dict]:
    """Collect text-chunk timestamps and final usage from an SSE completion."""
    request = urllib.request.Request(
        f"{api}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[tuple[float, str]] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            text = "".join(
                choice.get("text") or "" for choice in event.get("choices") or []
            )
            if text:
                chunks.append((time.perf_counter(), text))
    return chunks, usage


def stream_metrics(
    started: float,
    chunks: list[tuple[float, str]],
    completion_tokens: int,
    elapsed_s: float,
) -> dict[str, float | int | None]:
    """Calculate TTFT, token-normalized TPOT, and chunk-arrival diagnostics."""
    if not chunks:
        return {
            "ttft_s": None,
            "tpot_mean_s": None,
            "itl_p50_s": None,
            "itl_p90_s": None,
            "streamed_chunks": 0,
        }
    gaps = [current[0] - previous[0] for previous, current in zip(chunks, chunks[1:])]
    token_gaps = completion_tokens - 1
    tpot_mean_s = None
    if token_gaps > 0:
        decode_elapsed_s = started + elapsed_s - chunks[0][0]
        tpot_mean_s = decode_elapsed_s / token_gaps
    return {
        "ttft_s": chunks[0][0] - started,
        "tpot_mean_s": tpot_mean_s,
        "itl_p50_s": percentile(sorted(gaps), 50) if gaps else None,
        "itl_p90_s": percentile(sorted(gaps), 90) if gaps else None,
        "streamed_chunks": len(chunks),
    }


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def request_prompt(run: int, index: int, prompt_tag: str) -> str:
    return f"[tag={prompt_tag},run={run},request={index}] {PROMPT}"


def one_request(
    api: str,
    model: str,
    run: int,
    index: int,
    max_tokens: int,
    timeout: float,
    protocol: str,
    prompt_tag: str,
) -> dict:
    started = time.perf_counter()
    payload = {
        "model": model,
        "prompt": request_prompt(run, index, prompt_tag),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    if protocol == "stream":
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        chunks, usage = post_stream(api, payload, timeout)
    else:
        response = post(api, payload, timeout)
        usage = response.get("usage") or {}
        chunks = []
    elapsed = time.perf_counter() - started
    completion_tokens = int(usage.get("completion_tokens") or 0)
    result = {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
    }
    if protocol == "stream":
        result.update(stream_metrics(started, chunks, completion_tokens, elapsed))
    return result


def percentile(values: list[float], percentile_value: int) -> float:
    position = int(round((percentile_value / 100) * (len(values) - 1)))
    return values[min(len(values) - 1, max(0, position))]


def aggregate_stream_metrics(rows: list[dict]) -> dict[str, float]:
    """Aggregate per-request TTFT and TPOT without mixing SSE chunk counts with tokens."""
    ttfts = sorted(
        float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None
    )
    tpots = sorted(
        float(row["tpot_mean_s"]) for row in rows if row.get("tpot_mean_s") is not None
    )
    result: dict[str, float] = {}
    if ttfts:
        result["ttft_p50_s"] = percentile(ttfts, 50)
        result["ttft_p90_s"] = percentile(ttfts, 90)
    if tpots:
        result["tpot_mean_s"] = statistics.mean(tpots)
        result["tpot_p50_s"] = percentile(tpots, 50)
        result["tpot_p90_s"] = percentile(tpots, 90)
    return result


def measured_run(
    api: str,
    model: str,
    run: int,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()
    rows = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                one_request,
                api,
                model,
                run,
                index,
                args.max_tokens,
                args.timeout,
                args.protocol,
                args.prompt_tag,
            )
            for index in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors += 1
                rows.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
    wall_s = time.perf_counter() - started
    succeeded = [row for row in rows if "error" not in row]
    if not succeeded:
        return {
            "run": run,
            "ok": False,
            "errors": errors,
            "wall_s": wall_s,
            "rows": rows,
        }

    latencies = sorted(row["elapsed_s"] for row in succeeded)
    completion_tokens = sum(row["completion_tokens"] for row in succeeded)
    prompt_tokens = sum(row["prompt_tokens"] for row in succeeded)
    expected = args.requests * args.max_tokens
    fill_ratio = completion_tokens / expected
    output_tok_per_s = completion_tokens / wall_s if wall_s else 0.0
    stream_ok = args.protocol != "stream" or all(
        row.get("streamed_chunks", 0) > 0 for row in succeeded
    )
    result = {
        "run": run,
        "ok": errors == 0 and fill_ratio >= 0.9 and stream_ok,
        "errors": errors,
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "expected_completion_tokens": expected,
        "fill_ratio": fill_ratio,
        "latency_avg_s": statistics.mean(latencies),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p90_s": percentile(latencies, 90),
        "output_tok_per_s": output_tok_per_s,
        "total_tok_per_s": (completion_tokens + prompt_tokens) / wall_s,
        "rows": rows,
    }
    if args.protocol == "stream":
        result.update(aggregate_stream_metrics(succeeded))
    progress(
        f"run={run} ok={result['ok']} fill={fill_ratio:.4f} "
        f"output_tok_per_s={output_tok_per_s:.3f}"
    )
    return result


def paris_correctness(
    api: str, model: str, timeout: float, prompt_tag: str
) -> dict[str, Any]:
    request_id = int.from_bytes(
        hashlib.sha256(prompt_tag.encode("utf-8")).digest()[:4], "big"
    )
    response = post(
        api,
        {
            "model": model,
            "prompt": f"Request {request_id}: The capital of France is",
            "max_tokens": 32,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        min(timeout, 180.0),
    )
    text = (response.get("choices") or [{}])[0].get("text") or ""
    return {
        "ok": "paris" in text.lower(),
        "text": text[:200],
        "usage": response.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v41")
    parser.add_argument(
        "--engine", choices=("tokenspeed", "vllm-ascend"), required=True
    )
    parser.add_argument("--protocol", choices=("non-stream", "stream"), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--prompt-tag", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--weight-revision")
    parser.add_argument("--server-info-path")
    args = parser.parse_args()
    if (
        min(
            args.concurrency,
            args.requests,
            args.max_tokens,
            args.repeat,
            args.warmup_requests,
        )
        <= 0
    ):
        parser.error(
            "concurrency, requests, max-tokens, repeat, and warmup-requests must be positive"
        )

    if args.warmup_max_tokens is not None and args.warmup_max_tokens <= 0:
        parser.error("warmup-max-tokens must be positive")
    warmup_max_tokens = (
        args.max_tokens if args.warmup_max_tokens is None else args.warmup_max_tokens
    )

    output = {
        "engine": args.engine,
        "topology": args.topology,
        "load": {
            "api": args.api,
            "model": args.model,
            "protocol": args.protocol,
            "prompt_tag": args.prompt_tag,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "warmup_requests": args.warmup_requests,
            "warmup_max_tokens": warmup_max_tokens,
            "ignore_eos": True,
            "source_tree": str(args.source_tree) if args.source_tree else None,
            "model_path": str(args.model_path) if args.model_path else None,
            "weight_revision": args.weight_revision,
            "server_info_path": args.server_info_path,
        },
        "manifest": build_run_manifest(args),
        "warmup": {},
        "correctness": {},
        "runs": [],
    }
    try:
        if args.warmup_requests == 1:
            warmup_started = time.perf_counter()
            warmup_response = post(
                args.api,
                {
                    "model": args.model,
                    "prompt": f"[tag={args.prompt_tag},warmup] Hi",
                    "max_tokens": 4,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                min(args.timeout, 300.0),
            )
            output["warmup"] = {
                "ok": True,
                "elapsed_s": time.perf_counter() - warmup_started,
                "usage": warmup_response.get("usage"),
            }
        else:
            # Cover the same concurrent prefill/decode shapes as the measured
            # load without charging first-use compilation to a steady-state run.
            warmup_args = argparse.Namespace(**vars(args))
            warmup_args.requests = args.warmup_requests
            warmup_args.max_tokens = warmup_max_tokens
            batch = measured_run(args.api, args.model, -1, warmup_args)
            output["warmup"] = {
                "ok": batch["ok"],
                "elapsed_s": batch["wall_s"],
                "requests": args.warmup_requests,
                "batch": batch,
            }
            if not batch["ok"]:
                raise RuntimeError("concurrent warmup did not complete every request")
        output["correctness"] = paris_correctness(
            args.api, args.model, args.timeout, args.prompt_tag
        )
    except Exception as exc:
        if not output["warmup"]:
            output["warmup"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        output["correctness"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }

    if output["correctness"]["ok"]:
        for run in range(args.repeat):
            output["runs"].append(measured_run(args.api, args.model, run, args))

    throughputs = [
        run["output_tok_per_s"] for run in output["runs"] if "output_tok_per_s" in run
    ]
    if throughputs:
        output["summary"] = {
            "output_tok_per_s_mean": statistics.mean(throughputs),
            "output_tok_per_s_median": statistics.median(throughputs),
            "output_tok_per_s_min": min(throughputs),
            "output_tok_per_s_max": max(throughputs),
        }
        if args.protocol == "stream":
            output["summary"].update(
                aggregate_stream_metrics(
                    [
                        row
                        for run in output["runs"]
                        for row in run.get("rows", [])
                        if "error" not in row
                    ]
                )
            )
    if output["correctness"]["ok"] and len(output["runs"]) == args.repeat:
        try:
            output["correctness_post"] = paris_correctness(
                args.api, args.model, args.timeout, args.prompt_tag
            )
        except Exception as exc:
            output["correctness_post"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
    output["ok"] = output["correctness"]["ok"] and output["correctness_post"].get(
        "ok", False
    )
    output["ok"] = output["ok"] and len(output["runs"]) == args.repeat
    output["ok"] = output["ok"] and all(run["ok"] for run in output["runs"])
    print(json.dumps(output))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
