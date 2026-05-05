"""
Replay a JSONL workload through the skeleton serving pipeline.

Each workload line is one request. Supported request fields match MMRequest:
request_id, prompt, image_path, dataset, source, metadata. The optional
arrival_time field controls replay timing and is passed to MMRequest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmserve_skeleton.backend import MockBackend, VLLMBackend
from mmserve_skeleton.logging import JSONLLogWriter
from mmserve_skeleton.pipeline import ServingPipeline
from mmserve_skeleton.scheduler import FIFOScheduler, GMAXScheduler, LengthOnlyScheduler


def parse_args() -> argparse.Namespace:
    """Return the minimal configuration needed to replay a workload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, help="Input workload JSONL")
    parser.add_argument("--log", default="logs/run.jsonl", help="Output request log JSONL")
    parser.add_argument("--reset-log", action="store_true")
    parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--scheduler", choices=["fifo", "length-only", "gmax"], default="fifo")
    parser.add_argument("--gmax-window-size", type=int, default=None)
    return parser.parse_args()


def load_workload(path: str | Path) -> list[dict[str, Any]]:
    """Read workload requests from JSONL."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record:
                raise ValueError(f"Workload line {line_number} is missing required field 'prompt'")
            records.append(record)
    return records


def scheduled_arrival_timestamp(record: dict[str, Any], run_start: float) -> float | None:
    """Convert workload arrival_time into the timestamp used by MMRequest.

    Workload files can provide arrival_time as seconds from replay start. If the
    value already looks like a Unix timestamp, keep it as-is.
    """
    if "arrival_time" not in record:
        return None

    arrival_time = float(record["arrival_time"])
    if arrival_time > 1_000_000_000:
        return arrival_time
    return run_start + arrival_time


def submit_record(pipeline: ServingPipeline, record: dict[str, Any], arrival_time: float | None) -> None:
    """Submit one workload row using only fields defined on MMRequest."""
    pipeline.submit(
        request_id=record.get("request_id"),
        arrival_time=arrival_time,
        prompt=record["prompt"],
        image_path=record.get("image_path"),
        dataset=record.get("dataset"),
        source=record.get("source"),
        metadata=record.get("metadata"),
    )


def build_scheduler(args: argparse.Namespace):
    """Create the requested scheduler baseline."""
    if args.scheduler == "length-only":
        return LengthOnlyScheduler(max_batch_size=args.max_batch_size)
    if args.scheduler == "gmax":
        return GMAXScheduler(
            max_batch_size=args.max_batch_size,
            window_size=args.gmax_window_size,
        )
    return FIFOScheduler(max_batch_size=args.max_batch_size)


def main() -> None:
    """Replay every workload request through the local mock serving pipeline."""
    args = parse_args()
    log_path = Path(args.log)
    if args.reset_log and log_path.exists():
        log_path.unlink()

    backend = (
        VLLMBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if args.backend == "vllm"
        else MockBackend()
    )

    pipeline = ServingPipeline(
        backend=backend,
        log_writer=JSONLLogWriter(log_path),
        scheduler=build_scheduler(args),
    )

    run_start = time.time()
    for record in load_workload(args.workload):
        arrival_time = scheduled_arrival_timestamp(record, run_start)
        if arrival_time is not None:
            time.sleep(max(0.0, arrival_time - time.time()))
        submit_record(pipeline, record, arrival_time)
        pipeline.run_once()

    pipeline.drain()
    print(f"Wrote request logs to {log_path}")


if __name__ == "__main__":
    main()
