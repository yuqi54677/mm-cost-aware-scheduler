"""Benchmark runner that lets the request queue accumulate before dispatch.

Unlike run_workload.py, this script does not call pipeline.run_once() after
every submitted request. It replays arrivals, submits all requests that have
arrived, and dispatches on a small interval or when the queue reaches the
configured batch size. This gives schedulers such as GMAX a real candidate pool.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--log", default="logs/benchmark.jsonl")
    parser.add_argument("--reset-log", action="store_true")
    parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--scheduler", choices=["fifo", "length-only", "gmax"], default="fifo")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--gmax-window-size", type=int, default=None)
    parser.add_argument(
        "--gmax-tail-slo-ms",
        type=float,
        default=None,
        help="For GMAX, protect requests that have waited at least this long.",
    )
    parser.add_argument(
        "--dispatch-interval-ms",
        type=float,
        default=50.0,
        help="Dispatch at most once per interval so arrivals can accumulate.",
    )
    parser.add_argument(
        "--max-queue-delay-ms",
        type=float,
        default=250.0,
        help="Dispatch if the oldest queued request has waited this long.",
    )
    parser.add_argument(
        "--accumulate-all",
        action="store_true",
        help="Submit the full workload before dispatching; useful for scheduler stress tests.",
    )
    return parser.parse_args()


def load_workload(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record:
                raise ValueError(f"Workload line {line_number} is missing required field 'prompt'")
            records.append(record)
    return sorted(records, key=lambda record: float(record.get("arrival_time", 0.0)))


def build_backend(args: argparse.Namespace):
    if args.backend == "vllm":
        return VLLMBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
        )
    return MockBackend()


def build_scheduler(args: argparse.Namespace):
    if args.scheduler == "length-only":
        return LengthOnlyScheduler(max_batch_size=args.max_batch_size)
    if args.scheduler == "gmax":
        return GMAXScheduler(
            max_batch_size=args.max_batch_size,
            window_size=args.gmax_window_size,
            tail_slo_seconds=args.gmax_tail_slo_ms / 1000.0
            if args.gmax_tail_slo_ms is not None
            else None,
        )
    return FIFOScheduler(max_batch_size=args.max_batch_size)


def relative_arrival(record: dict[str, Any]) -> float:
    value = float(record.get("arrival_time", 0.0))
    if value > 1_000_000_000:
        return 0.0
    return value


def submit_record(pipeline: ServingPipeline, record: dict[str, Any], run_start: float) -> None:
    arrival_time = run_start + relative_arrival(record)
    pipeline.submit(
        request_id=record.get("request_id"),
        arrival_time=arrival_time,
        prompt=record["prompt"],
        image_path=record.get("image_path"),
        dataset=record.get("dataset"),
        source=record.get("source"),
        metadata=record.get("metadata"),
    )


def oldest_queue_wait_seconds(pipeline: ServingPipeline, now: float) -> float:
    waiting = pipeline.queue.snapshot()
    if not waiting:
        return 0.0
    queue_enter_time = waiting[0].timings.queue_enter_time
    if queue_enter_time is None:
        return 0.0
    return now - queue_enter_time


def should_dispatch(
    pipeline: ServingPipeline,
    now: float,
    next_dispatch_time: float,
    max_batch_size: int,
    max_queue_delay_seconds: float,
) -> bool:
    queue_size = len(pipeline.queue)
    if queue_size == 0:
        return False
    if queue_size >= max_batch_size:
        return True
    if oldest_queue_wait_seconds(pipeline, now) >= max_queue_delay_seconds:
        return True
    return now >= next_dispatch_time


def sleep_until_next_event(
    next_arrival_time: float | None,
    next_dispatch_time: float,
    pipeline: ServingPipeline,
    max_queue_delay_seconds: float,
) -> None:
    targets = [next_dispatch_time]
    if next_arrival_time is not None:
        targets.append(next_arrival_time)
    waiting = pipeline.queue.snapshot()
    if waiting and waiting[0].timings.queue_enter_time is not None:
        targets.append(waiting[0].timings.queue_enter_time + max_queue_delay_seconds)
    sleep_seconds = min(targets) - time.time()
    if sleep_seconds > 0:
        time.sleep(min(sleep_seconds, 0.01))


def run_accumulate_all(pipeline: ServingPipeline, records: list[dict[str, Any]], run_start: float) -> None:
    for record in records:
        submit_record(pipeline, record, run_start)
    pipeline.drain()


def run_timed_replay(pipeline: ServingPipeline, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    run_start = time.time()
    dispatch_interval_seconds = args.dispatch_interval_ms / 1000.0
    max_queue_delay_seconds = args.max_queue_delay_ms / 1000.0
    next_dispatch_time = run_start + dispatch_interval_seconds
    index = 0

    while index < len(records) or len(pipeline.queue) > 0:
        now = time.time()

        while index < len(records) and run_start + relative_arrival(records[index]) <= now:
            submit_record(pipeline, records[index], run_start)
            index += 1

        if should_dispatch(
            pipeline=pipeline,
            now=now,
            next_dispatch_time=next_dispatch_time,
            max_batch_size=args.max_batch_size,
            max_queue_delay_seconds=max_queue_delay_seconds,
        ):
            pipeline.run_once()
            next_dispatch_time = time.time() + dispatch_interval_seconds
            continue

        next_arrival_time = None
        if index < len(records):
            next_arrival_time = run_start + relative_arrival(records[index])
        sleep_until_next_event(
            next_arrival_time,
            next_dispatch_time,
            pipeline,
            max_queue_delay_seconds,
        )


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    if args.reset_log and log_path.exists():
        log_path.unlink()

    pipeline = ServingPipeline(
        backend=build_backend(args),
        log_writer=JSONLLogWriter(log_path),
        scheduler=build_scheduler(args),
    )
    records = load_workload(args.workload)

    if args.accumulate_all:
        run_accumulate_all(pipeline, records, time.time())
    else:
        run_timed_replay(pipeline, records, args)

    print(f"Wrote request logs to {log_path}")


if __name__ == "__main__":
    main()
