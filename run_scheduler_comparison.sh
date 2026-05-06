#!/usr/bin/env bash
# Run the stress workload under each scheduler and save separate logs.
# Usage: bash run_scheduler_comparison.sh

set -euo pipefail

WORKLOAD="workloads/stress_mixed.jsonl"
MAX_BATCH=8
BACKEND="vllm"

SCHEDULERS=("fifo" "length-only" "gmax")

for SCHED in "${SCHEDULERS[@]}"; do
    LOG="logs/stress_${SCHED}.jsonl"
    echo "================================================"
    echo " Running scheduler: $SCHED"
    echo " Log: $LOG"
    echo "================================================"

    CMD=(
        python scripts/run_benchmark.py
        --backend "$BACKEND"
        --workload "$WORKLOAD"
        --log "$LOG"
        --max-batch-size "$MAX_BATCH"
        --scheduler "$SCHED"
        --dispatch-interval-ms 100
        --max-queue-delay-ms 250
        --reset-log
    )

    if [[ "$SCHED" == "gmax" ]]; then
        CMD+=(--gmax-window-size 16)
    fi

    "${CMD[@]}"

    echo "Done: $SCHED"
    echo ""
done

echo "All schedulers complete. Logs:"
for SCHED in "${SCHEDULERS[@]}"; do
    echo "  logs/stress_${SCHED}.jsonl"
done
