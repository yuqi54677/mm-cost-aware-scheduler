#!/usr/bin/env bash
# Run the stress workload under each scheduler and save separate logs.
# Usage: bash run_scheduler_comparison.sh

set -euo pipefail

WORKLOAD="workloads/stress_mixed.jsonl"
MAX_BATCH=8
BACKEND="vllm"

SCHEDULERS=("gmax")

for SCHED in "${SCHEDULERS[@]}"; do
    LOG="logs/stress_${SCHED}.jsonl"
    echo "================================================"
    echo " Running scheduler: $SCHED"
    echo " Log: $LOG"
    echo "================================================"

    python scripts/run_workload.py \
        --backend "$BACKEND" \
        --workload "$WORKLOAD" \
        --log "$LOG" \
        --max-batch-size "$MAX_BATCH" \
        --scheduler "$SCHED" \
        --gmax-window-size 16 \
        --reset-log

    echo "Done: $SCHED"
    echo ""
done

echo "All schedulers complete. Logs:"
for SCHED in "${SCHEDULERS[@]}"; do
    echo "  logs/stress_${SCHED}.jsonl"
done
