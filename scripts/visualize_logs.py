"""
Generate plots from serving pipeline JSONL logs.

Produces six figures saved as PNG files:
  1. latency_hist.png         - End-to-end latency distribution
  2. ttft_hist.png            - Time-to-first-token distribution
  3. queue_wait_hist.png      - Queue wait time distribution
  4. dataset_latency_bar.png  - Mean latency per dataset with std-dev error bars
  5. throughput_over_time.png - Cumulative completions over wall time
  6. tokens_vs_latency.png    - Output token count vs end-to-end latency scatter

Usage:
    python scripts/visualize_logs.py --log logs/run.jsonl --output-dir plots/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/run.jsonl")
    parser.add_argument("--output-dir", default="plots")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def numeric(records: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in records if r.get(key) is not None]


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  saved {path}")


def plot_histogram(
    values: list[float],
    title: str,
    xlabel: str,
    output_path: Path,
    plt: Any,
    bins: int = 30,
) -> None:
    if not values:
        print(f"  skipped {output_path.name} (no data)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, edgecolor="white", linewidth=0.4)
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    ax.axvline(p50, color="orange", linestyle="--", linewidth=1.2, label=f"p50 {p50:.3f}")
    ax.axvline(p95, color="red", linestyle="--", linewidth=1.2, label=f"p95 {p95:.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("requests")
    ax.legend(fontsize=9)
    _save(fig, output_path)
    plt.close(fig)


def plot_dataset_latency_bar(records: list[dict], output_path: Path, plt: Any) -> None:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for r in records:
        dataset = r.get("dataset") or "unknown"
        if r.get("latency_seconds") is not None:
            by_dataset[dataset].append(float(r["latency_seconds"]))

    if not by_dataset:
        print(f"  skipped {output_path.name} (no data)")
        return

    labels = sorted(by_dataset)
    means = [mean(by_dataset[d]) for d in labels]
    errors = [stdev(by_dataset[d]) if len(by_dataset[d]) > 1 else 0.0 for d in labels]
    counts = [len(by_dataset[d]) for d in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 4))
    x = range(len(labels))
    ax.bar(x, means, yerr=errors, capsize=4, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\n(n={counts[i]})" for i, l in enumerate(labels)], fontsize=9)
    ax.set_ylabel("latency (s)")
    ax.set_title("Mean end-to-end latency per dataset (±1 std dev)")
    _save(fig, output_path)
    plt.close(fig)


def plot_throughput_over_time(records: list[dict], output_path: Path, plt: Any) -> None:
    completions = [(float(r["completion_time"]), r) for r in records if r.get("completion_time")]
    if not completions:
        print(f"  skipped {output_path.name} (no data)")
        return

    completions.sort(key=lambda t: t[0])
    t0 = min(float(r.get("arrival_time", t)) for t, r in completions)
    times = [(t - t0) for t, _ in completions]
    cumulative = list(range(1, len(times) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, cumulative, linewidth=1.5)
    ax.set_xlabel("wall time from first arrival (s)")
    ax.set_ylabel("cumulative completions")
    ax.set_title("Throughput over time")
    _save(fig, output_path)
    plt.close(fig)


def plot_tokens_vs_latency(records: list[dict], output_path: Path, plt: Any) -> None:
    pairs = [
        (float(r["output_token_count"]), float(r["latency_seconds"]))
        for r in records
        if r.get("output_token_count") is not None and r.get("latency_seconds") is not None
    ]
    if not pairs:
        print(f"  skipped {output_path.name} (no data)")
        return

    tokens, latencies = zip(*pairs)
    datasets = [r.get("dataset") or "unknown" for r in records
                if r.get("output_token_count") is not None and r.get("latency_seconds") is not None]
    unique_datasets = sorted(set(datasets))
    color_map = {d: i for i, d in enumerate(unique_datasets)}
    colors = [color_map[d] for d in datasets]

    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(tokens, latencies, c=colors, alpha=0.6, s=18, cmap="tab10")
    if len(unique_datasets) > 1:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=scatter.cmap(scatter.norm(color_map[d])),
                       markersize=7, label=d)
            for d in unique_datasets
        ]
        ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("output tokens")
    ax.set_ylabel("latency (s)")
    ax.set_title("Output token count vs end-to-end latency")
    _save(fig, output_path)
    plt.close(fig)


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Install matplotlib to generate plots: pip install matplotlib")

    # suppress the Any annotation used above
    global Any
    from typing import Any  # noqa: F811

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.log)
    print(f"Loaded {len(records)} records from {args.log}")

    plot_histogram(
        numeric(records, "latency_seconds"),
        "End-to-end latency distribution",
        "latency (s)",
        output_dir / "latency_hist.png",
        plt,
    )
    plot_histogram(
        numeric(records, "ttft_seconds"),
        "Time-to-first-token distribution",
        "TTFT (s)",
        output_dir / "ttft_hist.png",
        plt,
    )
    plot_histogram(
        numeric(records, "queue_wait_seconds"),
        "Queue wait time distribution",
        "queue wait (s)",
        output_dir / "queue_wait_hist.png",
        plt,
    )
    plot_dataset_latency_bar(records, output_dir / "dataset_latency_bar.png", plt)
    plot_throughput_over_time(records, output_dir / "throughput_over_time.png", plt)
    plot_tokens_vs_latency(records, output_dir / "tokens_vs_latency.png", plt)

    print(f"All plots written to {output_dir}/")


if __name__ == "__main__":
    main()
