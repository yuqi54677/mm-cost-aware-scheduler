"""Create report figures for output-length prediction risk.

Inputs:
  1. Evaluation JSONL from evaluate_output_prediction.py
  2. Structured metrics JSON from analyze_output_prediction_risk.py

Outputs:
  - coverage_by_dataset.png
  - underestimate_rate_by_dataset.png
  - overestimate_overhead_by_dataset.png
  - predicted_vs_actual.png
  - dataset_means.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_COLORS = {
    "coco": "#4C78A8",
    "textvqa": "#F58518",
    "mmmu": "#54A24B",
    "text-only": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True, help="Evaluation JSONL")
    parser.add_argument("--metrics-json", required=True, help="Structured metrics JSON")
    parser.add_argument("--output-dir", default="plots/output_prediction")
    return parser.parse_args()


def load_eval(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def actual_length(record: dict[str, Any]) -> int | None:
    for key in ("actual_output_length", "ground_truth_output_length"):
        if record.get(key) is not None:
            return int(record[key])
    return None


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def metric_by_dataset(section: dict[str, Any], metric: str) -> dict[str, float]:
    groups = section["groups"].get("dataset", {})
    return {
        dataset: float(summary[metric])
        for dataset, summary in groups.items()
        if summary.get("n", 0)
    }


def sections_for_plot(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (label, label)
        for label in sorted(metrics.get("percentile_ablation", {}))
    ]


def grouped_bar(
    data: dict[str, dict[str, float]],
    ylabel: str,
    title: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted({dataset for values in data.values() for dataset in values})
    labels = list(data)
    x = list(range(len(datasets)))
    width = 0.8 / max(1, len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, label in enumerate(labels):
        offsets = [value + (i - (len(labels) - 1) / 2) * width for value in x]
        values = [data[label].get(dataset, 0.0) for dataset in datasets]
        if percentage:
            values = [value * 100.0 for value in values]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_by_dataset(metrics: dict[str, Any], metric: str, ylabel: str, title: str, output_path: Path, percentage: bool = False) -> None:
    data: dict[str, dict[str, float]] = {}
    for label, _ in sections_for_plot(metrics):
        data[label] = metric_by_dataset(metrics["percentile_ablation"][label], metric)
    grouped_bar(data, ylabel, title, output_path, percentage=percentage)


def prediction_for_label(record: dict[str, Any], label: str) -> int | None:
    value = record.get("prediction_ablation", {}).get(label, {}).get("predicted_output_length")
    return int(value) if value is not None else None


def plot_predicted_vs_actual(records: list[dict[str, Any]], labels: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 4.8), squeeze=False)
    for ax, label in zip(axes[0], labels, strict=True):
        xs: list[int] = []
        ys: list[int] = []
        colors: list[str] = []
        for record in records:
            actual = actual_length(record)
            predicted = prediction_for_label(record, label)
            if actual is None or predicted is None:
                continue
            xs.append(actual)
            ys.append(predicted)
            colors.append(DATASET_COLORS.get(str(record.get("dataset")), "#777777"))

        ax.scatter(xs, ys, c=colors, alpha=0.75, s=36, edgecolors="white", linewidths=0.4)
        max_value = max(xs + ys + [1])
        ax.plot([0, max_value], [0, max_value], color="#333333", linestyle="--", linewidth=1)
        ax.set_xlabel("Actual output tokens")
        ax.set_ylabel("Predicted output tokens")
        ax.set_title(f"Predicted vs Actual ({label})")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dataset_means(metrics: dict[str, Any], output_path: Path) -> None:
    data: dict[str, dict[str, float]] = {}
    first_section = next(iter(metrics.get("percentile_ablation", {}).values()), None)
    if first_section is not None:
        data["actual"] = metric_by_dataset(first_section, "actual_mean")
    for label, _ in sections_for_plot(metrics):
        data[label] = metric_by_dataset(metrics["percentile_ablation"][label], "predicted_mean")
    grouped_bar(data, "Mean output tokens", "Actual vs Predicted Mean by Dataset", output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_eval(args.eval_jsonl)
    metrics = load_metrics(args.metrics_json)
    if not metrics.get("percentile_ablation"):
        raise SystemExit(
            "No percentile_ablation metrics found. Re-run evaluate_output_prediction.py "
            "with --ablation-percentiles 0.50,0.90, then re-run analyze_output_prediction_risk.py."
        )

    plot_metric_by_dataset(
        metrics,
        "coverage",
        "Coverage (%)",
        "Prediction Coverage by Dataset",
        output_dir / "coverage_by_dataset.png",
        percentage=True,
    )
    plot_metric_by_dataset(
        metrics,
        "underestimate_rate",
        "Underestimate rate (%)",
        "Underestimate Rate by Dataset",
        output_dir / "underestimate_rate_by_dataset.png",
        percentage=True,
    )
    plot_metric_by_dataset(
        metrics,
        "overestimate_mean",
        "Mean overestimate tokens",
        "Overestimation Overhead by Dataset",
        output_dir / "overestimate_overhead_by_dataset.png",
    )
    labels = [label for label, _ in sections_for_plot(metrics)]
    plot_predicted_vs_actual(records, labels, output_dir / "predicted_vs_actual.png")
    plot_dataset_means(metrics, output_dir / "dataset_means.png")

    print(f"Wrote output prediction figures to {output_dir}")


if __name__ == "__main__":
    main()
