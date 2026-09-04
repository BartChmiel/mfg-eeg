from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _mean(rows: list[dict[str, str]], column: str) -> float:
    values = [_safe_float(row.get(column)) for row in rows if row.get(column) not in ("", None)]
    return statistics.fmean(values) if values else 0.0


def _human_phase(value: str) -> str:
    return value.replace("__", " -> ").replace("_", " ")


def _scientific(value: Any) -> str:
    number = _safe_float(value, 0.0)
    if number <= 0:
        return "n/a"
    return f"{number:.1e}"


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    return plt


def _plot_top_edges(package_dir: Path, out_dir: Path) -> Path | None:
    rows = _read_csv(package_dir / "tables" / "top_edges.csv")[:10]
    if not rows:
        return None

    plt = _load_pyplot()
    labels = [
        (
            f"{row.get('src', '')}->{row.get('dst', '')}\n"
            f"{_human_phase(row.get('phase', ''))}, {row.get('pc', '').upper()}, "
            f"{row.get('lag_ms', '')} ms"
        )
        for row in rows
    ]
    values = [_safe_int(row.get("k")) for row in rows]
    totals = [_safe_int(row.get("n_subjects"), 12) for row in rows]
    colors = [
        "#f28e2b" if {row.get("src", ""), row.get("dst", "")} & {"Fp1", "Fp2"} else "#4c78a8"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y_positions = list(range(len(rows)))
    bars = ax.barh(y_positions, values, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(totals), 12) + 1)
    ax.set_xlabel("Subjects with replicated edge in top-k list")
    ax.set_title("Most replicated band-pass directed edge instances")
    ax.grid(axis="x", alpha=0.18)

    for bar, row, total in zip(bars, rows, totals):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{_safe_int(row.get('k'))}/{total}, q={_scientific(row.get('q_value'))}",
            va="center",
            fontsize=7,
        )

    ax.text(
        0.99,
        0.02,
        "orange: Fp1/Fp2 eye-proxy channels",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#7a4a12",
    )
    fig.tight_layout()

    out_path = out_dir / "top_edges_replication.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_predictive_group(rows: list[dict[str, str]]) -> dict[str, float] | None:
    groups: dict[tuple[float, float], list[dict[str, str]]] = {}
    for row in rows:
        if _safe_float(row.get("label_shift_ms")) != 0.0:
            continue
        key = (_safe_float(row.get("alpha")), _safe_float(row.get("fusion_weight")))
        groups.setdefault(key, []).append(row)

    best: dict[str, float] | None = None
    for (alpha, fusion_weight), group_rows in groups.items():
        stats = {
            "alpha": alpha,
            "fusion_weight": fusion_weight,
            "baseline": _mean(group_rows, "baseline_auc"),
            "mfg": _mean(group_rows, "mfg_auc"),
            "combined": _mean(group_rows, "combined_auc"),
            "fusion": _mean(group_rows, "fusion_auc"),
            "best_augmented": _mean(group_rows, "best_assisted_auc"),
        }
        if best is None or stats["best_augmented"] > best["best_augmented"]:
            best = stats
    return best


def _plot_classifier(package_dir: Path, out_dir: Path) -> Path | None:
    subject_rows = _read_csv(package_dir / "classification_subject" / "subject_results.csv")
    timing = _read_json(package_dir / "classification_subject" / "timing_control_summary.json")
    if not subject_rows and not timing:
        return None

    plt = _load_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35))

    best_group = _best_predictive_group(subject_rows)
    if best_group:
        labels = ["Baseline", "MFG only", "Combined", "Fusion", "Best aug."]
        values = [
            best_group["baseline"],
            best_group["mfg"],
            best_group["combined"],
            best_group["fusion"],
            best_group["best_augmented"],
        ]
        colors = ["#4c78a8", "#9c755f", "#59a14f", "#f28e2b", "#6b6ecf"]
        bars = axes[0].bar(labels, values, color=colors)
        y_min = max(0.0, min(values) - 0.02)
        y_max = min(1.0, max(values) + 0.015)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_ylabel("Mean ROC-AUC")
        axes[0].set_title(
            "Best predictive group\n"
            f"alpha={best_group['alpha']:.1f}, fusion={best_group['fusion_weight']:.2f}"
        )
        axes[0].tick_params(axis="x", labelrotation=30)
        axes[0].grid(axis="y", alpha=0.18)
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.002,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No subject AUC table available", ha="center", va="center")

    timing_values = [
        _safe_float(timing.get("mean_aligned_lift")),
        _safe_float(timing.get("mean_shifted_lift")),
        _safe_float(timing.get("mean_delta")),
    ]
    timing_labels = ["Aligned lift", "Shifted lift", "Matched delta"]
    bars = axes[1].bar(timing_labels, timing_values, color=["#59a14f", "#bab0ab", "#4c78a8"])
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    max_abs = max([abs(value) for value in timing_values] + [0.001])
    axes[1].set_ylim(-max_abs * 0.25, max_abs * 1.35)
    axes[1].set_title("500 ms shifted-label control")
    axes[1].set_ylabel("ROC-AUC lift over baseline")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, timing_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max_abs * 0.04,
            f"{value:+.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    if timing:
        axes[1].text(
            0.98,
            0.02,
            f"aligned wins: {timing.get('aligned_wins', 'n/a')}/{timing.get('matched_rows', 'n/a')}",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    out_path = out_dir / "classification_timing_control.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_article_figures(package_dir: Path | str = "out/article_package") -> list[Path]:
    package_path = Path(package_dir)
    out_dir = package_path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []
    for builder in (_plot_top_edges, _plot_classifier):
        path = builder(package_path, out_dir)
        if path is not None:
            figures.append(path)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article-ready summary figures from the evidence package.")
    parser.add_argument("--package-dir", default="out/article_package")
    args = parser.parse_args()

    figures = build_article_figures(args.package_dir)
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
