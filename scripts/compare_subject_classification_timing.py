from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


MATCH_COLUMNS = [
    "subject",
    "split",
    "classifier",
    "channel_set",
    "preprocess",
    "baseline_context",
    "smooth_proba_ms",
    "val_keep_all_positive",
    "edge_selection",
    "mode",
    "m",
    "max_edges",
    "alpha",
    "fusion_grid",
    "fusion_weight",
    "stride",
    "mfg_feature_mode",
]


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "nan"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return lines


def _has_fusion_grid(df: pd.DataFrame) -> bool:
    if "fusion_grid" not in df.columns:
        return False
    values = df["fusion_grid"].fillna("").astype(str).str.strip()
    return bool(values.ne("").any())


def _matching_columns(aligned: pd.DataFrame, shifted: pd.DataFrame) -> list[str]:
    key_cols = [col for col in MATCH_COLUMNS if col in aligned.columns and col in shifted.columns]
    if _has_fusion_grid(aligned) and _has_fusion_grid(shifted):
        key_cols = [col for col in key_cols if col != "fusion_weight"]
    return key_cols


def _fusion_column(df: pd.DataFrame) -> str:
    return "fusion_grid" if _has_fusion_grid(df) else "fusion_weight"


def compare_subject_timing(aligned_csv: Path, shifted_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_csv(aligned_csv)
    shifted = pd.read_csv(shifted_csv)
    key_cols = _matching_columns(aligned, shifted)
    if not key_cols:
        raise ValueError("No shared timing-control key columns found.")

    keep_cols = key_cols + [
        "baseline_auc",
        "best_assisted_auc",
        "best_assisted_feature_set",
        "best_assisted_minus_baseline",
    ]
    merged = aligned[keep_cols].merge(
        shifted[keep_cols],
        on=key_cols,
        how="inner",
        suffixes=("_aligned", "_shifted"),
    )
    if merged.empty:
        raise ValueError("Aligned and shifted result tables have no matched rows.")

    merged["aligned_lift"] = merged["best_assisted_minus_baseline_aligned"].astype(float)
    merged["shifted_lift"] = merged["best_assisted_minus_baseline_shifted"].astype(float)
    merged["aligned_minus_shifted"] = merged["aligned_lift"] - merged["shifted_lift"]
    merged["aligned_wins"] = merged["aligned_minus_shifted"] > 0.0
    merged["aligned_auc"] = merged["best_assisted_auc_aligned"].astype(float)
    merged["shifted_auc"] = merged["best_assisted_auc_shifted"].astype(float)
    merged["aligned_baseline_auc"] = merged["baseline_auc_aligned"].astype(float)
    merged["shifted_baseline_auc"] = merged["baseline_auc_shifted"].astype(float)

    fusion_col = _fusion_column(merged)
    group_cols = [col for col in ["alpha", fusion_col, "classifier", "channel_set", "preprocess"] if col in merged.columns]
    grouped = (
        merged.groupby(group_cols, as_index=False)
        .agg(
            matches=("aligned_minus_shifted", "count"),
            mean_aligned_lift=("aligned_lift", "mean"),
            mean_shifted_lift=("shifted_lift", "mean"),
            mean_delta=("aligned_minus_shifted", "mean"),
            aligned_wins=("aligned_wins", "sum"),
            mean_aligned_auc=("aligned_auc", "mean"),
            mean_shifted_auc=("shifted_auc", "mean"),
            mean_aligned_baseline_auc=("aligned_baseline_auc", "mean"),
            mean_shifted_baseline_auc=("shifted_baseline_auc", "mean"),
        )
    )
    grouped["summary_eligible"] = (grouped["mean_aligned_lift"] > 0.0) & (grouped["mean_delta"] > 0.0)
    grouped = grouped.sort_values(
        ["summary_eligible", "mean_delta", "mean_aligned_lift"],
        ascending=[False, False, False],
    )
    return merged.sort_values(["aligned_minus_shifted", "aligned_lift"], ascending=[False, False]), grouped


def write_report(out_path: Path, matched: pd.DataFrame, grouped: pd.DataFrame) -> None:
    overall = {
        "matches": int(len(matched)),
        "mean_aligned_lift": float(matched["aligned_lift"].mean()),
        "mean_shifted_lift": float(matched["shifted_lift"].mean()),
        "mean_delta": float(matched["aligned_minus_shifted"].mean()),
        "aligned_wins": int(matched["aligned_wins"].sum()),
    }
    best = grouped.iloc[0].to_dict() if not grouped.empty else {}
    fusion_col = _fusion_column(grouped) if not grouped.empty else _fusion_column(matched)

    group_rows = [
        [
            row.get("alpha", ""),
            row.get(fusion_col, ""),
            row.get("classifier", ""),
            row.get("channel_set", ""),
            row.get("preprocess", ""),
            int(row["matches"]),
            _format_float(row["mean_aligned_lift"]),
            _format_float(row["mean_shifted_lift"]),
            _format_float(row["mean_delta"]),
            int(row["aligned_wins"]),
        ]
        for _, row in grouped.iterrows()
    ]
    top_rows = [
        [
            int(row["subject"]),
            row["split"],
            row.get("alpha", ""),
            row.get(fusion_col, row.get("fusion_weight", "")),
            _format_float(row["aligned_lift"]),
            _format_float(row["shifted_lift"]),
            _format_float(row["aligned_minus_shifted"]),
            row.get("best_assisted_feature_set_aligned", ""),
        ]
        for _, row in matched.head(16).iterrows()
    ]

    lines = [
        "# Subject-Aware Timing Control",
        "",
        "Purpose: compare aligned classifier lift against a matched 500 ms shifted-label control.",
        "",
        "## Summary",
        "",
        f"- Matched rows: {overall['matches']}",
        f"- Mean aligned lift: {_format_float(overall['mean_aligned_lift'])}",
        f"- Mean shifted lift: {_format_float(overall['mean_shifted_lift'])}",
        f"- Mean aligned-minus-shifted delta: {_format_float(overall['mean_delta'])}",
        f"- Aligned wins: {overall['aligned_wins']}/{overall['matches']}",
        "",
        "## Best Hyperparameter Group",
        "",
        f"- Alpha: `{best.get('alpha', '')}`",
        f"- Fusion: `{best.get(fusion_col, '')}`",
        f"- Mean delta: `{_format_float(best.get('mean_delta'))}`",
        "",
        "## Hyperparameter Groups",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "alpha",
                "fusion",
                "classifier",
                "channels",
                "preprocess",
                "matches",
                "aligned_lift",
                "shifted_lift",
                "delta",
                "aligned_wins",
            ],
            group_rows,
        )
    )
    lines.extend(["", "## Top Matched Rows", ""])
    lines.extend(
        _markdown_table(
            ["subject", "split", "alpha", "fusion", "aligned_lift", "shifted_lift", "delta", "assisted"],
            top_rows,
        )
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched, grouped = compare_subject_timing(Path(args.aligned), Path(args.shifted))
    matched_path = out_dir / "timing_control.csv"
    grouped_path = out_dir / "timing_control_groups.csv"
    report_path = out_dir / "timing_control.md"
    summary_path = out_dir / "timing_control_summary.json"
    matched.to_csv(matched_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    write_report(report_path, matched, grouped)
    payload = {
        "matched_rows": int(len(matched)),
        "mean_aligned_lift": float(matched["aligned_lift"].mean()),
        "mean_shifted_lift": float(matched["shifted_lift"].mean()),
        "mean_delta": float(matched["aligned_minus_shifted"].mean()),
        "aligned_wins": int(matched["aligned_wins"].sum()),
        "best_group": grouped.iloc[0].to_dict() if not grouped.empty else None,
        "outputs": {
            "timing_control": str(matched_path),
            "timing_control_groups": str(grouped_path),
            "timing_control_report": str(report_path),
            "timing_control_summary": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare subject-aware aligned and shifted classifier runs.")
    parser.add_argument("--aligned", default="out/classification_subject/subject_results.csv")
    parser.add_argument("--shifted", default="out/classification_subject_shift500/subject_results.csv")
    parser.add_argument("--out-dir", default="out/classification_subject")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_compare(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
