from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.kaggle_classification_benchmark import BASELINE_CONTEXT_CHOICES, CLASSIFIER_CHOICES
from scripts.kaggle_classification_sweep import build_parser_sweep, run_sweep


def _as_strings(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values]


def _build_sweep_args(args: argparse.Namespace) -> argparse.Namespace:
    sweep_argv = [
        "--root",
        args.root,
        "--out-dir",
        args.out_dir,
        "--edge-table",
        args.edge_table,
        "--fallback-edge-table",
        args.fallback_edge_table,
        "--splits",
        *args.splits,
        "--modes",
        *_as_strings(args.modes),
        "--m-values",
        *_as_strings(args.m_values),
        "--max-edges",
        *_as_strings(args.max_edges),
        "--edge-selections",
        *_as_strings(args.edge_selections),
        "--channel-sets",
        *_as_strings(args.channel_sets),
        "--preprocessings",
        *_as_strings(args.preprocessings),
        "--label-shift-ms-values",
        *_as_strings(args.label_shift_ms_values),
        "--strides",
        *_as_strings(args.strides),
        "--mfg-feature-modes",
        *_as_strings(args.mfg_feature_modes),
        "--baseline-contexts",
        *_as_strings(args.baseline_contexts),
        "--smooth-proba-ms-values",
        *_as_strings(args.smooth_proba_ms_values),
        "--classifiers",
        *_as_strings(args.classifiers),
        "--baseline-windows-ms",
        *_as_strings(args.baseline_windows_ms),
        "--baseline-lags-ms",
        *_as_strings(args.baseline_lags_ms),
        "--ema-half-life-s",
        str(args.ema_half_life_s),
        "--epochs",
        str(args.epochs),
        "--alpha",
        str(args.alpha),
        "--tree-estimators",
        str(args.tree_estimators),
        "--tree-min-samples-leaf",
        str(args.tree_min_samples_leaf),
        "--random-state",
        str(args.random_state),
        "--cache-dir",
        args.cache_dir,
    ]
    if args.fusion_weights:
        sweep_argv.extend(["--fusion-weights", *_as_strings(args.fusion_weights)])
    if args.max_train_files is not None:
        sweep_argv.extend(["--max-train-files", str(args.max_train_files)])
    if args.max_val_files is not None:
        sweep_argv.extend(["--max-val-files", str(args.max_val_files)])
    if args.max_train_samples_per_file is not None:
        sweep_argv.extend(["--max-train-samples-per-file", str(args.max_train_samples_per_file)])
    if args.max_val_samples_per_file is not None:
        sweep_argv.extend(["--max-val-samples-per-file", str(args.max_val_samples_per_file)])
    if args.tree_max_depth is not None:
        sweep_argv.extend(["--tree-max-depth", str(args.tree_max_depth)])
    sweep_argv.append("--val-keep-all-positive" if args.val_keep_all_positive else "--no-val-keep-all-positive")
    if args.balance_classes:
        sweep_argv.append("--balance-classes")
    if args.quiet:
        sweep_argv.append("--quiet")
    return build_parser_sweep().parse_args(sweep_argv)


def _format_float(value: Any) -> str:
    try:
        if pd.isna(value):
            return "nan"
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


def _build_timing_control_table(rows: pd.DataFrame) -> tuple[list[list[Any]], int]:
    if rows.empty or "label_shift_ms" not in rows.columns:
        return [], 0

    candidate_keys = [
        "split",
        "mode",
        "m",
        "max_edges",
        "edge_selection",
        "classifier",
        "channel_set",
        "preprocess",
        "baseline_context",
        "smooth_proba_ms",
        "stride",
        "mfg_feature_mode",
        "fusion_weight",
    ]
    key_cols = [col for col in candidate_keys if col in rows.columns]
    if not key_cols:
        return [], 0

    aligned = rows[rows["label_shift_ms"].astype(float).eq(0.0)]
    shifted_values = sorted(
        float(value)
        for value in rows["label_shift_ms"].dropna().unique()
        if float(value) != 0.0
    )
    if aligned.empty or not shifted_values:
        return [], 0

    control_frames: list[pd.DataFrame] = []
    lift_col = "best_assisted_minus_baseline" if "best_assisted_minus_baseline" in rows.columns else "combined_minus_baseline"
    aligned_cols = key_cols + [lift_col]
    aligned_base = aligned[aligned_cols].rename(columns={lift_col: "aligned_lift"})

    for shift_ms in shifted_values:
        shifted = rows[rows["label_shift_ms"].astype(float).eq(shift_ms)]
        shifted_base = shifted[aligned_cols].rename(columns={lift_col: "shifted_lift"})
        matched = aligned_base.merge(shifted_base, on=key_cols, how="inner")
        if matched.empty:
            continue
        matched["label_shift_ms"] = shift_ms
        matched["aligned_minus_shifted"] = matched["aligned_lift"] - matched["shifted_lift"]
        matched["aligned_wins"] = matched["aligned_minus_shifted"] > 0.0
        control_frames.append(matched)

    if not control_frames:
        return [], 0

    matched_rows = pd.concat(control_frames, ignore_index=True)
    group_cols = [
        col
        for col in ["label_shift_ms", "classifier", "channel_set", "preprocess", "edge_selection"]
        if col in matched_rows.columns
    ]
    grouped = (
        matched_rows.groupby(group_cols, as_index=False)
        .agg(
            matched_configs=("aligned_minus_shifted", "count"),
            mean_aligned_lift=("aligned_lift", "mean"),
            mean_shifted_lift=("shifted_lift", "mean"),
            mean_delta=("aligned_minus_shifted", "mean"),
            aligned_wins=("aligned_wins", "sum"),
        )
        .sort_values(["mean_delta", "mean_aligned_lift"], ascending=[False, False])
    )

    table = [
        [
            _format_float(row.get("label_shift_ms")),
            row.get("classifier", ""),
            row.get("channel_set", ""),
            row.get("preprocess", ""),
            row.get("baseline_context", ""),
            _format_float(row.get("smooth_proba_ms")),
            row.get("edge_selection", ""),
            int(row["matched_configs"]),
            _format_float(row["mean_aligned_lift"]),
            _format_float(row["mean_shifted_lift"]),
            _format_float(row["mean_delta"]),
            int(row["aligned_wins"]),
        ]
        for _, row in grouped.iterrows()
    ]
    return table, int(len(matched_rows))


def _write_ablation_report(
    *,
    out_path: Path,
    rows: pd.DataFrame,
    event_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if rows.empty:
        out_path.write_text("# Classification Controls\n\nNo configurations completed.\n", encoding="utf-8")
        return {
            "configs_completed": 0,
            "best": None,
        }

    rows = rows.copy()
    if "classifier" not in rows.columns:
        rows["classifier"] = "sgd_logistic"
    if "label_shift_ms" not in rows.columns:
        rows["label_shift_ms"] = 0.0
    if "baseline_context" not in rows.columns:
        rows["baseline_context"] = "causal"
    if "smooth_proba_ms" not in rows.columns:
        rows["smooth_proba_ms"] = 0.0
    if "best_assisted_minus_baseline" not in rows.columns:
        rows["best_assisted_minus_baseline"] = rows["combined_minus_baseline"]
    if "best_assisted_auc" not in rows.columns:
        rows["best_assisted_auc"] = rows["combined_auc"]
    if "best_assisted_feature_set" not in rows.columns:
        rows["best_assisted_feature_set"] = "combined"
    rows["combined_minus_baseline"] = rows["combined_minus_baseline"].astype(float)
    rows["combined_auc"] = rows["combined_auc"].astype(float)
    rows["best_assisted_minus_baseline"] = rows["best_assisted_minus_baseline"].astype(float)
    rows["best_assisted_auc"] = rows["best_assisted_auc"].astype(float)
    rows = rows.sort_values(["best_assisted_minus_baseline", "best_assisted_auc"], ascending=[False, False])
    best = rows.iloc[0].to_dict()

    group_cols = [
        "classifier",
        "channel_set",
        "preprocess",
        "baseline_context",
        "smooth_proba_ms",
        "label_shift_ms",
        "edge_selection",
    ]
    grouped = (
        rows.groupby(group_cols, as_index=False)
        .agg(
            configs=("config_id", "count"),
            mean_lift=("best_assisted_minus_baseline", "mean"),
            best_lift=("best_assisted_minus_baseline", "max"),
            mean_assisted_auc=("best_assisted_auc", "mean"),
        )
        .sort_values(["mean_lift", "best_lift"], ascending=[False, False])
    )

    group_table = [
        [
            row["classifier"],
            row["channel_set"],
            row["preprocess"],
            row["baseline_context"],
            _format_float(row["smooth_proba_ms"]),
            _format_float(row["label_shift_ms"]),
            row["edge_selection"],
            int(row["configs"]),
            _format_float(row["mean_lift"]),
            _format_float(row["best_lift"]),
            _format_float(row["mean_assisted_auc"]),
        ]
        for _, row in grouped.iterrows()
    ]

    top_table = [
        [
            int(row["config_id"]),
            row.get("classifier", ""),
            row["channel_set"],
            row["preprocess"],
            row.get("baseline_context", "causal"),
            _format_float(row.get("smooth_proba_ms", 0.0)),
            _format_float(row["label_shift_ms"]),
            row["edge_selection"],
            row["mode"],
            int(row["max_edges"]),
            int(row["edges_used"]),
            row.get("best_assisted_feature_set", "combined"),
            _format_float(row["baseline_auc"]),
            _format_float(row["best_assisted_auc"]),
            _format_float(row["best_assisted_minus_baseline"]),
        ]
        for _, row in rows.head(12).iterrows()
    ]

    timing_table, matched_timing_rows = _build_timing_control_table(rows)

    event_table: list[list[Any]] = []
    if not event_rows.empty:
        event_rows = event_rows.copy()
        if "classifier" not in event_rows.columns:
            event_rows["classifier"] = "sgd_logistic"
        if "label_shift_ms" not in event_rows.columns:
            event_rows["label_shift_ms"] = 0.0
        if "baseline_context" not in event_rows.columns:
            event_rows["baseline_context"] = "causal"
        if "smooth_proba_ms" not in event_rows.columns:
            event_rows["smooth_proba_ms"] = 0.0
        event_lift_cols = ["combined_minus_baseline"]
        if "fusion_minus_baseline" in event_rows.columns:
            event_lift_cols.append("fusion_minus_baseline")
        event_rows["event_assisted_lift"] = event_rows[event_lift_cols].astype(float).max(axis=1)
        ev = (
            event_rows.groupby(
                ["event", "classifier", "channel_set", "preprocess", "baseline_context", "smooth_proba_ms", "label_shift_ms"],
                as_index=False,
            )["event_assisted_lift"]
            .mean()
            .sort_values("event_assisted_lift", ascending=False)
        )
        event_table = [
            [
                row["event"],
                row.get("classifier", ""),
                row["channel_set"],
                row["preprocess"],
                row["baseline_context"],
                _format_float(row["smooth_proba_ms"]),
                _format_float(row["label_shift_ms"]),
                _format_float(row["event_assisted_lift"]),
            ]
            for _, row in ev.head(18).iterrows()
        ]

    lines = [
        "# Classification Controls",
        "",
        "Purpose: test whether MFG-assisted classification survives channel-set and preprocessing controls.",
        "",
        "## Summary",
        "",
        f"- Requested configurations: {summary.get('configs_requested', len(rows))}",
        f"- Completed configurations: {len(rows)}",
        f"- Failed configurations: {summary.get('configs_failed', 0)}",
        f"- Best lift: {_format_float(best.get('best_assisted_minus_baseline'))}",
        f"- Best assisted feature set: `{best.get('best_assisted_feature_set', 'combined')}`",
        f"- Best classifier: `{best.get('classifier', '')}`",
        f"- Best channel set: `{best.get('channel_set')}`",
        f"- Best preprocessing: `{best.get('preprocess')}`",
        f"- Best label shift: `{best.get('label_shift_ms', 0.0)}` ms",
        "",
        "## Control Groups",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "classifier",
                "channel_set",
                "preprocess",
                "baseline_context",
                "smooth_ms",
                "label_shift_ms",
                "edge_selection",
                "configs",
                "mean_lift",
                "best_lift",
                "mean_assisted_auc",
            ],
            group_table,
        )
    )
    lines.extend(["", "## Top Configurations", ""])
    lines.extend(
        _markdown_table(
            [
                "config",
                "classifier",
                "channel_set",
                "preprocess",
                "baseline_context",
                "smooth_ms",
                "label_shift_ms",
                "edge_selection",
                "mode",
                "edge_cap",
                "edges_used",
                "assisted",
                "baseline",
                "assisted_auc",
                "lift",
            ],
            top_table,
        )
    )
    lines.extend(
        [
            "",
            "## Matched Timing Control",
            "",
            (
                "Positive `mean_delta` means the aligned labels produced more "
                "combined-vs-baseline lift than the shifted-label control under "
                "the same split and model settings."
            ),
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "shift_ms",
                "classifier",
                "channel_set",
                "preprocess",
                "baseline_context",
                "smooth_ms",
                "edge_selection",
                "matches",
                "mean_aligned_lift",
                "mean_shifted_lift",
                "mean_delta",
                "aligned_wins",
            ],
            timing_table,
        )
    )
    lines.extend(["", "## Event-Level Lift", ""])
    lines.extend(
        _markdown_table(
            ["event", "classifier", "channel_set", "preprocess", "baseline_context", "smooth_ms", "label_shift_ms", "mean_lift"],
            event_table,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation Criteria",
            "",
            (
                "`ocular_proxy_only` is a confound-control setting. Strong performance there indicates possible gaze or blink information. "
                "`no_fp1_fp2` and `motor_premotor` are stricter channel controls for brain-focused interpretation. "
                "Non-zero `label_shift_ms` rows are negative timing controls and are interpreted separately from aligned prediction."
            ),
            "",
        ]
    )
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "configs_completed": int(len(rows)),
        "best": best,
        "matched_timing_rows": matched_timing_rows,
        "outputs": {
            "ablation_report": str(out_path),
        },
    }


def run_controls(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_summary = run_sweep(_build_sweep_args(args))

    sweep_results = out_dir / "sweep_results.csv"
    sweep_event = out_dir / "event_lift.csv"
    ablation_results = out_dir / "ablation_results.csv"
    ablation_event = out_dir / "ablation_event_lift.csv"
    ablation_report = out_dir / "ablation_report.md"
    ablation_summary = out_dir / "ablation_summary.json"

    if sweep_results.exists():
        shutil.copy2(sweep_results, ablation_results)
        rows = pd.read_csv(sweep_results)
    else:
        rows = pd.DataFrame()

    if sweep_event.exists():
        shutil.copy2(sweep_event, ablation_event)
        event_rows = pd.read_csv(sweep_event)
    else:
        event_rows = pd.DataFrame()

    report_summary = _write_ablation_report(
        out_path=ablation_report,
        rows=rows,
        event_rows=event_rows,
        summary=sweep_summary,
    )
    payload = {
        "control_question": (
            "Does MFG-assisted classification survive channel-set and preprocessing controls?"
        ),
        "sweep_summary": sweep_summary,
        "report_summary": report_summary,
        "outputs": {
            "ablation_results": str(ablation_results),
            "ablation_event_lift": str(ablation_event),
            "ablation_report": str(ablation_report),
            "ablation_summary": str(ablation_summary),
            "sweep_results": str(sweep_results),
            "event_lift": str(sweep_event),
        },
    }
    with open(ablation_summary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def build_parser_controls() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the article control grid for MFG-assisted Kaggle EEG classification."
    )
    parser.add_argument("--root", default="data/grasp-and-lift-eeg-detection/train")
    parser.add_argument("--out-dir", default="out/classification_controls")
    parser.add_argument("--edge-table", default="out/article_meta_sensitivity/edge_stability.csv")
    parser.add_argument("--fallback-edge-table", default="out/article_meta_kaggle/meta_edges.csv")
    parser.add_argument("--splits", nargs="+", default=["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"])
    parser.add_argument("--modes", nargs="+", default=["corr"])
    parser.add_argument("--m-values", nargs="+", type=int, default=[2])
    parser.add_argument("--max-edges", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--edge-selections", nargs="+", default=["global", "event_union"])
    parser.add_argument(
        "--channel-sets",
        nargs="+",
        default=["all32", "no_fp1_fp2", "motor_premotor", "ocular_proxy_only"],
    )
    parser.add_argument("--preprocessings", nargs="+", default=["car_only", "bandpass_0_5_48"])
    parser.add_argument("--label-shift-ms-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--strides", nargs="+", type=int, default=[25])
    parser.add_argument("--mfg-feature-modes", nargs="+", default=["expanded"])
    parser.add_argument("--fusion-weights", nargs="*", type=float, default=[])
    parser.add_argument("--baseline-contexts", nargs="+", choices=BASELINE_CONTEXT_CHOICES, default=["causal"])
    parser.add_argument("--smooth-proba-ms-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIER_CHOICES, default=["sgd_logistic"])
    parser.add_argument("--baseline-windows-ms", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--baseline-lags-ms", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--ema-half-life-s", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--tree-estimators", type=int, default=120)
    parser.add_argument("--tree-max-depth", type=int, default=None)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--max-train-samples-per-file", type=int, default=1200)
    parser.add_argument("--max-val-samples-per-file", type=int, default=1200)
    parser.add_argument("--val-keep-all-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", default="out/classification_controls/cache")
    parser.add_argument("--balance-classes", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser_controls()
    args = parser.parse_args(argv)
    summary = run_controls(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
