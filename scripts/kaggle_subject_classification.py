from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.kaggle_classification_benchmark import (
    BASELINE_CONTEXT_CHOICES,
    CHANNEL_SETS,
    CLASSIFIER_CHOICES,
    FEATURE_SET_CHOICES,
    PREPROCESSING_CHOICES,
    build_parser,
    run_benchmark,
)


def _split_tokens(text: str) -> list[str]:
    return [token for token in str(text or "").replace(",", " ").split() if token]


def _parse_int_list(text: str) -> list[int]:
    return [int(token) for token in _split_tokens(text)]


def _parse_split(spec: str) -> tuple[list[int], list[int], str]:
    if "|" not in spec:
        raise ValueError(f"Split must use TRAIN|VAL syntax, got: {spec}")
    left, right = spec.split("|", 1)
    train = _parse_int_list(left)
    val = _parse_int_list(right)
    if not train or not val:
        raise ValueError(f"Split must contain non-empty train and validation series: {spec}")
    return train, val, f"{'-'.join(map(str, train))}|{'-'.join(map(str, val))}"


def _score_lookup(summary: dict[str, Any], metric: str) -> dict[str, float]:
    return {
        str(row["feature_set"]): float(row[metric])
        for row in summary.get("scores", [])
    }


def _best_assisted(auc: dict[str, float]) -> tuple[str, float]:
    candidates = {
        "combined": auc.get("combined", np.nan),
        "fusion": auc.get("fusion", np.nan),
    }
    valid = {name: value for name, value in candidates.items() if not np.isnan(float(value))}
    if not valid:
        return "", float("nan")
    name = max(valid, key=valid.get)
    return name, float(valid[name])


def _format_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _fusion_grid_tag(values: Sequence[float | None]) -> str:
    concrete = [float(value) for value in values if value is not None]
    if not concrete:
        return "none"
    if len(concrete) == 1:
        return _format_tag(concrete[0])
    return f"grid_{_format_tag(concrete[0])}_{_format_tag(concrete[-1])}_n{len(concrete)}"


def _has_fusion_grid(df: pd.DataFrame) -> bool:
    if "fusion_grid" not in df.columns:
        return False
    values = df["fusion_grid"].fillna("").astype(str).str.strip()
    return bool(values.ne("").any())


def _format_cell(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.6g}"


def _run_one(
    args: argparse.Namespace,
    *,
    subject: int,
    split: tuple[list[int], list[int], str],
    run_dir: Path,
    alpha: float,
    fusion_weights: Sequence[float | None],
    random_offset: int,
) -> dict[str, Any]:
    summary_path = run_dir / "benchmark_summary.json"
    if summary_path.exists() and not args.rerun_existing:
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    train_series, val_series, _split_label = split
    bench_args = [
        "--root",
        args.root,
        "--out-dir",
        str(run_dir),
        "--edge-table",
        args.edge_table,
        "--fallback-edge-table",
        args.fallback_edge_table,
        "--feature-sets",
        *args.feature_sets,
        "--subjects",
        str(subject),
        "--train-series",
        *map(str, train_series),
        "--val-series",
        *map(str, val_series),
        "--mode",
        args.mode,
        "--m",
        str(args.m),
        "--max-edges",
        str(args.max_edges),
        "--edge-selection",
        args.edge_selection,
        "--channel-set",
        args.channel_set,
        "--preprocess",
        args.preprocess,
        "--label-shift-ms",
        str(args.label_shift_ms),
        "--stride",
        str(args.stride),
        "--epochs",
        str(args.epochs),
        "--baseline-windows-ms",
        *map(str, args.baseline_windows_ms),
        "--baseline-lags-ms",
        *map(str, args.baseline_lags_ms),
        "--baseline-context",
        args.baseline_context,
        "--mfg-feature-mode",
        args.mfg_feature_mode,
        "--smooth-proba-ms",
        str(args.smooth_proba_ms),
        "--ema-half-life-s",
        str(args.ema_half_life_s),
        "--classifier",
        args.classifier,
        "--alpha",
        str(alpha),
        "--tree-estimators",
        str(args.tree_estimators),
        "--tree-min-samples-leaf",
        str(args.tree_min_samples_leaf),
        "--random-state",
        str(args.random_state + subject + random_offset),
        "--cache-dir",
        str(run_dir.parent / "feature_cache"),
        "--test-root",
        "",
        "--sample-submission",
        "",
    ]
    concrete_fusion_weights = [float(weight) for weight in fusion_weights if weight is not None]
    if len(concrete_fusion_weights) == 1:
        bench_args.extend(["--fusion-weight", str(concrete_fusion_weights[0])])
    elif len(concrete_fusion_weights) > 1:
        bench_args.extend(["--fusion-weights", *map(str, concrete_fusion_weights)])
    if args.max_train_samples_per_file is not None:
        bench_args.extend(["--max-train-samples-per-file", str(args.max_train_samples_per_file)])
    if args.max_val_samples_per_file is not None:
        bench_args.extend(["--max-val-samples-per-file", str(args.max_val_samples_per_file)])
    if args.tree_max_depth is not None:
        bench_args.extend(["--tree-max-depth", str(args.tree_max_depth)])
    if args.val_keep_all_positive:
        bench_args.append("--val-keep-all-positive")
    if args.balance_classes:
        bench_args.append("--balance-classes")

    parsed = build_parser().parse_args(bench_args)
    with contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext():
        return run_benchmark(parsed)


def _write_report(out_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        out_path.write_text("# Subject-Aware Classification\n\nNo runs completed.\n", encoding="utf-8")
        return

    df = pd.DataFrame(rows).sort_values(["best_assisted_minus_baseline", "best_assisted_auc"], ascending=[False, False])
    mean_lift = float(df["best_assisted_minus_baseline"].mean())
    mean_baseline = float(df["baseline_auc"].mean())
    mean_assisted = float(df["best_assisted_auc"].mean())
    fusion_col = "fusion_grid" if _has_fusion_grid(df) else "fusion_weight"
    grouped = (
        df.groupby(["alpha", fusion_col], as_index=False)
        .agg(
            runs=("subject", "count"),
            mean_baseline_auc=("baseline_auc", "mean"),
            mean_assisted_auc=("best_assisted_auc", "mean"),
            mean_lift=("best_assisted_minus_baseline", "mean"),
        )
        .sort_values(["mean_lift", "mean_assisted_auc"], ascending=[False, False])
    )

    lines = [
        "# Subject-Aware Classification",
        "",
        "Purpose: evaluate participant-specific Kaggle event decoders using the same MFG-assisted feature ablation.",
        "",
        "## Summary",
        "",
        f"- Completed subject/split runs: {len(df)}",
        f"- Mean baseline ROC-AUC: {mean_baseline:.6f}",
        f"- Mean assisted ROC-AUC: {mean_assisted:.6f}",
        f"- Mean assisted-baseline lift: {mean_lift:.6f}",
        "",
        "## Hyperparameter Summary",
        "",
        "| alpha | fusion | runs | baseline | assisted | lift |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            "| {alpha} | {fusion} | {runs:.0f} | {mean_baseline_auc:.6f} | {mean_assisted_auc:.6f} | {mean_lift:.6f} |".format(
                alpha=_format_cell(row["alpha"]),
                fusion=_format_cell(row[fusion_col]),
                runs=float(row["runs"]),
                mean_baseline_auc=float(row["mean_baseline_auc"]),
                mean_assisted_auc=float(row["mean_assisted_auc"]),
                mean_lift=float(row["mean_lift"]),
            )
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| subject | split | alpha | fusion | selected_w | baseline | mfg | combined | fusion_auc | best | lift |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for _, row in df.iterrows():
        fusion_value = row.get("fusion_grid", "") if fusion_col == "fusion_grid" else row.get("fusion_weight", "")
        lines.append(
            "| {subject} | {split} | {alpha} | {fusion} | {selected_w} | {baseline_auc:.6f} | {mfg_auc:.6f} | {combined_auc:.6f} | {fusion_auc:.6f} | {best_assisted_feature_set} | {best_assisted_minus_baseline:.6f} |".format(
                subject=row["subject"],
                split=row["split"],
                alpha=_format_cell(row["alpha"]),
                fusion=_format_cell(fusion_value),
                selected_w=_format_cell(row.get("fusion_weight", "")),
                baseline_auc=float(row["baseline_auc"]),
                mfg_auc=float(row["mfg_auc"]),
                combined_auc=float(row["combined_auc"]),
                fusion_auc=float(row["fusion_auc"]),
                best_assisted_feature_set=row["best_assisted_feature_set"],
                best_assisted_minus_baseline=float(row["best_assisted_minus_baseline"]),
            )
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_subject_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    splits = [_parse_split(spec) for spec in args.splits]
    alpha_values = args.alpha_values if args.alpha_values else [args.alpha]
    fusion_grids: list[list[float | None]]
    if args.fusion_weights and args.fusion_weight_values:
        raise ValueError("Use either --fusion-weights for validation selection or --fusion-weight-values for a fixed sweep.")
    if args.fusion_weights:
        fusion_grids = [[float(weight) for weight in args.fusion_weights]]
    elif args.fusion_weight_values:
        fusion_grids = [[float(weight)] for weight in args.fusion_weight_values]
    else:
        fusion_grids = [[args.fusion_weight]]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for subject in args.subjects:
        for split_idx, split in enumerate(splits, start=1):
            _train_series, _val_series, split_label = split
            for alpha_idx, alpha in enumerate(alpha_values):
                for fusion_idx, fusion_grid in enumerate(fusion_grids):
                    run_dir = (
                        runs_dir
                        / f"subj{int(subject):02d}_split{split_idx:02d}"
                        / f"alpha_{_format_tag(float(alpha))}_fusion_{_fusion_grid_tag(fusion_grid)}"
                    )
                    try:
                        summary = _run_one(
                            args,
                            subject=int(subject),
                            split=split,
                            run_dir=run_dir,
                            alpha=float(alpha),
                            fusion_weights=fusion_grid,
                            random_offset=0,
                        )
                        auc = _score_lookup(summary, "mean_roc_auc")
                        ap = _score_lookup(summary, "mean_average_precision")
                        best_name, best_auc = _best_assisted(auc)
                        baseline_auc = auc.get("baseline", np.nan)
                        parameters = summary.get("parameters", {})
                        selected_fusion_weight = parameters.get("selected_fusion_weight")
                        rows.append(
                            {
                                "subject": int(subject),
                                "split": split_label,
                                "classifier": args.classifier,
                                "channel_set": args.channel_set,
                                "preprocess": args.preprocess,
                                "baseline_context": args.baseline_context,
                                "smooth_proba_ms": float(args.smooth_proba_ms),
                                "val_keep_all_positive": bool(args.val_keep_all_positive),
                                "edge_selection": args.edge_selection,
                                "mode": args.mode,
                                "m": int(args.m),
                                "max_edges": int(args.max_edges),
                                "alpha": float(alpha),
                                "fusion_weight": float(selected_fusion_weight)
                                if selected_fusion_weight is not None
                                else (
                                    float(fusion_grid[0])
                                    if len(fusion_grid) == 1 and fusion_grid[0] is not None
                                    else np.nan
                                ),
                                "fusion_grid": " ".join(
                                    f"{float(weight):g}" for weight in fusion_grid if weight is not None
                                ),
                                "label_shift_ms": float(args.label_shift_ms),
                                "stride": int(args.stride),
                                "mfg_feature_mode": args.mfg_feature_mode,
                                "baseline_auc": baseline_auc,
                                "mfg_auc": auc.get("mfg", np.nan),
                                "combined_auc": auc.get("combined", np.nan),
                                "fusion_auc": auc.get("fusion", np.nan),
                                "baseline_ap": ap.get("baseline", np.nan),
                                "mfg_ap": ap.get("mfg", np.nan),
                                "combined_ap": ap.get("combined", np.nan),
                                "fusion_ap": ap.get("fusion", np.nan),
                                "best_assisted_feature_set": best_name,
                                "best_assisted_auc": best_auc,
                                "best_assisted_minus_baseline": best_auc - baseline_auc,
                                "result_dir": str(run_dir),
                            }
                        )
                        print(
                            f"subject={int(subject):02d} split={split_label} "
                            f"alpha={float(alpha):g} fusion={_fusion_grid_tag(fusion_grid)} "
                            f"baseline={baseline_auc:.6f} best={best_auc:.6f} "
                            f"lift={best_auc - baseline_auc:.6f}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - keep long benchmark runs going
                        failures.append(
                            {
                                "subject": int(subject),
                                "split": split_label,
                                "alpha": float(alpha),
                                "fusion_grid": " ".join(
                                    f"{float(weight):g}" for weight in fusion_grid if weight is not None
                                ),
                                "error": str(exc),
                            }
                        )
                        print(
                            f"subject={int(subject):02d} split={split_label} "
                            f"alpha={float(alpha):g} fusion={_fusion_grid_tag(fusion_grid)} failed: {exc}",
                            flush=True,
                        )

    rows_sorted = sorted(rows, key=lambda row: (row["best_assisted_minus_baseline"], row["best_assisted_auc"]), reverse=True)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "subject_results.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)
    _write_report(out_dir / "subject_report.md", rows_sorted)

    payload = {
        "runs_requested": len(args.subjects) * len(splits) * len(alpha_values) * len(fusion_grids),
        "runs_completed": len(rows_sorted),
        "runs_failed": len(failures),
        "best": rows_sorted[0] if rows_sorted else None,
        "outputs": {
            "subject_results": str(out_dir / "subject_results.csv"),
            "subject_report": str(out_dir / "subject_report.md"),
            "failures": str(out_dir / "failures.csv") if failures else None,
        },
    }
    with open(out_dir / "subject_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def build_parser_subject() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run subject-aware Kaggle EEG classification ablations.")
    parser.add_argument("--root", default="data/grasp-and-lift-eeg-detection/train")
    parser.add_argument("--out-dir", default="out/classification_subject")
    parser.add_argument("--edge-table", default="out/article_meta_sensitivity/edge_stability.csv")
    parser.add_argument("--fallback-edge-table", default="out/article_meta_kaggle/meta_edges.csv")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--splits", nargs="+", default=["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"])
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SET_CHOICES, default=["baseline", "mfg", "combined"])
    parser.add_argument("--mode", choices=["gc", "corr"], default="corr")
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--max-edges", type=int, default=8)
    parser.add_argument("--edge-selection", choices=["global", "event_union"], default="event_union")
    parser.add_argument("--channel-set", choices=sorted(CHANNEL_SETS), default="all32")
    parser.add_argument("--preprocess", choices=PREPROCESSING_CHOICES, default="bandpass_0_5_48")
    parser.add_argument("--label-shift-ms", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--baseline-windows-ms", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--baseline-lags-ms", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--baseline-context", choices=BASELINE_CONTEXT_CHOICES, default="causal")
    parser.add_argument("--mfg-feature-mode", choices=["energy", "expanded"], default="expanded")
    parser.add_argument("--smooth-proba-ms", type=float, default=0.0)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--fusion-weights", nargs="*", type=float, default=None)
    parser.add_argument("--fusion-weight-values", nargs="+", type=float, default=None)
    parser.add_argument("--ema-half-life-s", type=float, default=0.1)
    parser.add_argument("--classifier", choices=CLASSIFIER_CHOICES, default="sgd_logistic")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--alpha-values", nargs="+", type=float, default=None)
    parser.add_argument("--tree-estimators", type=int, default=120)
    parser.add_argument("--tree-max-depth", type=int, default=None)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-samples-per-file", type=int, default=1200)
    parser.add_argument("--max-val-samples-per-file", type=int, default=1200)
    parser.add_argument("--val-keep-all-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-classes", action="store_true", default=True)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser_subject()
    args = parser.parse_args(argv)
    summary = run_subject_benchmark(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
