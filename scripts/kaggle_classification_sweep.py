from __future__ import annotations

import argparse
import contextlib
import csv
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
    PREPROCESSING_CHOICES,
    build_parser,
    run_benchmark,
)
from src_mfg.io import KAGGLE_EVENT_COLS


def _split_tokens(text: str) -> list[str]:
    return [token for token in str(text or "").replace(",", " ").split() if token]


def _parse_int_list(text: str) -> list[int]:
    return [int(token) for token in _split_tokens(text)]


def _parse_str_list(text: str) -> list[str]:
    return _split_tokens(text)


def _parse_split(spec: str) -> tuple[list[int], list[int], str]:
    if "|" not in spec:
        raise ValueError(f"Split must use TRAIN|VAL syntax, got: {spec}")
    left, right = spec.split("|", 1)
    train = _parse_int_list(left)
    val = _parse_int_list(right)
    if not train or not val:
        raise ValueError(f"Split must contain non-empty train and validation series: {spec}")
    return train, val, f"{'-'.join(map(str, train))}|{'-'.join(map(str, val))}"


def _run_one(args: argparse.Namespace, *, config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    bench_args = [
        "--root",
        args.root,
        "--out-dir",
        str(out_dir),
        "--edge-table",
        args.edge_table,
        "--fallback-edge-table",
        args.fallback_edge_table,
        "--feature-sets",
        "baseline",
        "mfg",
        "combined",
        "--train-series",
        *map(str, config["train_series"]),
        "--val-series",
        *map(str, config["val_series"]),
        "--mode",
        config["mode"],
        "--m",
        str(config["m"]),
        "--max-edges",
        str(config["max_edges"]),
        "--edge-selection",
        config["edge_selection"],
        "--channel-set",
        config["channel_set"],
        "--preprocess",
        config["preprocess"],
        "--label-shift-ms",
        str(config["label_shift_ms"]),
        "--stride",
        str(config["stride"]),
        "--epochs",
        str(config["epochs"]),
        "--baseline-windows-ms",
        *map(str, config["baseline_windows_ms"]),
        "--baseline-lags-ms",
        *map(str, config["baseline_lags_ms"]),
        "--baseline-context",
        config["baseline_context"],
        "--mfg-feature-mode",
        config["mfg_feature_mode"],
        "--smooth-proba-ms",
        str(config["smooth_proba_ms"]),
        "--ema-half-life-s",
        str(config["ema_half_life_s"]),
        "--classifier",
        config["classifier"],
        "--alpha",
        str(args.alpha),
        "--tree-estimators",
        str(args.tree_estimators),
        "--tree-min-samples-leaf",
        str(args.tree_min_samples_leaf),
        "--random-state",
        str(args.random_state),
        "--test-root",
        "",
        "--sample-submission",
        "",
    ]
    if config["fusion_weight"] is not None:
        bench_args.extend(["--fusion-weight", str(config["fusion_weight"])])
    if args.max_train_files is not None:
        bench_args.extend(["--max-train-files", str(args.max_train_files)])
    if args.max_val_files is not None:
        bench_args.extend(["--max-val-files", str(args.max_val_files)])
    if args.max_train_samples_per_file is not None:
        bench_args.extend(["--max-train-samples-per-file", str(args.max_train_samples_per_file)])
    if args.max_val_samples_per_file is not None:
        bench_args.extend(["--max-val-samples-per-file", str(args.max_val_samples_per_file)])
    if args.tree_max_depth is not None:
        bench_args.extend(["--tree-max-depth", str(args.tree_max_depth)])
    if args.val_keep_all_positive:
        bench_args.append("--val-keep-all-positive")
    if args.cache_dir:
        bench_args.extend(["--cache-dir", str(Path(args.cache_dir) / f"cfg_{config['config_id']:03d}")])
    else:
        bench_args.extend(["--cache-dir", ""])
    if args.balance_classes:
        bench_args.append("--balance-classes")

    parsed = build_parser().parse_args(bench_args)
    with contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext():
        return run_benchmark(parsed)


def _score_lookup(summary: dict[str, Any]) -> dict[str, float]:
    return {
        row["feature_set"]: float(row["mean_roc_auc"])
        for row in summary.get("scores", [])
    }


def _ap_lookup(summary: dict[str, Any]) -> dict[str, float]:
    return {
        row["feature_set"]: float(row["mean_average_precision"])
        for row in summary.get("scores", [])
    }


def _nanmax_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.max())


def _assisted_feature_set(auc: dict[str, float]) -> str:
    candidates = {
        "combined": auc.get("combined", np.nan),
        "fusion": auc.get("fusion", np.nan),
    }
    valid = {name: value for name, value in candidates.items() if not np.isnan(float(value))}
    if not valid:
        return ""
    return max(valid, key=valid.get)


def _event_lift(config: dict[str, Any], result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "auc_by_event.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    out: list[dict[str, Any]] = []
    for event in KAGGLE_EVENT_COLS:
        sub = df[df["event"] == event]
        vals = {
            str(row["feature_set"]): float(row["roc_auc"])
            for _, row in sub.iterrows()
        }
        out.append(
            {
                "config_id": config["config_id"],
                "split": config["split"],
                "mode": config["mode"],
                "m": config["m"],
                "max_edges": config["max_edges"],
                "edge_selection": config["edge_selection"],
                "classifier": config["classifier"],
                "channel_set": config["channel_set"],
                "preprocess": config["preprocess"],
                "label_shift_ms": config["label_shift_ms"],
                "mfg_feature_mode": config["mfg_feature_mode"],
                "baseline_context": config["baseline_context"],
                "smooth_proba_ms": config["smooth_proba_ms"],
                "fusion_weight": config["fusion_weight"],
                "event": event,
                "baseline_auc": vals.get("baseline", np.nan),
                "mfg_auc": vals.get("mfg", np.nan),
                "combined_auc": vals.get("combined", np.nan),
                "fusion_auc": vals.get("fusion", np.nan),
                "combined_minus_baseline": vals.get("combined", np.nan) - vals.get("baseline", np.nan),
                "mfg_minus_baseline": vals.get("mfg", np.nan) - vals.get("baseline", np.nan),
                "fusion_minus_baseline": vals.get("fusion", np.nan) - vals.get("baseline", np.nan),
            }
        )
    return out


def _write_markdown(out_path: Path, rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> None:
    if not rows:
        out_path.write_text("# Classification Sweep\n\nNo configurations completed.\n", encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    if "best_assisted_minus_baseline" not in df.columns:
        df["best_assisted_minus_baseline"] = df["combined_minus_baseline"]
        df["best_assisted_auc"] = df["combined_auc"]
        df["best_assisted_feature_set"] = "combined"
    if "fusion_weight" not in df.columns:
        df["fusion_weight"] = np.nan
    if "fusion_auc" not in df.columns:
        df["fusion_auc"] = np.nan
        df["fusion_minus_baseline"] = np.nan
    if "baseline_context" not in df.columns:
        df["baseline_context"] = "causal"
    if "smooth_proba_ms" not in df.columns:
        df["smooth_proba_ms"] = 0.0
    df["fusion_weight"] = df["fusion_weight"].apply(
        lambda value: "none" if pd.isna(value) else f"{float(value):.3g}"
    )
    df = df.sort_values(
        ["best_assisted_minus_baseline", "best_assisted_auc"],
        ascending=[False, False],
    )
    if "label_shift_ms" not in df.columns:
        df["label_shift_ms"] = 0.0
    win_rate = float((df["best_assisted_minus_baseline"] > 0).mean())
    mean_lift = float(df["best_assisted_minus_baseline"].mean())
    best = df.iloc[0].to_dict()

    lines = [
        "# Classification Sweep",
        "",
        "Question: do MFG-derived features improve sample-level event classification beyond a baseline EEG predictor?",
        "",
        "## Summary",
        "",
        f"- Completed configurations: {len(df)}",
        f"- Assisted > baseline win rate: {win_rate:.3f}",
        f"- Mean best assisted-baseline ROC-AUC lift: {mean_lift:.6f}",
        f"- Best lift: {float(best['best_assisted_minus_baseline']):.6f} in config {int(best['config_id'])}",
        "",
        "## Top Configurations",
        "",
        "| config | split | classifier | channels | preprocess | context | smooth_ms | label_shift_ms | fusion_w | edge selection | mode | m | edge cap | used | stride | mfg_mode | baseline | mfg | combined | fusion | best | best_lift |",
        "|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in df.head(12).iterrows():
        lines.append(
            "| {config_id} | {split} | {classifier} | {channel_set} | {preprocess} | {baseline_context} | {smooth_proba_ms} | {label_shift_ms} | {fusion_weight} | {edge_selection} | {mode} | {m} | {max_edges} | {edges_used} | {stride} | {mfg_feature_mode} | "
            "{baseline_auc:.6f} | {mfg_auc:.6f} | {combined_auc:.6f} | {fusion_auc:.6f} | {best_assisted_feature_set} | {best_assisted_minus_baseline:.6f} |".format(
                **row.to_dict()
            )
        )

    if event_rows:
        ev = pd.DataFrame(event_rows)
        ev_summary = (
            ev.groupby("event", as_index=False)["combined_minus_baseline"]
            .mean()
            .sort_values("combined_minus_baseline", ascending=False)
        )
        lines.extend(
            [
                "",
                "## Mean Event-Level Lift",
                "",
                "| event | mean combined-baseline lift |",
                "|---|---:|",
            ]
        )
        for _, row in ev_summary.iterrows():
            lines.append(f"| {row['event']} | {float(row['combined_minus_baseline']):.6f} |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    splits = [_parse_split(spec) for spec in args.splits]
    fusion_weights = [None] if not args.fusion_weights else [float(weight) for weight in args.fusion_weights]
    for weight in fusion_weights:
        if weight is not None and not 0.0 <= weight <= 1.0:
            raise ValueError("--fusion-weights entries must be between 0 and 1.")
    config_id = 1
    for train_series, val_series, split_label in splits:
        for mode in args.modes:
            for m in args.m_values:
                for max_edges in args.max_edges:
                    for edge_selection in args.edge_selections:
                        for channel_set in args.channel_sets:
                            for preprocess in args.preprocessings:
                                for label_shift_ms in args.label_shift_ms_values:
                                    for stride in args.strides:
                                        for mfg_feature_mode in args.mfg_feature_modes:
                                            for classifier in args.classifiers:
                                                for baseline_context in args.baseline_contexts:
                                                    for smooth_proba_ms in args.smooth_proba_ms_values:
                                                        for fusion_weight in fusion_weights:
                                                            configs.append(
                                                                {
                                                                    "config_id": config_id,
                                                                    "split": split_label,
                                                                    "train_series": train_series,
                                                                    "val_series": val_series,
                                                                    "mode": mode,
                                                                    "m": int(m),
                                                                    "max_edges": int(max_edges),
                                                                    "edge_selection": edge_selection,
                                                                    "classifier": classifier,
                                                                    "channel_set": channel_set,
                                                                    "preprocess": preprocess,
                                                                    "label_shift_ms": float(label_shift_ms),
                                                                    "stride": int(stride),
                                                                    "epochs": int(args.epochs),
                                                                    "baseline_windows_ms": list(map(int, args.baseline_windows_ms)),
                                                                    "baseline_lags_ms": list(map(int, args.baseline_lags_ms)),
                                                                    "baseline_context": baseline_context,
                                                                    "smooth_proba_ms": float(smooth_proba_ms),
                                                                    "mfg_feature_mode": mfg_feature_mode,
                                                                    "ema_half_life_s": float(args.ema_half_life_s),
                                                                    "fusion_weight": fusion_weight,
                                                                }
                                                            )
                                                            config_id += 1
    return configs


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for config in configs:
        run_dir = runs_dir / f"cfg_{config['config_id']:03d}"
        try:
            summary = _run_one(args, config=config, out_dir=run_dir)
            auc = _score_lookup(summary)
            ap = _ap_lookup(summary)
            edges_used = int(summary.get("parameters", {}).get("edges_used", config["max_edges"]))
            baseline_auc = auc.get("baseline", np.nan)
            mfg_auc = auc.get("mfg", np.nan)
            combined_auc = auc.get("combined", np.nan)
            fusion_auc = auc.get("fusion", np.nan)
            best_assisted_auc = _nanmax_or_nan([combined_auc, fusion_auc])
            best_assisted_feature_set = _assisted_feature_set(auc)
            row = {
                **{
                    key: value
                    for key, value in config.items()
                    if key not in {"train_series", "val_series", "baseline_windows_ms", "baseline_lags_ms"}
                },
                "train_series": " ".join(map(str, config["train_series"])),
                "val_series": " ".join(map(str, config["val_series"])),
                "baseline_windows_ms": " ".join(map(str, config["baseline_windows_ms"])),
                "baseline_lags_ms": " ".join(map(str, config["baseline_lags_ms"])),
                "edges_used": edges_used,
                "baseline_auc": baseline_auc,
                "mfg_auc": mfg_auc,
                "combined_auc": combined_auc,
                "fusion_auc": fusion_auc,
                "baseline_ap": ap.get("baseline", np.nan),
                "mfg_ap": ap.get("mfg", np.nan),
                "combined_ap": ap.get("combined", np.nan),
                "fusion_ap": ap.get("fusion", np.nan),
                "combined_minus_baseline": combined_auc - baseline_auc,
                "mfg_minus_baseline": mfg_auc - baseline_auc,
                "fusion_minus_baseline": fusion_auc - baseline_auc,
                "best_assisted_auc": best_assisted_auc,
                "best_assisted_feature_set": best_assisted_feature_set,
                "best_assisted_minus_baseline": best_assisted_auc - baseline_auc,
                "val_keep_all_positive": bool(args.val_keep_all_positive),
                "result_dir": str(run_dir),
            }
            rows.append(row)
            event_rows.extend(_event_lift(config, run_dir))
            print(
                f"cfg={config['config_id']:03d} split={config['split']} mode={config['mode']} "
                f"classifier={config['classifier']} "
                f"channels={config['channel_set']} preprocess={config['preprocess']} "
                f"context={config['baseline_context']} smooth_ms={config['smooth_proba_ms']} "
                f"label_shift_ms={config['label_shift_ms']} "
                f"edge_selection={config['edge_selection']} edges={config['max_edges']} "
                f"fusion_weight={config['fusion_weight']} used={edges_used} "
                f"best_lift={row['best_assisted_minus_baseline']:.6f}"
            )
        except Exception as exc:  # noqa: BLE001 - sweep should continue and report failures
            failures.append({"config_id": str(config["config_id"]), "error": str(exc)})
            print(f"cfg={config['config_id']:03d} failed: {exc}")

    rows_sorted = sorted(
        rows,
        key=lambda row: (row["best_assisted_minus_baseline"], row["best_assisted_auc"]),
        reverse=True,
    )
    pd.DataFrame(rows_sorted).to_csv(out_dir / "sweep_results.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out_dir / "event_lift.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)
    _write_markdown(out_dir / "sweep_report.md", rows_sorted, event_rows)

    payload = {
        "configs_requested": len(configs),
        "configs_completed": len(rows_sorted),
        "configs_failed": len(failures),
        "best": rows_sorted[0] if rows_sorted else None,
        "outputs": {
            "sweep_results": str(out_dir / "sweep_results.csv"),
            "event_lift": str(out_dir / "event_lift.csv"),
            "sweep_report": str(out_dir / "sweep_report.md"),
            "failures": str(out_dir / "failures.csv") if failures else None,
        },
    }
    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def build_parser_sweep() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classification ablations for MFG-assisted Kaggle EEG prediction.")
    parser.add_argument("--root", default="data/grasp-and-lift-eeg-detection/train")
    parser.add_argument("--out-dir", default="out/classification_sweep")
    parser.add_argument("--edge-table", default="out/article_meta_sensitivity/edge_stability.csv")
    parser.add_argument("--fallback-edge-table", default="out/article_meta_kaggle/meta_edges.csv")
    parser.add_argument("--splits", nargs="+", default=["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"])
    parser.add_argument("--modes", nargs="+", default=["corr", "gc"], choices=["corr", "gc"])
    parser.add_argument("--m-values", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--max-edges", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--edge-selections", nargs="+", default=["global"], choices=["global", "event_union"])
    parser.add_argument("--channel-sets", nargs="+", default=["all32"], choices=sorted(CHANNEL_SETS))
    parser.add_argument("--preprocessings", nargs="+", default=["car_only"], choices=PREPROCESSING_CHOICES)
    parser.add_argument("--label-shift-ms-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--strides", nargs="+", type=int, default=[10, 25])
    parser.add_argument("--mfg-feature-modes", nargs="+", default=["energy", "expanded"], choices=["energy", "expanded"])
    parser.add_argument("--fusion-weights", nargs="*", type=float, default=[])
    parser.add_argument("--baseline-windows-ms", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--baseline-lags-ms", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--baseline-contexts", nargs="+", choices=BASELINE_CONTEXT_CHOICES, default=["causal"])
    parser.add_argument("--smooth-proba-ms-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--ema-half-life-s", type=float, default=0.1)
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIER_CHOICES, default=["sgd_logistic"])
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
    parser.add_argument("--cache-dir", default="out/classification_sweep/cache")
    parser.add_argument("--balance-classes", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser_sweep()
    args = parser.parse_args(argv)
    summary = run_sweep(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
