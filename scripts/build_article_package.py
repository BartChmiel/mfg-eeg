from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCENARIO_FIELDS = [
    "phase",
    "pc",
    "lag_ms",
    "n_subjects",
    "total_subjects_in_dir",
    "candidate_edges",
    "reported_edges",
    "dominant_flow_src",
    "dominant_flow_dst",
    "dominant_flow_count",
    "dominant_flow_di",
    "process_label",
    "process_desc",
]

EDGE_FIELDS = [
    "phase",
    "pc",
    "lag_ms",
    "src",
    "dst",
    "region_src",
    "region_dst",
    "k",
    "n_subjects",
    "total_subjects_in_dir",
    "p_value",
    "q_value",
    "significant",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scenario_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -_safe_int(row.get("reported_edges")),
        -_safe_int(row.get("candidate_edges")),
        -_safe_int(row.get("dominant_flow_count")),
        str(row.get("phase", "")),
        str(row.get("pc", "")),
        _safe_int(row.get("lag_ms")),
    )


def _edge_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    q_value = _safe_float(row.get("q_value"), _safe_float(row.get("p_value"), 1.0))
    return (
        not _safe_bool(row.get("significant")),
        -_safe_int(row.get("k")),
        q_value,
        str(row.get("phase", "")),
        str(row.get("pc", "")),
        _safe_int(row.get("lag_ms")),
        str(row.get("src", "")),
        str(row.get("dst", "")),
    )


def primary_sensitivity_rows(
    edges: list[dict[str, Any]], stability: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row["phase"], row["pc"], _safe_int(row["lag_ms"]), row["src"], row["dst"])

    indexed = {key(row): row for row in stability}
    result = []
    for edge in sorted(edges, key=_edge_sort_key):
        if not _safe_bool(edge.get("significant")):
            continue
        support = indexed.get(key(edge))
        if support is None:
            raise ValueError(f"Primary significant instance missing from sensitivity grid: {key(edge)}")
        result.append({**edge, **{name: support[name] for name in
            ("config_count", "successful_config_count", "stability_fraction")}})
    return result


def write_primary_sensitivity_table(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = EDGE_FIELDS + ["config_count", "successful_config_count", "stability_fraction"]
    _write_csv(out_dir / "primary_edge_sensitivity.csv", rows, fields)
    lines = [r"\begin{tabular}{p{0.34\textwidth}clcccc}", r"\toprule",
        r"Phase & PC & Edge & Lag (ms) & $K/N$ & $q$ & Settings \\", r"\midrule"]
    for row in rows:
        phase = str(row["phase"]).replace("__", " -> ").replace("_", r"\_")
        mantissa, exponent = f"{float(row['q_value']):.2e}".split("e")
        q = rf"${mantissa}\times10^{{{int(exponent)}}}$"
        cells = [rf"\code{{{phase}}}", str(row["pc"]).upper(),
            rf"{row['src']} $\to$ {row['dst']}", str(row["lag_ms"]),
            f"{row['k']}/{row['n_subjects']}", q,
            f"{row['config_count']}/{row['successful_config_count']}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "primary_edge_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_rglob(root: Path, pattern: str) -> Iterable[Path]:
    if not root.exists():
        return []
    try:
        return list(root.rglob(pattern))
    except OSError:
        return []


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return str(path)


def _copy_existing(src_dir: Path, out_dir: Path, names: list[str]) -> list[str]:
    copied: list[str] = []
    raw_dir = out_dir / "raw_meta_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = raw_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _copy_project_docs(out_dir: Path) -> list[str]:
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "documentation"
    dst_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        "METHODOLOGY.md",
        "eeg_mfg_article.pdf",
        "references.bib",
    }
    for existing in dst_dir.iterdir():
        if existing.name not in allowed:
            existing.unlink()
    for name in sorted(allowed):
        src = docs_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _copy_volume_conduction(src_dir: Path | None, out_dir: Path) -> list[str]:
    if src_dir is None or not src_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "volume_conduction"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "volume_conduction_report.md",
        "volume_conduction_summary.json",
        "volume_conduction_edges.csv",
        "vc_robust_edge_stability.csv",
    ]:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _volume_conduction_stats(vc_dir: Path | None) -> dict[str, Any] | None:
    if vc_dir is None:
        return None
    summary_path = vc_dir / "volume_conduction_summary.json"
    if not summary_path.exists():
        return None
    data = _read_json(summary_path)
    if not data:
        return None
    return {
        "total_edges": data.get("total_edges"),
        "short_range_fraction": data.get("short_range_fraction"),
        "vc_robust_edges": data.get("vc_robust_edges"),
        "vc_robust_fraction": data.get("vc_robust_fraction"),
        "observed_mean_distance_cm": (data.get("spatial_enrichment") or {}).get("observed_mean_distance_cm"),
        "permutation_p_shorter": (data.get("spatial_enrichment") or {}).get("permutation_p_shorter"),
    }


def _copy_sensitivity(src_dir: Path | None, out_dir: Path) -> list[str]:
    if src_dir is None or not src_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "sensitivity"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "sensitivity_summary.md",
        "sensitivity_edges.csv",
        "edge_stability.csv",
        "sensitivity_manifest.json",
    ]:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _copy_classification(src_dir: Path | None, out_dir: Path) -> list[str]:
    if src_dir is None or not src_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "classification"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "benchmark_report.md",
        "benchmark_summary.json",
        "model_comparison.csv",
        "auc_by_event.csv",
        "mfg_edges_used.csv",
    ]:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _copy_classification_sweep(src_dir: Path | None, out_dir: Path) -> list[str]:
    if src_dir is None or not src_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "classification_sweep"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "sweep_report.md",
        "sweep_summary.json",
        "sweep_results.csv",
        "event_lift.csv",
        "failures.csv",
    ]:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _copy_classification_controls(src_dir: Path | None, out_dir: Path) -> list[str]:
    if src_dir is None or not src_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "classification_controls"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "ablation_report.md",
        "ablation_summary.json",
        "ablation_results.csv",
        "ablation_event_lift.csv",
        "failures.csv",
    ]:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _copy_classification_subject(src_dir: Path | None, out_dir: Path) -> list[str]:
    if src_dir is None or not src_dir.exists():
        return []
    copied: list[str] = []
    dst_dir = out_dir / "classification_subject"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "subject_report.md",
        "subject_summary.json",
        "subject_results.csv",
        "timing_control.md",
        "timing_control.csv",
        "timing_control_groups.csv",
        "timing_control_summary.json",
        "failures.csv",
    ]:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _manifest_summary(path: Path, base: Path) -> dict[str, Any]:
    data = _read_json(path) or {}
    return {
        "path": _relative(path, base),
        "group": data.get("group"),
        "mode": data.get("mode"),
        "files_used": data.get("files_used"),
        "cycles_total": data.get("cycles_total"),
        "windows_total": data.get("windows_total"),
        "anchor_windows_total": data.get("anchor_windows_total"),
    }


def _md_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "/")


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return out


def _fmt_metric(value: Any, digits: int = 4, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.{digits}f}"


def _exact_positive_sign_p(wins: int, total: int) -> float | None:
    if total <= 0:
        return None
    return sum(math.comb(total, i) for i in range(wins, total + 1)) / float(2**total)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = 20000,
    seed: int = 20260617,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _classification_subject_stats(classification_subject_dir: Path | None) -> dict[str, Any] | None:
    if classification_subject_dir is None:
        return None
    timing_path = classification_subject_dir / "timing_control.csv"
    rows = _read_csv(timing_path)
    if not rows:
        return None

    deltas = [_safe_float(row.get("aligned_minus_shifted"), 0.0) for row in rows]
    wins = sum(delta > 0 for delta in deltas)
    subject_values: dict[str, list[float]] = {}
    for row, delta in zip(rows, deltas):
        subject_values.setdefault(str(row.get("subject", "")), []).append(delta)
    subject_means = [
        statistics.fmean(values)
        for subject, values in sorted(subject_values.items())
        if subject and values
    ]
    subject_wins = sum(value > 0 for value in subject_means)
    ci_low, ci_high = _bootstrap_mean_ci(subject_means)

    return {
        "matched_rows": len(rows),
        "row_mean_delta": statistics.fmean(deltas),
        "row_median_delta": statistics.median(deltas),
        "row_positive": wins,
        "row_sign_p": _exact_positive_sign_p(wins, len(deltas)),
        "subjects": len(subject_means),
        "subject_mean_delta": statistics.fmean(subject_means) if subject_means else None,
        "subject_median_delta": statistics.median(subject_means) if subject_means else None,
        "subject_positive": subject_wins,
        "subject_sign_p": _exact_positive_sign_p(subject_wins, len(subject_means)),
        "subject_bootstrap_ci_low": ci_low,
        "subject_bootstrap_ci_high": ci_high,
    }


def _sensitivity_stats(sensitivity_dir: Path | None) -> dict[str, Any] | None:
    if sensitivity_dir is None:
        return None
    edge_path = sensitivity_dir / "edge_stability.csv"
    rows = _read_csv(edge_path)
    if not rows:
        return None
    full_stable = [
        row
        for row in rows
        if _safe_float(row.get("stability_fraction"), 0.0) >= 1.0
    ]
    region_counts: dict[tuple[str, str], int] = {}
    for row in full_stable:
        key = (str(row.get("region_src", "")), str(row.get("region_dst", "")))
        region_counts[key] = region_counts.get(key, 0) + 1
    top_region = max(region_counts.items(), key=lambda item: item[1], default=(("", ""), 0))

    return {
        "unique_stable_edges": len(rows),
        "fully_stable_edges": len(full_stable),
        "stable_90_edges": sum(_safe_float(row.get("stability_fraction"), 0.0) >= 0.9 for row in rows),
        "stable_75_edges": sum(_safe_float(row.get("stability_fraction"), 0.0) >= 0.75 for row in rows),
        "stable_50_edges": sum(_safe_float(row.get("stability_fraction"), 0.0) >= 0.5 for row in rows),
        "top_full_stable_region_flow": f"{top_region[0][0]}->{top_region[0][1]}" if top_region[1] else "",
        "top_full_stable_region_count": top_region[1],
    }


def _write_evidence_tables(
    out_dir: Path,
    classifier_stats: dict[str, Any] | None,
    sensitivity_stats: dict[str, Any] | None,
    preprocessing_comparison: list[dict[str, Any]] | None = None,
) -> list[str]:
    copied: list[str] = []
    tables_dir = out_dir / "tables"
    if classifier_stats:
        classifier_path = tables_dir / "classifier_evidence.csv"
        _write_csv(
            classifier_path,
            [classifier_stats],
            [
                "matched_rows",
                "row_mean_delta",
                "row_median_delta",
                "row_positive",
                "row_sign_p",
                "subjects",
                "subject_mean_delta",
                "subject_median_delta",
                "subject_positive",
                "subject_sign_p",
                "subject_bootstrap_ci_low",
                "subject_bootstrap_ci_high",
            ],
        )
        copied.append(str(classifier_path))
    if sensitivity_stats:
        sensitivity_path = tables_dir / "sensitivity_evidence.csv"
        _write_csv(
            sensitivity_path,
            [sensitivity_stats],
            [
                "unique_stable_edges",
                "fully_stable_edges",
                "stable_90_edges",
                "stable_75_edges",
                "stable_50_edges",
                "top_full_stable_region_flow",
                "top_full_stable_region_count",
            ],
        )
        copied.append(str(sensitivity_path))
    if preprocessing_comparison:
        comparison_path = tables_dir / "preprocessing_comparison.csv"
        _write_csv(
            comparison_path,
            preprocessing_comparison,
            [
                "preprocessing",
                "stable_edge_count",
                "fully_stable_edges",
                "vc_robust_edges",
                "vc_robust_fraction",
                "short_range_fraction",
                "observed_mean_distance_cm",
                "permutation_p_shorter",
            ],
        )
        copied.append(str(comparison_path))
    return copied


def _preprocessing_comparison_row(
    label: str,
    sensitivity_dir: Path | None,
    volume_conduction_dir: Path | None,
) -> dict[str, Any] | None:
    sensitivity = _sensitivity_stats(sensitivity_dir)
    volume_conduction = _volume_conduction_stats(volume_conduction_dir)
    if not sensitivity and not volume_conduction:
        return None
    return {
        "preprocessing": label,
        "stable_edge_count": (sensitivity or {}).get("unique_stable_edges"),
        "fully_stable_edges": (sensitivity or {}).get("fully_stable_edges"),
        "vc_robust_edges": (volume_conduction or {}).get("vc_robust_edges"),
        "vc_robust_fraction": (volume_conduction or {}).get("vc_robust_fraction"),
        "short_range_fraction": (volume_conduction or {}).get("short_range_fraction"),
        "observed_mean_distance_cm": (volume_conduction or {}).get("observed_mean_distance_cm"),
        "permutation_p_shorter": (volume_conduction or {}).get("permutation_p_shorter"),
    }


def _build_classifier_snapshot(
    classification_controls_dir: Path | None,
    classification_subject_dir: Path | None,
    classifier_stats: dict[str, Any] | None = None,
) -> list[str]:
    lines: list[str] = []

    timing = (
        _read_json(classification_subject_dir / "timing_control_summary.json")
        if classification_subject_dir
        else None
    )
    if timing:
        lines.append(
            (
                "- Subject-aware timing control: "
                f"{timing.get('matched_rows', 'n/a')} matched rows, aligned lift "
                f"{_fmt_metric(timing.get('mean_aligned_lift'), signed=True)}, shifted lift "
                f"{_fmt_metric(timing.get('mean_shifted_lift'), signed=True)}, delta "
                f"{_fmt_metric(timing.get('mean_delta'), signed=True)}, aligned wins "
                f"{timing.get('aligned_wins', 'n/a')}/{timing.get('matched_rows', 'n/a')}."
            )
        )
        best_group = timing.get("best_group", {})
        if best_group:
            fusion_value = best_group.get("fusion_weight", best_group.get("fusion_grid", "n/a"))
            lines.append(
                (
                    "- Best timing-control group: "
                    f"alpha={best_group.get('alpha', 'n/a')}, "
                    f"fusion={fusion_value}, "
                    f"baseline AUC {_fmt_metric(best_group.get('mean_aligned_baseline_auc'))}, "
                    f"assisted AUC {_fmt_metric(best_group.get('mean_aligned_auc'))}, "
                    f"lift {_fmt_metric(best_group.get('mean_aligned_lift'), signed=True)}, "
                    f"matched delta {_fmt_metric(best_group.get('mean_delta'), signed=True)}."
                )
            )

    if classifier_stats:
        ci_low = classifier_stats.get("subject_bootstrap_ci_low")
        ci_high = classifier_stats.get("subject_bootstrap_ci_high")
        lines.append(
            (
                "- Subject-level robustness: "
                f"{classifier_stats.get('subject_positive', 'n/a')}/"
                f"{classifier_stats.get('subjects', 'n/a')} subjects have positive mean delta; "
                f"subject-mean delta {_fmt_metric(classifier_stats.get('subject_mean_delta'), signed=True)}; "
                f"bootstrap 95% CI [{_fmt_metric(ci_low, signed=True)}, {_fmt_metric(ci_high, signed=True)}]; "
                f"one-sided sign-test p={_fmt_metric(classifier_stats.get('subject_sign_p'))}."
            )
        )

    if lines:
        lines.append(
            (
                "- The classifier increment is descriptive; selection and validation "
                "are documented in [Methodology](documentation/METHODOLOGY.md#classification-benchmark)."
            )
        )
    return lines


def _build_markdown(
    *,
    generated_at: str,
    phase_dir: Path,
    pre_event_dir: Path,
    meta_dir: Path,
    sensitivity_dir: Path | None,
    classification_dir: Path | None,
    classification_sweep_dir: Path | None,
    classification_controls_dir: Path | None,
    classification_subject_dir: Path | None,
    volume_conduction_dir: Path | None,
    top_scenarios: list[dict[str, Any]],
    top_edges: list[dict[str, Any]],
    manifest_count: int,
    figure_count: int,
    article_figure_count: int,
    sensitivity_count: int,
    classification_count: int,
    classification_sweep_count: int,
    classification_controls_count: int,
    classification_subject_count: int,
    volume_conduction_count: int,
    classifier_snapshot: list[str],
    sensitivity_snapshot: list[str],
    volume_conduction_snapshot: list[str],
    evidence_table_count: int,
    warnings: list[str],
) -> str:
    scenario_rows = [
        [
            row.get("phase", ""),
            row.get("pc", ""),
            row.get("lag_ms", ""),
            f"{row.get('n_subjects', '')}/{row.get('total_subjects_in_dir', '')}",
            row.get("reported_edges", ""),
            f"{row.get('dominant_flow_src', '')}->{row.get('dominant_flow_dst', '')}",
        ]
        for row in top_scenarios[:10]
    ]
    edge_rows = [
        [
            row.get("phase", ""),
            row.get("pc", ""),
            row.get("lag_ms", ""),
            f"{row.get('src', '')}->{row.get('dst', '')}",
            f"{row.get('k', '')}/{row.get('n_subjects', '')}",
            row.get("q_value", row.get("p_value", "")),
            row.get("significant", ""),
        ]
        for row in top_edges[:15]
    ]

    lines: list[str] = [
        "# Article Evidence Bundle",
        "",
        f"Generated at UTC: {generated_at}",
        "",
        (
            "This directory contains the results used in the Grasp-and-Lift EEG "
            "article: phase-wise directed lagged dependence, cross-subject "
            "meta-analysis, sensitivity checks, classifier controls, and provenance."
        ),
        "",
        "Input paths and the file inventory are recorded in [package_manifest.json](package_manifest.json).",
        "",
        "## Top Scenarios",
        "",
    ]
    lines.extend(
        _md_table(
            [
                "phase",
                "pc",
                "lag_ms",
                "subjects",
                "reported_edges",
                "dominant_flow",
            ],
            scenario_rows,
        )
    )
    lines.extend(["", "## Top Replicated Edges", ""])
    lines.extend(
        _md_table(
            ["phase", "pc", "lag_ms", "edge", "k/N", "q_or_p", "significant"],
            edge_rows,
        )
    )
    lines.extend(["", "## Classifier Evidence Snapshot", ""])
    if classifier_snapshot:
        lines.extend(classifier_snapshot)
    else:
        lines.append("- No classifier evidence snapshot available.")
    lines.extend(["", "## Sensitivity Evidence Snapshot", ""])
    if sensitivity_snapshot:
        lines.extend(sensitivity_snapshot)
    else:
        lines.append("- No sensitivity evidence snapshot available.")
    lines.extend(["", "## Volume-Conduction Snapshot", ""])
    if volume_conduction_snapshot:
        lines.extend(volume_conduction_snapshot)
    else:
        lines.append("- No volume-conduction snapshot available.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The reported discoveries are significant in the fixed reference setting. "
                "They describe sensor-level lagged dependence, not anatomical pathways."
            ),
            "",
            "## Package Contents",
            "",
            "- `tables/top_scenarios.csv`: compact table of strongest phase/PC/lag scenarios.",
            "- `tables/top_edges.csv`: compact table of most replicated directed edges.",
            "- `tables/primary_edge_sensitivity.csv`: reference-setting instances, q-values, and sensitivity retention.",
            "- `tables/classifier_evidence.csv`: compact uncertainty summary for the subject-aware timing control.",
            "- `tables/sensitivity_evidence.csv`: compact sensitivity-grid stability summary.",
            "- `tables/preprocessing_comparison.csv`: CAR-only versus band-pass robustness comparison, when supplied.",
            "- `package_manifest.json`: provenance index for inputs, manifests, and figures.",
            "- `raw_meta_exports/`: exact meta-analysis exports copied from the source run.",
            "- `sensitivity/`: robustness-grid exports, when supplied.",
            "- `classification/`: optional Kaggle-style classifier ablation exports, if provided.",
            "- `classification_sweep/`: optional classification-grid robustness exports, if provided.",
            "- `classification_controls/`: optional artefact and channel-set control exports, if provided.",
            "- `classification_subject/`: participant-specific classifier results and timing controls, when supplied.",
            "- `volume_conduction/`: distance/lag/asymmetry screen, when supplied.",
            "- `documentation/`: article PDF, references, and the technical methodology note.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def _generate_article_figures(out_dir: Path) -> tuple[list[str], str | None]:
    try:
        from scripts.build_article_figures import build_article_figures
    except Exception as exc:  # pragma: no cover - defensive import guard
        return [], f"Article figure generation is unavailable: {exc}"

    try:
        figures = build_article_figures(out_dir)
    except Exception as exc:  # pragma: no cover - figure failures are surfaced as package warnings
        return [], f"Article figure generation failed: {exc}"
    return [str(path) for path in figures], None


def build_package(
    *,
    phase_dir: str | Path,
    pre_event_dir: str | Path,
    meta_dir: str | Path,
    sensitivity_dir: str | Path | None = None,
    classification_dir: str | Path | None = None,
    classification_sweep_dir: str | Path | None = None,
    classification_controls_dir: str | Path | None = None,
    classification_subject_dir: str | Path | None = None,
    volume_conduction_dir: str | Path | None = None,
    comparison_baseline_label: str = "",
    comparison_baseline_sensitivity_dir: str | Path | None = None,
    comparison_baseline_volume_conduction_dir: str | Path | None = None,
    comparison_current_label: str = "",
    out_dir: str | Path,
    top_scenarios: int = 20,
    top_edges: int = 50,
) -> dict[str, Any]:
    base = Path.cwd()
    phase_path = Path(phase_dir)
    pre_path = Path(pre_event_dir)
    meta_path = Path(meta_dir)
    sensitivity_path = Path(sensitivity_dir) if sensitivity_dir else None
    classification_path = Path(classification_dir) if classification_dir else None
    classification_sweep_path = Path(classification_sweep_dir) if classification_sweep_dir else None
    classification_controls_path = Path(classification_controls_dir) if classification_controls_dir else None
    classification_subject_path = Path(classification_subject_dir) if classification_subject_dir else None
    volume_conduction_path = Path(volume_conduction_dir) if volume_conduction_dir else None
    comparison_baseline_sensitivity_path = (
        Path(comparison_baseline_sensitivity_dir)
        if comparison_baseline_sensitivity_dir
        else None
    )
    comparison_baseline_volume_conduction_path = (
        Path(comparison_baseline_volume_conduction_dir)
        if comparison_baseline_volume_conduction_dir
        else None
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    scenarios = _read_csv(meta_path / "meta_scenarios.csv")
    edges = _read_csv(meta_path / "meta_edges.csv")
    if not scenarios:
        warnings.append("No meta_scenarios.csv rows were found.")
    if not edges:
        warnings.append("No meta_edges.csv rows were found.")

    scenarios_sorted = sorted((row for row in scenarios if _safe_int(row.get("reported_edges")) > 0), key=_scenario_sort_key)
    edges_sorted = sorted((row for row in edges if _safe_bool(row.get("significant"))), key=_edge_sort_key)
    scenario_top_rows = scenarios_sorted[: int(top_scenarios)]
    edge_top_rows = edges_sorted[: int(top_edges)]

    tables_dir = out_path / "tables"
    _write_csv(tables_dir / "top_scenarios.csv", scenario_top_rows, SCENARIO_FIELDS)
    _write_csv(tables_dir / "top_edges.csv", edge_top_rows, EDGE_FIELDS)
    if sensitivity_path is not None:
        primary_rows = primary_sensitivity_rows(edges, _read_csv(sensitivity_path / "edge_stability.csv"))
        write_primary_sensitivity_table(tables_dir, primary_rows)

    copied = _copy_existing(
        meta_path,
        out_path,
        ["meta_report.txt", "meta_edges.csv", "meta_scenarios.csv", "meta_report.json"],
    )
    copied_docs = _copy_project_docs(out_path)
    copied_sensitivity = _copy_sensitivity(sensitivity_path, out_path)
    copied_classification = _copy_classification(classification_path, out_path)
    copied_classification_sweep = _copy_classification_sweep(classification_sweep_path, out_path)
    copied_classification_controls = _copy_classification_controls(classification_controls_path, out_path)
    copied_classification_subject = _copy_classification_subject(classification_subject_path, out_path)
    copied_volume_conduction = _copy_volume_conduction(volume_conduction_path, out_path)
    if classification_path is not None and not copied_classification:
        warnings.append("No classification benchmark files were copied.")
    if classification_sweep_path is not None and not copied_classification_sweep:
        warnings.append("No classification sweep files were copied.")
    if classification_controls_path is not None and not copied_classification_controls:
        warnings.append("No classification control files were copied.")
    if classification_subject_path is not None and not copied_classification_subject:
        warnings.append("No subject classification files were copied.")
    if volume_conduction_path is not None and not copied_volume_conduction:
        warnings.append("No volume-conduction control files were copied.")

    classifier_stats = _classification_subject_stats(classification_subject_path)
    sensitivity_stats_payload = _sensitivity_stats(sensitivity_path)
    volume_conduction_stats_payload = _volume_conduction_stats(volume_conduction_path)
    preprocessing_comparison: list[dict[str, Any]] = []
    if comparison_baseline_label:
        baseline_row = _preprocessing_comparison_row(
            comparison_baseline_label,
            comparison_baseline_sensitivity_path,
            comparison_baseline_volume_conduction_path,
        )
        if baseline_row:
            preprocessing_comparison.append(baseline_row)
        else:
            warnings.append("No baseline preprocessing comparison row could be built.")
    if comparison_current_label:
        current_row = _preprocessing_comparison_row(
            comparison_current_label,
            sensitivity_path,
            volume_conduction_path,
        )
        if current_row:
            preprocessing_comparison.append(current_row)
        else:
            warnings.append("No current preprocessing comparison row could be built.")
    copied_evidence_tables = _write_evidence_tables(
        out_path,
        classifier_stats,
        sensitivity_stats_payload,
        preprocessing_comparison,
    )

    classifier_snapshot = _build_classifier_snapshot(
        classification_controls_path,
        classification_subject_path,
        classifier_stats,
    )
    sensitivity_snapshot: list[str] = []
    if sensitivity_stats_payload:
        sensitivity_snapshot.extend(
            [
                (
                    "- Sensitivity grid: "
                    f"{sensitivity_stats_payload['unique_stable_edges']} instances significant in at least one setting; "
                    f"{sensitivity_stats_payload['fully_stable_edges']} remain significant in all grid settings."
                ),
                (
                    "- Fully stable region flow: "
                    f"{sensitivity_stats_payload['top_full_stable_region_flow'] or 'none'} "
                    f"({sensitivity_stats_payload['top_full_stable_region_count']} edge instances)."
                ),
            ]
        )
        sensitivity_snapshot.append(
            "- Per-instance retention is listed in [primary_edge_sensitivity.csv]"
            "(tables/primary_edge_sensitivity.csv)."
        )
    volume_conduction_snapshot: list[str] = []
    if volume_conduction_stats_payload and volume_conduction_stats_payload["total_edges"] == 0:
        volume_conduction_snapshot.append(
            "- The secondary screen has zero fully stable candidates; spatial enrichment is unassessed."
        )
    elif volume_conduction_stats_payload:
        volume_conduction_snapshot.extend(
            [
                (
                    "- Distance/lag/asymmetry screen: "
                    f"{volume_conduction_stats_payload['vc_robust_edges']} / "
                    f"{volume_conduction_stats_payload['total_edges']} edges pass "
                    f"({float(volume_conduction_stats_payload['vc_robust_fraction']) * 100:.1f}%)."
                ),
                (
                    "- Short-range fraction among fully stable edges: "
                    f"{float(volume_conduction_stats_payload['short_range_fraction']) * 100:.1f}% "
                    f"(permutation p shorter than chance: "
                    f"{volume_conduction_stats_payload['permutation_p_shorter']})."
                ),
            ]
        )

    manifest_paths = list(_safe_rglob(phase_path, "run_manifest.json")) + list(
        _safe_rglob(pre_path, "run_manifest.json")
    )
    manifests = [_manifest_summary(path, base) for path in sorted(manifest_paths)]
    figures = [
        _relative(path, base)
        for path in sorted(
            list(_safe_rglob(phase_path, "*.png")) + list(_safe_rglob(pre_path, "*.png"))
        )
    ]
    if not manifests:
        warnings.append(
            "No run_manifest.json files were found. Re-run the current pipeline before final submission to capture full provenance."
        )
    if not figures:
        warnings.append("No PNG figure candidates were found.")

    article_figures, figure_warning = _generate_article_figures(out_path)
    if figure_warning:
        warnings.append(figure_warning)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary_text = _build_markdown(
        generated_at=generated_at,
        phase_dir=phase_path,
        pre_event_dir=pre_path,
        meta_dir=meta_path,
        sensitivity_dir=sensitivity_path,
        classification_dir=classification_path,
        classification_sweep_dir=classification_sweep_path,
        classification_controls_dir=classification_controls_path,
        classification_subject_dir=classification_subject_path,
        volume_conduction_dir=volume_conduction_path,
        top_scenarios=scenario_top_rows,
        top_edges=edge_top_rows,
        manifest_count=len(manifests),
        figure_count=len(figures),
        article_figure_count=len(article_figures),
        sensitivity_count=len(copied_sensitivity),
        classification_count=len(copied_classification),
        classification_sweep_count=len(copied_classification_sweep),
        classification_controls_count=len(copied_classification_controls),
        classification_subject_count=len(copied_classification_subject),
        volume_conduction_count=len(copied_volume_conduction),
        classifier_snapshot=classifier_snapshot,
        sensitivity_snapshot=sensitivity_snapshot,
        volume_conduction_snapshot=volume_conduction_snapshot,
        evidence_table_count=len(copied_evidence_tables),
        warnings=warnings,
    )
    (out_path / "article_summary.md").write_text(summary_text, encoding="utf-8")

    payload: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "inputs": {
            "phase_dir": str(phase_path),
            "pre_event_dir": str(pre_path),
            "meta_dir": str(meta_path),
            "sensitivity_dir": str(sensitivity_path) if sensitivity_path else None,
            "classification_dir": str(classification_path) if classification_path else None,
            "classification_sweep_dir": str(classification_sweep_path) if classification_sweep_path else None,
            "classification_controls_dir": str(classification_controls_path) if classification_controls_path else None,
            "classification_subject_dir": str(classification_subject_path) if classification_subject_path else None,
            "volume_conduction_dir": str(volume_conduction_path) if volume_conduction_path else None,
            "comparison_baseline_label": comparison_baseline_label or None,
            "comparison_baseline_sensitivity_dir": (
                str(comparison_baseline_sensitivity_path)
                if comparison_baseline_sensitivity_path
                else None
            ),
            "comparison_baseline_volume_conduction_dir": (
                str(comparison_baseline_volume_conduction_path)
                if comparison_baseline_volume_conduction_path
                else None
            ),
            "comparison_current_label": comparison_current_label or None,
        },
        "outputs": {
            "article_summary": str(out_path / "article_summary.md"),
            "top_scenarios": str(tables_dir / "top_scenarios.csv"),
            "top_edges": str(tables_dir / "top_edges.csv"),
            "raw_meta_exports": copied,
            "sensitivity": copied_sensitivity,
            "classification": copied_classification,
            "classification_sweep": copied_classification_sweep,
            "classification_controls": copied_classification_controls,
            "classification_subject": copied_classification_subject,
            "volume_conduction": copied_volume_conduction,
            "evidence_tables": copied_evidence_tables,
            "documentation": copied_docs,
            "article_figures": article_figures,
        },
        "counts": {
            "meta_scenarios": len(scenarios),
            "meta_edges": len(edges),
            "top_scenarios": len(scenario_top_rows),
            "top_edges": len(edge_top_rows),
            "run_manifests": len(manifests),
            "figure_candidates": len(figures),
            "sensitivity_files": len(copied_sensitivity),
            "classification_files": len(copied_classification),
            "classification_sweep_files": len(copied_classification_sweep),
            "classification_controls_files": len(copied_classification_controls),
            "classification_subject_files": len(copied_classification_subject),
            "volume_conduction_files": len(copied_volume_conduction),
            "evidence_table_files": len(copied_evidence_tables),
            "article_figures": len(article_figures),
        },
        "run_manifests": manifests,
        "figure_candidates": figures,
        "article_figures": article_figures,
        "warnings": warnings,
        "classifier_snapshot": classifier_snapshot,
        "sensitivity_snapshot": sensitivity_snapshot,
        "volume_conduction_snapshot": volume_conduction_snapshot,
    }
    _write_json(out_path / "package_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an article-ready package from Kaggle EEG pipeline outputs."
    )
    parser.add_argument("--phase-dir", default="out/phase_mats_pca_by_subject")
    parser.add_argument("--pre-event-dir", default="out/experimental_kaggle_by_subject")
    parser.add_argument("--meta-dir", default="out/article_meta_kaggle")
    parser.add_argument("--sensitivity-dir", default="")
    parser.add_argument("--classification-dir", default="")
    parser.add_argument("--classification-sweep-dir", default="")
    parser.add_argument("--classification-controls-dir", default="")
    parser.add_argument("--classification-subject-dir", default="")
    parser.add_argument("--volume-conduction-dir", default="out/article_volume_conduction")
    parser.add_argument("--comparison-baseline-label", default="")
    parser.add_argument("--comparison-baseline-sensitivity-dir", default="")
    parser.add_argument("--comparison-baseline-volume-conduction-dir", default="")
    parser.add_argument("--comparison-current-label", default="")
    parser.add_argument("--out", default="out/article_package")
    parser.add_argument("--top-scenarios", type=int, default=20)
    parser.add_argument("--top-edges", type=int, default=50)
    args = parser.parse_args()

    payload = build_package(
        phase_dir=args.phase_dir,
        pre_event_dir=args.pre_event_dir,
        meta_dir=args.meta_dir,
        sensitivity_dir=args.sensitivity_dir or None,
        classification_dir=args.classification_dir or None,
        classification_sweep_dir=args.classification_sweep_dir or None,
        classification_controls_dir=args.classification_controls_dir or None,
        classification_subject_dir=args.classification_subject_dir or None,
        volume_conduction_dir=args.volume_conduction_dir or None,
        comparison_baseline_label=args.comparison_baseline_label,
        comparison_baseline_sensitivity_dir=args.comparison_baseline_sensitivity_dir or None,
        comparison_baseline_volume_conduction_dir=args.comparison_baseline_volume_conduction_dir or None,
        comparison_current_label=args.comparison_current_label,
        out_dir=args.out,
        top_scenarios=int(args.top_scenarios),
        top_edges=int(args.top_edges),
    )
    print(f"Saved article package: {args.out}")
    print(f"  scenarios={payload['counts']['top_scenarios']}")
    print(f"  edges={payload['counts']['top_edges']}")
    print(f"  manifests={payload['counts']['run_manifests']}")


if __name__ == "__main__":
    main()
