from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    allowed = {"METHODOLOGY.md", "PROJECT_OVERVIEW.md"}
    for existing in dst_dir.glob("*.md"):
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


def _build_markdown(
    *,
    generated_at: str,
    phase_dir: Path,
    pre_event_dir: Path,
    meta_dir: Path,
    sensitivity_dir: Path | None,
    top_scenarios: list[dict[str, Any]],
    top_edges: list[dict[str, Any]],
    manifest_count: int,
    figure_count: int,
    sensitivity_count: int,
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
            row.get("process_label", ""),
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
        "# Kaggle Article Package",
        "",
        f"Generated at UTC: {generated_at}",
        "",
        "## Purpose",
        "",
        (
            "This package collects the final outputs from the Grasp-and-Lift EEG "
            "pipeline: phase-wise directed lagged coupling, pre-event EMA dynamics, "
            "cross-subject meta-analysis, sensitivity analysis, and provenance."
        ),
        "",
        "## Inputs",
        "",
        f"- Phase directory: `{phase_dir}`",
        f"- Pre-event directory: `{pre_event_dir}`",
        f"- Meta-analysis directory: `{meta_dir}`",
        f"- Sensitivity directory: `{sensitivity_dir}`" if sensitivity_dir else "- Sensitivity directory: not provided",
        f"- Run manifests indexed: {manifest_count}",
        f"- Figure candidates indexed: {figure_count}",
        f"- Sensitivity files copied: {sensitivity_count}",
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
                "process",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The primary supported result is reproducible, phase-locked, directed "
                "lagged dependence between EEG channels after marginal normalization "
                "and innovation-style whitening. The meta-analysis evaluates whether "
                "the same edges recur across subjects under a binomial null with "
                "optional BH-FDR."
            ),
            "",
            (
                "The results should not be described as direct proof of anatomical "
                "causality. Recommended terms are directional lagged innovation-coupling, "
                "directed functional connectivity, or candidate information flow. "
                "Interpretation should refer to replication, event phase, lag, and "
                "region-level convergence."
            ),
            "",
            "## Files To Use In The Article",
            "",
            "- `tables/top_scenarios.csv`: compact table of strongest phase/PC/lag scenarios.",
            "- `tables/top_edges.csv`: compact table of most replicated directed edges.",
            "- `article_summary.md`: summary of the final output package.",
            "- `package_manifest.json`: provenance index for inputs, manifests, and figures.",
            "- `raw_meta_exports/`: exact meta-analysis exports copied from the source run.",
            "- `sensitivity/`: optional robustness-grid exports, if provided.",
            "- `documentation/`: methodology and project overview copied from the repo.",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No packaging warnings.")
    lines.append("")
    return "\n".join(lines)


def build_package(
    *,
    phase_dir: str | Path,
    pre_event_dir: str | Path,
    meta_dir: str | Path,
    sensitivity_dir: str | Path | None = None,
    out_dir: str | Path,
    top_scenarios: int = 20,
    top_edges: int = 50,
) -> dict[str, Any]:
    base = Path.cwd()
    phase_path = Path(phase_dir)
    pre_path = Path(pre_event_dir)
    meta_path = Path(meta_dir)
    sensitivity_path = Path(sensitivity_dir) if sensitivity_dir else None
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    scenarios = _read_csv(meta_path / "meta_scenarios.csv")
    edges = _read_csv(meta_path / "meta_edges.csv")
    if not scenarios:
        warnings.append("No meta_scenarios.csv rows were found.")
    if not edges:
        warnings.append("No meta_edges.csv rows were found.")

    scenarios_sorted = sorted(scenarios, key=_scenario_sort_key)
    edges_sorted = sorted(edges, key=_edge_sort_key)
    scenario_top_rows = scenarios_sorted[: int(top_scenarios)]
    edge_top_rows = edges_sorted[: int(top_edges)]

    tables_dir = out_path / "tables"
    _write_csv(tables_dir / "top_scenarios.csv", scenario_top_rows, SCENARIO_FIELDS)
    _write_csv(tables_dir / "top_edges.csv", edge_top_rows, EDGE_FIELDS)

    copied = _copy_existing(
        meta_path,
        out_path,
        ["meta_report.txt", "meta_edges.csv", "meta_scenarios.csv", "meta_report.json"],
    )
    copied_docs = _copy_project_docs(out_path)
    copied_sensitivity = _copy_sensitivity(sensitivity_path, out_path)

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

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary_text = _build_markdown(
        generated_at=generated_at,
        phase_dir=phase_path,
        pre_event_dir=pre_path,
        meta_dir=meta_path,
        sensitivity_dir=sensitivity_path,
        top_scenarios=scenario_top_rows,
        top_edges=edge_top_rows,
        manifest_count=len(manifests),
        figure_count=len(figures),
        sensitivity_count=len(copied_sensitivity),
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
        },
        "outputs": {
            "article_summary": str(out_path / "article_summary.md"),
            "top_scenarios": str(tables_dir / "top_scenarios.csv"),
            "top_edges": str(tables_dir / "top_edges.csv"),
            "raw_meta_exports": copied,
            "sensitivity": copied_sensitivity,
            "documentation": copied_docs,
        },
        "counts": {
            "meta_scenarios": len(scenarios),
            "meta_edges": len(edges),
            "top_scenarios": len(scenario_top_rows),
            "top_edges": len(edge_top_rows),
            "run_manifests": len(manifests),
            "figure_candidates": len(figures),
            "sensitivity_files": len(copied_sensitivity),
        },
        "run_manifests": manifests,
        "figure_candidates": figures,
        "warnings": warnings,
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
    parser.add_argument("--out", default="out/article_package")
    parser.add_argument("--top-scenarios", type=int, default=20)
    parser.add_argument("--top-edges", type=int, default=50)
    args = parser.parse_args()

    payload = build_package(
        phase_dir=args.phase_dir,
        pre_event_dir=args.pre_event_dir,
        meta_dir=args.meta_dir,
        sensitivity_dir=args.sensitivity_dir or None,
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
