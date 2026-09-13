from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.meta_analysis import EdgeResult, build_reports


EDGE_OBSERVATION_FIELDS = [
    "config_id",
    "topk",
    "min_subjects_param",
    "p0_inflate",
    "p0_effective",
    "use_fdr",
    "alpha",
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

STABILITY_FIELDS = [
    "phase",
    "pc",
    "lag_ms",
    "src",
    "dst",
    "region_src",
    "region_dst",
    "config_count",
    "successful_config_count",
    "stability_fraction",
    "max_k",
    "max_n_subjects",
    "min_p_value",
    "min_q_value",
    "topk_values",
    "min_subjects_values",
    "p0_inflate_values",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _edge_key(row: EdgeResult | dict[str, Any]) -> tuple[Any, ...]:
    data = asdict(row) if isinstance(row, EdgeResult) else row
    return (
        data["phase"],
        data["pc"],
        int(data["lag_ms"]),
        data["src"],
        data["dst"],
    )


def _sort_stability(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["config_count"]),
        -float(row["stability_fraction"]),
        -int(row["max_k"]),
        _safe_float(row["min_q_value"], _safe_float(row["min_p_value"], 1.0)),
        str(row["phase"]),
        str(row["pc"]),
        int(row["lag_ms"]),
        str(row["src"]),
        str(row["dst"]),
    )


def _format_values(values: set[Any]) -> str:
    def key(value: Any) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    return " ".join(str(value) for value in sorted(values, key=key))


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No instances retained in the sensitivity grid."]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |")
    return out


def _build_summary(
    *,
    dir_root: str,
    configs_requested: int,
    configs_successful: int,
    observation_count: int,
    stability_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    top_rows = [
        [
            row["phase"],
            row["pc"],
            row["lag_ms"],
            f"{row['src']}->{row['dst']}",
            f"{row['config_count']}/{row['successful_config_count']}",
            row["stability_fraction"],
            row["max_k"],
            row["min_q_value"] or row["min_p_value"],
        ]
        for row in stability_rows[:20]
    ]
    lines = [
        "# Meta-Analysis Sensitivity Report",
        "",
        f"Input directory: `{dir_root}`",
        f"Requested parameter configurations: {configs_requested}",
        f"Successful configurations: {configs_successful}",
        f"Significant edge observations: {observation_count}",
        f"Instances significant in at least one setting: {len(stability_rows)}",
        "",
        "## Most Frequently Retained Instances",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "phase",
                "pc",
                "lag_ms",
                "edge",
                "configs",
                "stability",
                "max_k",
                "best_q_or_p",
            ],
            top_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Columns",
            "",
            (
                "`configs` counts retention across the grid; `best_q_or_p` is the minimum "
                "across those settings. Reference-setting q-values are reported in the "
                "meta-analysis edge table."
            ),
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No sensitivity warnings.")
    lines.append("")
    return "\n".join(lines)


def build_sensitivity(
    *,
    dir_root: str | Path,
    out_dir: str | Path,
    topk_values: list[int],
    min_subjects_values: list[int],
    p0_inflate_values: list[float],
    mode: str = "gc",
    channels: int = 32,
    alpha: float = 0.05,
    use_fdr: bool = True,
    significant_only: bool = True,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, Any]] = []
    warnings: list[str] = []
    config_results: list[dict[str, Any]] = []
    config_id = 0

    for topk in topk_values:
        for min_subjects in min_subjects_values:
            for p0_inflate in p0_inflate_values:
                config_id += 1
                config_label = (
                    f"cfg{config_id:03d}_topk{topk}_min{min_subjects}_p0x{p0_inflate:g}"
                )
                try:
                    edge_rows, _scenario_rows, _report_text, metadata = build_reports(
                        dir_root=str(dir_root),
                        topk=int(topk),
                        min_subjects=int(min_subjects),
                        channels=int(channels),
                        p0=None,
                        p0_inflate=float(p0_inflate),
                        mode_filter=str(mode),
                        use_fdr=bool(use_fdr),
                        alpha=float(alpha),
                        significant_only=bool(significant_only),
                        dominant_flow_scope="reported",
                        max_edges=999999,
                    )
                except Exception as exc:  # Keep grid runs robust and report failures.
                    warnings.append(f"{config_label} failed: {exc}")
                    config_results.append(
                        {
                            "config_id": config_label,
                            "topk": topk,
                            "min_subjects": min_subjects,
                            "p0_inflate": p0_inflate,
                            "ok": False,
                            "error": str(exc),
                        }
                    )
                    continue

                config_results.append(
                    {
                        "config_id": config_label,
                        "topk": topk,
                        "min_subjects": min_subjects,
                        "p0_inflate": p0_inflate,
                        "ok": True,
                        "edge_rows": len(edge_rows),
                        "p0_effective": metadata.get("p0"),
                        "fdr_total_tests": metadata.get("fdr_total_tests"),
                        "min_subjects_applied_after_fdr": True,
                    }
                )

                kept_rows = [
                    row for row in edge_rows if (row.significant or not significant_only)
                ]
                for row in kept_rows:
                    data = asdict(row)
                    observations.append(
                        {
                            "config_id": config_label,
                            "topk": int(topk),
                            "min_subjects_param": int(min_subjects),
                            "p0_inflate": float(p0_inflate),
                            "p0_effective": metadata.get("p0"),
                            "use_fdr": bool(use_fdr),
                            "alpha": float(alpha),
                            **data,
                        }
                    )

    successful_config_count = sum(1 for item in config_results if item.get("ok"))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[_edge_key(row)].append(row)

    stability_rows: list[dict[str, Any]] = []
    for (_phase, _pc, _lag_ms, _src, _dst), rows in grouped.items():
        first = rows[0]
        q_values = [
            _safe_float(row.get("q_value"), 1.0)
            for row in rows
            if row.get("q_value") not in ("", None)
        ]
        p_values = [_safe_float(row.get("p_value"), 1.0) for row in rows]
        stability_rows.append(
            {
                "phase": first["phase"],
                "pc": first["pc"],
                "lag_ms": int(first["lag_ms"]),
                "src": first["src"],
                "dst": first["dst"],
                "region_src": first["region_src"],
                "region_dst": first["region_dst"],
                "config_count": len({row["config_id"] for row in rows}),
                "successful_config_count": successful_config_count,
                "stability_fraction": (
                    len({row["config_id"] for row in rows}) / float(successful_config_count)
                    if successful_config_count
                    else 0.0
                ),
                "max_k": max(int(row["k"]) for row in rows),
                "max_n_subjects": max(int(row["n_subjects"]) for row in rows),
                "min_p_value": min(p_values) if p_values else "",
                "min_q_value": min(q_values) if q_values else "",
                "topk_values": _format_values({row["topk"] for row in rows}),
                "min_subjects_values": _format_values(
                    {row["min_subjects_param"] for row in rows}
                ),
                "p0_inflate_values": _format_values(
                    {row["p0_inflate"] for row in rows}
                ),
            }
        )
    stability_rows.sort(key=_sort_stability)

    _write_csv(out_path / "sensitivity_edges.csv", observations, EDGE_OBSERVATION_FIELDS)
    _write_csv(out_path / "edge_stability.csv", stability_rows, STABILITY_FIELDS)

    summary = _build_summary(
        dir_root=str(dir_root),
        configs_requested=config_id,
        configs_successful=successful_config_count,
        observation_count=len(observations),
        stability_rows=stability_rows,
        warnings=warnings,
    )
    (out_path / "sensitivity_summary.md").write_text(summary, encoding="utf-8")

    payload = {
        "dir": str(dir_root),
        "configs_requested": config_id,
        "configs_successful": successful_config_count,
        "topk_values": list(map(int, topk_values)),
        "min_subjects_values": list(map(int, min_subjects_values)),
        "p0_inflate_values": list(map(float, p0_inflate_values)),
        "mode": mode,
        "channels": int(channels),
        "alpha": float(alpha),
        "use_fdr": bool(use_fdr),
        "significant_only": bool(significant_only),
        "observation_count": len(observations),
        "stable_edge_count": len(stability_rows),
        "warnings": warnings,
        "config_results": config_results,
    }
    _write_json(out_path / "sensitivity_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a parameter-grid sensitivity analysis over meta-analysis settings."
    )
    parser.add_argument("--dir", default="out/phase_mats_pca_by_subject")
    parser.add_argument("--out-dir", default="out/article_meta_sensitivity")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--min-subjects", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--p0-inflate", type=float, nargs="+", default=[5.0, 10.0, 15.0])
    parser.add_argument("--mode", choices=["gc", "corr", "any"], default="gc")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--use-fdr", action="store_true")
    parser.add_argument("--include-nonsignificant", action="store_true")
    args = parser.parse_args()

    payload = build_sensitivity(
        dir_root=args.dir,
        out_dir=args.out_dir,
        topk_values=list(args.topk),
        min_subjects_values=list(args.min_subjects),
        p0_inflate_values=list(args.p0_inflate),
        mode=str(args.mode),
        channels=int(args.channels),
        alpha=float(args.alpha),
        use_fdr=bool(args.use_fdr),
        significant_only=not bool(args.include_nonsignificant),
    )
    print(f"Saved meta sensitivity outputs: {args.out_dir}")
    print(f"  configs_successful={payload['configs_successful']}")
    print(f"  observations={payload['observation_count']}")
    print(f"  stable_edges={payload['stable_edge_count']}")


if __name__ == "__main__":
    main()
