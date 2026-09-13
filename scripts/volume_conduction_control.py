"""
Volume-conduction control for reproducible directed MFG edges.

Motivation
----------
The strongest reproducible edges in the initial analysis connect physically
adjacent posterior electrodes (for example PO9->O1, PO10->O2). Such a pattern is
compatible with volume conduction and shared-reference leakage. This control
summarizes distance, lag, and directional asymmetry so those alternatives remain
visible in the interpretation.

Three independent checks are applied to the reproducible / stable edges:

1. Spatial enrichment. Volume conduction produces dependence between nearby
   sensors. We compare the geodesic inter-electrode distance of the reproducible
   edges against a null in which the same number of directed pairs is drawn
   uniformly from the montage. If reproducible edges are significantly *shorter*
   than chance, short-range leakage is a concern.

2. Lag structure. Zero-lag edges are more compatible with instantaneous field
   spread, although a non-zero estimated lag does not rule it out.

3. Directional asymmetry. Reproducible reverse edges increase concern about a
   shared source; asymmetry alone does not establish directionality.

The implementation retains the field name `vc_robust` for compatibility. It
means only that an edge passes the long-range, non-zero-lag, and asymmetric
screen; it is not proof of cortical connectivity.

Inputs
------
Either the sensitivity stability table (`edge_stability.csv`, preferred) or the
flat meta-analysis edge table (`meta_edges.csv`). Both expose
phase, pc, lag_ms, src, dst columns.

Outputs
-------
- volume_conduction_edges.csv   per-edge annotation (distance, lag, symmetry)
- volume_conduction_summary.json machine-readable metrics
- volume_conduction_report.md    human-readable interpretation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src_mfg.montage import (  # noqa: E402
    GAL_CHANNELS,
    all_directed_pair_distances_cm,
    geodesic_distance_cm,
    has_channel,
)


@dataclass(frozen=True)
class AnnotatedEdge:
    phase: str
    pc: str
    lag_ms: int
    src: str
    dst: str
    region_src: str
    region_dst: str
    distance_cm: float
    short_range: bool
    zero_lag: bool
    reverse_present: bool
    asymmetric: bool
    vc_robust: bool


def _read_edge_rows(path: str, min_stability: float) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return []

    fieldnames = rows[0].keys()
    filtered: List[Dict[str, str]] = []
    for row in rows:
        # edge_stability.csv: keep edges that are stable enough.
        if "stability_fraction" in fieldnames:
            try:
                if float(row.get("stability_fraction", 0.0) or 0.0) < min_stability:
                    continue
            except (TypeError, ValueError):
                continue
        # meta_edges.csv: keep only significant rows when the column exists.
        elif "significant" in fieldnames:
            if str(row.get("significant", "")).strip().lower() not in ("true", "1"):
                continue
        filtered.append(row)
    return filtered


def annotate_edges(
    rows: List[Dict[str, str]],
    *,
    short_range_cm: float,
) -> List[AnnotatedEdge]:
    # Set of reproducible directed edges keyed by (phase, pc, lag, src, dst),
    # used to detect whether the reverse direction is also reproducible.
    present = {
        (
            r["phase"],
            r["pc"],
            int(r["lag_ms"]),
            r["src"],
            r["dst"],
        )
        for r in rows
        if has_channel(r["src"]) and has_channel(r["dst"])
    }

    annotated: List[AnnotatedEdge] = []
    for r in rows:
        src, dst = r["src"], r["dst"]
        if not (has_channel(src) and has_channel(dst)):
            continue
        if src == dst:
            continue
        lag_ms = int(r["lag_ms"])
        dist = geodesic_distance_cm(src, dst)
        reverse_key = (r["phase"], r["pc"], lag_ms, dst, src)
        reverse_present = reverse_key in present
        short_range = dist <= short_range_cm
        zero_lag = lag_ms == 0
        asymmetric = not reverse_present
        vc_robust = (not short_range) and (not zero_lag) and asymmetric
        annotated.append(
            AnnotatedEdge(
                phase=r["phase"],
                pc=r["pc"],
                lag_ms=lag_ms,
                src=src,
                dst=dst,
                region_src=r.get("region_src", ""),
                region_dst=r.get("region_dst", ""),
                distance_cm=round(dist, 3),
                short_range=short_range,
                zero_lag=zero_lag,
                reverse_present=reverse_present,
                asymmetric=asymmetric,
                vc_robust=vc_robust,
            )
        )
    return annotated


def spatial_enrichment(
    annotated: List[AnnotatedEdge],
    *,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Permutation test: are reproducible edges spatially shorter than random pairs?

    One-sided p-value = fraction of random same-size samples whose mean distance
    is <= the observed mean distance.
    """
    observed = np.array([e.distance_cm for e in annotated], dtype=float)
    null_universe = np.array(all_directed_pair_distances_cm(GAL_CHANNELS), dtype=float)
    if observed.size == 0 or null_universe.size == 0:
        return {
            "n_edges": int(observed.size),
            "observed_mean_distance_cm": None,
            "null_mean_distance_cm": float(null_universe.mean()) if null_universe.size else None,
            "permutation_p_shorter": None,
            "z_score": None,
            "n_permutations": int(n_permutations),
        }

    rng = np.random.default_rng(seed)
    obs_mean = float(observed.mean())
    sample_means = np.empty(n_permutations, dtype=float)
    n = observed.size
    for i in range(n_permutations):
        idx = rng.integers(0, null_universe.size, size=n)
        sample_means[i] = null_universe[idx].mean()

    p_shorter = float((np.sum(sample_means <= obs_mean) + 1) / (n_permutations + 1))
    null_mu = float(sample_means.mean())
    null_sd = float(sample_means.std(ddof=1)) if n_permutations > 1 else 0.0
    z = float((obs_mean - null_mu) / null_sd) if null_sd > 0 else None
    return {
        "n_edges": int(n),
        "observed_mean_distance_cm": round(obs_mean, 3),
        "null_mean_distance_cm": round(null_mu, 3),
        "null_sd_distance_cm": round(null_sd, 3),
        "permutation_p_shorter": p_shorter,
        "z_score": round(z, 3) if z is not None else None,
        "n_permutations": int(n_permutations),
    }


def summarize(
    annotated: List[AnnotatedEdge],
    enrichment: Dict[str, Any],
    *,
    short_range_cm: float,
) -> Dict[str, Any]:
    total = len(annotated)

    def frac(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    short = sum(1 for e in annotated if e.short_range)
    zero_lag = sum(1 for e in annotated if e.zero_lag)
    asym = sum(1 for e in annotated if e.asymmetric)
    robust = [e for e in annotated if e.vc_robust]
    robust_flow = Counter(
        (e.region_src, e.region_dst) for e in robust if e.region_src and e.region_dst
    )
    return {
        "total_edges": total,
        "short_range_cm_threshold": short_range_cm,
        "short_range_edges": short,
        "short_range_fraction": frac(short),
        "long_range_edges": total - short,
        "zero_lag_edges": zero_lag,
        "nonzero_lag_edges": total - zero_lag,
        "nonzero_lag_fraction": frac(total - zero_lag),
        "asymmetric_edges": asym,
        "asymmetric_fraction": frac(asym),
        "vc_robust_edges": len(robust),
        "vc_robust_fraction": frac(len(robust)),
        "vc_robust_region_flow_top": [
            {"flow": f"{a}->{b}", "count": c} for (a, b), c in robust_flow.most_common(10)
        ],
        "spatial_enrichment": enrichment,
    }


def _verdict(summary: Dict[str, Any]) -> str:
    if summary["total_edges"] == 0:
        return "- No fully stable candidates enter this screen; spatial enrichment and leakage cannot be assessed."
    enr = summary["spatial_enrichment"]
    p_shorter = enr.get("permutation_p_shorter")
    robust = summary["vc_robust_edges"]
    short_frac = summary["short_range_fraction"]

    spatial_flag = p_shorter is not None and p_shorter < 0.05
    lines = []
    if spatial_flag:
        lines.append(
            f"WARNING: reproducible edges are significantly shorter-range than chance "
            f"(permutation p={p_shorter:.4g}); short-range volume conduction is a real concern."
        )
    else:
        lines.append(
            f"No significant short-range enrichment was detected "
            f"(permutation p={p_shorter}); this does not rule out spatial-adjacency leakage."
        )
    if short_frac >= 0.5:
        lines.append(
            f"A majority of reproducible edges are short-range "
            f"(<= {summary['short_range_cm_threshold']} cm): interpret raw edge maps cautiously."
        )
    if robust > 0:
        lines.append(
            f"{robust} edge instance(s) pass all three screening criteria "
            f"(long-range AND non-zero lag AND directionally asymmetric). "
            f"They still require sensor-level interpretation and are not proof of cortical connectivity."
        )
    else:
        lines.append(
            "No edge instance passes all three screening criteria simultaneously. The "
            "connectivity claim should be downgraded to a sensor-level, adjacency-"
            "limited observation, or recomputed with an estimator less sensitive to "
            "field spread (surface Laplacian / imaginary coherence / wPLI)."
        )
    return "\n".join(f"- {line}" for line in lines)


def build_report(
    summary: Dict[str, Any],
    annotated: List[AnnotatedEdge],
    *,
    source_table: str,
) -> str:
    enr = summary["spatial_enrichment"]
    lines: List[str] = [
        "# Volume-Conduction Control",
        "",
        f"Source edge table: `{source_table}`",
        f"Reproducible edge instances analysed: {summary['total_edges']}",
        "",
        "## Purpose",
        "",
        (
            "Summarizes whether reproducible directed edges are short-range, zero-lag, "
            "or bidirectional. These features can be consistent with volume conduction "
            "or shared-reference effects; this heuristic screen does not identify "
            "genuine cortical coupling."
        ),
        "",
        "## Spatial enrichment (are edges just nearest neighbours?)",
        "",
        f"- Observed mean inter-electrode distance: {enr.get('observed_mean_distance_cm')} cm",
        f"- Null mean distance (random directed pairs): {enr.get('null_mean_distance_cm')} cm",
        f"- One-sided permutation p (edges shorter than chance): {enr.get('permutation_p_shorter')}",
        f"- z-score vs null: {enr.get('z_score')}",
        "",
        "## Lag structure",
        "",
        f"- Zero-lag edge instances: {summary['zero_lag_edges']} / {summary['total_edges']}",
        f"- Non-zero-lag edge instances: {summary['nonzero_lag_edges']} "
        f"({summary['nonzero_lag_fraction'] * 100:.1f}%)",
        "",
        "## Directional asymmetry",
        "",
        f"- Asymmetric edge instances (reverse not reproducible): "
        f"{summary['asymmetric_edges']} ({summary['asymmetric_fraction'] * 100:.1f}%)",
        "",
        "## Short-range fraction",
        "",
        f"- Short-range (<= {summary['short_range_cm_threshold']} cm): "
        f"{summary['short_range_edges']} ({summary['short_range_fraction'] * 100:.1f}%)",
        f"- Long-range: {summary['long_range_edges']}",
        "",
        "## Distance/lag/asymmetry screen",
        "",
        f"Edges that are long-range AND non-zero-lag AND asymmetric: "
        f"{summary['vc_robust_edges']} ({summary['vc_robust_fraction'] * 100:.1f}%).",
        "",
    ]
    if summary["vc_robust_region_flow_top"]:
        lines.append("Region flow of the screen-passing subset:")
        lines.append("")
        lines.append("| flow | count |")
        lines.append("| --- | --- |")
        for item in summary["vc_robust_region_flow_top"]:
            lines.append(f"| {item['flow']} | {item['count']} |")
        lines.append("")

    robust_examples = [e for e in annotated if e.vc_robust]
    robust_examples.sort(key=lambda e: (-e.distance_cm, e.phase, e.pc, e.lag_ms))
    if robust_examples:
        lines.append("Top screen-passing edges (by distance):")
        lines.append("")
        lines.append("| phase | pc | lag_ms | edge | distance_cm |")
        lines.append("| --- | --- | --- | --- | --- |")
        for e in robust_examples[:15]:
            lines.append(
                f"| {e.phase} | {e.pc} | {e.lag_ms} | {e.src}->{e.dst} | {e.distance_cm} |"
            )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(_verdict(summary))
    lines.append("")
    return "\n".join(lines)


def write_edges_csv(path: str, annotated: List[AnnotatedEdge]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(asdict(annotated[0]).keys()) if annotated else [
        "phase", "pc", "lag_ms", "src", "dst", "region_src", "region_dst",
        "distance_cm", "short_range", "zero_lag", "reverse_present",
        "asymmetric", "vc_robust",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for e in annotated:
            writer.writerow(asdict(e))


def export_vc_robust_edge_table(
    *,
    stability_table: str,
    vc_edges_csv: str,
    out_path: str,
    min_stability: float = 1.0,
) -> int:
    """
    Export a classifier-compatible edge_stability table containing only
    edges that pass the distance/lag/asymmetry screen.
    """
    with open(vc_edges_csv, "r", encoding="utf-8", newline="") as handle:
        vc_rows = list(csv.DictReader(handle))
    robust_keys = {
        (r["phase"], r["pc"], str(r["lag_ms"]), r["src"], r["dst"])
        for r in vc_rows
        if str(r.get("vc_robust", "")).strip().lower() == "true"
    }

    stability_rows = _read_edge_rows(stability_table, min_stability)
    filtered = [
        r
        for r in stability_rows
        if (r["phase"], r["pc"], str(r["lag_ms"]), r["src"], r["dst"]) in robust_keys
    ]
    if robust_keys and not filtered:
        raise ValueError("Screen-passing edges did not match the stability table.")
    with open(stability_table, "r", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered:
            writer.writerow(row)
    return len(filtered)


def run_control(
    *,
    edge_table: str,
    out_dir: str,
    short_range_cm: float,
    min_stability: float,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rows = _read_edge_rows(edge_table, min_stability)
    annotated = annotate_edges(rows, short_range_cm=short_range_cm)
    enrichment = spatial_enrichment(
        annotated, n_permutations=n_permutations, seed=seed
    )
    summary = summarize(annotated, enrichment, short_range_cm=short_range_cm)

    os.makedirs(out_dir, exist_ok=True)
    edges_csv = os.path.join(out_dir, "volume_conduction_edges.csv")
    summary_json = os.path.join(out_dir, "volume_conduction_summary.json")
    report_md = os.path.join(out_dir, "volume_conduction_report.md")

    write_edges_csv(edges_csv, annotated)
    payload = {
        "source_edge_table": edge_table,
        "short_range_cm_threshold": short_range_cm,
        "min_stability": min_stability,
        "seed": seed,
        **summary,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(report_md, "w", encoding="utf-8") as handle:
        handle.write(build_report(summary, annotated, source_table=edge_table))

    robust_table = os.path.join(out_dir, "vc_robust_edge_stability.csv")
    robust_count = export_vc_robust_edge_table(
        stability_table=edge_table,
        vc_edges_csv=edges_csv,
        out_path=robust_table,
        min_stability=min_stability,
    )
    payload["vc_robust_edge_table"] = robust_table
    payload["vc_robust_edge_table_rows"] = robust_count

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Volume-conduction control for reproducible directed MFG edges."
    )
    ap.add_argument(
        "--edge-table",
        default="out/article_meta_sensitivity/edge_stability.csv",
        help="Stability table (edge_stability.csv) or flat meta_edges.csv.",
    )
    ap.add_argument(
        "--out-dir",
        default="out/article_volume_conduction",
        help="Output directory for the control exports.",
    )
    ap.add_argument(
        "--short-range-cm",
        type=float,
        default=4.5,
        help="Edges at or below this geodesic distance are treated as short-range "
        "(adjacency-dominated / volume-conduction suspect).",
    )
    ap.add_argument(
        "--min-stability",
        type=float,
        default=1.0,
        help="For edge_stability.csv: minimum stability_fraction to include an edge.",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=20000,
        help="Number of permutations for the spatial-enrichment null.",
    )
    ap.add_argument("--seed", type=int, default=20240617, help="RNG seed.")
    ap.add_argument("--quiet", action="store_true", help="Suppress console output.")
    args = ap.parse_args()

    payload = run_control(
        edge_table=args.edge_table,
        out_dir=args.out_dir,
        short_range_cm=float(args.short_range_cm),
        min_stability=float(args.min_stability),
        n_permutations=int(args.permutations),
        seed=int(args.seed),
    )

    if not args.quiet:
        enr = payload["spatial_enrichment"]
        print(f"Volume-conduction control written to: {args.out_dir}")
        print(f"  edges analysed              = {payload['total_edges']}")
        print(f"  short-range fraction        = {payload['short_range_fraction']}")
        print(f"  non-zero-lag fraction       = {payload['nonzero_lag_fraction']}")
        print(f"  asymmetric fraction         = {payload['asymmetric_fraction']}")
        print(f"  observed mean distance (cm) = {enr.get('observed_mean_distance_cm')}")
        print(f"  null mean distance (cm)     = {enr.get('null_mean_distance_cm')}")
        print(f"  permutation p (shorter)     = {enr.get('permutation_p_shorter')}")
        print(f"  Screen-passing edges        = {payload['vc_robust_edges']}")
        if payload.get("vc_robust_edge_table"):
            print(f"  Screen-passing edge table   = {payload['vc_robust_edge_table']} ({payload.get('vc_robust_edge_table_rows')} rows)")


if __name__ == "__main__":
    main()
