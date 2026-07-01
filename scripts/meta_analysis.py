"""
MFG EEG Meta-Analysis

Goal:
- Aggregate top-k directed edges per subject for each (phase, pc, lag).
- Reproducibility = number of subjects in which a given edge appears in top-k.
- Significance: one-sided binomial test P(X >= k) for X ~ Binomial(N_subjects, p0).
- Optional multiple-testing correction: BH FDR (q-values).
- Region-level interpretation + Directionality Index (DI) for dominant region flow.
- Article-ready exports: text report, edge table CSV, scenario summary CSV, JSON.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest


REGIONS: Dict[str, List[str]] = {
    "Frontal": ["Fp1", "Fp2", "Fz"],
    "Frontocentral": ["F3", "F4", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "Parietal": ["P7", "P3", "Pz", "P4", "P8", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["PO9", "O1", "Oz", "O2", "PO10"],
    "Temporal": ["T7", "T8", "TP9", "TP10"],
}


def get_region(channel: str) -> str:
    for region, chans in REGIONS.items():
        if channel in chans:
            return region
    return "Other"


class RegionSummary:
    @staticmethod
    def get_context(r_src: str, r_dst: str) -> Tuple[str, str]:
        if r_src == r_dst:
            return (
                f"WITHIN {r_src.upper()} SENSOR GROUP",
                f"Dominant reported edges connect sensors within the {r_src.lower()} group.",
            )
        return (
            f"{r_src.upper()} TO {r_dst.upper()} SENSOR FLOW",
            f"Dominant reported edges run from the {r_src.lower()} to the {r_dst.lower()} sensor group.",
        )


_EDGE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+(?P<src>[A-Za-z0-9]+)\s+->\s+(?P<dst>[A-Za-z0-9]+)\s+:\s+(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
_FILE_RE = re.compile(
    r"^phase_top_edges_(?P<phase>.+?)_(?P<mode>gc|corr)_m(?P<m>\d+)_pc(?P<pc>\d+)_lag(?P<lag>\d+)ms\.txt$",
    re.IGNORECASE,
)


def parse_edge_line(line: str) -> Optional[Tuple[str, str, float]]:
    match = _EDGE_RE.match(line.strip())
    if not match:
        return None
    return match.group("src"), match.group("dst"), float(match.group("val"))


def parse_filename(fname: str) -> Optional[Tuple[str, str, int, str, int]]:
    match = _FILE_RE.match(fname)
    if not match:
        return None
    phase = match.group("phase")
    mode = match.group("mode").lower()
    pc = f"pc{int(match.group('pc'))}"
    lag_ms = int(match.group("lag"))
    deg_m = int(match.group("m"))
    return phase, pc, lag_ms, mode, deg_m


def fdr_bh(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float)
    m = p.size
    if m == 0:
        return []
    idx = np.argsort(p)
    ranked = p[idx]
    q = ranked * (m / np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[idx] = q
    return out.tolist()


def directionality_index(flow: Counter, a: str, b: str) -> float:
    ab = int(flow.get((a, b), 0))
    ba = int(flow.get((b, a), 0))
    denom = ab + ba
    return 0.0 if denom == 0 else (ab - ba) / float(denom)


@dataclass(frozen=True)
class EdgeResult:
    phase: str
    pc: str
    lag_ms: int
    src: str
    dst: str
    region_src: str
    region_dst: str
    k: int
    n_subjects: int
    total_subjects_in_dir: int
    p_value: float
    q_value: Optional[float]
    significant: bool


@dataclass(frozen=True)
class ScenarioResult:
    phase: str
    pc: str
    lag_ms: int
    n_subjects: int
    total_subjects_in_dir: int
    candidate_edges: int
    reported_edges: int
    dominant_flow_src: str
    dominant_flow_dst: str
    dominant_flow_count: int
    dominant_flow_di: float
    process_label: str
    process_desc: str


def _scenario_sort_key(scenario: Tuple[str, str, int]) -> tuple[str, str, int]:
    return scenario[0], scenario[1], scenario[2]


def _edge_sort_key(edge: EdgeResult) -> tuple[Any, ...]:
    q_for_sort = edge.q_value if edge.q_value is not None else 1.0
    return (-edge.k, edge.p_value, q_for_sort, edge.src, edge.dst)


def _sig_marker(p_value: float, q_value: Optional[float], alpha: float, use_fdr: bool) -> str:
    score = q_value if use_fdr else p_value
    if score is None:
        return ""
    if score < 0.001:
        return "***"
    if score < 0.01:
        return "**"
    if score < alpha:
        return "*"
    return ""


def _is_significant(p_value: float, q_value: Optional[float], alpha: float, use_fdr: bool) -> bool:
    score = q_value if use_fdr else p_value
    if score is None:
        return False
    return bool(score < alpha)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def collect_edge_presence(
    *,
    dir_root: str,
    topk: int,
    mode_filter: str,
) -> tuple[
    dict[tuple[str, str, int], dict[tuple[str, str], set[str]]],
    dict[tuple[str, str, int], list[tuple[str, str]]],
    dict[tuple[str, str, int], set[str]],
    set[str],
]:
    pattern = os.path.join(dir_root, "**", "phase_top_edges_*.txt")
    all_files = glob.glob(pattern, recursive=True)
    if not all_files:
        raise RuntimeError("No phase_top_edges_*.txt files found in the given directory.")

    subj_edge_presence: dict[tuple[str, str, int], dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    region_flow_by_scenario: dict[tuple[str, str, int], list[tuple[str, str]]] = defaultdict(list)
    scenario_subjects: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    subjects_all: set[str] = set()

    for fpath in all_files:
        subj = os.path.basename(os.path.dirname(fpath))

        meta = parse_filename(os.path.basename(fpath))
        if meta is None:
            continue
        phase, pc, lag_ms, mode, _deg_m = meta

        if mode_filter != "any" and mode != mode_filter:
            continue

        scenario = (phase, pc, lag_ms)
        subjects_all.add(subj)
        scenario_subjects[scenario].add(subj)

        edges_taken = 0
        try:
            with open(fpath, "r", encoding="utf-8") as handle:
                for line in handle:
                    parsed = parse_edge_line(line)
                    if parsed is None:
                        continue
                    src, dst, _val = parsed
                    subj_edge_presence[scenario][(src, dst)].add(subj)
                    region_flow_by_scenario[scenario].append((get_region(src), get_region(dst)))
                    edges_taken += 1
                    if edges_taken >= int(topk):
                        break
        except OSError:
            continue

    if not subjects_all:
        raise RuntimeError("No matching files remained after mode filtering.")

    return subj_edge_presence, region_flow_by_scenario, scenario_subjects, subjects_all


def build_reports(
    *,
    dir_root: str,
    topk: int,
    min_subjects: int,
    channels: int,
    p0: Optional[float],
    p0_inflate: float,
    mode_filter: str,
    use_fdr: bool,
    alpha: float,
    significant_only: bool,
    dominant_flow_scope: str,
    max_edges: int,
) -> tuple[list[EdgeResult], list[ScenarioResult], str, dict[str, Any]]:
    subj_edge_presence, region_flow_by_scenario, scenario_subjects, subjects_all = collect_edge_presence(
        dir_root=dir_root,
        topk=topk,
        mode_filter=mode_filter,
    )

    total_subjects = len(subjects_all)
    C = int(channels)
    M = C * (C - 1)
    if p0 is None:
        p0_eff = min(1.0, float(p0_inflate) * (float(topk) / float(M)))
    else:
        p0_eff = float(p0)

    scenarios = sorted(subj_edge_presence.keys(), key=_scenario_sort_key)

    all_edge_rows: list[EdgeResult] = []
    for scenario in scenarios:
        phase, pc, lag_ms = scenario
        N = len(scenario_subjects[scenario])
        for (src, dst), subjs in subj_edge_presence[scenario].items():
            k = len(subjs)
            if k < int(min_subjects):
                continue
            p_value = binomtest(k, N, p=p0_eff, alternative="greater").pvalue
            all_edge_rows.append(
                EdgeResult(
                    phase=phase,
                    pc=pc,
                    lag_ms=lag_ms,
                    src=src,
                    dst=dst,
                    region_src=get_region(src),
                    region_dst=get_region(dst),
                    k=k,
                    n_subjects=N,
                    total_subjects_in_dir=total_subjects,
                    p_value=p_value,
                    q_value=None,
                    significant=False,
                )
            )

    if use_fdr and all_edge_rows:
        qvals = fdr_bh([row.p_value for row in all_edge_rows])
        all_edge_rows = [
            EdgeResult(
                **{
                    **asdict(row),
                    "q_value": float(q),
                    "significant": _is_significant(row.p_value, float(q), alpha, True),
                }
            )
            for row, q in zip(all_edge_rows, qvals)
        ]
    else:
        all_edge_rows = [
            EdgeResult(
                **{
                    **asdict(row),
                    "q_value": None,
                    "significant": _is_significant(row.p_value, None, alpha, False),
                }
            )
            for row in all_edge_rows
        ]

    rows_by_scenario: dict[tuple[str, str, int], list[EdgeResult]] = defaultdict(list)
    for row in all_edge_rows:
        rows_by_scenario[(row.phase, row.pc, row.lag_ms)].append(row)
    for scenario in rows_by_scenario:
        rows_by_scenario[scenario].sort(key=_edge_sort_key)

    scenario_rows: list[ScenarioResult] = []
    report_lines: list[str] = [
        (
            f"N_subjects_total={total_subjects} | mode={mode_filter} | topk={topk} | "
            f"p0={p0_eff:.6g} | FDR={use_fdr} | significant_only={significant_only}"
        )
    ]

    for scenario in scenarios:
        phase, pc, lag_ms = scenario
        scenario_edge_rows = rows_by_scenario.get(scenario, [])
        reported_rows = [
            row for row in scenario_edge_rows if (row.significant or not significant_only)
        ]

        flow_source: Iterable[tuple[str, str]]
        if dominant_flow_scope == "reported" and reported_rows:
            flow_source = [(row.region_src, row.region_dst) for row in reported_rows]
        else:
            flow_source = region_flow_by_scenario[scenario]

        flow = Counter(flow_source)
        if flow:
            (dom_src, dom_dst), dom_cnt = flow.most_common(1)[0]
            dom_di = directionality_index(flow, dom_src, dom_dst)
        else:
            dom_src, dom_dst, dom_cnt, dom_di = "X_Other", "X_Other", 0, 0.0
        process_label, process_desc = RegionSummary.get_context(dom_src, dom_dst)

        scenario_rows.append(
            ScenarioResult(
                phase=phase,
                pc=pc,
                lag_ms=lag_ms,
                n_subjects=len(scenario_subjects[scenario]),
                total_subjects_in_dir=total_subjects,
                candidate_edges=len(scenario_edge_rows),
                reported_edges=len(reported_rows),
                dominant_flow_src=dom_src,
                dominant_flow_dst=dom_dst,
                dominant_flow_count=dom_cnt,
                dominant_flow_di=dom_di,
                process_label=process_label,
                process_desc=process_desc,
            )
        )

        if not reported_rows:
            continue

        report_lines.append(f"{phase} | {pc} | lag={lag_ms}ms")
        report_lines.append(
            f"  scenario_subjects={len(scenario_subjects[scenario])}/{total_subjects}  reported_edges={len(reported_rows)}"
        )
        if use_fdr:
            report_lines.append("  edge        k/N    p-value      q(FDR)      sig")
        else:
            report_lines.append("  edge        k/N    p-value      sig")

        for row in reported_rows[: int(max_edges)]:
            marker = _sig_marker(row.p_value, row.q_value, alpha, use_fdr)
            if use_fdr:
                report_lines.append(
                    f"  {row.src:4}->{row.dst:4}   {row.k:>2}/{row.n_subjects:<2}  "
                    f"{row.p_value:>10.4g}  {row.q_value if row.q_value is not None else 1.0:>10.4g}  {marker}"
                )
            else:
                report_lines.append(
                    f"  {row.src:4}->{row.dst:4}   {row.k:>2}/{row.n_subjects:<2}  {row.p_value:>10.4g}  {marker}"
                )

        report_lines.append(
            f"  dominant_flow {dom_src}->{dom_dst}  DI={dom_di:+.2f}  ({process_label})"
        )
        report_lines.append(f"  {process_desc}")
        report_lines.append("")

    report_text = "\n".join(report_lines).rstrip() + "\n"
    metadata = {
        "dir": dir_root,
        "mode": mode_filter,
        "topk": int(topk),
        "min_subjects": int(min_subjects),
        "channels": int(channels),
        "p0": float(p0_eff),
        "use_fdr": bool(use_fdr),
        "alpha": float(alpha),
        "significant_only": bool(significant_only),
        "dominant_flow_scope": dominant_flow_scope,
        "total_subjects": int(total_subjects),
    }
    return all_edge_rows, scenario_rows, report_text, metadata


def _default_export_paths(out_dir: Optional[str]) -> dict[str, Optional[str]]:
    if not out_dir:
        return {"text": None, "edges_csv": None, "scenarios_csv": None, "json": None}
    os.makedirs(out_dir, exist_ok=True)
    return {
        "text": os.path.join(out_dir, "meta_report.txt"),
        "edges_csv": os.path.join(out_dir, "meta_edges.csv"),
        "scenarios_csv": os.path.join(out_dir, "meta_scenarios.csv"),
        "json": os.path.join(out_dir, "meta_report.json"),
    }


def write_edge_csv(path: str, rows: list[EdgeResult]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [
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
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_scenario_csv(path: str, rows: list[ScenarioResult]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0]).keys()) if rows else [
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json_report(
    path: str,
    *,
    metadata: dict[str, Any],
    edge_rows: list[EdgeResult],
    scenario_rows: list[ScenarioResult],
) -> None:
    _ensure_parent(path)
    payload = {
        "metadata": metadata,
        "scenarios": [asdict(row) for row in scenario_rows],
        "edges": [asdict(row) for row in edge_rows],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="MFG EEG Meta-Analysis")
    ap.add_argument(
        "--dir",
        default="out/phase_mats_pca_by_subject",
        help="Root directory containing subject subfolders with phase_top_edges_*.txt files.",
    )
    ap.add_argument("--topk", type=int, default=10, help="Top-k edges to read per file.")
    ap.add_argument(
        "--min-subjects",
        type=int,
        default=4,
        help="Minimum number of subjects required to report an edge.",
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=32,
        help="Number of EEG channels C (used for default p0).",
    )
    ap.add_argument(
        "--p0",
        type=float,
        default=None,
        help="Null occurrence probability p0. If not provided, estimated from topk and channels.",
    )
    ap.add_argument(
        "--p0-inflate",
        type=float,
        default=10.0,
        help="Inflation factor for default p0 (accounts for non-uniform ranking biases).",
    )
    ap.add_argument(
        "--mode",
        choices=["gc", "corr", "any"],
        default="gc",
        help="Filter by mode encoded in filenames.",
    )
    ap.add_argument(
        "--use-fdr",
        action="store_true",
        help="Compute BH FDR q-values across all reported edges.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for p (and q if --use-fdr).",
    )
    ap.add_argument(
        "--max-edges",
        type=int,
        default=3,
        help="Maximum number of edges printed per (phase, pc, lag).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    ap.add_argument(
        "--significant-only",
        action="store_true",
        help="Report and export only statistically significant edges.",
    )
    ap.add_argument(
        "--dominant-flow-scope",
        choices=["all", "reported"],
        default="reported",
        help="Whether dominant flow should use all parsed top-k edges or only reported rows.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="If set, write article-ready exports into this directory.",
    )
    ap.add_argument("--out-text", default=None, help="Optional path for the text report.")
    ap.add_argument("--out-csv", default=None, help="Optional path for the flat edge CSV.")
    ap.add_argument(
        "--out-scenarios-csv",
        default=None,
        help="Optional path for the scenario summary CSV.",
    )
    ap.add_argument("--out-json", default=None, help="Optional path for the JSON report.")
    args = ap.parse_args()

    edge_rows, scenario_rows, report_text, metadata = build_reports(
        dir_root=args.dir,
        topk=int(args.topk),
        min_subjects=int(args.min_subjects),
        channels=int(args.channels),
        p0=args.p0,
        p0_inflate=float(args.p0_inflate),
        mode_filter=str(args.mode),
        use_fdr=bool(args.use_fdr),
        alpha=float(args.alpha),
        significant_only=bool(args.significant_only),
        dominant_flow_scope=str(args.dominant_flow_scope),
        max_edges=int(args.max_edges),
    )

    filtered_edge_rows = [
        row for row in edge_rows if (row.significant or not args.significant_only)
    ]
    rows_by_scenario: dict[tuple[str, str, int], list[EdgeResult]] = defaultdict(list)
    for row in filtered_edge_rows:
        rows_by_scenario[(row.phase, row.pc, row.lag_ms)].append(row)
    for scenario in rows_by_scenario:
        rows_by_scenario[scenario].sort(key=_edge_sort_key)
        rows_by_scenario[scenario] = rows_by_scenario[scenario][: int(args.max_edges)]

    default_paths = _default_export_paths(args.out_dir)
    out_text = args.out_text or default_paths["text"]
    out_csv = args.out_csv or default_paths["edges_csv"]
    out_scenarios_csv = args.out_scenarios_csv or default_paths["scenarios_csv"]
    out_json = args.out_json or default_paths["json"]

    if out_text:
        _ensure_parent(out_text)
        with open(out_text, "w", encoding="utf-8") as handle:
            handle.write(report_text)
    if out_csv:
        write_edge_csv(out_csv, filtered_edge_rows)
    if out_scenarios_csv:
        write_scenario_csv(out_scenarios_csv, scenario_rows)
    if out_json:
        write_json_report(
            out_json,
            metadata=metadata,
            edge_rows=filtered_edge_rows,
            scenario_rows=scenario_rows,
        )

    if not args.quiet:
        print(report_text, end="")
        for label, path in [
            ("text", out_text),
            ("edges_csv", out_csv),
            ("scenarios_csv", out_scenarios_csv),
            ("json", out_json),
        ]:
            if path:
                print(f"[OK] wrote {label}: {path}")


if __name__ == "__main__":
    main()
