"""
MFG EEG Meta-Analysis

Goal:
- Aggregate top-k directed edges per subject for each (phase, pc, lag).
- Reproducibility = number of subjects in which a given edge appears in top-k.
- Significance: one-sided binomial test P(X >= k) for X ~ Binomial(N_subjects, p0).
- Optional multiple-testing correction: BH FDR (q-values).
- Region-level interpretation + Directionality Index (DI) for dominant region flow.
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import binomtest


# -----------------------------------------------------------------------------
# Regions (channel -> anatomical macro-region)
# -----------------------------------------------------------------------------
REGIONS: Dict[str, List[str]] = {
    "F_Exec": ["Fp1", "Fp2", "Fz"],
    "F_Motor": ["F3", "F4", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "P_Space": ["P7", "P3", "Pz", "P4", "P8", "CP1", "CP2", "CP5", "CP6"],
    "O_Vis": ["PO9", "O1", "Oz", "O2", "PO10"],
    "T_Temp": ["T7", "T8", "TP9", "TP10"],
}


def get_region(channel: str) -> str:
    for region, chans in REGIONS.items():
        if channel in chans:
            return region
    return "X_Other"


# -----------------------------------------------------------------------------
# Interpretation engine (region-to-region)
# -----------------------------------------------------------------------------
class BrainInsights:
    @staticmethod
    def get_context(r_src: str, r_dst: str) -> Tuple[str, str]:
        if r_src == r_dst:
            if r_src == "O_Vis":
                return (
                    "VISUAL RECURRENCE / BINDING",
                    "Recurrent processing within the visual network supporting stable representation.",
                )
            if r_src == "P_Space":
                return (
                    "SPATIAL INTEGRATION",
                    "Integration of multisensory inputs into a coherent body/world spatial map.",
                )
            if r_src == "F_Motor":
                return (
                    "INTRA-MOTOR COORDINATION",
                    "Coordination and synchronization within the motor planning/execution network.",
                )
            return (
                "LOCAL PROCESSING",
                f"Predominantly local processing within region {r_src}.",
            )

        if r_src == "O_Vis" and (r_dst == "P_Space" or r_dst == "F_Motor"):
            return (
                "VISUO-MOTOR TRANSFORMATION",
                "Transformation of visual information into spatial/motor coordinates.",
            )
        if r_src == "P_Space" and r_dst == "O_Vis":
            return (
                "TOP-DOWN VISUAL ATTENTION",
                "Top-down feedback from parietal areas biasing visual processing (predictive attention).",
            )
        if r_src == "F_Motor" and (r_dst == "P_Space" or r_dst == "O_Vis"):
            return (
                "EFFERENCE COPY (PREDICTION)",
                "Efference copy sent to sensory systems to predict the consequences of motor commands.",
            )
        if r_src == "F_Exec":
            return (
                "EXECUTIVE CONTROL",
                "Executive control signals related to intention, inhibition, and task control.",
            )

        return ("FUNCTIONAL CONNECTIVITY", f"Information transfer: {r_src} -> {r_dst}.")


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------
_EDGE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+(?P<src>[A-Za-z0-9]+)\s+->\s+(?P<dst>[A-Za-z0-9]+)\s+:\s+(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)

_FILE_RE = re.compile(
    r"^phase_top_edges_(?P<phase>.+?)_(?P<mode>gc|corr)_m(?P<m>\d+)_pc(?P<pc>\d+)_lag(?P<lag>\d+)ms\.txt$",
    re.IGNORECASE,
)


def parse_edge_line(line: str) -> Optional[Tuple[str, str, float]]:
    m = _EDGE_RE.match(line.strip())
    if not m:
        return None
    return m.group("src"), m.group("dst"), float(m.group("val"))


def parse_filename(fname: str) -> Optional[Tuple[str, str, int, str, int]]:
    m = _FILE_RE.match(fname)
    if not m:
        return None
    phase = m.group("phase")
    mode = m.group("mode").lower()
    pc = f"pc{int(m.group('pc'))}"
    lag_ms = int(m.group("lag"))
    deg_m = int(m.group("m"))
    return phase, pc, lag_ms, mode, deg_m


# -----------------------------------------------------------------------------
# Statistics: BH FDR
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Directionality Index on region flows
# -----------------------------------------------------------------------------
def directionality_index(flow: Counter, a: str, b: str) -> float:
    ab = int(flow.get((a, b), 0))
    ba = int(flow.get((b, a), 0))
    denom = ab + ba
    return 0.0 if denom == 0 else (ab - ba) / float(denom)


def main() -> None:
    ap = argparse.ArgumentParser(description="MFG EEG Meta-Analysis")
    ap.add_argument(
        "--dir",
        default="out/phase_mats_pca_by_subject",
        help="Root directory containing subject subfolders with phase_top_edges_*.txt files.",
    )
    ap.add_argument(
        "--topk", type=int, default=10, help="Top-k edges to read per file."
    )
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
        help="Suppress non-essential output (prints only the core report).",
    )
    args = ap.parse_args()

    C = int(args.channels)
    M = C * (C - 1)
    if args.p0 is None:
        p0 = min(1.0, float(args.p0_inflate) * (float(args.topk) / float(M)))
    else:
        p0 = float(args.p0)

    pattern = os.path.join(args.dir, "**", "phase_top_edges_*.txt")
    all_files = glob.glob(pattern, recursive=True)
    if not all_files:
        print("No phase_top_edges_*.txt files found in the given directory.")
        return

    # subj_edge_presence[(phase, pc, lag)][(src,dst)] = set(subjects)
    subj_edge_presence = defaultdict(lambda: defaultdict(set))
    region_flow_by_scenario = defaultdict(list)

    subjects = set()

    for fpath in all_files:
        subj = os.path.basename(os.path.dirname(fpath))
        subjects.add(subj)

        meta = parse_filename(os.path.basename(fpath))
        if meta is None:
            continue
        phase, pc, lag_ms, mode, _deg_m = meta

        if args.mode != "any" and mode != args.mode:
            continue

        scenario = (phase, pc, lag_ms)

        edges_taken = 0
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_edge_line(line)
                    if parsed is None:
                        continue
                    src, dst, _val = parsed

                    subj_edge_presence[scenario][(src, dst)].add(subj)

                    r_src = get_region(src)
                    r_dst = get_region(dst)
                    region_flow_by_scenario[scenario].append((r_src, r_dst))

                    edges_taken += 1
                    if edges_taken >= int(args.topk):
                        break
        except OSError:
            continue

    N = len(subjects)
    scenarios = sorted(subj_edge_presence.keys(), key=lambda x: (x[0], x[1], x[2]))

    # Collect global report items for optional FDR
    report_items: List[Tuple[Tuple[str, str, int], Tuple[str, str], int, float]] = []
    for scenario in scenarios:
        for edge, subjs in subj_edge_presence[scenario].items():
            k = len(subjs)
            if k >= int(args.min_subjects):
                pval = binomtest(k, N, p=p0, alternative="greater").pvalue
                report_items.append((scenario, edge, k, pval))

    q_map = {}
    if args.use_fdr and report_items:
        qvals = fdr_bh([it[3] for it in report_items])
        for it, q in zip(report_items, qvals):
            scenario, edge, _k, _p = it
            q_map[(scenario, edge)] = q

    if not args.quiet:
        print(
            f"N_subjects={N} | mode={args.mode} | topk={args.topk} | p0={p0:.6g} | FDR={args.use_fdr}"
        )

    for scenario in scenarios:
        phase, pc, lag_ms = scenario
        edge2subj = subj_edge_presence[scenario]

        candidates = []
        for (src, dst), subjs in edge2subj.items():
            k = len(subjs)
            if k < int(args.min_subjects):
                continue
            pval = binomtest(k, N, p=p0, alternative="greater").pvalue
            qval = q_map.get((scenario, (src, dst)), None) if args.use_fdr else None
            candidates.append((k, pval, (qval if qval is not None else 1.0), src, dst))

        if not candidates:
            continue

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        flow = Counter(region_flow_by_scenario[scenario])
        (dom_src, dom_dst), dom_cnt = flow.most_common(1)[0]
        di = directionality_index(flow, dom_src, dom_dst)
        proc_label, proc_desc = BrainInsights.get_context(dom_src, dom_dst)

        # Scenario header
        print(f"{phase} | {pc} | lag={lag_ms}ms")

        # Edge table header
        if args.use_fdr:
            print("  edge        k/N    p-value      q(FDR)      sig")
        else:
            print("  edge        k/N    p-value      sig")

        # Print top edges
        shown = 0
        for k, pval, qtmp, src, dst in candidates:
            if shown >= int(args.max_edges):
                break
            if args.use_fdr:
                qval = q_map.get((scenario, (src, dst)), 1.0)
                sig = (
                    "***"
                    if qval < 0.001
                    else ("**" if qval < 0.01 else ("*" if qval < args.alpha else ""))
                )
                print(
                    f"  {src:4}->{dst:4}   {k:>2}/{N:<2}  {pval:>10.4g}  {qval:>10.4g}  {sig}"
                )
            else:
                sig = (
                    "***"
                    if pval < 0.001
                    else ("**" if pval < 0.01 else ("*" if pval < args.alpha else ""))
                )
                print(f"  {src:4}->{dst:4}   {k:>2}/{N:<2}  {pval:>10.4g}  {sig}")
            shown += 1

        # Interpretation line
        print(f"  dominant_flow {dom_src}->{dom_dst}  DI={di:+.2f}  ({proc_label})")
        print(f"  {proc_desc}")
        print()


if __name__ == "__main__":
    main()
