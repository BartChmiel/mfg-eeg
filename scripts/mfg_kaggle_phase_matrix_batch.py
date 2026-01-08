"""
Batch version of mfg_kaggle_phase_matrix.py

Goal:
- compute phase matrices (C x C) for canonical phases
- aggregate across MANY Kaggle train files
- optionally group by subject (to compare repeatability across subjects)

This outputs:
- heatmaps per phase: phase_<ev0>__<ev1>_<mode>_m<m>_<metric>_<summary>.png
- top edges per phase: phase_top_edges_<...>.txt
- a short metadata header in each txt with config + counts

Notes:
- Works on TRAIN split (needs *_events.csv).
- Uses fixed-length windows via phase_window_bounds for consistent lag grid.
- Normalization is global per series (per file), not per window (better comparability).
"""

from __future__ import annotations

import os
import re
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from src.mfg.config import load_config
from src.mfg.basis import legendre_orthonormal
from src.mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src.mfg.io import (
    load_kaggle_data_with_events,
    extract_grasp_cycles,
    phase_window_bounds,
)


PHASES = [
    ("HandStart", "FirstDigitTouch"),
    ("FirstDigitTouch", "BothStartLoadPhase"),
    ("BothStartLoadPhase", "LiftOff"),
    ("LiftOff", "Replace"),
    ("Replace", "BothReleased"),
]


_RX = re.compile(r"subj(?P<subj>\d+)_series(?P<series>\d+)_data\.csv$", re.IGNORECASE)


def _parse_subj_series(p: Path) -> Tuple[Optional[int], Optional[int]]:
    m = _RX.search(p.name)
    if not m:
        return None, None
    return int(m.group("subj")), int(m.group("series"))


def _iter_train_data_files(root: Path) -> List[Path]:
    if (root / "train").is_dir():
        train_dir = root / "train"
    else:
        train_dir = root
    return sorted(train_dir.glob("subj*_series*_data.csv"))


def _events_path_for_data(data_path: Path) -> Path:
    return data_path.with_name(data_path.name.replace("_data.csv", "_events.csv"))


def _basis_mixed(u_1d: np.ndarray, m: int) -> np.ndarray:
    """
    Legendre orthonormal basis on [0,1], keep only degrees 1..m (mixed_only).
    Returns shape (T, m).
    """
    F = legendre_orthonormal(u_1d, m)  # (T, m+1)
    return F[:, 1:]  # drop degree-0


def _compute_coeffs_all_pairs_for_lag(
    Fy: np.ndarray,  # (n_eff, C, m)
    Fz: np.ndarray,  # (n_eff, C, m)
    subtract_marginals: bool,
) -> np.ndarray:
    """
    Compute M for all ordered pairs (i -> j):
      M[i,j,a,b] = E[ Fy_i,a * Fz_j,b ]  (optionally - marginals)
    Returns shape (C, C, m, m).
    """
    n_eff = Fy.shape[0]
    M = np.einsum("tcm,tdn->cdmn", Fy, Fz) / float(n_eff)

    if subtract_marginals:
        my = Fy.mean(axis=0)  # (C, m)
        mz = Fz.mean(axis=0)  # (C, m)
        M = M - my[:, None, :, None] * mz[None, :, None, :]

    return M


def _robust_vmax(A: np.ndarray) -> float:
    x = A[np.isfinite(A)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(x, 0.98))


def _save_top_edges(
    path: str, mat: np.ndarray, ch_names: List[str], topk: int = 40
) -> None:
    A = np.array(mat, float)
    np.fill_diagonal(A, -np.inf)
    flat_idx = np.argsort(A.ravel())[::-1]
    with open(path, "a", encoding="utf-8") as f:
        k = 0
        for idx in flat_idx:
            val = float(A.ravel()[idx])
            if not np.isfinite(val):
                continue
            i = idx // A.shape[1]
            j = idx % A.shape[1]
            f.write(f"{k+1:03d}. {ch_names[i]} -> {ch_names[j]} : {val:.6f}\n")
            k += 1
            if k >= topk:
                break


def _normalize_series(
    X: np.ndarray, mode: str, fs: int, cfg
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce Y (reason) and Z (result) arrays, shape (T, C) each, for the WHOLE series.
    - corr: both gaussian-CDF
    - gc:   reason = EDF, result = pnorm_student
    """
    T, C = X.shape
    Y = np.empty((T, C), dtype=float)
    Z = np.empty((T, C), dtype=float)

    if mode == "corr":
        for c in range(C):
            u = normalize_gauss(X[:, c])
            Y[:, c] = u
            Z[:, c] = u
        return Y, Z

    for c in range(C):
        Y[:, c] = normalize_edf(X[:, c])
        Z[:, c] = pnorm_student(
            X[:, c],
            fs,
            cfg.ar_order,
            cfg.ema_half_life_s,
            cfg.student_nu,
        )
    return Y, Z


def _filter_files(
    files: List[Path],
    subjects: Optional[List[int]],
    series: Optional[List[int]],
    max_files: Optional[int],
) -> List[Path]:
    out: List[Path] = []
    for p in files:
        s, se = _parse_subj_series(p)
        if s is None or se is None:
            continue
        if subjects is not None and s not in subjects:
            continue
        if series is not None and se not in series:
            continue
        out.append(p)
        if max_files is not None and len(out) >= max_files:
            break
    return out


def _run_group(
    *,
    group_name: str,
    files: List[Path],
    out_dir: Path,
    mode: str,
    epoch_len_s: float,
    max_cycle_s: float,
    min_cycles: int,
    m: int,
    lags_ms: List[int],
    summary: str,
    metric: str,
) -> None:
    cfg = load_config()
    fs = int(cfg.fs)

    if not files:
        print(f"[WARN] group={group_name}: no files -> skipping")
        return

    # Setup from first file
    first_data = files[0]
    first_events = _events_path_for_data(first_data)
    if not first_events.exists():
        print(f"[WARN] group={group_name}: missing events for first file -> skipping")
        return

    _, X0, E0, ch_names0, _ = load_kaggle_data_with_events(
        str(first_data), str(first_events)
    )
    C = X0.shape[1]

    # Lag setup
    seg_len = int(round(float(epoch_len_s) * float(fs)))
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]
    K = m * m  # mixed_only=True

    P = len(PHASES)
    L = len(lag_samples)

    # Accumulators per phase, per lag
    coeffs_sum = np.zeros((P, L, C, C, K), dtype=float)
    n_eff_sum = np.zeros((P, L), dtype=float)

    subtract_marginals = True

    n_files_used = 0
    n_cycles_total = 0
    n_windows_total = 0

    for data_path in files:
        events_path = _events_path_for_data(data_path)
        if not events_path.exists():
            print(f"[WARN] Missing events for {data_path.name} -> skipping")
            continue

        ids, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(events_path)
        )

        # Verify channel order consistency; if not, attempt to reorder to match first file
        if ch_names != ch_names0:
            # Build mapping from current file channels to reference ordering
            idx_map = []
            ok = True
            for nm in ch_names0:
                if nm not in ch_names:
                    ok = False
                    break
                idx_map.append(ch_names.index(nm))
            if not ok:
                raise RuntimeError(
                    f"Channel mismatch in {data_path.name}. Cannot align to reference order."
                )
            X = X[:, idx_map]
            ch_names = ch_names0

        T = X.shape[0]
        Y_all, Z_all = _normalize_series(X, mode=mode, fs=fs, cfg=cfg)

        cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=float(max_cycle_s))
        if len(cycles) < int(min_cycles):
            print(
                f"[WARN] {data_path.name}: cycles={len(cycles)} < min_cycles={min_cycles}"
            )

        n_files_used += 1
        n_cycles_total += int(len(cycles))

        for cyc in cycles:
            for pi, (ev0, ev1) in enumerate(PHASES):
                bounds = phase_window_bounds(
                    T,
                    int(cyc[ev0]),
                    int(cyc[ev1]),
                    fs=fs,
                    epoch_len_s=float(epoch_len_s),
                )
                if bounds is None:
                    continue
                a, b = bounds
                if (b - a) != seg_len:
                    continue

                Yr = Y_all[a:b, :]  # (seg_len, C)
                Zr = Z_all[a:b, :]  # (seg_len, C)

                # Precompute basis per channel for this window: (seg_len, C, m)
                Fy_full = np.empty((seg_len, C, m), dtype=float)
                Fz_full = np.empty((seg_len, C, m), dtype=float)
                for c in range(C):
                    Fy_full[:, c, :] = _basis_mixed(Yr[:, c], m)
                    Fz_full[:, c, :] = _basis_mixed(Zr[:, c], m)

                for li, Ls in enumerate(lag_samples):
                    n_eff = seg_len - Ls
                    if n_eff <= 5:
                        continue

                    Fy = Fy_full[:n_eff, :, :]  # (n_eff, C, m)
                    Fz = Fz_full[Ls:, :, :]  # (n_eff, C, m)

                    M = _compute_coeffs_all_pairs_for_lag(
                        Fy, Fz, subtract_marginals=subtract_marginals
                    )
                    M_flat = M.reshape(C, C, K)

                    coeffs_sum[pi, li] += float(n_eff) * M_flat
                    n_eff_sum[pi, li] += float(n_eff)

                n_windows_total += 1

    # Finalize + save per phase
    group_out = out_dir / group_name
    group_out.mkdir(parents=True, exist_ok=True)

    for pi, (ev0, ev1) in enumerate(PHASES):
        if np.all(n_eff_sum[pi] <= 0):
            print(
                f"[WARN] group={group_name} phase={ev0}->{ev1}: no samples -> skipping outputs"
            )
            continue

        coeffs_mean = coeffs_sum[pi] / n_eff_sum[pi][:, None, None, None]  # (L,C,C,K)
        energy_per_lag = np.sqrt(np.sum(coeffs_mean**2, axis=-1))  # (L,C,C)

        chi2_stat = n_eff_sum[pi][:, None, None] * np.sum(
            coeffs_mean**2, axis=-1
        )  # (L,C,C)
        p = chi2.sf(chi2_stat, df=K)
        p = np.clip(p, 1e-300, 1.0)
        neglog10p = -np.log10(p)

        if metric == "energy":
            per_lag = energy_per_lag
            cbar_label = "||coeffs||"
        elif metric == "chi2":
            per_lag = chi2_stat
            cbar_label = f"chi2(df={K})"
        elif metric == "neglog10p":
            per_lag = neglog10p
            cbar_label = "-log10(p)"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if summary == "mean":
            mat = np.mean(per_lag, axis=0)
        elif summary == "max":
            mat = np.max(per_lag, axis=0)
        else:
            raise ValueError(f"Unknown summary: {summary}")

        mat_disp = np.array(mat, float)
        np.fill_diagonal(mat_disp, 0.0)

        title = (
            f"{ev0}__{ev1} | group={group_name} | mode={mode} | m={m} | "
            f"metric={metric} | summary={summary} | lags_ms={lags_ms} | "
            f"files={n_files_used} cycles={n_cycles_total} windows={n_windows_total}"
        )

        # Heatmap
        plt.figure(figsize=(11.5, 10))
        vmax = _robust_vmax(mat_disp)
        im = plt.imshow(mat_disp, origin="lower", vmin=0.0, vmax=vmax)
        plt.colorbar(im, label=cbar_label)
        plt.xticks(range(C), ch_names0, rotation=90, fontsize=8)
        plt.yticks(range(C), ch_names0, fontsize=8)
        plt.title(title)
        plt.tight_layout()

        out_png = group_out / f"phase_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.png"
        plt.savefig(out_png, dpi=220)
        plt.close()

        # Top edges
        out_txt = (
            group_out
            / f"phase_top_edges_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.txt"
        )
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"phase={ev0}__{ev1}\n")
            f.write(f"group={group_name}\n")
            f.write(
                f"mode={mode} m={m} metric={metric} summary={summary} lags_ms={lags_ms}\n"
            )
            f.write(f"epoch_len_s={epoch_len_s} fs={fs}\n")
            f.write(
                f"files={n_files_used} cycles_total={n_cycles_total} windows_total={n_windows_total}\n"
            )
            f.write(f"config={asdict(cfg)}\n\n")
        _save_top_edges(str(out_txt), mat, ch_names0, topk=40)

        print("Saved:", str(out_png))
        print("Saved:", str(out_txt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root or train/ directory.")
    ap.add_argument(
        "--out", required=True, help="Output directory (e.g. out/phase_mats_batch)."
    )

    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")
    ap.add_argument("--epoch-len-s", type=float, default=2.0)
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)

    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lags-ms", type=int, nargs="+", default=[50, 100, 150, 200])
    ap.add_argument("--summary", choices=["mean", "max"], default="mean")
    ap.add_argument(
        "--metric", choices=["energy", "chi2", "neglog10p"], default="neglog10p"
    )

    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--series", type=int, nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)

    ap.add_argument(
        "--group-by",
        choices=["all", "subject"],
        default="all",
        help="all: single global aggregate; subject: separate outputs for each subject.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_all = _iter_train_data_files(root)
    files = _filter_files(files_all, args.subjects, args.series, args.max_files)
    if not files:
        raise RuntimeError("No matching train data files found.")

    if args.group_by == "all":
        _run_group(
            group_name="ALL",
            files=files,
            out_dir=out_dir,
            mode=args.mode,
            epoch_len_s=float(args.epoch_len_s),
            max_cycle_s=float(args.max_cycle_s),
            min_cycles=int(args.min_cycles),
            m=int(args.m),
            lags_ms=list(args.lags_ms),
            summary=str(args.summary),
            metric=str(args.metric),
        )
        return

    # group_by == "subject"
    groups: Dict[str, List[Path]] = {}
    for p in files:
        subj, se = _parse_subj_series(p)
        if subj is None:
            continue
        key = f"subj{subj:02d}"
        groups.setdefault(key, []).append(p)

    for key in sorted(groups.keys()):
        _run_group(
            group_name=key,
            files=groups[key],
            out_dir=out_dir,
            mode=args.mode,
            epoch_len_s=float(args.epoch_len_s),
            max_cycle_s=float(args.max_cycle_s),
            min_cycles=int(args.min_cycles),
            m=int(args.m),
            lags_ms=list(args.lags_ms),
            summary=str(args.summary),
            metric=str(args.metric),
        )


if __name__ == "__main__":
    main()


# python -m scripts.mfg_kaggle_phase_matrix_batch `
#   --root data/grasp-and-lift-eeg-detection/train `
#   --out out/phase_mats_batch `
#   --mode gc --epoch-len-s 2.0 --m 4 `
#   --lags-ms 50 100 150 200 `
#   --metric neglog10p --summary mean
