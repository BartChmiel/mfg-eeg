"""
Build PCA bases for ALL ordered channel pairs (i -> j) from Kaggle TRAIN split.

For each ordered pair (i->j), we collect K=m*m HCR mixed-only features at a fixed set
of lags (e.g. 0,50,100,150,200 ms) across:
- many files (train),
- all canonical phases,
- all cycles/windows.

Then we compute PCA per pair and keep r=3 components.

Output .npz contains:
- ch_names (object array of strings), C
- mean:   (C, C, K)
- U:      (C, C, r, K)   (top-r PCA directions, row-major)
- eigvals:(C, C, r)
- lags_ms, lag_samples
- metadata: fs, m, mode, epoch_len_s, counts
"""

from __future__ import annotations

import os
import re
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src_mfg.config import load_config
from src_mfg.basis import legendre_orthonormal
from src_mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src_mfg.io import (
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
    M[i,j,a,b] = E[ Fy_i,a * Fz_j,b ] - E[Fy_i,a]E[Fz_j,b]
    Returns shape (C, C, m, m).
    """
    n_eff = Fy.shape[0]
    M = np.einsum("tcm,tdn->cdmn", Fy, Fz) / float(n_eff)

    if subtract_marginals:
        my = Fy.mean(axis=0)  # (C, m)
        mz = Fz.mean(axis=0)  # (C, m)
        M = M - my[:, None, :, None] * mz[None, :, None, :]

    return M


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


def _fix_pca_signs(U: np.ndarray) -> np.ndarray:
    """
    Deterministic PCA sign convention:
    for each component u, make the largest-abs entry positive.
    U shape: (r, K)
    """
    U = np.asarray(U, float).copy()
    for k in range(U.shape[0]):
        idx = int(np.argmax(np.abs(U[k])))
        if U[k, idx] < 0:
            U[k] *= -1.0
    return U


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root or train/ directory.")
    ap.add_argument(
        "--out", required=True, help="Output .npz path, e.g. out/basis_allpairs.npz"
    )

    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")
    ap.add_argument("--epoch-len-s", type=float, default=2.0)
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)

    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lags-ms", type=int, nargs="+", default=[0, 50, 100, 150, 200])
    ap.add_argument("--pca-r", type=int, default=3)

    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--series", type=int, nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    root = Path(args.root)
    files_all = _iter_train_data_files(root)
    files = _filter_files(files_all, args.subjects, args.series, args.max_files)
    if not files:
        raise RuntimeError("No matching train data files found.")

    if args.dry_run:
        print(f"Matched {len(files)} files:")
        for p in files[:50]:
            print(" -", p)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fs = int(cfg.fs)
    m = int(args.m)
    K = m * m
    r = int(args.pca_r)

    lags_ms = list(map(int, args.lags_ms))
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]
    seg_len = int(round(float(args.epoch_len_s) * float(fs)))

    # Initialize from first file: reference channel ordering
    first_data = files[0]
    first_events = _events_path_for_data(first_data)
    if not first_events.exists():
        raise RuntimeError("Missing events for first file (expected TRAIN split).")

    _, X0, _, ch_names0, _ = load_kaggle_data_with_events(
        str(first_data), str(first_events)
    )
    C = X0.shape[1]

    # Streaming moments per pair: sum_x and sum_xxT
    sum_x = np.zeros((C, C, K), dtype=float)
    sum_xxT = np.zeros((C, C, K, K), dtype=float)
    # n_samples = np.zeros((C, C), dtype=np.int64)
    w_sum = np.zeros((C, C), dtype=float)

    n_files_used = 0
    n_cycles_total = 0
    n_windows_total = 0

    subtract_marginals = True

    for data_path in files:
        events_path = _events_path_for_data(data_path)
        if not events_path.exists():
            print(f"[WARN] Missing events for {data_path.name} -> skipping")
            continue

        _, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(events_path)
        )

        # align channel ordering to ch_names0 if needed
        if ch_names != ch_names0:
            idx_map = []
            ok = True
            for nm in ch_names0:
                if nm not in ch_names:
                    ok = False
                    break
                idx_map.append(ch_names.index(nm))
            if not ok:
                raise RuntimeError(
                    f"Channel mismatch in {data_path.name}, cannot align."
                )
            X = X[:, idx_map]
            ch_names = ch_names0

        T = X.shape[0]
        Y_all, Z_all = _normalize_series(X, mode=str(args.mode), fs=fs, cfg=cfg)

        cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=float(args.max_cycle_s))
        if len(cycles) < int(args.min_cycles):
            print(
                f"[WARN] {data_path.name}: cycles={len(cycles)} < min_cycles={args.min_cycles}"
            )

        n_files_used += 1
        n_cycles_total += int(len(cycles))

        for cyc in cycles:
            for ev0, ev1 in PHASES:
                bounds = phase_window_bounds(
                    T,
                    int(cyc[ev0]),
                    int(cyc[ev1]),
                    fs=fs,
                    epoch_len_s=float(args.epoch_len_s),
                )
                if bounds is None:
                    continue
                a, b = bounds
                if (b - a) != seg_len:
                    continue

                Yr = Y_all[a:b, :]  # (seg_len, C)
                Zr = Z_all[a:b, :]  # (seg_len, C)

                Fy_full = np.empty((seg_len, C, m), dtype=float)
                Fz_full = np.empty((seg_len, C, m), dtype=float)
                for c in range(C):
                    Fy_full[:, c, :] = _basis_mixed(Yr[:, c], m)
                    Fz_full[:, c, :] = _basis_mixed(Zr[:, c], m)

                for Ls in lag_samples:
                    n_eff = seg_len - Ls
                    if n_eff <= 5:
                        continue

                    Fy = Fy_full[:n_eff, :, :]
                    Fz = Fz_full[Ls:, :, :]

                    M = _compute_coeffs_all_pairs_for_lag(
                        Fy, Fz, subtract_marginals=subtract_marginals
                    )
                    Xpair = M.reshape(C, C, K)  # (C,C,K)

                    # # Accumulate per pair: each (window,lag) is one sample in R^K
                    # sum_x += Xpair
                    # # Outer products per pair (vectorized)
                    # sum_xxT += np.einsum("cdk,cdl->cdkl", Xpair, Xpair)
                    # n_samples += 1

                    # Weighted accumulation per pair: weight by n_eff (effective window length)
                    w = float(n_eff)
                    sum_x += w * Xpair
                    sum_xxT += w * np.einsum("cdk,cdl->cdkl", Xpair, Xpair)
                    w_sum += w

                n_windows_total += 1

    # PCA per pair
    mean = np.zeros((C, C, K), dtype=float)
    U = np.zeros((C, C, r, K), dtype=float)
    eigvals = np.zeros((C, C, r), dtype=float)

    for i in range(C):
        for j in range(C):
            # if n_samples[i, j] < 2:
            if w_sum[i, j] <= 0.0:
                # leave zeros
                continue

            # mu = sum_x[i, j] / float(n_samples[i, j])
            # Exx = sum_xxT[i, j] / float(n_samples[i, j])
            mu = sum_x[i, j] / float(w_sum[i, j])
            Exx = sum_xxT[i, j] / float(w_sum[i, j])
            Cov = Exx - np.outer(mu, mu)

            # Symmetrize tiny numerical asymmetry
            Cov = 0.5 * (Cov + Cov.T)

            w, V = np.linalg.eigh(Cov)  # ascending
            idx = np.argsort(w)[::-1]  # descending
            w = w[idx]
            V = V[:, idx]  # columns are eigenvectors

            # store top-r
            mu = np.asarray(mu, float)
            Uk = V[:, :r].T  # (r,K), row = component
            Uk = _fix_pca_signs(Uk)

            mean[i, j] = mu
            U[i, j] = Uk
            eigvals[i, j] = w[:r]

    np.savez_compressed(
        args.out,
        ch_names=np.array(ch_names0, dtype=object),
        mean=mean,
        U=U,
        eigvals=eigvals,
        lags_ms=np.array(lags_ms, dtype=int),
        lag_samples=np.array(lag_samples, dtype=int),
        fs=int(fs),
        m=int(m),
        mixed_only=np.int8(1),
        mode=str(args.mode),
        epoch_len_s=float(args.epoch_len_s),
        n_files=int(n_files_used),
        n_cycles_total=int(n_cycles_total),
        n_windows_total=int(n_windows_total),
        # n_samples_per_pair=n_samples,
        weight_sum_per_pair=w_sum,
        config=str(asdict(cfg)),
    )

    print(
        f"Saved all-pairs PCA basis: {args.out}\n"
        f"  files={n_files_used} cycles_total={n_cycles_total} windows_total={n_windows_total}\n"
        f"  C={C} K={K} r={r} lags_ms={lags_ms} mode={args.mode} epoch_len_s={args.epoch_len_s}"
    )


if __name__ == "__main__":
    main()
