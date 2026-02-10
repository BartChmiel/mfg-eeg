from __future__ import annotations
import os, re, argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

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


def _normalize_series(
    X: np.ndarray, mode: str, fs: int, cfg
) -> Tuple[np.ndarray, np.ndarray]:
    T, C = X.shape
    Y, Z = np.empty((T, C)), np.empty((T, C))
    for c in range(C):
        if mode == "corr":
            Y[:, c] = Z[:, c] = normalize_gauss(X[:, c])
        else:
            Y[:, c] = normalize_edf(X[:, c])
            Z[:, c] = pnorm_student(
                X[:, c], fs, cfg.ar_order, cfg.ema_half_life_s, cfg.student_nu
            )
    return Y, Z


def _save_top_edges(
    path: str, mat: np.ndarray, ch_names: List[str], topk: int = 40
) -> None:
    A = np.array(mat, float)
    np.fill_diagonal(A, -np.inf)
    flat_idx = np.argsort(A.ravel())[::-1]
    with open(path, "a", encoding="utf-8") as f:
        for count, idx in enumerate(flat_idx[:topk]):
            val = A.ravel()[idx]
            if not np.isfinite(val):
                break
            i, j = divmod(idx, A.shape[1])
            f.write(f"{count+1:03d}. {ch_names[i]} -> {ch_names[j]} : {val:.6f}\n")


def _run_group(
    group_name: str,
    files: List[Path],
    out_dir: Path,
    args: argparse.Namespace,
    cfg: Any,
) -> None:
    fs, m = int(cfg.fs), int(args.m)
    K, r = m * m, int(args.pca_r)
    lags_ms = list(args.lags_ms)
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]

    # Initialization from first file
    _, X0, _, ch_names0, _ = load_kaggle_data_with_events(
        str(files[0]), str(files[0]).replace("_data.csv", "_events.csv")
    )
    C, P, L = X0.shape[1], len(PHASES), len(lag_samples)

    coeffs_sum = np.zeros((P, L, C, C, K))
    n_eff_sum = np.zeros((P, L))
    n_files, n_cycles = 0, 0

    for data_path in files:
        ev_path = Path(str(data_path).replace("_data.csv", "_events.csv"))
        if not ev_path.exists():
            continue

        _, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(ev_path)
        )
        if ch_names != ch_names0:
            continue

        Y_all, Z_all = _normalize_series(X, args.mode, fs, cfg)
        cycles = extract_grasp_cycles(E, fs)
        n_files += 1
        n_cycles += len(cycles)

        for cyc in cycles:
            for pi, (ev0, ev1) in enumerate(PHASES):
                bounds = phase_window_bounds(X.shape[0], cyc[ev0], cyc[ev1], fs, 2.0)
                if not bounds:
                    continue
                a, b = bounds

                # Precompute Legendre basis for the window
                Fy_full = np.stack(
                    [legendre_orthonormal(Y_all[a:b, c], m)[:, 1:] for c in range(C)],
                    axis=1,
                )
                Fz_full = np.stack(
                    [legendre_orthonormal(Z_all[a:b, c], m)[:, 1:] for c in range(C)],
                    axis=1,
                )

                for li, Ls in enumerate(lag_samples):
                    n_eff = (b - a) - Ls
                    if n_eff <= 5:
                        continue

                    Fy, Fz = Fy_full[:n_eff], Fz_full[Ls:]
                    M = np.einsum("tcm,tdn->cdmn", Fy, Fz) / float(n_eff)
                    M -= (
                        Fy.mean(axis=0)[:, None, :, None]
                        * Fz.mean(axis=0)[None, :, None, :]
                    )

                    coeffs_sum[pi, li] += n_eff * M.reshape(C, C, K)
                    n_eff_sum[pi, li] += n_eff

    # Result generation per phase
    group_path = out_dir / group_name
    group_path.mkdir(exist_ok=True)

    for pi, (ev0, ev1) in enumerate(PHASES):
        if np.all(n_eff_sum[pi] <= 0):
            continue

        c_mean = coeffs_sum[pi] / n_eff_sum[pi][:, None, None, None]

        if args.basis_allpairs:
            # PCA Projection Mode
            B = np.load(args.basis_allpairs, allow_pickle=True)
            U, mu = B["U"][:, :, :r, :], B["mean"]
            scores = np.einsum("cdrk,lck->rlcd", U, c_mean - mu)

            for pc_idx in range(r):
                for li, ms in enumerate(lags_ms):
                    mat = scores[pc_idx, li]
                    fname = f"phase_{ev0}_{ev1}_{group_name}_pc{pc_idx+1}_lag{ms}ms"
                    _save_top_edges(str(group_path / f"{fname}.txt"), mat, ch_names0)
        else:
            # Statistical Metric Mode
            chi2_stat = n_eff_sum[pi][:, None, None] * np.sum(c_mean**2, axis=-1)
            if args.metric == "neglog10p":
                mat = np.mean(
                    -np.log10(np.clip(chi2.sf(chi2_stat, df=K), 1e-300, 1.0)), axis=0
                )
            else:
                mat = np.mean(chi2_stat, axis=0)

            plt.figure(figsize=(10, 8))
            plt.imshow(mat, origin="lower")
            plt.title(f"{ev0} to {ev1} - {args.metric}")
            plt.savefig(group_path / f"phase_{ev0}_{ev1}_{args.metric}.png")
            plt.close()
            _save_top_edges(
                str(group_path / f"phase_{ev0}_{ev1}_{args.metric}.txt"), mat, ch_names0
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-by", choices=["all", "subject"], default="all")
    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")
    ap.add_argument(
        "--metric", choices=["energy", "chi2", "neglog10p"], default="neglog10p"
    )
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lags-ms", type=int, nargs="+", default=[50, 100, 150, 200])
    ap.add_argument("--pca-r", type=int, default=3)
    ap.add_argument("--basis-allpairs", default=None)
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(list(Path(args.root).glob("**/subj*_series*_data.csv")))

    if args.group_by == "all":
        _run_group("ALL", all_files, out_dir, args, cfg)
    else:
        subjects = set(re.search(r"subj(\d+)", f.name).group(1) for f in all_files)
        for s in sorted(subjects):
            s_files = [f for f in all_files if f"subj{s}" in f.name]
            _run_group(f"subj{s}", s_files, out_dir, args, cfg)


if __name__ == "__main__":
    main()
