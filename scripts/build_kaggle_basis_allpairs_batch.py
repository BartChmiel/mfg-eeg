from __future__ import annotations
import os, re, argparse
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


def _fix_pca_signs(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, float).copy()
    for k in range(U.shape[0]):
        if U[k, np.argmax(np.abs(U[k]))] < 0:
            U[k] *= -1.0
    return U


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description="Build global PCA basis for all channel pairs."
    )
    ap.add_argument("--root", required=True, help="Dataset root directory.")
    ap.add_argument("--out", required=True, help="Output path for .npz basis file.")
    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lags-ms", type=int, nargs="+", default=[0, 50, 100, 150, 200])
    ap.add_argument("--pca-r", type=int, default=3)
    args = ap.parse_args()

    # Initialization
    root = Path(args.root)
    files = sorted(list(root.glob("**/subj*_series*_data.csv")))
    if not files:
        raise RuntimeError("No data files found.")

    fs, m, r = int(cfg.fs), int(args.m), int(args.pca_r)
    K = m * m
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in args.lags_ms]
    seg_len = int(round(2.0 * fs))

    # Reference channel setup from first file
    _, X0, _, ch_names0, _ = load_kaggle_data_with_events(
        str(files[0]), str(files[0]).replace("_data.csv", "_events.csv")
    )
    C = X0.shape[1]

    sum_x = np.zeros((C, C, K))
    sum_xxT = np.zeros((C, C, K, K))
    w_sum = np.zeros((C, C))

    # Main processing loop
    for data_path in files:
        events_path = Path(str(data_path).replace("_data.csv", "_events.csv"))
        if not events_path.exists():
            continue

        _, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(events_path)
        )
        if ch_names != ch_names0:
            continue  # Basic safety

        Y_all, Z_all = _normalize_series(X, args.mode, fs, cfg)
        cycles = extract_grasp_cycles(E, fs)

        for cyc in cycles:
            for ev0, ev1 in PHASES:
                bounds = phase_window_bounds(X.shape[0], cyc[ev0], cyc[ev1], fs, 2.0)
                if not bounds:
                    continue
                a, b = bounds

                # Windowed basis projection
                for li, Ls in enumerate(lag_samples):
                    n_eff = (b - a) - Ls
                    if n_eff <= 5:
                        continue

                    # Compute coefficients using vectorized ops
                    Fy = np.stack(
                        [
                            legendre_orthonormal(Y_all[a : b - Ls, c], m)[:, 1:]
                            for c in range(C)
                        ],
                        axis=1,
                    )
                    Fz = np.stack(
                        [
                            legendre_orthonormal(Z_all[a + Ls : b, c], m)[:, 1:]
                            for c in range(C)
                        ],
                        axis=1,
                    )

                    M = np.einsum("tcm,tdn->cdmn", Fy, Fz) / float(n_eff)
                    # Subtract marginals
                    M -= (
                        Fy.mean(axis=0)[:, None, :, None]
                        * Fz.mean(axis=0)[None, :, None, :]
                    )

                    M_flat = M.reshape(C, C, K)
                    sum_x += n_eff * M_flat
                    sum_xxT += n_eff * np.einsum("cdk,cdl->cdkl", M_flat, M_flat)
                    w_sum += n_eff

    # Final PCA computation
    U = np.zeros((C, C, r, K))
    mean = sum_x / w_sum[:, :, None]

    for i in range(C):
        for j in range(C):
            if w_sum[i, j] <= 0:
                continue
            mu = mean[i, j]
            Cov = (sum_xxT[i, j] / w_sum[i, j]) - np.outer(mu, mu)
            w, V = np.linalg.eigh(0.5 * (Cov + Cov.T))
            idx = np.argsort(w)[::-1]
            U[i, j] = _fix_pca_signs(V[:, idx[:r]].T)

    np.savez_compressed(
        args.out,
        ch_names=np.array(ch_names0),
        mean=mean,
        U=U,
        lags_ms=args.lags_ms,
        fs=fs,
        m=m,
    )
    print(f"Basis saved to {args.out}")


if __name__ == "__main__":
    main()
