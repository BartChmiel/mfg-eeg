# scripts/build_kaggle_basis.py
"""
Build a shared PCA basis for a fixed Kaggle electrode pair across ALL phases.

Output: .npz containing:
- mean (K,), V (K,K), eigvals (K,), lag_samples (L,)
- metadata: fs, maxlag_ms, lag_step_ms, m, mixed_only, chan_a, chan_b, mode, epoch_len_s

This basis can be reused in scripts/mfg_kaggle_phase_pair.py to make
PC1/PC2/PC3 comparable across phases.
"""

from __future__ import annotations

import os
import argparse
import numpy as np

from src.mfg.config import load_config
from src.mfg.io import (
    load_kaggle_data_with_events,
    extract_grasp_cycles,
    phase_window_bounds,
)
from src.mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src.mfg.hcr import hcr_coeffs_over_lags
from src.mfg.global_basis import pca_from_second_moments


PHASES = [
    ("HandStart", "FirstDigitTouch"),
    ("FirstDigitTouch", "BothStartLoadPhase"),
    ("BothStartLoadPhase", "LiftOff"),
    ("LiftOff", "Replace"),
    ("Replace", "BothReleased"),
]


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--chan-a", required=True)
    ap.add_argument("--chan-b", required=True)
    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")

    ap.add_argument("--epoch-len-s", type=float, default=2.0)
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)

    ap.add_argument(
        "--m", type=int, default=4
    )  # <-- m=4 => 16 features (mixed_only=True)
    ap.add_argument("--out", required=True, help="Output .npz path for the basis")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ids, X, E, ch_names, _ = load_kaggle_data_with_events(args.data, args.events)
    fs = int(cfg.fs)

    try:
        ia = ch_names.index(args.chan_a)
        ib = ch_names.index(args.chan_b)
    except ValueError as exc:
        raise ValueError(
            f"Channel not found. Available example: {ch_names[:10]} ..."
        ) from exc

    cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=args.max_cycle_s)
    if len(cycles) < args.min_cycles:
        print(
            f"[WARN] cycles={len(cycles)} < min_cycles={args.min_cycles}. Basis may be noisy."
        )

    T = X.shape[0]

    # Global normalization over full series (NOT per window)
    if args.mode == "corr":
        a_reason_all = normalize_gauss(X[:, ia])
        b_result_all = normalize_gauss(X[:, ib])
    else:
        a_reason_all = normalize_edf(X[:, ia])
        b_result_all = pnorm_student(
            X[:, ib],
            fs,
            cfg.ar_order,
            cfg.ema_half_life_s,
            cfg.student_nu,
        )

    # Fixed window length => fixed maxlag => fixed lag grid
    seg_len = int(round(args.epoch_len_s * fs))
    min_pairs = 10
    maxlag_samples_data = max(1, seg_len - min_pairs)
    maxlag_ms_data = int(1000.0 * maxlag_samples_data / fs)
    maxlag_ms = int(min(cfg.maxlag_ms, maxlag_ms_data))

    lag_step_ms = int(cfg.lag_step_ms)

    m = int(args.m)
    mixed_only = True
    K = m * m  # 16 when m=4

    # Online accumulation of samples x in R^K (each lag-column of coeffs is a sample)
    n_samples = 0
    sum_x = np.zeros(K, dtype=float)
    sum_xxT = np.zeros((K, K), dtype=float)
    lag_ref = None

    n_windows = 0

    for cyc in cycles:
        for ev0, ev1 in PHASES:
            bounds = phase_window_bounds(
                T, cyc[ev0], cyc[ev1], fs=fs, epoch_len_s=args.epoch_len_s
            )
            if bounds is None:
                continue
            a, b = bounds

            y = a_reason_all[a:b]
            z = b_result_all[a:b]
            if y.size != seg_len or z.size != seg_len:
                continue

            coeffs, lag_samples = hcr_coeffs_over_lags(
                y,
                z,
                fs,
                maxlag_ms,
                lag_step_ms,
                m,
                subtract_marginals=True,
                mixed_only=mixed_only,
            )

            if coeffs.shape[0] != K:
                raise RuntimeError(f"Expected K={K} features, got {coeffs.shape[0]}")

            if lag_ref is None:
                lag_ref = lag_samples
            else:
                if not np.array_equal(lag_ref, lag_samples):
                    raise RuntimeError(
                        "Lag grid mismatch. Check epoch_len_s / fs / lag_step_ms consistency."
                    )

            # Accumulate across ALL lag-columns: each column is one sample in R^K
            sum_x += coeffs.sum(axis=1)
            sum_xxT += coeffs @ coeffs.T
            n_samples += int(coeffs.shape[1])
            n_windows += 1

    if n_samples < 2:
        raise RuntimeError(
            "Not enough samples to build PCA basis. Increase data / windows / epoch length."
        )

    mean, V, eigvals = pca_from_second_moments(sum_x, sum_xxT, n_samples)

    np.savez_compressed(
        args.out,
        mean=mean,
        V=V,
        eigvals=eigvals,
        lag_samples=np.asarray(lag_ref, dtype=int),
        fs=int(fs),
        maxlag_ms=int(maxlag_ms),
        lag_step_ms=int(lag_step_ms),
        m=int(m),
        mixed_only=np.int8(1),
        chan_a=str(args.chan_a),
        chan_b=str(args.chan_b),
        mode=str(args.mode),
        epoch_len_s=float(args.epoch_len_s),
        n_cycles=int(len(cycles)),
        n_windows=int(n_windows),
        n_samples=int(n_samples),
    )

    print(
        f"Saved basis: {args.out}\n"
        f"  cycles={len(cycles)} windows={n_windows} samples={n_samples}\n"
        f"  K={K} m={m} mixed_only={mixed_only}\n"
        f"  fs={fs} epoch_len_s={args.epoch_len_s} maxlag_ms={maxlag_ms} step_ms={lag_step_ms}"
    )


if __name__ == "__main__":
    main()
