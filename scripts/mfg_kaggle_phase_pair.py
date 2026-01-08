from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mfg.config import load_config
from src.mfg.io import (
    load_kaggle_data_with_events,
    extract_grasp_cycles,
    phase_window_bounds,
    KAGGLE_EVENT_COLS,
)
from src.mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src.mfg.hcr import hcr_coeffs_over_lags
from src.mfg.diagnostics import pearson_over_lags
from src.mfg.pca_features import select_r


PHASES = list(zip(KAGGLE_EVENT_COLS[:-1], KAGGLE_EVENT_COLS[1:]))


def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.asarray(x, float)
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x = np.asarray(x, float)
    xpad = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xpad, np.ones(win) / win, mode="valid")


def load_basis(path: str) -> dict:
    D = np.load(path, allow_pickle=False)
    return {
        "mean": D["mean"],
        "V": D["V"],
        "eigvals": D["eigvals"],
        "lag_samples": D["lag_samples"],
        "fs": int(D["fs"]),
        "maxlag_ms": int(D["maxlag_ms"]),
        "lag_step_ms": int(D["lag_step_ms"]),
        "m": int(D["m"]),
        "mixed_only": bool(int(D["mixed_only"])),
        "chan_a": str(D["chan_a"]),
        "chan_b": str(D["chan_b"]),
        "mode": str(D["mode"]),
        "epoch_len_s": float(D["epoch_len_s"]),
    }


def main() -> None:
    cfg = load_config()
    fs = int(cfg.fs)
    maxlag_ms_cfg = int(cfg.maxlag_ms)
    lag_step_ms = int(cfg.lag_step_ms)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--chan-a", required=True)
    ap.add_argument("--chan-b", required=True)
    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")

    ap.add_argument("--epoch-len-s", type=float, default=2.0)
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)

    ap.add_argument("--m", type=int, default=4)  # m=4 => 16 features (mixed_only=True)

    ap.add_argument(
        "--basis",
        required=True,
        help="Path to PCA basis .npz built by scripts/build_kaggle_basis.py",
    )

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ids, X, E, ch_names, _ = load_kaggle_data_with_events(args.data, args.events)
    T = X.shape[0]

    try:
        ia = ch_names.index(args.chan_a)
        ib = ch_names.index(args.chan_b)
    except ValueError as exc:
        raise ValueError(
            f"Channel not found. Example available: {ch_names[:10]} ..."
        ) from exc

    cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=args.max_cycle_s)
    if len(cycles) < args.min_cycles:
        print(
            f"[WARN] cycles={len(cycles)} < min_cycles={args.min_cycles}. Results may be noisy."
        )

    basis = load_basis(args.basis)

    # Validate basis compatibility
    if (basis["chan_a"], basis["chan_b"]) != (args.chan_a, args.chan_b):
        raise ValueError("Basis was built for a different channel pair.")
    if basis["mode"] != args.mode:
        raise ValueError("Basis was built for a different mode (corr/gc).")
    if abs(basis["epoch_len_s"] - float(args.epoch_len_s)) > 1e-9:
        raise ValueError("Basis was built for a different epoch_len_s.")
    if basis["fs"] != fs:
        raise ValueError("Basis was built for a different fs.")
    if basis["lag_step_ms"] != lag_step_ms:
        raise ValueError("Basis was built for a different lag_step_ms.")
    if basis["m"] != int(args.m) or not basis["mixed_only"]:
        raise ValueError("Basis was built with different m or not mixed_only=True.")

    # Global normalization over the whole series (must match basis building)
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

    seg_len = int(round(args.epoch_len_s * fs))
    min_pairs = 10
    maxlag_samples_data = max(1, seg_len - min_pairs)
    maxlag_ms_data = int(1000.0 * maxlag_samples_data / fs)
    maxlag_ms = int(min(maxlag_ms_cfg, maxlag_ms_data))

    m = int(args.m)
    mixed_only = True
    K = m * m

    # Smoothing in lag-step units
    smooth_ms = 10
    win = max(1, int(round(smooth_ms / lag_step_ms)))

    fig, ax = plt.subplots(len(PHASES), 1, figsize=(12, 2.5 * len(PHASES)), sharex=True)
    if len(PHASES) == 1:
        ax = [ax]

    for pi, (ev0, ev1) in enumerate(PHASES):
        coeffs_list = []
        pear_list = []
        lag_ref = None
        n_windows = 0

        for cyc in cycles:
            bounds = phase_window_bounds(
                T, cyc[ev0], cyc[ev1], fs=fs, epoch_len_s=args.epoch_len_s
            )
            if bounds is None:
                continue
            a0, b0 = bounds

            y = a_reason_all[a0:b0]
            z = b_result_all[a0:b0]
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
            pear, lag_p = pearson_over_lags(y, z, fs, maxlag_ms, lag_step_ms)

            if coeffs.shape[0] != K:
                raise RuntimeError(f"Expected K={K}, got {coeffs.shape[0]}")

            if lag_ref is None:
                lag_ref = lag_samples
            else:
                if not np.array_equal(lag_ref, lag_samples) or not np.array_equal(
                    lag_ref, lag_p
                ):
                    raise RuntimeError(
                        "Lag grid mismatch across windows (unexpected with fixed windows)."
                    )

            coeffs_list.append(coeffs)
            pear_list.append(pear)
            n_windows += 1

        if n_windows == 0:
            ax[pi].set_title(f"{ev0} → {ev1} | windows=0 (no valid fixed windows)")
            ax[pi].grid(True, alpha=0.15)
            continue

        coeffs_mean = np.mean(np.stack(coeffs_list, axis=0), axis=0)  # (K, L)
        pear_mean = np.mean(np.stack(pear_list, axis=0), axis=0)  # (L,)

        if not np.array_equal(lag_ref, basis["lag_samples"]):
            raise ValueError(
                "Basis lag_samples differ from analysis lag grid. Rebuild basis with the same settings."
            )

        Xc = coeffs_mean - basis["mean"][:, None]
        scores_full = basis["V"].T @ Xc  # (K, L)

        r = select_r(basis["eigvals"], cfg.pca_var_thresh, cfg.pca_max_r)
        r = min(int(r), 3)

        ms = np.asarray(lag_ref, float) * 1000.0 / float(fs)

        ax[pi].plot(ms, moving_avg(pear_mean, win), label="corr", linewidth=1.1)

        for k in range(r):
            yk = scores_full[k]
            yk = yk / (np.max(np.abs(yk)) + 1e-12)
            ax[pi].plot(ms, moving_avg(yk, win), label=f"PC{k+1}", linewidth=1.0)

        ax[pi].axvline(0, color="k", lw=0.6)
        ax[pi].set_title(f"{ev0} → {ev1} | windows={n_windows} | mode={args.mode}")
        ax[pi].grid(True, alpha=0.15)
        ax[pi].set_ylabel("norm proj")

        if pi == 0:
            ax[pi].legend(ncol=6, frameon=False, loc="upper right")

    ax[-1].set_xlabel("lag [ms]")
    out_png = os.path.join(
        args.out, f"phases_{args.chan_a}_{args.chan_b}_{args.mode}_m{m}.png"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
