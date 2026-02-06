from __future__ import annotations

import os
import argparse
from dataclasses import asdict

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
    KAGGLE_EVENT_COLS,
)


# Canonical phases derived from event order (single source of truth)
PHASES = list(zip(KAGGLE_EVENT_COLS[:-1], KAGGLE_EVENT_COLS[1:]))


def _basis_mixed(u_1d: np.ndarray, m: int) -> np.ndarray:
    """
    Legendre orthonormal basis on [0,1], keep only degrees 1..m (mixed_only).
    Returns shape (T, m).
    """
    F = legendre_orthonormal(u_1d, m)  # (T, m+1)
    return F[:, 1:]  # drop degree-0


def _normalize_window(
    W: np.ndarray, mode: str, fs: int, cfg
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (Y,Z) from raw window W (T,C).
    - corr: Y=Z=Gaussian-CDF normalization
    - gc:   Y=EDF (reason), Z=Student-t pnorm (result)
    """
    T, C = W.shape
    Y = np.empty((T, C), dtype=float)
    Z = np.empty((T, C), dtype=float)

    if mode == "corr":
        for c in range(C):
            u = normalize_gauss(W[:, c])
            Y[:, c] = u
            Z[:, c] = u
        return Y, Z

    # mode == "gc"
    for c in range(C):
        Y[:, c] = normalize_edf(W[:, c])
        Z[:, c] = pnorm_student(
            W[:, c],
            fs,
            cfg.ar_order,
            cfg.ema_half_life_s,
            cfg.student_nu,
        )
    return Y, Z


def _compute_coeffs_all_pairs_for_lag(
    Fy: np.ndarray,  # (n_eff, C, m)
    Fz: np.ndarray,  # (n_eff, C, m)
    subtract_marginals: bool,
) -> np.ndarray:
    """
    Cross-moments for all ordered pairs (i -> j):
      M[i,j,a,b] = E[ Fy_i,a * Fz_j,b ]   (optionally centered by marginals)
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


def _append_top_edges(
    path: str, mat: np.ndarray, ch_names: list[str], topk: int = 40
) -> None:
    """
    Append top directed edges (i->j) to an existing text file.
    Diagonal is ignored.
    """
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


def run_one_phase(
    *,
    data_csv: str,
    events_csv: str,
    ev0: str,
    ev1: str,
    out_dir: str,
    mode: str,
    epoch_len_s: float,
    max_cycle_s: float,
    min_cycles: int,
    m: int,
    lags_ms: list[int],
    summary: str,
    metric: str,
) -> None:
    cfg = load_config()
    fs = cfg.fs

    ids, X, E, ch_names, _ = load_kaggle_data_with_events(data_csv, events_csv)
    T = X.shape[0]
    cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=max_cycle_s)
    if len(cycles) < min_cycles:
        print(f"[WARN] cycles={len(cycles)} < min_cycles={min_cycles} for {ev0}->{ev1}")

    windows: list[np.ndarray] = []
    seg_len = int(round(epoch_len_s * fs))

    for cyc in cycles:
        t0 = cyc[ev0]
        t1 = cyc[ev1]
        bounds = phase_window_bounds(T, t0, t1, fs=fs, epoch_len_s=epoch_len_s)
        if bounds is None:
            continue
        a, b = bounds
        W = X[a:b]
        if W.shape[0] != seg_len:
            continue
        windows.append(W)

    if not windows:
        raise RuntimeError(f"No usable windows for phase {ev0}->{ev1}")

    C = windows[0].shape[1]
    K = m * m
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]

    # Pooled sums per lag (weighted by n_eff)
    coeffs_sum = np.zeros((len(lag_samples), C, C, K), dtype=float)
    n_eff_sum = np.zeros((len(lag_samples),), dtype=float)

    subtract_marginals = True

    for W in windows:
        Y, Z = _normalize_window(W, mode=mode, fs=fs, cfg=cfg)
        T = Y.shape[0]

        Fy_full = np.empty((T, C, m), dtype=float)
        Fz_full = np.empty((T, C, m), dtype=float)
        for c in range(C):
            Fy_full[:, c, :] = _basis_mixed(Y[:, c], m)
            Fz_full[:, c, :] = _basis_mixed(Z[:, c], m)

        for li, L in enumerate(lag_samples):
            n_eff = T - L
            if n_eff <= 5:
                continue

            Fy = Fy_full[:n_eff, :, :]
            Fz = Fz_full[L:, :, :]

            M = _compute_coeffs_all_pairs_for_lag(
                Fy, Fz, subtract_marginals=subtract_marginals
            )
            M_flat = M.reshape(C, C, K)

            coeffs_sum[li] += float(n_eff) * M_flat
            n_eff_sum[li] += float(n_eff)

    if np.any(n_eff_sum <= 0):
        bad = [lags_ms[i] for i in range(len(lags_ms)) if n_eff_sum[i] <= 0]
        raise RuntimeError(f"No samples accumulated for lags_ms={bad}")

    coeffs_mean = coeffs_sum / n_eff_sum[:, None, None, None]  # pooled estimate
    energy_per_lag = np.sqrt(np.sum(coeffs_mean**2, axis=-1))  # (L,C,C)

    # Significance proxy: under null, n_eff * sum(coeff^2) ~ chi2(df=K) (heuristic)
    chi2_stat = n_eff_sum[:, None, None] * np.sum(coeffs_mean**2, axis=-1)
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

    # Display matrix with cleared diagonal
    mat_disp = np.array(mat, float)
    np.fill_diagonal(mat_disp, 0.0)

    os.makedirs(out_dir, exist_ok=True)

    title = (
        f"{ev0}__{ev1} | mode={mode} | m={m} | metric={metric} | summary={summary} "
        f"| lags_ms={lags_ms} | windows={len(windows)}"
    )

    # Heatmap
    plt.figure(figsize=(11.5, 10))
    vmax = _robust_vmax(mat_disp)
    im = plt.imshow(mat_disp, origin="lower", vmin=0.0, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(C), ch_names, rotation=90, fontsize=8)
    plt.yticks(range(C), ch_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()

    out_png = os.path.join(
        out_dir, f"phase_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.png"
    )
    plt.savefig(out_png, dpi=220)
    plt.close()

    # Text report (header + appended top edges)
    out_txt = os.path.join(
        out_dir, f"phase_top_edges_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.txt"
    )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"phase={ev0}__{ev1}\n")
        f.write(
            f"mode={mode} m={m} metric={metric} summary={summary} lags_ms={lags_ms}\n"
        )
        f.write(f"epoch_len_s={epoch_len_s} fs={fs} windows={len(windows)}\n")
        f.write(f"config={asdict(cfg)}\n\n")

    _append_top_edges(out_txt, mat, ch_names, topk=40)

    print("Saved:", out_png)
    print("Saved:", out_txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)

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

    ap.add_argument(
        "--phase",
        nargs=2,
        default=None,
        help="Two event names: ev0 ev1. If omitted, run all canonical phases.",
    )
    args = ap.parse_args()

    if args.phase is not None:
        ev0, ev1 = args.phase
        run_one_phase(
            data_csv=args.data,
            events_csv=args.events,
            ev0=ev0,
            ev1=ev1,
            out_dir=args.out,
            mode=args.mode,
            epoch_len_s=args.epoch_len_s,
            max_cycle_s=args.max_cycle_s,
            min_cycles=args.min_cycles,
            m=args.m,
            lags_ms=args.lags_ms,
            summary=args.summary,
            metric=args.metric,
        )
        return

    for ev0, ev1 in PHASES:
        run_one_phase(
            data_csv=args.data,
            events_csv=args.events,
            ev0=ev0,
            ev1=ev1,
            out_dir=args.out,
            mode=args.mode,
            epoch_len_s=args.epoch_len_s,
            max_cycle_s=args.max_cycle_s,
            min_cycles=args.min_cycles,
            m=args.m,
            lags_ms=args.lags_ms,
            summary=args.summary,
            metric=args.metric,
        )


if __name__ == "__main__":
    main()
