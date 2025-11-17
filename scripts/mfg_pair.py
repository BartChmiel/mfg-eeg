"""
Pairwise multi-feature HCR / PCA analysis for EEG channels.

- Supports both Kaggle CSV (continuous signals) and mfgin ROI .mat files.
- For ROI .mat:
    * uses fixed window inside trials (configured in io.load_mfgin_mat),
    * processes each trial separately,
    * averages HCR coefficients and Pearson correlation over trials.
- For CSV:
    * treats the whole series as a single "trial".

Outputs:
- Simple Granger-like plot (PC1-PC3 vs lag >= 0) for A -> B.
- DICC-style curves (direct / indirect / common cause) for lag >= 0.
- Full 4-row panel:
    1) trial-based hexbin plots at selected lags,
    2) reconstructed joint densities from HCR,
    3) coefficient matrices a[j, k] at selected lags,
    4) features + Pearson + GC(lag).
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src.mfg.hcr import hcr_coeffs_over_lags
from src.mfg.pca_features import pca_over_lags, select_r
from src.mfg.density import coeffs_vec_to_matrix, reconstruct_density_model
from src.mfg.diagnostics import pearson_over_lags
from src.mfg.config import load_config
from src.mfg.viz import plot_granger_simple, plot_granger_dicc
from src.mfg.io import load_pair_for_analysis


_cfg = load_config()
FS_DEFAULT, MAXLAG_MS, LAG_STEP_MS = _cfg.fs, _cfg.maxlag_ms, _cfg.lag_step_ms
M = _cfg.m
AR_ORDER, EMA_HALF_LIFE_S, STUDENT_NU = (
    _cfg.ar_order,
    _cfg.ema_half_life_s,
    _cfg.student_nu,
)

# Left panel: A later (+200 ms), center: 0 ms, right: A earlier (-200 ms)
PANEL_LAGS_MS = [200, 0, -200]
DISPLAY_FEATURES = 4
SMOOTH_MS = 10


def hexbin_for_lag_trials(
    y_trials: np.ndarray,
    z_trials: np.ndarray,
    lag_ms: float,
    fs: float,
    ax: plt.Axes,
    title: str,
) -> None:
    """
    Draw a hexbin scatter for many trials at a given lag.

    Parameters
    ----------
    y_trials, z_trials : np.ndarray
        Arrays of shape (n_trials, L), normalized to U(0, 1).
    lag_ms : float
        Time lag in milliseconds (can be negative).
    fs : float
        Sampling frequency in Hz.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    title : str
        Plot title.
    """
    y_trials = np.asarray(y_trials, float)
    z_trials = np.asarray(z_trials, float)
    assert y_trials.shape == z_trials.shape
    n_trials, n = y_trials.shape

    lag_samples = int(round(lag_ms * fs / 1000.0))

    ya_list = []
    zb_list = []

    if lag_samples == 0:
        for tr in range(n_trials):
            ya_list.append(y_trials[tr])
            zb_list.append(z_trials[tr])
    elif lag_samples > 0:
        # A earlier -> B later
        for tr in range(n_trials):
            if n <= lag_samples:
                continue
            ya_list.append(y_trials[tr, : n - lag_samples])
            zb_list.append(z_trials[tr, lag_samples:])
    else:
        # lag_samples < 0: A later, B earlier
        L = -lag_samples
        for tr in range(n_trials):
            if n <= L:
                continue
            ya_list.append(y_trials[tr, L:])
            zb_list.append(z_trials[tr, : n - L])

    if not ya_list:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title + " (no samples)", fontsize=10)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        return

    ya = np.concatenate(ya_list)
    zb = np.concatenate(zb_list)

    gridsize = int(np.clip(ya.size // 900, 60, 160))
    ax.hexbin(
        ya,
        zb,
        gridsize=gridsize,
        cmap="Greys",
        bins="log",
        mincnt=1,
        linewidths=0.0,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])


def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    """
    Symmetric moving average (odd window, edge-padded).

    Parameters
    ----------
    x : np.ndarray
        Input 1D array.
    win : int
        Window size in "number of lag steps".

    Returns
    -------
    np.ndarray
        Smoothed array of the same length as x.
    """
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x = np.asarray(x, float)
    xpad = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xpad, np.ones(win) / win, mode="valid")


def nearest_idx_ms(ms_array: np.ndarray, target_ms: float) -> int:
    """
    Return index of ms_array closest to target_ms.
    """
    return int(np.argmin(np.abs(ms_array - float(target_ms))))


def reshape_trials_from_meta(
    a_1d: np.ndarray,
    b_1d: np.ndarray,
    meta: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For ROI .mat data: reshape flattened (T,) arrays back to (n_trials, trial_len).

    Parameters
    ----------
    a_1d, b_1d : np.ndarray
        Flattened signals of shape (n_trials * trial_len,).
    meta : dict
        Metadata returned by load_pair_for_analysis, containing:
        - 'n_trials'
        - 'trial_len'

    Returns
    -------
    a_trials, b_trials : np.ndarray
        Arrays of shape (n_trials, trial_len).
    """
    n_trials = int(meta["n_trials"])
    trial_len = int(meta["trial_len"])

    if a_1d.size != n_trials * trial_len or b_1d.size != n_trials * trial_len:
        raise ValueError(
            f"Flattened length mismatch for ROI: "
            f"{a_1d.size} / {b_1d.size} vs n_trials*trial_len = "
            f"{n_trials}*{trial_len} = {n_trials * trial_len}"
        )

    a_trials = np.asarray(a_1d, float).reshape(n_trials, trial_len)
    b_trials = np.asarray(b_1d, float).reshape(n_trials, trial_len)
    return a_trials, b_trials


def main() -> None:
    # Basic matplotlib style for all plots from this script
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 6,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    parser = argparse.ArgumentParser(
        description="Multi-feature HCR/PCA analysis for a pair of EEG channels."
    )
    parser.add_argument("--file", required=True, help="CSV or MAT file")
    parser.add_argument("--chan-a", required=True, help="reason (A)")
    parser.add_argument("--chan-b", required=True, help="result (B)")
    parser.add_argument("--mode", choices=["corr", "gc"], default="corr")
    parser.add_argument("--out", required=True, help="Output directory for plots")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    path = args.file

    # 1) Unified loader: supports CSV and mfgin ROI .mat
    a_1d, b_1d, fs, meta = load_pair_for_analysis(
        path,
        args.chan_a,
        args.chan_b,
        fs_fallback=FS_DEFAULT,
    )

    if meta.get("type") == "roi":
        # Many trials: reshape flattened (T,) back to (n_trials, trial_len)
        a_trials, b_trials = reshape_trials_from_meta(a_1d, b_1d, meta)
    else:
        # CSV: treat as a single trial
        a_trials = np.asarray(a_1d, float)[None, :]
        b_trials = np.asarray(b_1d, float)[None, :]

    assert a_trials.shape == b_trials.shape, "A and B must have the same length"
    n_trials, seg_len = a_trials.shape

    # 2) Maximum lag constrained by single-trial length
    min_pairs = 10  # minimal number of sample pairs for the largest lag
    maxlag_samples_data = max(1, seg_len - min_pairs)
    maxlag_ms_data = int(1000.0 * maxlag_samples_data / fs)
    maxlag_ms = min(MAXLAG_MS, maxlag_ms_data)

    # 3) Trial-wise normalization
    y_trials = np.empty_like(a_trials, float)
    z_trials = np.empty_like(b_trials, float)

    for tr in range(n_trials):
        if args.mode == "corr":
            y_trials[tr] = normalize_gauss(a_trials[tr])
            z_trials[tr] = normalize_gauss(b_trials[tr])
        else:
            y_trials[tr] = normalize_edf(a_trials[tr])
            z_trials[tr] = pnorm_student(
                b_trials[tr],
                fs,
                AR_ORDER,
                EMA_HALF_LIFE_S,
                STUDENT_NU,
            )

    # 4) HCR for A -> B and B -> A, per trial, then averaging
    coeffs_ab_list = []
    coeffs_ba_list = []
    pearson_list = []
    lag_samples_ref = None

    for tr in range(n_trials):
        y_a = y_trials[tr]
        z_b = z_trials[tr]

        coeffs_ab_tr, lag_samples_tr = hcr_coeffs_over_lags(
            y_a,
            z_b,
            fs,
            maxlag_ms,
            LAG_STEP_MS,
            M,
            subtract_marginals=True,
        )
        coeffs_ba_tr, _ = hcr_coeffs_over_lags(
            z_b,
            y_a,
            fs,
            maxlag_ms,
            LAG_STEP_MS,
            M,
            subtract_marginals=True,
        )
        pearson_tr, lag_samples_p = pearson_over_lags(
            y_a,
            z_b,
            fs,
            maxlag_ms,
            LAG_STEP_MS,
        )

        if lag_samples_ref is None:
            lag_samples_ref = lag_samples_tr
        else:
            if not np.array_equal(lag_samples_ref, lag_samples_tr):
                raise RuntimeError("Inconsistent lag_samples across trials")
        if not np.array_equal(lag_samples_ref, lag_samples_p):
            raise RuntimeError("HCR and Pearson lag grids differ")

        coeffs_ab_list.append(coeffs_ab_tr)
        coeffs_ba_list.append(coeffs_ba_tr)
        pearson_list.append(pearson_tr)

    coeffs_ab = np.mean(np.stack(coeffs_ab_list, axis=0), axis=0)
    coeffs_ba = np.mean(np.stack(coeffs_ba_list, axis=0), axis=0)
    pearson_pos = np.mean(np.stack(pearson_list, axis=0), axis=0)
    lag_samples = lag_samples_ref

    # 5) PCA and GC on averaged coefficients
    v_ab, a_ab, eig_ab = pca_over_lags(coeffs_ab)
    r_ab = select_r(eig_ab, 0.90, DISPLAY_FEATURES)

    v_ba, a_ba, eig_ba = pca_over_lags(coeffs_ba)
    r_ba = select_r(eig_ba, 0.90, DISPLAY_FEATURES)

    r = min(DISPLAY_FEATURES, r_ab, r_ba)

    frac_ab = eig_ab / (eig_ab.sum() + 1e-12)
    frac_ba = eig_ba / (eig_ba.sum() + 1e-12)
    lambdas_frac = 0.5 * (frac_ab[:r] + frac_ba[:r])

    ms_pos = lag_samples * 1000.0 / fs  # lags >= 0 in ms
    ms_sym = np.concatenate((-ms_pos[::-1], ms_pos))

    scores_ab_raw = a_ab[:r]
    scores_ba_raw = a_ba[:r]

    # Align PCA signs between A -> B and B -> A
    for idx in range(r):
        if np.corrcoef(scores_ab_raw[idx], scores_ba_raw[idx, ::-1])[0, 1] < 0:
            scores_ba_raw[idx] *= -1
    scores_sym_raw = np.concatenate((scores_ba_raw[:, ::-1], scores_ab_raw), axis=1)

    # 6) Pearson to fix the sign of the first feature
    pearson_sym = np.concatenate((pearson_pos[::-1], pearson_pos))
    idx_zero = nearest_idx_ms(ms_sym, 0.0)
    if r > 0 and idx_zero < len(pearson_sym):
        if scores_sym_raw[0, idx_zero] * pearson_sym[idx_zero] < 0:
            scores_sym_raw[0] *= -1

    # GC(delta) = ||a(delta)||_2
    gc_sym = np.sqrt((scores_sym_raw**2).sum(axis=0))

    # 7) Simple PC1-PC3 plot for lag >= 0
    simple_out = os.path.join(args.out, f"pair_{args.chan_a}_{args.chan_b}_simple.png")
    scores_pos = a_ab[: min(3, r)]
    plot_granger_simple(
        lag_samples=lag_samples,
        scores=scores_pos,
        fs=fs,
        title=f"{args.chan_a} → {args.chan_b} (lag ≥ 0)",
        outpath=simple_out,
    )

    # Smoothing window in units of lag steps
    win = max(1, int(round(SMOOTH_MS / LAG_STEP_MS)))

    # 8) DICC-style curves: only non-negative lags
    mask_pos = ms_sym >= 0.0
    ms_pos_sec_dicc = ms_sym[mask_pos] / 1000.0
    scores_dicc = scores_sym_raw[: min(3, r), :][:, mask_pos]
    scores_dicc_disp = scores_dicc.copy()
    for idx in range(scores_dicc_disp.shape[0]):
        max_abs = float(np.max(np.abs(scores_dicc_disp[idx])) or 1.0)
        scores_dicc_disp[idx] = moving_avg(scores_dicc_disp[idx] / max_abs, win)

    dicc_out = os.path.join(args.out, f"pair_{args.chan_a}_{args.chan_b}_dicc.png")
    plot_granger_dicc(
        t=ms_pos_sec_dicc,
        scores=scores_dicc_disp,
        title=f"{args.chan_a} → {args.chan_b}",
        outpath=dicc_out,
    )

    # 9) Full 4-row panel: hexbin, densities, coeff matrices, features + GC
    scores_disp = scores_sym_raw.copy()
    for idx in range(r):
        max_abs = float(np.max(np.abs(scores_disp[idx])) or 1.0)
        scores_disp[idx] = moving_avg(scores_disp[idx] / max_abs, win)
    pearson_disp = moving_avg(pearson_sym, win)

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.05, 1.2, 1.1, 1.5])

    # Row 1: trial-based hexbin plots at ±200 ms and 0 ms
    titles = [
        f"{args.chan_a} 200 ms later",
        "same time",
        f"{args.chan_a} 200 ms earlier",
    ]
    for j, lag_ms in enumerate(PANEL_LAGS_MS):
        axis = fig.add_subplot(grid[0, j])
        hexbin_for_lag_trials(y_trials, z_trials, lag_ms, fs, axis, titles[j])

    # Row 2: reconstructed densities from averaged coefficients
    from scipy.ndimage import gaussian_filter  # local import

    densities = []
    for lag_ms in PANEL_LAGS_MS:
        if lag_ms < 0:
            idx = nearest_idx_ms(ms_pos, abs(lag_ms))
            mat_coeffs = coeffs_vec_to_matrix(coeffs_ba[:, idx], M)
        else:
            idx = nearest_idx_ms(ms_pos, lag_ms)
            mat_coeffs = coeffs_vec_to_matrix(coeffs_ab[:, idx], M)
        u, v, rho = reconstruct_density_model(
            mat_coeffs,
            M,
            grid_n=192,
            renormalize=True,
            clip_q=0.99,
        )
        rho = gaussian_filter(rho, sigma=0.6, mode="nearest")
        densities.append((u, v, rho))

    vmin = min(arr.min() for _, _, arr in densities)
    vmax = max(arr.max() for _, _, arr in densities)
    levels = MaxNLocator(nbins=9).tick_values(vmin, vmax)
    for j, (u, v, rho) in enumerate(densities):
        axis = fig.add_subplot(grid[1, j])
        cf = axis.contourf(u, v, rho, levels=levels, cmap="Spectral_r")
        axis.contour(
            u,
            v,
            rho,
            levels=levels,
            colors="k",
            linewidths=0.35,
            alpha=0.55,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(f"density at lag={PANEL_LAGS_MS[j]} ms", fontsize=11)
        cb = fig.colorbar(cf, ax=axis, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=8)

    # Row 3: coefficient matrices a[j, k] at selected lags
    mats = []
    for lag_ms in PANEL_LAGS_MS:
        if lag_ms < 0:
            idx = nearest_idx_ms(ms_pos, abs(lag_ms))
            mats.append(coeffs_vec_to_matrix(coeffs_ba[:, idx], M))
        else:
            idx = nearest_idx_ms(ms_pos, lag_ms)
            mats.append(coeffs_vec_to_matrix(coeffs_ab[:, idx], M))
    lim = float(np.percentile(np.abs(np.stack(mats)), 98))
    vmin_h, vmax_h = -lim, lim

    for j, mat_coeffs in enumerate(mats):
        axis = fig.add_subplot(grid[2, j])
        im = axis.imshow(
            mat_coeffs,
            origin="lower",
            cmap="RdYlBu_r",
            vmin=vmin_h,
            vmax=vmax_h,
            interpolation="nearest",
            aspect="auto",
        )
        cb = fig.colorbar(im, ax=axis, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=8)
        axis.set_title(f"coeffs a[j,k] at {PANEL_LAGS_MS[j]} ms", fontsize=11)
        axis.set_xlabel("k")
        axis.set_ylabel("j")

    # Row 4: features + Pearson + GC
    axis = fig.add_subplot(grid[3, :])
    axis.grid(True, alpha=0.15)
    axis.plot(ms_sym, pearson_disp, label="correlation", linewidth=1.2, zorder=2)
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx in range(r):
        label = f"feature {idx+1} (λ={lambdas_frac[idx]:.3f})"
        axis.plot(
            ms_sym,
            scores_disp[idx],
            label=label,
            linewidth=1.0,
            zorder=2,
            color=colors[idx % len(colors)],
        )
    axis.set_xlim(-maxlag_ms, maxlag_ms)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("lag [ms]")
    axis.set_ylabel("value (features, corr)")
    axis.axvline(0, color="k", lw=0.7)
    axis.axvline(200, color="k", lw=0.5, ls="--")
    axis.axvline(-200, color="k", lw=0.5, ls="--")

    axis2 = axis.twinx()
    axis2.plot(
        ms_sym,
        gc_sym,
        color="k",
        linewidth=2.0,
        label="GC(lag)",
        zorder=3,
    )
    axis2.set_ylabel("GC(lag)")
    axis2.set_ylim(0.0, 1.05 * float(np.max(gc_sym) or 1.0))

    lines_left, labels_left = axis.get_legend_handles_labels()
    lines_right, labels_right = axis2.get_legend_handles_labels()
    axis.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"{args.chan_a} to {args.chan_b}: HCR+PCA panel", y=0.995)

    out_png = os.path.join(args.out, f"pair_{args.chan_a}_{args.chan_b}.png")
    fig.savefig(out_png, dpi=220)
    print(
        f"Saved: {out_png} | r_ab={r_ab}, r_ba={r_ba}, r_used={r} | "
        f"n_trials={n_trials}, seg_len={seg_len}, maxlag_ms={maxlag_ms}"
    )


if __name__ == "__main__":
    main()
