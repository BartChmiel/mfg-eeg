# scripts/mfg_pair.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.mfg.normalize import normalize_edf, normalize_gauss, pnorm_student
from src.mfg.hcr import hcr_coeffs_over_lags
from src.mfg.pca_features import pca_over_lags, select_r
from src.mfg.density import coeffs_vec_to_matrix, reconstruct_density_model
from src.mfg.diagnostics import pearson_over_lags
from src.mfg.config import load_config

# ---------- config (numerics) ----------
_cfg = load_config()
FS = _cfg.fs
MAXLAG_MS = _cfg.maxlag_ms
LAG_STEP_MS = _cfg.lag_step_ms
M = _cfg.m
AR_ORDER = _cfg.ar_order
EMA_HALF_LIFE_S = _cfg.ema_half_life_s
STUDENT_NU = _cfg.student_nu

# which panels (negative = A earlier, positive = A later)
PANEL_LAGS_MS = [-200, 0, 200]

# how many features to draw (paper shows 3)
DISPLAY_FEATURES = 3

# optional smoothing (for display only; not used for computation)
SMOOTH_MS = 15  # window in ms for Savitzky-Golay-like simple moving avg
# --------------------------------------


def pair_by_lag(y, z, lag_ms, fs):
    """
    Align (y,z) for a given lag in milliseconds.
    For lag_ms > 0, z is shifted forward by +lag (A earlier -> B later).
    Ensures equal length of y,z.
    """
    L = int(round(lag_ms * fs / 1000))
    n = len(y)
    if L > 0:
        return y[: n - L], z[L:]
    elif L < 0:
        L = -L
        return y[L:], z[: n - L]
    else:
        return y, z


def scatter_for_lag(y, z, lag_ms, fs, ax, title):
    ya, zb = pair_by_lag(y, z, lag_ms, fs)
    idx = np.linspace(0, len(ya) - 1, num=min(120000, len(ya)), dtype=int)
    ax.plot(ya[idx], zb[idx], ",", color="k", markersize=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13)


def hexbin_for_lag(y, z, lag_ms, fs, ax, title):
    """High-density scatter replacement to avoid overplotting."""
    ya, zb = pair_by_lag(y, z, lag_ms, fs)
    gridsize = int(np.clip(len(ya) // 900, 60, 160))
    hb = ax.hexbin(
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


def moving_avg(x, win):
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xpad = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    ker = np.ones(win, dtype=float) / win
    y = np.convolve(xpad, ker, mode="valid")
    return y


def main():
    # style: clean and compact
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--chan-a", required=True, help="reason (A)")
    ap.add_argument("--chan-b", required=True, help="result (B)")
    ap.add_argument("--mode", choices=["corr", "gc"], default="corr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # load
    df = pd.read_csv(args.csv)
    a = df[args.chan_a].to_numpy()
    b = df[args.chan_b].to_numpy()

    if args.mode == "corr":
        yA = normalize_edf(a)
        zB = normalize_edf(b)
    else:
        yA = normalize_gauss(a)
        zB = pnorm_student(b, FS, AR_ORDER, EMA_HALF_LIFE_S, STUDENT_NU)

    # HCR A->B (lags >= 0)
    coeffs_AB, lag_samples = hcr_coeffs_over_lags(
        yA, zB, FS, MAXLAG_MS, LAG_STEP_MS, M, subtract_marginals=True
    )
    V_ab, A_ab, eig_ab = pca_over_lags(coeffs_AB)
    r_ab = select_r(eig_ab, 0.90, DISPLAY_FEATURES)

    # HCR B->A (for negative lags)
    coeffs_BA, _ = hcr_coeffs_over_lags(
        zB, yA, FS, MAXLAG_MS, LAG_STEP_MS, M, subtract_marginals=True
    )
    V_ba, A_ba, eig_ba = pca_over_lags(coeffs_BA)
    r_ba = select_r(eig_ba, 0.90, DISPLAY_FEATURES)

    r = min(DISPLAY_FEATURES, r_ab, r_ba)

    # build symmetric lag axis
    ms_pos = lag_samples * 1000.0 / FS  # >=0
    ms_sym = np.concatenate((-ms_pos[::-1], ms_pos))  # negative + positive

    # raw scores for GC
    scores_ab_raw = A_ab[:r]  # (r, Tpos)
    scores_ba_raw = A_ba[:r]  # (r, Tpos)

    # align PCA signs across directions so that mirrored features match
    for v in range(r):
        if np.corrcoef(scores_ab_raw[v], scores_ba_raw[v, ::-1])[0, 1] < 0:
            scores_ba_raw[v] *= -1

    scores_sym_raw = np.concatenate((scores_ba_raw[:, ::-1], scores_ab_raw), axis=1)

    # GC uses raw amplitudes
    gc_sym = np.sqrt((scores_sym_raw**2).sum(axis=0))

    # display: normalize features to [-1, 1] per component; optional smoothing
    step_ms = LAG_STEP_MS
    win = max(1, int(round(SMOOTH_MS / step_ms)))
    scores_disp = scores_sym_raw.copy()
    for v in range(r):
        m = float(np.max(np.abs(scores_disp[v])) or 1.0)
        scores_disp[v] = moving_avg(scores_disp[v] / m, win)

    # Pearson correlation (symmetric)
    pearson_pos, _ = pearson_over_lags(yA, zB, FS, MAXLAG_MS, LAG_STEP_MS)
    pearson_sym = np.concatenate((pearson_pos[::-1], pearson_pos))
    pearson_disp = moving_avg(pearson_sym, win)

    # figure with constrained layout to avoid overlaps
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.05, 1.2, 1.1, 1.5])

    # ----- Row 1: hexbin scatter -----
    titles = [
        f"{args.chan_a} 200ms earlier",
        "same time",
        f"{args.chan_a} 200ms later",
    ]
    for j, d in enumerate(PANEL_LAGS_MS):
        ax = fig.add_subplot(gs[0, j])
        hexbin_for_lag(yA, zB, d, FS, ax, titles[j])

    # ----- Row 2: modeled full density with shared color scale + contours -----
    dens = []
    for d in PANEL_LAGS_MS:
        if d < 0:
            idx = int(np.searchsorted(ms_pos, abs(d)))
            Mmat = coeffs_vec_to_matrix(coeffs_BA[:, idx], M)
        else:
            idx = int(np.searchsorted(ms_pos, d))
            Mmat = coeffs_vec_to_matrix(coeffs_AB[:, idx], M)
        u, v, rho = reconstruct_density_model(
            Mmat, M, grid_n=192, renormalize=True, clip_q=0.99
        )
        dens.append((u, v, rho))

    vmin = min(arr.min() for _, _, arr in dens)
    vmax = max(arr.max() for _, _, arr in dens)
    levels = MaxNLocator(nbins=9).tick_values(vmin, vmax)

    for j, (u, v, rho) in enumerate(dens):
        ax = fig.add_subplot(gs[1, j])
        cf = ax.contourf(u, v, rho, levels=levels, cmap="Spectral_r")
        ax.contour(u, v, rho, levels=levels, colors="k", linewidths=0.35, alpha=0.55)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"modeled density at lag={PANEL_LAGS_MS[j]}ms", fontsize=11)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.01)
        cbar.ax.tick_params(labelsize=8)

    # ----- Row 3: a[j,k] heatmaps with shared scale -----
    mats = []
    for d in PANEL_LAGS_MS:
        if d < 0:
            idx = int(np.searchsorted(ms_pos, abs(d)))
            mats.append(coeffs_vec_to_matrix(coeffs_BA[:, idx], M))
        else:
            idx = int(np.searchsorted(ms_pos, d))
            mats.append(coeffs_vec_to_matrix(coeffs_AB[:, idx], M))

    # robust shared limits (98th percentile of abs)
    lim = float(np.percentile(np.abs(np.stack(mats)), 98))
    vmin_h, vmax_h = -lim, lim

    for j, Mmat in enumerate(mats):
        ax = fig.add_subplot(gs[2, j])
        im = ax.imshow(
            Mmat,
            origin="lower",
            cmap="RdYlBu_r",
            vmin=vmin_h,
            vmax=vmax_h,
            interpolation="nearest",
            aspect="auto",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(f"coeffs a[j,k] at lag={PANEL_LAGS_MS[j]}ms", fontsize=11)
        ax.set_xlabel("k")
        ax.set_ylabel("j")

    # ----- Row 4: features (left y) and GC (right y) on symmetric lag axis -----
    ax = fig.add_subplot(gs[3, :])
    ax.grid(True, alpha=0.15)

    # left y: features + Pearson in [-1, 1]
    ax.plot(ms_sym, pearson_disp, label="correlation", linewidth=1.2, zorder=2)
    colors = ["tab:orange", "tab:green", "tab:red"]
    for v in range(r):
        ax.plot(
            ms_sym,
            scores_disp[v],
            label=f"feature {v+1}",
            linewidth=1.0,
            zorder=2,
            color=colors[v % len(colors)],
        )
    ax.set_xlim(-MAXLAG_MS, MAXLAG_MS)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("lag [ms]")
    ax.set_ylabel("value (features, corr)")
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(200, color="k", lw=0.5, ls="--")
    ax.axvline(-200, color="k", lw=0.5, ls="--")

    # right y: GC (raw scale)
    ax2 = ax.twinx()
    ax2.plot(ms_sym, gc_sym, color="k", linewidth=2.0, label="GC(lag)", zorder=3)
    ax2.set_ylabel("GC(lag)")
    gmax = float(np.max(gc_sym) or 1.0)
    ax2.set_ylim(0.0, 1.05 * gmax)

    # one combined legend outside
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax2.get_legend_handles_labels()
    ax.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
    )

    fig.suptitle(f"{args.chan_a} to {args.chan_b}: HCR+PCA panel", y=0.995)
    out_png = os.path.join(args.out, f"pair_{args.chan_a}_{args.chan_b}.png")
    fig.savefig(out_png, dpi=220)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
