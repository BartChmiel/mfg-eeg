# mfg_batch.py
# Batch pipeline for HCR+PCA over many EEG channel pairs and (optionally) many CSV files.
# - Supports two modes: "corr" (Section II in the paper) and "gc" (density-based multi-feature Granger, Section III)
# - Produces per-lag Granger-causality heatmaps (like Fig. 6), numeric artifacts, and optional per-pair panels.
# - Keeps outputs neatly separated in subfolders per CSV, per mode, per artifact type.
#
# Usage examples:
#   python mfg_batch.py \
#       --csv data/grasp-and-lift-eeg-detection/train/subj7_series8_data.csv \
#       --mode both --outdir out_batch --lags 0,100,200,300,400,500
#
#   python mfg_batch.py \
#       --csv-dir data/grasp-and-lift-eeg-detection/train \
#       --mode gc --outdir out_batch --lags 0,100,200,300,400,500
#
# Optional:
#   --channels "Fp1,Fp2,F3,F4,FC5,FC6"   (to limit the set)
#   --include-auto                         (include autocorrelations on diagonal)
#   --save-panels                          (save per-pair full panels like scripts/mfg_pair)
#   --pca-scope {perpair,fit,load}         (default: perpair)
#   --ref-pair "Fp1,Fp2"                   (when --pca-scope=fit, PCA basis fit on this pair)
#   --pca-basis-path path/to/basis.npz     (when --pca-scope=load)
#
# Outputs (for each CSV base name and mode):
#   <outdir>/<csv_base>/<mode>/
#       data/
#           gc_at_lags.npy            # shape (n_lags, n_ch, n_ch), NaN on diagonal if not include-auto
#           channels.txt               # channel order used
#           meta.json                 # parameters
#       gc_mats/
#           lag_000ms.png, lag_100ms.png, ...
#       panels/                       # optional per-pair rich panels (heavy)

from __future__ import annotations
import os
import re
import json
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mfg.config import load_config
from src.mfg.io import load_eeg_csv
from src.mfg.normalize import normalize_gauss, pnorm_student
from src.mfg.hcr import hcr_coeffs_over_lags
from src.mfg.pca_features import pca_over_lags, select_r
from src.mfg.granger import gc_scalar

# ---------------------------- helpers ----------------------------


def _numeric_channel_names(df: pd.DataFrame) -> List[str]:
    """Pick numeric columns in a CSV as channel candidates."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ms_to_index(ms_array: np.ndarray, target_ms: float) -> int:
    """Get nearest index in a monotonic ms array."""
    idx = int(np.argmin(np.abs(ms_array - float(target_ms))))
    return idx


def _robust_limits(A: np.ndarray, q: float = 98.0) -> Tuple[float, float]:
    v = np.percentile(np.abs(A[np.isfinite(A)]), q)
    if v == 0:
        return (-1.0, 1.0)
    return (-v, v)


def _plot_gc_matrix(
    gc_mat: np.ndarray, ch_names: List[str], title: str, outpath: str
) -> None:
    if gc_mat.size == 0 or np.all(np.isnan(gc_mat)):
        print(f"[warn] Empty or NaN-only GC matrix skipped: {outpath}")
        return
    plt.figure(figsize=(8, 7))
    vmin, vmax = (np.nanmin(gc_mat), np.nanmax(gc_mat))
    im = plt.imshow(
        gc_mat, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    plt.colorbar(im, label="GC(lag)")
    plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=8)
    plt.yticks(range(len(ch_names)), ch_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------- core computations ----------------------------


def _normalize_pair(
    a: np.ndarray, b: np.ndarray, mode: str, cfg
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (yA, zB) normalized as in the paper for the given mode.
    mode: "corr" → both Gaussian→U(0,1); "gc" → A Gaussian, B p-normalized.
    """
    if mode == "corr":
        return normalize_gauss(a), normalize_gauss(b)
    elif mode == "gc":
        yA = normalize_gauss(a)
        zB = pnorm_student(b, cfg.fs, cfg.ar_order, cfg.ema_half_life_s, cfg.student_nu)
        return yA, zB
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _per_pair_hcr_pca(
    yA: np.ndarray, zB: np.ndarray, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute HCR over lags (nonnegative), then PCA.
    Returns: (coeffs, lags_samp, V, A)
    coeffs shape: (K, Tpos), A shape: (r, Tpos)
    """
    coeffs, lag_samples = hcr_coeffs_over_lags(
        yA, zB, cfg.fs, cfg.maxlag_ms, cfg.lag_step_ms, cfg.m, cfg.subtract_marginals
    )
    V, A, eig = pca_over_lags(coeffs)
    r = select_r(eig, cfg.pca_var_thresh, cfg.pca_max_r)
    return coeffs, lag_samples, V[:, :r], A[:r]


def _transform_with_V(coeffs: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Project coeffs (K,T) on a fixed basis V (K,r) and return scores (r,T)."""
    X = coeffs - coeffs.mean(axis=1, keepdims=True)
    return V.T @ X


# ---------------------------- main batch ----------------------------


def process_csv(
    csv_path: str,
    out_root: str,
    mode: str,
    cfg,
    lags_ms: List[int],
    channels: Optional[List[str]] = None,
    include_auto: bool = False,
    save_panels: bool = False,
    pca_scope: str = "perpair",
    ref_pair: Optional[Tuple[str, str]] = None,
    pca_basis_path: Optional[str] = None,
) -> None:
    """Process a single CSV file for all channel pairs.

    pca_scope:
      - "perpair": PCA fitted separately for each pair (closest to the paper)
      - "fit": fit PCA on ref_pair, reuse for all other pairs
      - "load": load V from pca_basis_path and reuse for all pairs
    """
    base = os.path.splitext(os.path.basename(csv_path))[0]
    outdir = os.path.join(out_root, base, mode)
    out_data = os.path.join(outdir, "data")
    out_mats = os.path.join(outdir, "gc_mats")
    out_pan = os.path.join(outdir, "panels")
    for d in (outdir, out_data, out_mats):
        _ensure_dir(d)
    if save_panels:
        _ensure_dir(out_pan)

    # Load data & channel names
    df = pd.read_csv(csv_path)
    all_ch = _numeric_channel_names(df)
    if channels is not None:
        # Keep only those that exist in the file, preserve requested order
        ch = [c for c in channels if c in all_ch]
    else:
        ch = all_ch

    X = df[ch].to_numpy(dtype=float)  # (T, C)
    nC = len(ch)

    # Prepare PCA reference basis if needed
    V_ref: Optional[np.ndarray] = None
    if pca_scope == "fit":
        if not ref_pair:
            raise ValueError("--pca-scope=fit requires --ref-pair like 'Fp1,Fp2'")
        a_name, b_name = ref_pair
        if a_name not in ch or b_name not in ch:
            raise ValueError(f"ref-pair {ref_pair} not found in channels of {csv_path}")
        ia, ib = ch.index(a_name), ch.index(b_name)
        yA, zB = _normalize_pair(X[:, ia], X[:, ib], mode, cfg)
        coeffs_ref, lag_samples, V_ref_full, A_ref = _per_pair_hcr_pca(yA, zB, cfg)
        V_ref = V_ref_full  # (K, r)
        # Save the basis
        np.savez_compressed(
            os.path.join(out_data, "pca_basis_refpair.npz"),
            V=V_ref,
            ref_pair=np.array([a_name, b_name]),
        )
    elif pca_scope == "load":
        if not pca_basis_path:
            raise ValueError("--pca-scope=load requires --pca-basis-path")
        dat = np.load(pca_basis_path)
        V_ref = dat["V"]
        lag_samples = None  # will recompute later
    else:
        lag_samples = None

    # Convert requested lags to indices once we have ms array
    ms_pos: Optional[np.ndarray] = None

    # Allocate gc matrices for requested lags
    lag_list = [int(x) for x in lags_ms]
    L = len(lag_list)
    gc_at_lags = np.full((L, nC, nC), np.nan, dtype=float)

    # Iterate pairs
    for i in range(nC):
        for j in range(nC):
            if i == j and not include_auto:
                continue
            a = X[:, i]
            b = X[:, j]
            yA, zB = _normalize_pair(a, b, mode, cfg)

            coeffs, lag_samples, V_pair, A_scores = _per_pair_hcr_pca(yA, zB, cfg)
            # Establish ms axis
            if ms_pos is None:
                ms_pos = lag_samples * 1000.0 / cfg.fs  # nonnegative

            # If using a shared basis, override scores
            if pca_scope in ("fit", "load") and V_ref is not None:
                A_scores = _transform_with_V(coeffs, V_ref)

            # GC over nonnegative lags
            gc_pos = gc_scalar(A_scores)  # shape (Tpos,)

            # Collect GC at requested lags (nonnegative)
            for li, Lms in enumerate(lag_list):
                idx = _ms_to_index(ms_pos, Lms)
                gc_at_lags[li, i, j] = float(gc_pos[idx])

            # Optional: per-pair panel figure (heavy)
            if save_panels:
                # Only draw a compact features+GC figure (no scatter for batch)
                fig = plt.figure(figsize=(8, 3))
                ax = fig.add_subplot(111)
                for v in range(min(A_scores.shape[0], 4)):
                    ax.plot(
                        ms_pos,
                        A_scores[v] / (np.max(np.abs(A_scores[v])) + 1e-12),
                        linewidth=0.9,
                        label=f"feature {v+1}",
                    )
                ax2 = ax.twinx()
                ax.plot(ms_pos, np.zeros_like(ms_pos), color="k", lw=0.3)
                ax2.plot(ms_pos, gc_pos, color="k", lw=1.4, label="GC(lag)")
                ax.set_xlabel("lag [ms]")
                ax.set_ylabel("features (norm)")
                ax2.set_ylabel("GC(lag)")
                ax.set_title(f"{ch[i]} → {ch[j]} ({mode})")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(out_pan, f"pair_{ch[i]}__{ch[j]}.png"), dpi=160
                )
                plt.close(fig)

        print(f"Processed row {i+1}/{nC} channels for {base} [{mode}]")

    # Save numeric outputs and meta
    np.save(os.path.join(out_data, "gc_at_lags.npy"), gc_at_lags)
    with open(os.path.join(out_data, "channels.txt"), "w", encoding="utf-8") as f:
        for name in ch:
            f.write(name + "\n")

    meta = {
        "csv": csv_path,
        "mode": mode,
        "fs": cfg.fs,
        "maxlag_ms": cfg.maxlag_ms,
        "lag_step_ms": cfg.lag_step_ms,
        "m": cfg.m,
        "subtract_marginals": cfg.subtract_marginals,
        "pca_var_thresh": cfg.pca_var_thresh,
        "pca_max_r": cfg.pca_max_r,
        "lags_ms": lag_list,
        "include_auto": include_auto,
        "pca_scope": pca_scope,
    }
    _save_json(os.path.join(out_data, "meta.json"), meta)

    # Plot heatmaps for each requested lag
    for li, Lms in enumerate(lag_list):
        title = f"GC at lag={Lms} ms ({mode}) — {base}"
        out_png = os.path.join(out_mats, f"lag_{Lms:03d}ms.png")
        _plot_gc_matrix(gc_at_lags[li], ch, title, out_png)

    print(f"Saved GC matrices for {base} [{mode}] → {out_mats}")


# ---------------------------- CLI ----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Batch HCR+PCA EEG processing across many pairs/CSVs"
    )
    g_src = ap.add_mutually_exclusive_group(required=True)
    g_src.add_argument("--csv", help="Single CSV path")
    g_src.add_argument("--csv-dir", help="Directory with CSV files (recursive)")
    ap.add_argument("--mode", choices=["corr", "gc", "both"], default="gc")
    ap.add_argument("--outdir", required=True, help="Root directory for outputs")
    ap.add_argument(
        "--lags",
        default="0,100,200,300,400,500",
        help="Comma-separated nonnegative lags in ms for GC matrices",
    )
    ap.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel names to include (default: all numeric columns)",
    )
    ap.add_argument(
        "--include-auto",
        action="store_true",
        help="Include autocorrelations on diagonal",
    )
    ap.add_argument(
        "--save-panels",
        action="store_true",
        help="Save compact per-pair panels with features+GC (heavy)",
    )
    ap.add_argument(
        "--pca-scope",
        choices=["perpair", "fit", "load"],
        default="perpair",
        help="Use PCA per pair (default), or fit/load a shared basis",
    )
    ap.add_argument(
        "--ref-pair", default=None, help="Ref pair 'A,B' for --pca-scope=fit"
    )
    ap.add_argument(
        "--pca-basis-path", default=None, help=".npz with V for --pca-scope=load"
    )

    args = ap.parse_args()

    cfg = load_config()
    lags_ms = [int(x) for x in args.lags.split(",") if x.strip()]
    channels = [s.strip() for s in args.channels.split(",")] if args.channels else None

    # Gather CSV list
    csv_list: List[str] = []
    if args.csv:
        csv_list = [args.csv]
    else:
        for root, _, files in os.walk(args.csv_dir):
            for fn in files:
                if fn.lower().endswith(".csv"):
                    csv_list.append(os.path.join(root, fn))
        csv_list.sort()

    # PCA scope prep
    ref_pair_tuple: Optional[Tuple[str, str]] = None
    if args.ref_pair:
        parts = [p.strip() for p in args.ref_pair.split(",")]
        if len(parts) != 2:
            raise ValueError("--ref-pair must look like 'Fp1,Fp2'")
        ref_pair_tuple = (parts[0], parts[1])

    for csvp in csv_list:
        modes = [args.mode] if args.mode != "both" else ["corr", "gc"]
        for m in modes:
            process_csv(
                csvp,
                out_root=args.outdir,
                mode=m,
                cfg=cfg,
                lags_ms=lags_ms,
                channels=channels,
                include_auto=args.include_auto,
                save_panels=args.save_panels,
                pca_scope=args.pca_scope,
                ref_pair=ref_pair_tuple,
                pca_basis_path=args.pca_basis_path,
            )


if __name__ == "__main__":
    main()
