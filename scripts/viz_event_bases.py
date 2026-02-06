#!/usr/bin/env python3
"""
Visualize per-phase (event) all-pairs PCA bases saved as .npz.

Expected keys in each .npz:
- ch_names: list/array of channel names, length C
- mean:     (C, C, K)
- U:        (C, C, r, K)  where K = m*m
- eigvals:  (C, C, r)
- pk:       phase key string (optional; otherwise inferred from filename)

Usage examples:
  python viz_event_bases.py --glob "out/basis/*_*.npz" --pair "Ch1->Ch2" --outdir viz/
  python viz_event_bases.py --glob "out/basis/*_*.npz" --pair "0->3" --outdir viz/ --r 3
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


def infer_phase_key(path: Path) -> str:
    # tries to extract "..._HandStart__FirstDigitTouch.npz" -> "HandStart__FirstDigitTouch"
    m = re.search(r"_([A-Za-z]+__[_A-Za-z]+)\.npz$", path.name)
    if m:
        return m.group(1)
    return path.stem


def parse_pair(pair: str, ch_names: list[str]) -> tuple[int, int]:
    pair = pair.strip()
    if "->" not in pair:
        raise ValueError("pair must be like 'A->B' or '0->3'")
    a, b = [x.strip() for x in pair.split("->", 1)]

    def to_idx(x: str) -> int:
        if x.isdigit():
            return int(x)
        if x in ch_names:
            return ch_names.index(x)
        raise ValueError(f"Unknown channel '{x}'. Available example: {ch_names[:5]}...")

    i = to_idx(a)
    j = to_idx(b)
    return i, j


def reshape_mm(vec: np.ndarray) -> np.ndarray:
    K = vec.shape[-1]
    m = int(round(K**0.5))
    if m * m != K:
        raise ValueError(f"K={K} is not a perfect square; cannot reshape to m×m.")
    return vec.reshape(m, m)


def save_pc_grid(
    out_png: Path, U_ij: np.ndarray, eig_ij: np.ndarray, title: str
) -> None:
    """
    U_ij: (r, K)
    eig_ij: (r,)
    """
    r = U_ij.shape[0]
    fig, axes = plt.subplots(1, r, figsize=(4.2 * r, 4.0))
    if r == 1:
        axes = [axes]

    for k in range(r):
        mat = reshape_mm(U_ij[k])
        vmax = np.quantile(np.abs(mat), 0.98) if np.isfinite(mat).any() else 1.0
        im = axes[k].imshow(mat, origin="lower", vmin=-vmax, vmax=vmax)
        axes[k].set_title(f"PC{k+1}\nλ={eig_ij[k]:.3g}", fontsize=10)
        axes[k].set_xticks([])
        axes[k].set_yticks([])
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_mean(out_png: Path, mean_ij: np.ndarray, title: str) -> None:
    mat = reshape_mm(mean_ij)
    vmax = np.quantile(np.abs(mat), 0.98) if np.isfinite(mat).any() else 1.0

    plt.figure(figsize=(4.6, 4.0))
    im = plt.imshow(mat, origin="lower", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="mean coeff")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_eigvals_compare(
    out_png: Path, phase_to_eigs: dict[str, np.ndarray], title: str
) -> None:
    phases = list(phase_to_eigs.keys())
    R = max(len(v) for v in phase_to_eigs.values())

    # bar-like plot: each phase is a line of eigenvalues
    plt.figure(figsize=(7.5, 4.5))
    for ph in phases:
        y = phase_to_eigs[ph]
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, marker="o", label=ph)

    plt.xlabel("component")
    plt.ylabel("eigenvalue")
    plt.title(title)
    plt.xticks(np.arange(1, R + 1))
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob", required=True, help="Glob for .npz files, e.g. out/basis/*_*.npz"
    )
    ap.add_argument("--pair", required=True, help="Channel pair like 'A->B' or '0->3'")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs")
    ap.add_argument(
        "--r",
        type=int,
        default=None,
        help="How many PCs to visualize (default: all in file)",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not paths:
        raise RuntimeError("No .npz files matched.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load first file for channel names + pair parsing
    D0 = np.load(paths[0], allow_pickle=True)
    ch_names = list(D0["ch_names"])
    i, j = parse_pair(args.pair, ch_names)

    phase_to_eigs = {}
    for p in paths:
        D = np.load(p, allow_pickle=True)
        phase = str(D["pk"]) if "pk" in D else infer_phase_key(p)

        mean = D["mean"]  # (C,C,K)
        U = D["U"]  # (C,C,r,K)
        eigvals = D["eigvals"]  # (C,C,r)

        r_file = U.shape[2]
        r = min(r_file, args.r) if args.r is not None else r_file

        U_ij = np.array(U[i, j, :r, :], float)  # (r,K)
        eig_ij = np.array(eigvals[i, j, :r], float)  # (r,)
        mean_ij = np.array(mean[i, j, :], float)  # (K,)

        pair_label = f"{ch_names[i]} -> {ch_names[j]}"
        save_pc_grid(
            outdir / f"pcs_{phase}_{i}_{j}.png",
            U_ij,
            eig_ij,
            title=f"{phase} | {pair_label} | PC loadings",
        )
        save_mean(
            outdir / f"mean_{phase}_{i}_{j}.png",
            mean_ij,
            title=f"{phase} | {pair_label} | mean vector",
        )

        phase_to_eigs[phase] = eig_ij

    save_eigvals_compare(
        outdir / f"eigvals_compare_{i}_{j}.png",
        phase_to_eigs,
        title=f"Eigenvalues by phase | {ch_names[i]} -> {ch_names[j]}",
    )

    print(f"[OK] Saved visualizations to: {outdir}")


if __name__ == "__main__":
    main()
