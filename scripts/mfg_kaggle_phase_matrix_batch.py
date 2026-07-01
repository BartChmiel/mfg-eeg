from __future__ import annotations

import json
import os
import re
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from src_mfg.config import load_config_yaml
from src_mfg.basis import legendre_orthonormal
from src_mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src_mfg.io import (
    load_kaggle_data_with_events,
    extract_grasp_cycles,
    phase_window_bounds,
)
from src_mfg.preprocessing import PREPROCESSING_CHOICES, apply_preprocessing


PHASES = [
    ("HandStart", "FirstDigitTouch"),
    ("FirstDigitTouch", "BothStartLoadPhase"),
    ("BothStartLoadPhase", "LiftOff"),
    ("LiftOff", "Replace"),
    ("Replace", "BothReleased"),
]


_RX = re.compile(r"subj(?P<subj>\d+)_series(?P<series>\d+)_data\.csv$", re.IGNORECASE)


def _load_allpairs_basis(path: str) -> dict[str, Any]:
    D = np.load(path, allow_pickle=True)
    return {
        "ch_names": list(D["ch_names"]),
        "mean": D["mean"],
        "U": D["U"],
        "eigvals": D["eigvals"],
        "lags_ms": list(map(int, D["lags_ms"])),
        "lag_samples": list(map(int, D["lag_samples"])),
        "fs": int(D["fs"]),
        "m": int(D["m"]),
        "mode": str(D["mode"]),
        "epoch_len_s": float(D["epoch_len_s"]),
    }


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


def _basis_mixed(u_1d: np.ndarray, m: int) -> np.ndarray:
    """
    Legendre orthonormal basis on [0,1], keep only degrees 1..m (mixed_only).
    Returns shape (T, m).
    """
    F = legendre_orthonormal(u_1d, m)  # (T, m+1)
    return F[:, 1:]  # drop degree-0


def _compute_coeffs_all_pairs_for_lag(
    Fy: np.ndarray,
    Fz: np.ndarray,
    subtract_marginals: bool,
) -> np.ndarray:
    """
    Compute M for all ordered pairs (i -> j):
      M[i,j,a,b] = E[ Fy_i,a * Fz_j,b ]  (optionally - marginals)
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


def _save_top_edges(
    path: str,
    mat: np.ndarray,
    ch_names: List[str],
    topk: int = 40,
    sort_abs: bool = False,
) -> None:
    A = np.array(mat, float)
    np.fill_diagonal(A, -np.inf)
    score = np.abs(A) if sort_abs else A
    flat_idx = np.argsort(score.ravel())[::-1]
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


def _save_npz_payload(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize_series(
    X: np.ndarray, mode: str, fs: int, cfg
) -> Tuple[np.ndarray, np.ndarray]:
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


def _run_group(
    *,
    group_name: str,
    files: List[Path],
    out_dir: Path,
    mode: str,
    epoch_len_s: float,
    max_cycle_s: float,
    min_cycles: int,
    m: int,
    lags_ms: List[int],
    summary: str,
    metric: str,
    basis_allpairs: Optional[str],
    pca_r: int,
    topk: int,
    save_npz: bool,
    preprocess: str,
) -> None:
    cfg = load_config_yaml()
    fs = int(cfg.fs)

    if not files:
        print(f"[WARN] group={group_name}: no files -> skipping")
        return

    # Setup from first file
    first_data = files[0]
    first_events = _events_path_for_data(first_data)
    if not first_events.exists():
        print(f"[WARN] group={group_name}: missing events for first file -> skipping")
        return

    _, X0, E0, ch_names0, _ = load_kaggle_data_with_events(
        str(first_data), str(first_events)
    )
    C = X0.shape[1]

    # Lag setup
    seg_len = int(round(float(epoch_len_s) * float(fs)))
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]
    K = m * m  # mixed_only=True

    P = len(PHASES)
    L = len(lag_samples)

    # Accumulators per phase, per lag
    coeffs_sum = np.zeros((P, L, C, C, K), dtype=float)
    n_eff_sum = np.zeros((P, L), dtype=float)

    subtract_marginals = True

    n_files_used = 0
    n_cycles_total = 0
    n_windows_total = 0
    manifest: dict[str, Any] = {
        "group": group_name,
        "mode": mode,
        "epoch_len_s": float(epoch_len_s),
        "max_cycle_s": float(max_cycle_s),
        "min_cycles": int(min_cycles),
        "m": int(m),
        "lags_ms": list(map(int, lags_ms)),
        "summary": summary,
        "metric": metric,
        "basis_allpairs": basis_allpairs,
        "pca_r": int(pca_r),
        "topk": int(topk),
        "save_npz": bool(save_npz),
        "preprocess": preprocess,
        "config": asdict(cfg),
        "input_files": [path.name for path in files],
        "used_files": [],
        "phases": [],
    }

    for data_path in files:
        events_path = _events_path_for_data(data_path)
        if not events_path.exists():
            print(f"[WARN] Missing events for {data_path.name} -> skipping")
            continue

        ids, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(events_path)
        )

        # Verify channel order consistency; if not, attempt to reorder to match first file
        if ch_names != ch_names0:
            # Build mapping from current file channels to reference ordering
            idx_map = []
            ok = True
            for nm in ch_names0:
                if nm not in ch_names:
                    ok = False
                    break
                idx_map.append(ch_names.index(nm))
            if not ok:
                raise RuntimeError(
                    f"Channel mismatch in {data_path.name}. Cannot align to reference order."
                )
            X = X[:, idx_map]
            ch_names = ch_names0

        X = apply_preprocessing(
            X,
            preprocess=preprocess,
            fs=fs,
            all_channels=ch_names,
        )

        T = X.shape[0]
        Y_all, Z_all = _normalize_series(X, mode=mode, fs=fs, cfg=cfg)

        cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=float(max_cycle_s))
        if len(cycles) < int(min_cycles):
            print(
                f"[WARN] {data_path.name}: cycles={len(cycles)} < min_cycles={min_cycles}"
            )

        n_files_used += 1
        n_cycles_total += int(len(cycles))
        manifest["used_files"].append(data_path.name)

        for cyc in cycles:
            for pi, (ev0, ev1) in enumerate(PHASES):
                bounds = phase_window_bounds(
                    T,
                    int(cyc[ev0]),
                    int(cyc[ev1]),
                    fs=fs,
                    epoch_len_s=float(epoch_len_s),
                )
                if bounds is None:
                    continue
                a, b = bounds
                if (b - a) != seg_len:
                    continue

                Yr = Y_all[a:b, :]  # (seg_len, C)
                Zr = Z_all[a:b, :]  # (seg_len, C)

                # Precompute basis per channel for this window: (seg_len, C, m)
                Fy_full = np.empty((seg_len, C, m), dtype=float)
                Fz_full = np.empty((seg_len, C, m), dtype=float)
                for c in range(C):
                    Fy_full[:, c, :] = _basis_mixed(Yr[:, c], m)
                    Fz_full[:, c, :] = _basis_mixed(Zr[:, c], m)

                for li, Ls in enumerate(lag_samples):
                    n_eff = seg_len - Ls
                    if n_eff <= 5:
                        continue

                    Fy = Fy_full[:n_eff, :, :]  # (n_eff, C, m)
                    Fz = Fz_full[Ls:, :, :]  # (n_eff, C, m)

                    M = _compute_coeffs_all_pairs_for_lag(
                        Fy, Fz, subtract_marginals=subtract_marginals
                    )
                    M_flat = M.reshape(C, C, K)

                    coeffs_sum[pi, li] += float(n_eff) * M_flat
                    n_eff_sum[pi, li] += float(n_eff)

                n_windows_total += 1

    # Finalize + save per phase
    group_out = out_dir / group_name
    group_out.mkdir(parents=True, exist_ok=True)
    basis_payload = _load_allpairs_basis(basis_allpairs) if basis_allpairs is not None else None

    for pi, (ev0, ev1) in enumerate(PHASES):
        if np.all(n_eff_sum[pi] <= 0):
            print(
                f"[WARN] group={group_name} phase={ev0}->{ev1}: no samples -> skipping outputs"
            )
            continue

        coeffs_mean = coeffs_sum[pi] / n_eff_sum[pi][:, None, None, None]  # (L,C,C,K)
        phase_manifest: dict[str, Any] = {
            "phase": f"{ev0}__{ev1}",
            "group": group_name,
            "windows_per_lag": n_eff_sum[pi].tolist(),
            "outputs": [],
        }
        if basis_payload is not None:
            B = basis_payload

            # sanity checks
            if B["fs"] != fs:
                raise ValueError("Basis fs mismatch.")
            if B["mode"] != mode:
                raise ValueError("Basis mode mismatch.")
            if abs(B["epoch_len_s"] - float(epoch_len_s)) > 1e-9:
                raise ValueError("Basis epoch_len_s mismatch.")
            if B["m"] != int(m):
                raise ValueError("Basis m mismatch.")
            if list(B["ch_names"]) != list(ch_names0):
                raise ValueError("Basis channel order mismatch (ch_names).")

            if list(map(int, B["lag_samples"])) != list(map(int, lag_samples)):
                raise ValueError(
                    f"Basis lag_samples mismatch.\n"
                    f"  basis={list(B['lag_samples'])}\n"
                    f"  run  ={list(lag_samples)}"
                )

            r = int(pca_r)
            U = B["U"][:, :, :r, :]  # (C,C,r,K)
            mu = B["mean"]  # (C,C,K)

            # scores: (r, L, C, C)
            scores = np.zeros((r, L, C, C), dtype=float)

            for li in range(L):
                Xlag = coeffs_mean[li]  # (C,C,K)
                Xc = Xlag - mu  # (C,C,K)
                S = np.einsum("cdrk,cdk->cdr", U, Xc)  # (C,C,r)
                for pc in range(r):
                    scores[pc, li] = S[:, :, pc]

            # Plot as a grid r x L (recommended for paper)
            fig, axes = plt.subplots(
                r, L, figsize=(4.2 * L, 4.0 * r), sharex=True, sharey=True
            )
            if r == 1:
                axes = np.array([axes])

            for pc in range(r):
                for li, ms in enumerate(lags_ms):
                    ax = axes[pc, li]
                    mat = np.array(scores[pc, li], float)
                    np.fill_diagonal(mat, 0.0)
                    vmax = _robust_vmax(np.abs(mat))
                    im = ax.imshow(mat, origin="lower", vmin=-vmax, vmax=vmax)
                    ax.set_title(f"PC{pc+1} | lag={ms} ms", fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])

            title = (
                f"{ev0}__{ev1} | group={group_name} | mode={mode} | m={m} | "
                f"pca_r={r} | lags_ms={lags_ms} | files={n_files_used} windows={n_windows_total}"
            )
            fig.suptitle(title, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            out_png = group_out / (
                f"phasegrid_{ev0}__{ev1}_{mode}_m{m}_pca_r{r}_lags{'-'.join(map(str,lags_ms))}.png"
            )
            fig.savefig(out_png, dpi=220)
            plt.close(fig)
            phase_manifest["outputs"].append(out_png.name)
            print("Saved:", str(out_png))

            # Optional: top edges per (pc, lag)
            for pc in range(r):
                for li, ms in enumerate(lags_ms):
                    mat = np.array(scores[pc, li], float)
                    np.fill_diagonal(mat, 0.0)
                    out_txt = group_out / (
                        f"phase_top_edges_{ev0}__{ev1}_{mode}_m{m}_pc{pc+1}_lag{ms}ms.txt"
                    )
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(f"phase={ev0}__{ev1}\n")
                        f.write(f"group={group_name}\n")
                        f.write(f"mode={mode} m={m} pca_r={r} pc={pc+1} lag_ms={ms}\n")
                        f.write(f"epoch_len_s={epoch_len_s} fs={fs}\n")
                        f.write(
                            f"files={n_files_used} cycles_total={n_cycles_total} windows_total={n_windows_total}\n"
                        )
                        f.write(f"config={asdict(cfg)}\n\n")
                    _save_top_edges(
                        str(out_txt), mat, ch_names0, topk=topk, sort_abs=True
                    )
                    phase_manifest["outputs"].append(out_txt.name)
                    print("Saved:", str(out_txt))

            if save_npz:
                out_npz = group_out / (
                    f"phasegrid_data_{ev0}__{ev1}_{mode}_m{m}_pca_r{r}_lags{'-'.join(map(str,lags_ms))}.npz"
                )
                _save_npz_payload(
                    out_npz,
                    scores=scores.astype(np.float32),
                    lags_ms=np.array(lags_ms, dtype=int),
                    ch_names=np.array(ch_names0, dtype=object),
                    phase=np.array(f"{ev0}__{ev1}"),
                    group=np.array(group_name),
                    mode=np.array(mode),
                    m=np.array(int(m)),
                    pca_r=np.array(int(r)),
                    files_used=np.array(int(n_files_used)),
                    cycles_total=np.array(int(n_cycles_total)),
                    windows_total=np.array(int(n_windows_total)),
                )
                phase_manifest["outputs"].append(out_npz.name)
                print("Saved:", str(out_npz))

            manifest["phases"].append(phase_manifest)
            continue

        energy_per_lag = np.sqrt(np.sum(coeffs_mean**2, axis=-1))

        chi2_stat = n_eff_sum[pi][:, None, None] * np.sum(coeffs_mean**2, axis=-1)
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

        mat_disp = np.array(mat, float)
        np.fill_diagonal(mat_disp, 0.0)

        title = (
            f"{ev0}__{ev1} | group={group_name} | mode={mode} | m={m} | "
            f"metric={metric} | summary={summary} | lags_ms={lags_ms} | "
            f"files={n_files_used} cycles={n_cycles_total} windows={n_windows_total}"
        )

        # Heatmap
        plt.figure(figsize=(11.5, 10))
        vmax = _robust_vmax(mat_disp)
        im = plt.imshow(mat_disp, origin="lower", vmin=0.0, vmax=vmax)
        plt.colorbar(im, label=cbar_label)
        plt.xticks(range(C), ch_names0, rotation=90, fontsize=8)
        plt.yticks(range(C), ch_names0, fontsize=8)
        plt.title(title)
        plt.tight_layout()

        out_png = group_out / f"phase_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.png"
        plt.savefig(out_png, dpi=220)
        plt.close()

        # Top edges
        out_txt = (
            group_out
            / f"phase_top_edges_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.txt"
        )
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"phase={ev0}__{ev1}\n")
            f.write(f"group={group_name}\n")
            f.write(
                f"mode={mode} m={m} metric={metric} summary={summary} lags_ms={lags_ms}\n"
            )
            f.write(f"epoch_len_s={epoch_len_s} fs={fs}\n")
            f.write(
                f"files={n_files_used} cycles_total={n_cycles_total} windows_total={n_windows_total}\n"
            )
            f.write(f"config={asdict(cfg)}\n\n")
        _save_top_edges(str(out_txt), mat, ch_names0, topk=topk)

        phase_manifest["outputs"].extend([out_png.name, out_txt.name])

        if save_npz:
            out_npz = (
                group_out
                / f"phase_data_{ev0}__{ev1}_{mode}_m{m}_{metric}_{summary}.npz"
            )
            _save_npz_payload(
                out_npz,
                mat_summary=np.asarray(mat, dtype=np.float32),
                per_lag=np.asarray(per_lag, dtype=np.float32),
                coeffs_mean=np.asarray(coeffs_mean, dtype=np.float32),
                energy_per_lag=np.asarray(energy_per_lag, dtype=np.float32),
                chi2_stat=np.asarray(chi2_stat, dtype=np.float32),
                neglog10p=np.asarray(neglog10p, dtype=np.float32),
                lags_ms=np.array(lags_ms, dtype=int),
                ch_names=np.array(ch_names0, dtype=object),
                phase=np.array(f"{ev0}__{ev1}"),
                group=np.array(group_name),
                mode=np.array(mode),
                m=np.array(int(m)),
                metric=np.array(metric),
                summary=np.array(summary),
                files_used=np.array(int(n_files_used)),
                cycles_total=np.array(int(n_cycles_total)),
                windows_total=np.array(int(n_windows_total)),
            )
            phase_manifest["outputs"].append(out_npz.name)
            print("Saved:", str(out_npz))

        print("Saved:", str(out_png))
        print("Saved:", str(out_txt))
        manifest["phases"].append(phase_manifest)

    manifest["files_requested"] = len(files)
    manifest["files_used"] = int(n_files_used)
    manifest["cycles_total"] = int(n_cycles_total)
    manifest["windows_total"] = int(n_windows_total)
    _write_json(group_out / "run_manifest.json", manifest)
    print("Saved:", str(group_out / "run_manifest.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root or train/ directory.")
    ap.add_argument(
        "--out", required=True, help="Output directory (e.g. out/phase_mats_batch)."
    )

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
        "--basis-allpairs",
        default=None,
        help="If set, run PCA-projection mode using an all-pairs basis .npz",
    )

    ap.add_argument("--pca-r", type=int, default=3)
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--series", type=int, nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument(
        "--save-npz",
        action="store_true",
        help="Save machine-readable .npz payloads alongside plots and text files.",
    )

    ap.add_argument(
        "--group-by",
        choices=["all", "subject"],
        default="all",
        help="all: single global aggregate; subject: separate outputs for each subject.",
    )
    ap.add_argument(
        "--preprocess",
        choices=PREPROCESSING_CHOICES,
        default="car_only",
        help="EEG cleaning before MFG normalization. Use bandpass_0_5_48 for publication reruns.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_all = _iter_train_data_files(root)
    files = _filter_files(files_all, args.subjects, args.series, args.max_files)
    if not files:
        raise RuntimeError("No matching train data files found.")

    if args.group_by == "all":
        _run_group(
            group_name="ALL",
            files=files,
            out_dir=out_dir,
            mode=args.mode,
            epoch_len_s=float(args.epoch_len_s),
            max_cycle_s=float(args.max_cycle_s),
            min_cycles=int(args.min_cycles),
            m=int(args.m),
            lags_ms=list(args.lags_ms),
            summary=str(args.summary),
            metric=str(args.metric),
            basis_allpairs=args.basis_allpairs,
            pca_r=int(args.pca_r),
            topk=int(args.topk),
            save_npz=bool(args.save_npz),
            preprocess=str(args.preprocess),
        )
        return

    # group_by == "subject"
    groups: Dict[str, List[Path]] = {}
    for p in files:
        subj, se = _parse_subj_series(p)
        if subj is None:
            continue
        key = f"subj{subj:02d}"
        groups.setdefault(key, []).append(p)

    for key in sorted(groups.keys()):
        _run_group(
            group_name=key,
            files=groups[key],
            out_dir=out_dir,
            mode=args.mode,
            epoch_len_s=float(args.epoch_len_s),
            max_cycle_s=float(args.max_cycle_s),
            min_cycles=int(args.min_cycles),
            m=int(args.m),
            lags_ms=list(args.lags_ms),
            summary=str(args.summary),
            metric=str(args.metric),
            basis_allpairs=args.basis_allpairs,
            pca_r=int(args.pca_r),
            topk=int(args.topk),
            save_npz=bool(args.save_npz),
            preprocess=str(args.preprocess),
        )


if __name__ == "__main__":
    main()
