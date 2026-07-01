from __future__ import annotations

import os
import re
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

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


def _default_checkpoint_path(out_path: str | Path) -> Path:
    return Path(f"{out_path}.checkpoint.npz")


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "ch_names": list(data["ch_names"]),
        "sum_x": data["sum_x"],
        "sum_xxT": data["sum_xxT"],
        "w_sum": data["w_sum"],
        "n_files_used": int(data["n_files_used"]),
        "n_cycles_total": int(data["n_cycles_total"]),
        "n_windows_total": int(data["n_windows_total"]),
        "processed_files": set(map(str, data["processed_files"].tolist())),
        "fs": int(data["fs"]),
        "m": int(data["m"]),
        "pca_r": int(data["pca_r"]),
        "lags_ms": list(map(int, data["lags_ms"].tolist())),
        "lag_samples": list(map(int, data["lag_samples"].tolist())),
        "mode": str(data["mode"]),
        "preprocess": str(data["preprocess"]),
        "epoch_len_s": float(data["epoch_len_s"]),
    }


def _write_checkpoint(
    path: Path,
    *,
    ch_names: list[str],
    sum_x: np.ndarray,
    sum_xxT: np.ndarray,
    w_sum: np.ndarray,
    n_files_used: int,
    n_cycles_total: int,
    n_windows_total: int,
    processed_files: set[str],
    fs: int,
    m: int,
    pca_r: int,
    lags_ms: list[int],
    lag_samples: list[int],
    mode: str,
    preprocess: str,
    epoch_len_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            ch_names=np.array(ch_names, dtype=object),
            sum_x=sum_x,
            sum_xxT=sum_xxT,
            w_sum=w_sum,
            n_files_used=int(n_files_used),
            n_cycles_total=int(n_cycles_total),
            n_windows_total=int(n_windows_total),
            processed_files=np.array(sorted(processed_files), dtype=object),
            fs=int(fs),
            m=int(m),
            pca_r=int(pca_r),
            lags_ms=np.array(lags_ms, dtype=int),
            lag_samples=np.array(lag_samples, dtype=int),
            mode=str(mode),
            preprocess=str(preprocess),
            epoch_len_s=float(epoch_len_s),
        )
    tmp_path.replace(path)


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    fs: int,
    m: int,
    pca_r: int,
    lags_ms: list[int],
    lag_samples: list[int],
    mode: str,
    preprocess: str,
    epoch_len_s: float,
) -> None:
    expected = {
        "fs": int(fs),
        "m": int(m),
        "pca_r": int(pca_r),
        "lags_ms": list(map(int, lags_ms)),
        "lag_samples": list(map(int, lag_samples)),
        "mode": str(mode),
        "preprocess": str(preprocess),
        "epoch_len_s": float(epoch_len_s),
    }
    mismatches = [
        f"{key}: checkpoint={checkpoint.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Checkpoint settings do not match this run. Use a different checkpoint "
            "path or remove the old checkpoint.\n" + "\n".join(mismatches)
        )


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


def _basis_mixed(u_1d: np.ndarray, m: int) -> np.ndarray:
    F = legendre_orthonormal(u_1d, m)
    return F[:, 1:]


def _compute_coeffs_all_pairs_for_lag(
    Fy: np.ndarray,
    Fz: np.ndarray,
    subtract_marginals: bool,
) -> np.ndarray:
    n_eff = Fy.shape[0]
    M = np.einsum("tcm,tdn->cdmn", Fy, Fz) / float(n_eff)

    if subtract_marginals:
        my = Fy.mean(axis=0)  # (C, m)
        mz = Fz.mean(axis=0)  # (C, m)
        M = M - my[:, None, :, None] * mz[None, :, None, :]

    return M


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


def _fix_pca_signs(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, float).copy()
    for k in range(U.shape[0]):
        idx = int(np.argmax(np.abs(U[k])))
        if U[k, idx] < 0:
            U[k] *= -1.0
    return U


def main() -> None:
    cfg = load_config_yaml()

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root or train/ directory.")
    ap.add_argument(
        "--out", required=True, help="Output .npz path, e.g. out/basis_allpairs.npz"
    )

    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")
    ap.add_argument("--epoch-len-s", type=float, default=2.0)
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)

    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lags-ms", type=int, nargs="+", default=[0, 50, 100, 150, 200])
    ap.add_argument("--pca-r", type=int, default=3)
    ap.add_argument("--preprocess", choices=PREPROCESSING_CHOICES, default="car_only")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a checkpoint written by a previous interrupted run.",
    )
    ap.add_argument(
        "--checkpoint-path",
        default="",
        help="Checkpoint .npz path. Defaults to '<out>.checkpoint.npz'.",
    )
    ap.add_argument(
        "--checkpoint-every-files",
        type=int,
        default=1,
        help="Write a checkpoint after this many newly processed files.",
    )
    ap.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writing.",
    )

    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--series", type=int, nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    root = Path(args.root)
    files_all = _iter_train_data_files(root)
    files = _filter_files(files_all, args.subjects, args.series, args.max_files)
    if not files:
        raise RuntimeError("No matching train data files found.")

    if args.dry_run:
        print(f"Matched {len(files)} files:")
        for p in files[:50]:
            print(" -", p)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fs = int(cfg.fs)
    m = int(args.m)
    K = m * m
    r = int(args.pca_r)

    lags_ms = list(map(int, args.lags_ms))
    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]
    seg_len = int(round(float(args.epoch_len_s) * float(fs)))

    checkpoint_path = (
        Path(args.checkpoint_path)
        if str(args.checkpoint_path).strip()
        else _default_checkpoint_path(args.out)
    )
    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else None

    if checkpoint is not None:
        _validate_checkpoint(
            checkpoint,
            fs=fs,
            m=m,
            pca_r=r,
            lags_ms=lags_ms,
            lag_samples=lag_samples,
            mode=str(args.mode),
            preprocess=str(args.preprocess),
            epoch_len_s=float(args.epoch_len_s),
        )
        ch_names0 = checkpoint["ch_names"]
        C = len(ch_names0)
        sum_x = checkpoint["sum_x"]
        sum_xxT = checkpoint["sum_xxT"]
        w_sum = checkpoint["w_sum"]
        n_files_used = int(checkpoint["n_files_used"])
        n_cycles_total = int(checkpoint["n_cycles_total"])
        n_windows_total = int(checkpoint["n_windows_total"])
        processed_files = set(checkpoint["processed_files"])
        print(
            f"[RESUME] loaded checkpoint {checkpoint_path} "
            f"({len(processed_files)} files complete)",
            flush=True,
        )
    else:
        first_data = files[0]
        first_events = _events_path_for_data(first_data)
        if not first_events.exists():
            raise RuntimeError("Missing events for first file (expected TRAIN split).")

        _, X0, _, ch_names0, _ = load_kaggle_data_with_events(
            str(first_data), str(first_events)
        )
        C = X0.shape[1]

        sum_x = np.zeros((C, C, K), dtype=float)
        sum_xxT = np.zeros((C, C, K, K), dtype=float)
        w_sum = np.zeros((C, C), dtype=float)

        n_files_used = 0
        n_cycles_total = 0
        n_windows_total = 0
        processed_files: set[str] = set()

    subtract_marginals = True
    checkpoint_every = max(1, int(args.checkpoint_every_files))
    newly_processed_since_checkpoint = 0

    for data_path in files:
        file_key = data_path.name
        if file_key in processed_files:
            print(f"[SKIP] basis file already checkpointed: {file_key}", flush=True)
            continue
        events_path = _events_path_for_data(data_path)
        if not events_path.exists():
            print(f"[WARN] Missing events for {data_path.name} -> skipping")
            continue

        _, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(events_path)
        )

        if ch_names != ch_names0:
            idx_map = []
            ok = True
            for nm in ch_names0:
                if nm not in ch_names:
                    ok = False
                    break
                idx_map.append(ch_names.index(nm))
            if not ok:
                raise RuntimeError(
                    f"Channel mismatch in {data_path.name}, cannot align."
                )
            X = X[:, idx_map]
            ch_names = ch_names0

        X = apply_preprocessing(
            X,
            preprocess=str(args.preprocess),
            fs=fs,
            all_channels=ch_names,
        )

        T = X.shape[0]
        Y_all, Z_all = _normalize_series(X, mode=str(args.mode), fs=fs, cfg=cfg)

        cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=float(args.max_cycle_s))
        if len(cycles) < int(args.min_cycles):
            print(
                f"[WARN] {data_path.name}: cycles={len(cycles)} < min_cycles={args.min_cycles}"
            )

        n_files_used += 1
        n_cycles_total += int(len(cycles))

        for cyc in cycles:
            for ev0, ev1 in PHASES:
                bounds = phase_window_bounds(
                    T,
                    int(cyc[ev0]),
                    int(cyc[ev1]),
                    fs=fs,
                    epoch_len_s=float(args.epoch_len_s),
                )
                if bounds is None:
                    continue
                a, b = bounds
                if (b - a) != seg_len:
                    continue

                Yr = Y_all[a:b, :]
                Zr = Z_all[a:b, :]

                Fy_full = np.empty((seg_len, C, m), dtype=float)
                Fz_full = np.empty((seg_len, C, m), dtype=float)
                for c in range(C):
                    Fy_full[:, c, :] = _basis_mixed(Yr[:, c], m)
                    Fz_full[:, c, :] = _basis_mixed(Zr[:, c], m)

                for Ls in lag_samples:
                    n_eff = seg_len - Ls
                    if n_eff <= 5:
                        continue

                    Fy = Fy_full[:n_eff, :, :]
                    Fz = Fz_full[Ls:, :, :]

                    M = _compute_coeffs_all_pairs_for_lag(
                        Fy, Fz, subtract_marginals=subtract_marginals
                    )
                    Xpair = M.reshape(C, C, K)

                    # Weighted accumulation per pair: weight by n_eff (effective window length)
                    w = float(n_eff)
                    sum_x += w * Xpair
                    sum_xxT += w * np.einsum("cdk,cdl->cdkl", Xpair, Xpair)
                    w_sum += w

                n_windows_total += 1

        processed_files.add(file_key)
        newly_processed_since_checkpoint += 1
        print(
            f"[OK] basis file {len(processed_files)}/{len(files)}: {file_key} "
            f"(cycles_total={n_cycles_total}, windows_total={n_windows_total})",
            flush=True,
        )
        if not args.no_checkpoint and newly_processed_since_checkpoint >= checkpoint_every:
            _write_checkpoint(
                checkpoint_path,
                ch_names=ch_names0,
                sum_x=sum_x,
                sum_xxT=sum_xxT,
                w_sum=w_sum,
                n_files_used=n_files_used,
                n_cycles_total=n_cycles_total,
                n_windows_total=n_windows_total,
                processed_files=processed_files,
                fs=fs,
                m=m,
                pca_r=r,
                lags_ms=lags_ms,
                lag_samples=lag_samples,
                mode=str(args.mode),
                preprocess=str(args.preprocess),
                epoch_len_s=float(args.epoch_len_s),
            )
            newly_processed_since_checkpoint = 0
            print(f"[CHECKPOINT] wrote {checkpoint_path}", flush=True)

    if not args.no_checkpoint and newly_processed_since_checkpoint:
        _write_checkpoint(
            checkpoint_path,
            ch_names=ch_names0,
            sum_x=sum_x,
            sum_xxT=sum_xxT,
            w_sum=w_sum,
            n_files_used=n_files_used,
            n_cycles_total=n_cycles_total,
            n_windows_total=n_windows_total,
            processed_files=processed_files,
            fs=fs,
            m=m,
            pca_r=r,
            lags_ms=lags_ms,
            lag_samples=lag_samples,
            mode=str(args.mode),
            preprocess=str(args.preprocess),
            epoch_len_s=float(args.epoch_len_s),
        )
        print(f"[CHECKPOINT] wrote {checkpoint_path}", flush=True)

    mean = np.zeros((C, C, K), dtype=float)
    U = np.zeros((C, C, r, K), dtype=float)
    eigvals = np.zeros((C, C, r), dtype=float)

    for i in range(C):
        for j in range(C):
            if w_sum[i, j] <= 0.0:
                continue

            mu = sum_x[i, j] / float(w_sum[i, j])
            Exx = sum_xxT[i, j] / float(w_sum[i, j])
            Cov = Exx - np.outer(mu, mu)

            Cov = 0.5 * (Cov + Cov.T)

            w, V = np.linalg.eigh(Cov)
            idx = np.argsort(w)[::-1]
            w = w[idx]
            V = V[:, idx]

            mu = np.asarray(mu, float)
            Uk = V[:, :r].T
            Uk = _fix_pca_signs(Uk)

            mean[i, j] = mu
            U[i, j] = Uk
            eigvals[i, j] = w[:r]

    np.savez_compressed(
        args.out,
        ch_names=np.array(ch_names0, dtype=object),
        mean=mean,
        U=U,
        eigvals=eigvals,
        lags_ms=np.array(lags_ms, dtype=int),
        lag_samples=np.array(lag_samples, dtype=int),
        fs=int(fs),
        m=int(m),
        mixed_only=np.int8(1),
        mode=str(args.mode),
        preprocess=str(args.preprocess),
        epoch_len_s=float(args.epoch_len_s),
        n_files=int(n_files_used),
        n_cycles_total=int(n_cycles_total),
        n_windows_total=int(n_windows_total),
        # n_samples_per_pair=n_samples,
        weight_sum_per_pair=w_sum,
        config=str(asdict(cfg)),
    )

    print(
        f"Saved all-pairs PCA basis: {args.out}\n"
        f"  files={n_files_used} cycles_total={n_cycles_total} windows_total={n_windows_total}\n"
        f"  C={C} K={K} r={r} lags_ms={lags_ms} mode={args.mode} "
        f"preprocess={args.preprocess} epoch_len_s={args.epoch_len_s}"
    )


if __name__ == "__main__":
    main()
