# scripts/build_kaggle_basis_batch.py
"""
Build a shared PCA basis for a fixed Kaggle electrode pair across MANY Kaggle files.

This is the "full version" of build_kaggle_basis.py:
- iterates over many subj/series in train/
- aggregates HCR feature samples across ALL phases and cycles (streaming moments)
- outputs one .npz that can be reused to compare PC1/PC2/PC3 across phases AND across files

Important:
- Requires *_events.csv -> works on TRAIN split (series 1..8).
- TEST split has no events, so basis cannot be built from test unless you have predicted events.

Output .npz contains:
- mean (K,), V (K,K), eigvals (K,), lag_samples (L,)
- metadata: fs, maxlag_ms, lag_step_ms, m, mixed_only, chan_a, chan_b, mode, epoch_len_s
- batch metadata: n_files, n_cycles_total, n_windows_total, n_samples_total, included_files
"""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


_RX = re.compile(r"subj(?P<subj>\d+)_series(?P<series>\d+)_data\.csv$", re.IGNORECASE)


def _parse_subj_series(p: Path) -> Tuple[Optional[int], Optional[int]]:
    m = _RX.search(p.name)
    if not m:
        return None, None
    return int(m.group("subj")), int(m.group("series"))


def _iter_train_data_files(root: Path) -> Iterable[Path]:
    # Kaggle extracted layout usually: root/train/*.csv
    # Accept either root pointing to dataset root OR directly to train/
    if (root / "train").is_dir():
        train_dir = root / "train"
    else:
        train_dir = root
    yield from sorted(train_dir.glob("subj*_series*_data.csv"))


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


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Path to Kaggle dataset root or train/ directory (with subj*_series*_data.csv).",
    )
    ap.add_argument("--chan-a", required=True)
    ap.add_argument("--chan-b", required=True)
    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")

    ap.add_argument("--epoch-len-s", type=float, default=2.0)
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)

    ap.add_argument("--m", type=int, default=4)  # m=4 => K=16 with mixed_only=True
    ap.add_argument(
        "--out",
        required=True,
        help="Output .npz path for the basis (e.g., out/basis_Fp1_Fp2_gc_e2s_m4_batch.npz).",
    )

    ap.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of subject ids to include (e.g. --subjects 1 2 3).",
    )
    ap.add_argument(
        "--series",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of series ids to include (e.g. --series 1 2 3 4 5 6 7 8).",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files to process (debug).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matched files and exit.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    files_all = list(_iter_train_data_files(root))
    files = _filter_files(files_all, args.subjects, args.series, args.max_files)

    if not files:
        raise RuntimeError(
            f"No train data files found under {root!r} with given filters."
        )

    if args.dry_run:
        print(f"Matched {len(files)} files:")
        for p in files[:50]:
            print(" -", p)
        if len(files) > 50:
            print(" ...")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fs = int(cfg.fs)
    lag_step_ms = int(cfg.lag_step_ms)

    m = int(args.m)
    mixed_only = True
    K = m * m  # 16 when m=4

    # Streaming moments for PCA
    n_samples_total = 0
    sum_x = np.zeros(K, dtype=float)
    sum_xxT = np.zeros((K, K), dtype=float)
    lag_ref = None

    included_files: List[str] = []
    n_cycles_total = 0
    n_windows_total = 0

    # We also need a consistent lag grid -> depends on epoch_len_s/fs/maxlag_ms/lag_step_ms
    # maxlag_ms is limited by cfg.maxlag_ms and data window length
    seg_len = int(round(float(args.epoch_len_s) * float(fs)))
    min_pairs = 10
    maxlag_samples_data = max(1, seg_len - min_pairs)
    maxlag_ms_data = int(1000.0 * maxlag_samples_data / fs)
    maxlag_ms = int(min(cfg.maxlag_ms, maxlag_ms_data))

    for data_path in files:
        events_path = _events_path_for_data(data_path)
        if not events_path.exists():
            # Should not happen for train split; skip safely.
            print(
                f"[WARN] Missing events file for {data_path.name}: {events_path.name} -> skipping"
            )
            continue

        ids, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path), str(events_path)
        )
        T = X.shape[0]

        try:
            ia = ch_names.index(args.chan_a)
            ib = ch_names.index(args.chan_b)
        except ValueError as exc:
            raise ValueError(
                f"Channel not found in {data_path.name}. "
                f"Looking for ({args.chan_a}, {args.chan_b}). "
                f"Example channels: {ch_names[:10]} ..."
            ) from exc

        cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=float(args.max_cycle_s))
        if len(cycles) < int(args.min_cycles):
            print(
                f"[WARN] {data_path.name}: cycles={len(cycles)} < min_cycles={args.min_cycles} (still using it)"
            )

        # Global normalization on full series (NOT per window)
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

        included_files.append(str(data_path))
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
                    raise RuntimeError(
                        f"Expected K={K} features, got {coeffs.shape[0]}"
                    )

                if lag_ref is None:
                    lag_ref = lag_samples
                else:
                    if not np.array_equal(lag_ref, lag_samples):
                        raise RuntimeError(
                            "Lag grid mismatch across files/windows. "
                            "Check fs/epoch_len_s/lag_step_ms consistency."
                        )

                # Each lag-column is one sample x in R^K
                sum_x += coeffs.sum(axis=1)
                sum_xxT += coeffs @ coeffs.T
                n_samples_total += int(coeffs.shape[1])
                n_windows_total += 1

    if n_samples_total < 2:
        raise RuntimeError(
            "Not enough samples to build PCA basis. "
            "Increase files/windows or verify event extraction."
        )

    mean, V, eigvals = pca_from_second_moments(sum_x, sum_xxT, n_samples_total)

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
        n_files=int(len(included_files)),
        n_cycles_total=int(n_cycles_total),
        n_windows_total=int(n_windows_total),
        n_samples_total=int(n_samples_total),
        included_files=np.array(included_files, dtype=object),
    )

    print(
        f"Saved batch basis: {args.out}\n"
        f"  files={len(included_files)} cycles_total={n_cycles_total}\n"
        f"  windows_total={n_windows_total} samples_total={n_samples_total}\n"
        f"  chan=({args.chan_a}->{args.chan_b}) mode={args.mode} m={m} K={K}\n"
        f"  fs={fs} epoch_len_s={args.epoch_len_s} maxlag_ms={maxlag_ms} step_ms={lag_step_ms}"
    )


if __name__ == "__main__":
    main()
