import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mfg.normalize import normalize_gauss, normalize_edf, pnorm_student
from src.mfg.hcr import hcr_coeffs_over_lags
from src.mfg.pca_features import pca_over_lags, select_r
from src.mfg.diagnostics import pearson_over_lags
from src.mfg.config import load_config
from src.mfg.viz import plot_granger_simple, plot_granger_dicc
from src.mfg.io import load_pair_for_analysis, save_npz


_cfg = load_config()
FS_DEFAULT, MAXLAG_MS, LAG_STEP_MS = _cfg.fs, _cfg.maxlag_ms, _cfg.lag_step_ms
M = _cfg.m
AR_ORDER, EMA_HALF_LIFE_S, STUDENT_NU = (
    _cfg.ar_order,
    _cfg.ema_half_life_s,
    _cfg.student_nu,
)

DISPLAY_FEATURES = 4
SMOOTH_MS = 10  # wygładzanie po lagach w ms


def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xpad = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xpad, np.ones(win) / win, mode="valid")


def _nearest_idx(ms_array: np.ndarray, target_ms: float) -> int:
    return int(np.argmin(np.abs(ms_array - float(target_ms))))


def main() -> None:
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

    ap = argparse.ArgumentParser(
        description=(
            "Per-trial HCR+PCA+GC for a pair of EEG channels "
            "(CSV or mfgin .mat; każdy trial osobno)."
        )
    )
    ap.add_argument("--file", required=True, help="CSV or MAT file")
    ap.add_argument("--chan-a", required=True, help="source (A)")
    ap.add_argument("--chan-b", required=True, help="target (B)")
    ap.add_argument(
        "--mode",
        choices=["corr", "gc"],
        default="gc",
        help="corr = Pearson-only; gc = EDF+AR+Student (Atlantis-style)",
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="opcjonalny limit liczby triali do policzenia (debug)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 1) Ujednolicony loader (CSV / mfgin .mat) ----
    a_flat, b_flat, fs, meta = load_pair_for_analysis(
        args.file,
        args.chan_a,
        args.chan_b,
        fs_fallback=FS_DEFAULT,
    )

    assert len(a_flat) == len(b_flat), "A i B muszą mieć tę samą długość"

    data_type = meta.get("type", "unknown")

    # ---- 2) Rekonstrukcja triali ----
    if data_type == "roi":
        # mfgin .mat: mamy info o liczbie triali i długości okna
        n_trials = int(meta["n_trials"])
        W = int(meta["trial_len"])
        if n_trials * W != len(a_flat):
            raise ValueError(
                f"Flattened length {len(a_flat)} != n_trials*trial_len = {n_trials}*{W}"
            )
    elif data_type == "csv":
        # CSV: traktujemy jako 1 trial
        n_trials = 1
        W = len(a_flat)
    else:
        raise ValueError(f"Unsupported data type for per-trial analysis: {data_type!r}")

    a_trials = a_flat.reshape(n_trials, W)
    b_trials = b_flat.reshape(n_trials, W)

    if args.max_trials is not None:
        n_trials = min(n_trials, int(args.max_trials))

    print(
        f"[INFO] file={os.path.basename(args.file)}, type={data_type}, "
        f"fs={fs} Hz, n_trials={n_trials}, trial_len={W}"
    )

    # okno wygładzania w jednostkach liczby lagów
    win = max(1, int(round(SMOOTH_MS / LAG_STEP_MS)))

    # ---- 3) Pętla po trialach ----
    for tr in range(n_trials):
        a = a_trials[tr]
        b = b_trials[tr]

        # maksymalny sensowny lag z długości triala - chcemy zostawić >= min_pairs punktów
        min_pairs = 10
        maxlag_samples_data = max(1, W - min_pairs)
        maxlag_ms_data = int(1000.0 * maxlag_samples_data / fs)
        maxlag_ms = min(MAXLAG_MS, maxlag_ms_data)

        if maxlag_ms <= 0:
            print(f"[WARN] trial {tr+1}: maxlag_ms<=0, pomijam.")
            continue

        # ---- 3.1) Normalizacja ----
        if args.mode == "corr":
            yA = normalize_gauss(a)
            zB = normalize_gauss(b)
        else:
            # Atlantis-style: A → EDF, B → AR+Student
            yA = normalize_edf(a)
            zB = pnorm_student(b, fs, AR_ORDER, EMA_HALF_LIFE_S, STUDENT_NU)

        # ---- 3.2) HCR dla A→B i B→A ----
        coeffs_AB, lag_samples = hcr_coeffs_over_lags(
            yA, zB, fs, maxlag_ms, LAG_STEP_MS, M, subtract_marginals=True
        )
        coeffs_BA, _ = hcr_coeffs_over_lags(
            zB, yA, fs, maxlag_ms, LAG_STEP_MS, M, subtract_marginals=True
        )

        if coeffs_AB.shape[1] < 2:
            print(
                f"[WARN] trial {tr+1}: za mało lagów ({coeffs_AB.shape[1]}), pomijam."
            )
            continue

        # ---- 3.3) PCA ----
        V_ab, A_ab, eig_ab = pca_over_lags(coeffs_AB)
        r_ab = select_r(eig_ab, 0.90, DISPLAY_FEATURES)

        V_ba, A_ba, eig_ba = pca_over_lags(coeffs_BA)
        r_ba = select_r(eig_ba, 0.90, DISPLAY_FEATURES)

        r = min(DISPLAY_FEATURES, r_ab, r_ba)
        if r == 0:
            print(f"[WARN] trial {tr+1}: r=0 (brak stabilnych PC), pomijam.")
            continue

        frac_ab = eig_ab / (eig_ab.sum() + 1e-12)
        frac_ba = eig_ba / (eig_ba.sum() + 1e-12)
        lambdas_frac = 0.5 * (frac_ab[:r] + frac_ba[:r])

        ms_pos = lag_samples * 1000.0 / fs  # lags>=0 w ms
        ms_sym = np.concatenate((-ms_pos[::-1], ms_pos))

        scores_ab_raw = A_ab[:r]
        scores_ba_raw = A_ba[:r]

        # dopasowanie znaków PCA: A→B vs B→A
        for v in range(r):
            if np.corrcoef(scores_ab_raw[v], scores_ba_raw[v, ::-1])[0, 1] < 0:
                scores_ba_raw[v] *= -1
        scores_sym_raw = np.concatenate((scores_ba_raw[:, ::-1], scores_ab_raw), axis=1)

        # ---- 3.4) Pearson do ustalenia znaku cechy #1 ----
        pearson_pos, _ = pearson_over_lags(yA, zB, fs, maxlag_ms, LAG_STEP_MS)
        pearson_sym = np.concatenate((pearson_pos[::-1], pearson_pos))

        i0 = _nearest_idx(ms_sym, 0.0)
        if i0 < len(pearson_sym) and r > 0:
            if scores_sym_raw[0, i0] * pearson_sym[i0] < 0:
                scores_sym_raw[0] *= -1

        # GC(delta) = ||a(delta)||_2
        gc_sym = np.sqrt((scores_sym_raw**2).sum(axis=0))

        # ---- 3.5) Prosty wykres (lag>=0) ----
        simple_out = os.path.join(
            args.out,
            f"trial_{tr+1:03d}_{args.chan_a}_{args.chan_b}_simple.png",
        )
        scores_pos = A_ab[: min(3, r)]
        plot_granger_simple(
            lag_samples=lag_samples,
            scores=scores_pos,
            fs=fs,
            title=f"{args.chan_a} → {args.chan_b} (lag ≥ 0), trial {tr+1}",
            outpath=simple_out,
        )

        # ---- 3.6) DICC (tylko dodatnie lags) ----
        mask_pos = ms_sym >= 0.0
        ms_pos_sec_dicc = ms_sym[mask_pos] / 1000.0
        scores_dicc = scores_sym_raw[: min(3, r), :][:, mask_pos]
        scores_dicc_disp = scores_dicc.copy()
        for v in range(scores_dicc_disp.shape[0]):
            m = float(np.max(np.abs(scores_dicc_disp[v])) or 1.0)
            scores_dicc_disp[v] = moving_avg(scores_dicc_disp[v] / m, win)

        dicc_out = os.path.join(
            args.out,
            f"trial_{tr+1:03d}_{args.chan_a}_{args.chan_b}_dicc.png",
        )
        plot_granger_dicc(
            t=ms_pos_sec_dicc,
            scores=scores_dicc_disp,
            title=f"{args.chan_a} → {args.chan_b} (trial {tr+1})",
            outpath=dicc_out,
        )

        # ---- 3.7) Zapis surowych wyników do .npz ----
        npz_out = os.path.join(
            args.out,
            f"trial_{tr+1:03d}_{args.chan_a}_{args.chan_b}.npz",
        )
        save_npz(
            npz_out,
            lag_samples=lag_samples,
            ms_pos=ms_pos,
            ms_sym=ms_sym,
            eig_ab=eig_ab,
            eig_ba=eig_ba,
            lambdas_frac=lambdas_frac,
            scores_ab=A_ab,
            scores_ba=A_ba,
            gc_sym=gc_sym,
            pearson_pos=pearson_pos,
            pearson_sym=pearson_sym,
            fs=fs,
            trial_index=tr + 1,
            chan_a=args.chan_a,
            chan_b=args.chan_b,
            mode=args.mode,
        )

        print(
            f"[OK] trial {tr+1:03d}: r_ab={r_ab}, r_ba={r_ba}, "
            f"r_used={r}, PNG simple+dicc + NPZ zapisane."
        )


if __name__ == "__main__":
    main()
