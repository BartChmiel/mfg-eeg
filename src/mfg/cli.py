import argparse, os, numpy as np
from .io import load_eeg_csv, save_npz
from .normalize import normalize_gauss, pnorm_student
from .hcr import compute_hcr_lags
from .pca_features import pca_over_lags, select_r
from .granger import gc_scalar
from .config import load_config


def main():
    ap = argparse.ArgumentParser(prog="mfg")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_norm = sub.add_parser("norm", help="Normalize A (Gauss) and B (p-norm)")
    s_norm.add_argument("--csv", required=True)
    s_norm.add_argument("--chan-a", required=True)
    s_norm.add_argument("--chan-b", required=True)
    s_norm.add_argument("--out-a", required=True)
    s_norm.add_argument("--out-b", required=True)

    s_hcr = sub.add_parser("hcr", help="Compute HCR coeffs over lags (pair)")
    s_hcr.add_argument("--csv", required=True)
    s_hcr.add_argument("--chan-a", required=True)
    s_hcr.add_argument("--chan-b", required=True)
    s_hcr.add_argument("--out", required=True)

    s_mfg = sub.add_parser("mfg", help="Full pipeline -> features and GC (pair)")
    s_mfg.add_argument("--csv", required=True)
    s_mfg.add_argument("--chan-a", required=True)
    s_mfg.add_argument("--chan-b", required=True)
    s_mfg.add_argument("--outdir", required=True)

    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "norm":
        X, chans = load_eeg_csv(args.csv, [args.chan_a, args.chan_b])
        a = normalize_gauss(X[:, 0])
        b = pnorm_student(
            X[:, 1], cfg.fs, cfg.ar_order, cfg.ema_half_life_s, cfg.student_nu
        )
        np.save(args.out_a, a)
        np.save(args.out_b, b)
        return

    if args.cmd == "hcr":
        X, chans = load_eeg_csv(args.csv, [args.chan_a, args.chan_b])
        a = normalize_gauss(X[:, 0])
        b = pnorm_student(
            X[:, 1], cfg.fs, cfg.ar_order, cfg.ema_half_life_s, cfg.student_nu
        )
        coeffs, lags = compute_hcr_lags(
            a, b, cfg.fs, cfg.maxlag_ms, cfg.lag_step_ms, cfg.m, cfg.subtract_marginals
        )
        save_npz(args.out, coeffs=coeffs, lags=lags, m=cfg.m, fs=cfg.fs)
        return

    if args.cmd == "mfg":
        os.makedirs(args.outdir, exist_ok=True)
        X, _ = load_eeg_csv(args.csv, [args.chan_a, args.chan_b])
        a = normalize_gauss(X[:, 0])
        b = pnorm_student(
            X[:, 1], cfg.fs, cfg.ar_order, cfg.ema_half_life_s, cfg.student_nu
        )

        coeffs, lags = compute_hcr_lags(
            a, b, cfg.fs, cfg.maxlag_ms, cfg.lag_step_ms, cfg.m, cfg.subtract_marginals
        )
        V, A, eigvals = pca_over_lags(coeffs)
        r = select_r(eigvals, cfg.pca_var_thresh, cfg.pca_max_r)
        scores = A[:r]
        gc = gc_scalar(scores)

        ms = lags * 1000.0 / cfg.fs
        save_npz(
            os.path.join(args.outdir, f"mfg_{args.chan_a}_{args.chan_b}.npz"),
            coeffs=coeffs,
            lags=lags,
            eigvals=eigvals,
            scores=scores,
            loadings=V[:, :r],
            gc=gc,
            ms=ms,
            m=cfg.m,
            fs=cfg.fs,
        )
        return


if __name__ == "__main__":
    main()
