import argparse
from .io import load_eeg_csv
from .normalize import normalize_gauss, pnorm_student
from .hcr import compute_hcr_lags
from .pca_features import pca_over_lags, select_r
from .granger import gc_scalar
from .viz import plot_features, plot_gc_matrix
from .config import load_config


def main():
    ap = argparse.ArgumentParser(prog="mfg")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_norm = sub.add_parser("norm", help="Normalize A (Gauss) and B (p-norm)")
    s_norm.add_argument("--file-a", required=True)
    s_norm.add_argument("--file-b", required=True)
    s_norm.add_argument("--out-a", required=True)
    s_norm.add_argument("--out-b", required=True)

    s_hcr = sub.add_parser("hcr", help="Compute HCR coefficients over lags for a pair")
    s_hcr.add_argument("--file-a", required=True)
    s_hcr.add_argument("--file-b", required=True)
    s_hcr.add_argument("--out", required=True)

    s_mfg = sub.add_parser("mfg", help="Full pipeline -> features and GC")
    s_mfg.add_argument("--file-a", required=True)
    s_mfg.add_argument("--file-b", required=True)
    s_mfg.add_argument("--outdir", required=True)

    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "norm":
        a = load_eeg_csv(args.file_a)
        b = load_eeg_csv(args.file_b)
        y = normalize_gauss(a)
        z = pnorm_student(b, fs=cfg["fs"], **cfg["pnorm"])
        # save to parquet...

    if args.cmd == "hcr":
        # load normalized y,z and compute a_{jk}(delta) and save to npy/parquet
        pass

    if args.cmd == "mfg":
        # norm -> hcr -> pca -> gc -> plots
        pass


if __name__ == "__main__":
    main()
