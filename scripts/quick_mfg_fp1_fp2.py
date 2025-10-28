import numpy as np
import numpy.polynomial.legendre as legval
import pandas as pd
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import argparse
import os

# Configuration
# CSV = r"data\grasp-and-lift-eeg-detection\train\subj7_series8_data.csv"
FS = 500
MAXLAG_MS = 1000
LAG_STEP_MS = 2
M = 12
AR_ORDER = 10
EMA_HALF_LIFE_S = 1.0
STUDENT_NU = 10


def legendre_orthonorm(x, m):
    x2 = 2 * x - 1
    F = np.empty((len(x), m + 1), float)
    for j in range(m + 1):
        F[:, j] = np.sqrt(2 * j + 1) * legval.legval(x2, [0] * j + [1])
    return F


def normalize_gauss(x: np.ndarray):

    x = np.asarray(x, float)
    mu, sd = np.mean(x), np.std(x) + 1e-12

    return norm.cdf((x - mu) / sd)


def pnorm_student(x, fs, ar_order=10, ema_half_life_s=1.0, student_nu=10):

    from statsmodels.tsa.tsatools import lagmat

    x = np.asarray(x, float)
    X = lagmat(x, maxlag=ar_order, trim="both")
    y = x[ar_order:]
    Phi = np.c_[np.ones(len(X)), X]
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    pred = Phi @ w
    res = y - pred

    alpha = 1 - np.exp(-np.log(2) / (ema_half_life_s * fs))
    s2 = np.empty_like(res)
    s2[0] = np.var(res)
    for i in range(1, len(res)):
        s2[i] = (1 - alpha) * s2[i - 1] + alpha * res[i - 1] ** 2
    s = np.sqrt(s2) + 1e-9
    u = t.cdf(res / s, df=student_nu)
    u = np.r_[np.full(ar_order, 0.5), u]
    return u


def hcr_coeffs_over_lags(y, z, fs, maxlag_ms, lag_step_ms, m, subtract_marginals=True):

    n = len(y)
    assert n == len(z)
    lag_samples = (np.arange(0, maxlag_ms + 1, lag_step_ms) * fs // 1000).astype(int)
    F = legendre_orthonorm(y, m)
    G = legendre_orthonorm(z, m)
    K = (m + 1) * (m + 1)
    T = len(lag_samples)
    coeffs = np.zeros((K, T), float)
    for li, L in enumerate(lag_samples):
        Fy, Gz = F[: n - L], G[L:]
        M = (Fy.T @ Gz) / (n - L)
        if subtract_marginals:
            my = Fy.mean(axis=0, keepdims=True)
            mz = Gz.mean(axis=0, keepdims=True)
            M -= my.T @ mz
        coeffs[:, li] = M.reshape(-1)
    return coeffs, lag_samples


def pca_over_lags(coeffs):
    X = coeffs - coeffs.mean(axis=1, keepdims=True)
    C = X @ X.T / (X.shape[1] - 1)
    w, V = np.linalg.eigh(C)
    idx = w.argsort()[::-1]
    w = w[idx]
    V = V[:, idx]
    A = V.T @ X
    return V, A, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--chan_a", default="Fp1")
    p.add_argument("--chan_b", default="Fp2")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    a = df[args.chan_a].values
    b = df[args.chan_b].values

    y = normalize_gauss(a)
    z = pnorm_student(b, FS, AR_ORDER, EMA_HALF_LIFE_S, STUDENT_NU)

    coeffs, lags = hcr_coeffs_over_lags(y, z, FS, MAXLAG_MS, LAG_STEP_MS, M)
    V, A, lam = pca_over_lags(coeffs)

    cvar = np.cumsum(lam) / lam.sum()
    r = int(np.searchsorted(cvar, 0.9) + 1)
    r = min(r, 4)

    ms = lags * 1000 // FS
    import matplotlib

    matplotlib.use("Agg")

    plt.figure(figsize=(10, 5))
    for i in range(r):
        plt.plot(ms, A[i], label=f"feature {i+1}")
    plt.axvline(0, color="k", lw=0.5)
    plt.title("a_v(Δ) for Fp1 → Fp2")
    plt.xlabel("Δ [ms]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.outdir}\\features_{args.chan_a}_{args.chan_b}.png", dpi=160)

    GC = np.sqrt((A[:r] ** 2).sum(axis=0))
    plt.figure(figsize=(10, 4))
    plt.plot(ms, GC)
    plt.axvline(0, color="k", lw=0.5)
    plt.title("GC(Δ) for Fp1 → Fp2")
    plt.xlabel("Δ [ms]")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}\\gc_{args.chan_a}_{args.chan_b}.png", dpi=160)


if __name__ == "__main__":
    main()
