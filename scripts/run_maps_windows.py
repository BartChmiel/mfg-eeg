# scripts/run_maps_windows.py
import numpy as np, pandas as pd, os
from numpy.polynomial.legendre import legval
from scipy.stats import norm, t
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 500
MAXLAG_MS = 1000
LAG_STEP_MS = 2
LAGS_TO_PLOT_MS = [0, 100, 200, 300, 400, 500]
M = 12
AR_ORDER = 10
EMA_HALF_LIFE_S = 1.0
STUDENT_NU = 10


def legendre_orthonorm(x, m):
    x2 = 2.0 * x - 1.0
    F = np.empty((len(x), m + 1))
    for j in range(m + 1):
        F[:, j] = np.sqrt(2 * j + 1) * legval(x2, [0] * j + [1])
    return F


def normalize_gauss(x):
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std() + 1e-12
    return norm.cdf((x - mu) / sd)


def pnorm_student(x, fs, ar_order=10, ema_half_life_s=1.0, nu=10):
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
    s2[0] = res.var()
    for i in range(1, len(res)):
        s2[i] = (1 - alpha) * s2[i - 1] + alpha * res[i - 1] ** 2
    s = np.sqrt(s2) + 1e-9
    u = t.cdf(res / s, df=nu)
    return np.r_[np.full(ar_order, 0.5), u]


def hcr_coeffs_over_lags(y, z, fs, maxlag_ms, lag_step_ms, m):
    n = len(y)
    assert n == len(z)
    lag_samples = (np.arange(0, maxlag_ms + 1, lag_step_ms) * fs // 1000).astype(int)
    Fy = legendre_orthonorm(y, m)
    Gz = legendre_orthonorm(z, m)
    K, T = (m + 1) * (m + 1), len(lag_samples)
    coeffs = np.zeros((K, T))
    for li, L in enumerate(lag_samples):
        FyL, GzL = Fy[: n - L], Gz[L:]
        M = (FyL.T @ GzL) / (n - L)
        my, mz = FyL.mean(axis=0, keepdims=True), GzL.mean(axis=0, keepdims=True)
        M = M - my.T @ mz
        coeffs[:, li] = M.reshape(-1)
    return coeffs, lag_samples


def pca_over_lags(coeffs):
    X = coeffs - coeffs.mean(axis=1, keepdims=True)
    C = X @ X.T / (X.shape[1] - 1)
    w, V = np.linalg.eigh(C)
    idx = w.argsort()[::-1]
    return V[:, idx], (V[:, idx].T @ X), w[idx]


def run(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    chans = [c for c in df.columns if c.lower() != "id"]
    data = df[chans].to_numpy()

    Y = np.column_stack([normalize_gauss(data[:, i]) for i in range(len(chans))])

    Z = np.column_stack(
        [
            pnorm_student(data[:, i], FS, AR_ORDER, EMA_HALF_LIFE_S, STUDENT_NU)
            for i in tqdm(range(len(chans)), desc="pnorm B")
        ]
    )

    lag_samples = (np.arange(0, MAXLAG_MS + 1, LAG_STEP_MS) * FS // 1000).astype(int)
    idx_plot = [np.searchsorted(lag_samples, ms * FS // 1000) for ms in LAGS_TO_PLOT_MS]

    maps = {ms: np.zeros((len(chans), len(chans))) for ms in LAGS_TO_PLOT_MS}

    for ia in tqdm(range(len(chans)), desc="pairs A->B"):
        ya = Y[:, ia]
        for ib in range(len(chans)):
            zb = Z[:, ib]
            coeffs, lags = hcr_coeffs_over_lags(ya, zb, FS, MAXLAG_MS, LAG_STEP_MS, M)
            V, A, lam = pca_over_lags(coeffs)
            cvar = np.cumsum(lam) / lam.sum()
            r = min(4, int(np.searchsorted(cvar, 0.9) + 1))
            GC = np.sqrt((A[:r] ** 2).sum(axis=0))
            for ms, idx in zip(LAGS_TO_PLOT_MS, idx_plot):
                maps[ms][ia, ib] = GC[idx]

    for ms in LAGS_TO_PLOT_MS:
        plt.figure(figsize=(7, 6))
        plt.imshow(maps[ms], origin="lower", interpolation="nearest")
        plt.colorbar(label=f"GC(Δ={ms} ms)")
        plt.xticks(range(len(chans)), chans, rotation=90, fontsize=8)
        plt.yticks(range(len(chans)), chans, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"gc_map_{ms:03d}ms.png"), dpi=160)
        plt.close()
    print("Gotowe mapy:", [f"gc_map_{ms:03d}ms.png" for ms in LAGS_TO_PLOT_MS])


if __name__ == "__main__":
    import argparse, os

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    run(args.csv, args.outdir)
