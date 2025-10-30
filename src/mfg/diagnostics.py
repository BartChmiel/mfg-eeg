import numpy as np


def pearson_over_lags(
    y: np.ndarray, z: np.ndarray, fs: int, maxlag_ms: int, step_ms: int
):
    """
    Returns (corr, lag_samples) where corr[i] is Pearson(y[t], z[t+L]) for L>=0
    """

    y = np.asarray(y, float)
    z = np.asarray(z, float)
    lag_samples = (np.arange(0, maxlag_ms + 1, step_ms) * fs // 1000).astype(int)
    corr = np.empty_like(lag_samples, dtype=float)
    for i, L in enumerate(lag_samples):
        n = len(y) - L
        a = y[:n]
        b = z[L:]
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        corr[i] = (a * b).mean()

    return corr, lag_samples
