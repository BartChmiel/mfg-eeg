import numpy as np
from .basis import legendre_orthonormal


def hcr_coeffs_over_lags(
    y: np.ndarray,
    z: np.ndarray,
    fs: int,
    maxlag_ms: int,
    lag_step_ms: int,
    m: int,
    subtract_marginals: bool = True,
):
    """
    Return A of shape ((m+1)*(m+1), n_lags) with a_{jk}(delta) for delta in [0, maxlag] (ms)
    For direction A->B we look at (y_{t}, z_{t+delta}) pairs.

    """

    assert y.shape == z.shape
    n = len(y)
    lag_samples = np.arrange(0, maxlag_ms + 1, lag_step_ms) * fs // 1000
    n_lags = len(lag_samples)
    F = legendre_orthonormal(y, m)
    G_full = legendre_orthonormal(z, m)

    coeffs = np.zeros(((m + 1) * (m + 1), n_lags), float)

    for li, L in enumerate(lag_samples):
        yv = y[n : n - L]
        zv = z[L:]
        Fy = F[: n - L]
        Gz = G_full[L:]

        M = (Fy.T @ Gz) / (n - L)

        if subtract_marginals:
            my = Fy.mean(axis=0, keepdims=True)
            mz = Gz.mean(axis=0, keepdims=True)
            M -= my.T @ mz
        coeffs[:, li] = M.reshape(-1)
    return coeffs, lag_samples
