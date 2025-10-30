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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute polynomial cross-correlation coefficients for all lags.
    Returns:
      coeffs: (K, T) where K=(m+1)^2, T=n_lags
      lag_samples: (T,) sample offsets
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    assert y.shape == z.shape
    n = len(y)
    lag_samples = (np.arange(0, maxlag_ms + 1, lag_step_ms) * fs // 1000).astype(int)

    Fy = legendre_orthonormal(y, m)  # (n, m+1)
    Gz = legendre_orthonormal(z, m)  # (n, m+1)

    K, T = (m + 1) * (m + 1), len(lag_samples)
    coeffs = np.zeros((K, T), float)

    for li, L in enumerate(lag_samples):
        FyL, GzL = Fy[: n - L], Gz[L:]
        M = (FyL.T @ GzL) / (n - L)
        if subtract_marginals:
            my, mz = FyL.mean(axis=0, keepdims=True), GzL.mean(axis=0, keepdims=True)
            M = M - my.T @ mz
        coeffs[:, li] = M.reshape(-1)

    return coeffs, lag_samples


# Backwards-compatible alias used by cli.py earlier
compute_hcr_lags = hcr_coeffs_over_lags
