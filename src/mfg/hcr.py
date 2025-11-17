# src/mfg/hcr.py
"""
Hierarchical Correlation Reconstruction (HCR) over time lags.

This module implements polynomial cross-correlation analysis between
two normalized time series y(t), z(t + lag), using an orthonormal
Legendre basis on [0,1].

For each non-negative lag L (in samples) we compute a matrix:

    M_jk(L) = E[ f_j(y_t) * f_k(z_{t+L}) ],

where f_j, f_k are 1D orthonormal basis functions (here: Legendre
polynomials mapped to [0,1]) up to degree m. Optionally, we subtract
marginal contributions:

    M_tilde_jk = M_jk - E[f_j(y)] * E[f_k(z)],

to focus on pure dependencies (mixed moments), as in eq. (3) of the paper.

All such matrices M(L) are then flattened and stacked into a 2D array
of coefficients over lags.
"""

from typing import Tuple

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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute polynomial cross-correlation coefficients for all non-negative lags.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        First normalized time series (reason). Expected to be in (0,1),
        e.g. after normalize_gauss / normalize_edf / pnorm_student.
    z : np.ndarray, shape (T,)
        Second normalized time series (result). Must have the same length as y.
    fs : int
        Sampling frequency in Hz.
    maxlag_ms : int
        Maximum lag (in milliseconds) to consider, starting from 0 ms.
    lag_step_ms : int
        Step between consecutive lags in milliseconds.
    m : int
        Maximum polynomial degree for Legendre basis; we use 0..m in each
        coordinate, giving (m+1)^2 coefficients per lag.
    subtract_marginals : bool, default True
        If True, subtract product of marginals:
            M <- M - mean(FyL)^T * mean(GzL),
        which removes purely marginal effects and focuses on dependence.

    Returns
    -------
    coeffs : np.ndarray, shape (K, T_lags)
        Flattened HCR coefficients for all lags, where K = (m+1)^2 and
        T_lags is the number of valid lags (those with n - L > 1 samples).
        Column t corresponds to lag lag_samples[t].
    lag_samples : np.ndarray, shape (T_lags,)
        Non-negative lags in samples (integer), monotonically increasing.

    Notes
    -----
    - For each lag L >= 0 we only use pairs (y[t], z[t+L]) where both indices
      are inside [0, n). Hence the effective sample size decreases with L.
    - The loop stops when n - L <= 1, i.e. not enough pairs remain.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    if y.shape != z.shape:
        raise ValueError(
            f"hcr_coeffs_over_lags expects y and z of the same shape, "
            f"got {y.shape} and {z.shape}"
        )

    n = len(y)

    # Candidate lags (in samples) from 0 to maxlag_ms with given step
    lag_samples_full = (np.arange(0, maxlag_ms + 1, lag_step_ms) * fs // 1000).astype(
        int
    )

    # Legendre features on [0,1]; output: (T, m+1)
    Fy = legendre_orthonormal(y, m)
    Gz = legendre_orthonormal(z, m)

    if Fy.shape != (n, m + 1) or Gz.shape != (n, m + 1):
        raise ValueError(
            "legendre_orthonormal returned unexpected shapes: "
            f"Fy={Fy.shape}, Gz={Gz.shape}, expected (n, m+1) with n={n}"
        )

    K = (m + 1) * (m + 1)
    coeffs_list = []
    lags_list = []

    for L in lag_samples_full:
        n_eff = n - L
        if n_eff <= 1:
            # No enough pairs for this and further lags
            break

        # Align windows: (y_t, z_{t+L}), t = 0..n_eff-1
        FyL = Fy[:n_eff]  # shape (n_eff, m+1)
        GzL = Gz[L:]  # shape (n_eff, m+1)

        # Raw mixed moments: E[ f_j(y) f_k(z) ]
        M = (FyL.T @ GzL) / float(n_eff)  # shape (m+1, m+1)

        if subtract_marginals:
            my = FyL.mean(axis=0, keepdims=True)  # shape (1, m+1)
            mz = GzL.mean(axis=0, keepdims=True)  # shape (1, m+1)
            M = M - my.T @ mz  # outer product, same shape

        coeffs_list.append(M.reshape(-1))  # flatten to length K
        lags_list.append(int(L))

    if coeffs_list:
        coeffs = np.stack(coeffs_list, axis=1)  # shape (K, T_lags)
    else:
        coeffs = np.zeros((K, 0), dtype=float)

    lag_samples = np.asarray(lags_list, dtype=int)

    return coeffs, lag_samples


# Backwards-compatible alias used earlier by CLI code
compute_hcr_lags = hcr_coeffs_over_lags
