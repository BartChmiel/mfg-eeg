# src/mfg/hcr.py
"""
Hierarchical Correlation Reconstruction (HCR) over non-negative time lags.

Given two normalized time series y(t), z(t) with values in (0,1), we compute
polynomial cross-moments using an orthonormal Legendre basis on [0,1].

For each lag L >= 0 (in samples):
    M_jk(L) = E[ f_j(y_t) * f_k(z_{t+L}) ],  j,k = 0..m

Optionally we subtract the product of marginals:
    M <- M - E[f_j(y)] * E[f_k(z)]

If mixed_only=True, we keep only degrees 1..m in both dimensions (drop degree-0),
so the feature vector length becomes K = m*m (e.g. m=4 -> K=16).
Otherwise K = (m+1)^2.
"""

from __future__ import annotations

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
    mixed_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute polynomial cross-correlation coefficients for all non-negative lags.

    Parameters
    ----------
    y, z : np.ndarray, shape (T,)
        Normalized time series in (0,1). Must have equal shape.
    fs : int
        Sampling frequency in Hz.
    maxlag_ms : int
        Max lag in milliseconds (>= 0).
    lag_step_ms : int
        Lag step in milliseconds (> 0).
    m : int
        Max polynomial degree (>= 0). Basis uses degrees 0..m.
    subtract_marginals : bool
        If True, subtract outer product of basis means to focus on dependence.
    mixed_only : bool
        If True, keep only degrees 1..m in both dimensions (K = m*m).
        If False, keep full (m+1)^2 coefficients.

    Returns
    -------
    coeffs : np.ndarray, shape (K, n_lags)
        Stacked flattened coefficient vectors over lags.
        Column i corresponds to lag lag_samples[i].
    lag_samples : np.ndarray, shape (n_lags,)
        Non-negative lags in samples.

    Notes
    -----
    For lag L we only use pairs (y[t], z[t+L]) for t=0..(n-L-1),
    hence the effective sample size decreases with L.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    if y.shape != z.shape:
        raise ValueError(
            f"y and z must have the same shape, got {y.shape} and {z.shape}"
        )
    if y.ndim != 1:
        raise ValueError(f"y and z must be 1D arrays, got y.ndim={y.ndim}")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise ValueError("y/z contain NaN or inf")
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    if maxlag_ms < 0:
        raise ValueError(f"maxlag_ms must be >= 0, got {maxlag_ms}")
    if lag_step_ms <= 0:
        raise ValueError(f"lag_step_ms must be > 0, got {lag_step_ms}")
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")

    n = int(y.size)

    # Lag grid in samples
    step_samples = int(round(lag_step_ms * fs / 1000.0))
    step_samples = max(1, step_samples)  # safety (still keep step >= 1 sample)
    maxlag_samples = int(round(maxlag_ms * fs / 1000.0))
    lag_samples_full = np.arange(0, maxlag_samples + 1, step_samples, dtype=int)

    # Basis features: (n, m+1)
    Fy = legendre_orthonormal(y, m)
    Gz = legendre_orthonormal(z, m)

    # Feature dimensionality
    K = (m * m) if mixed_only else ((m + 1) * (m + 1))
    coeffs_list: list[np.ndarray] = []
    lags_list: list[int] = []

    for L in lag_samples_full:
        n_eff = n - int(L)
        if n_eff <= 1:
            break

        FyL = Fy[:n_eff]  # (n_eff, m+1)
        GzL = Gz[L:]  # (n_eff, m+1)

        # Cross-moments matrix: (m+1, m+1)
        M = (FyL.T @ GzL) / float(n_eff)

        if subtract_marginals:
            my = FyL.mean(axis=0, keepdims=True)  # (1, m+1)
            mz = GzL.mean(axis=0, keepdims=True)  # (1, m+1)
            M = M - (my.T @ mz)

        if mixed_only:
            # degrees 1..m only -> (m, m) -> length m*m
            coeffs_list.append(M[1:, 1:].reshape(-1))
        else:
            coeffs_list.append(M.reshape(-1))

        lags_list.append(int(L))

    if coeffs_list:
        coeffs = np.stack(coeffs_list, axis=1)  # (K, n_lags)
    else:
        coeffs = np.zeros((K, 0), dtype=float)

    lag_samples = np.asarray(lags_list, dtype=int)
    return coeffs, lag_samples


# Backwards-compatible alias used earlier by CLI code
compute_hcr_lags = hcr_coeffs_over_lags
