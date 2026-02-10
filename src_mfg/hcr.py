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
    Computes polynomial cross-correlation coefficients for non-negative lags.

    Processing Steps:
    1. Grid Initialization: Defines lag offsets in samples based on fs and step_ms.
    2. Basis Projection: Evaluates Legendre polynomials for both signals across the entire series.
    3. Lag-Iterative Moments: For each lag L, calculates the expectation E[f_j(y_t) * f_k(z_{t+L})].
    4. Centering: Subtracts marginal products if subtract_marginals is True to focus on coupling.

    Args:
        y, z: np.ndarray - Normalized 1D signals in (0, 1).
        fs: int - Sampling frequency.
        maxlag_ms: int - Maximum time shift.
        lag_step_ms: int - Resolution of the lag grid.
        m: int - Maximum polynomial degree.
        subtract_marginals: bool - Focus on dependency by removing marginal expectations.
        mixed_only: bool - Exclude degree-0 interactions (reduces K from (m+1)^2 to m^2).

    Returns:
        coeffs: np.ndarray - Flattened coefficient matrix of shape (K, n_lags).
        lag_samples: np.ndarray - Integer lag offsets in samples.
    """
    y, z = np.asarray(y, float), np.asarray(z, float)
    if y.shape != z.shape or y.ndim != 1:
        raise ValueError("y and z must be identical 1D arrays")

    # Sample-based lag grid calculation
    step_samples = max(1, int(round(lag_step_ms * fs / 1000.0)))
    maxlag_samples = int(round(maxlag_ms * fs / 1000.0))
    lag_samples_full = np.arange(0, maxlag_samples + 1, step_samples, dtype=int)

    # Precompute basis functions
    Fy = legendre_orthonormal(y, m)
    Gz = legendre_orthonormal(z, m)

    K = (m * m) if mixed_only else ((m + 1) * (m + 1))
    coeffs_list, lags_list = [], []

    for L in lag_samples_full:
        n_eff = len(y) - int(L)
        if n_eff <= 1:
            break

        # Compute cross-moments matrix M for current lag L
        M = (Fy[:n_eff].T @ Gz[L:]) / float(n_eff)

        if subtract_marginals:
            my = Fy[:n_eff].mean(axis=0, keepdims=True)
            mz = Gz[L:].mean(axis=0, keepdims=True)
            M -= my.T @ mz

        # Reshape and store coefficients
        val = M[1:, 1:] if mixed_only else M
        coeffs_list.append(val.reshape(-1))
        lags_list.append(int(L))

    if not coeffs_list:
        return np.zeros((K, 0), dtype=float), np.array([], dtype=int)

    coeffs = np.stack(coeffs_list, axis=1)
    lag_samples = np.asarray(lags_list, dtype=int)
    return coeffs, lag_samples


# Backwards-compatible alias
compute_hcr_lags = hcr_coeffs_over_lags
