"""
PCA over HCR coefficient trajectories across lags.

Given HCR coefficients over lags:
    coeffs[k, t],  k = 0..K-1, t = 0..T-1

we perform PCA in the K-dimensional coefficient space, using lags t
as "samples". This yields:

- eigenvectors V[:, i] describing orthogonal directions in coefficient space,
- scores A[i, t] = projection of coeffs at lag t onto component i,
- eigenvalues eigvals[i] = variance explained by component i.
"""

from typing import Tuple

import numpy as np


def pca_over_lags(coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on coefficient trajectories over lags.

    Parameters
    ----------
    coeffs : np.ndarray, shape (K, T)
        HCR coefficients over lags:
            K = (m + 1)^2  (flattened polynomial coefficients),
            T = number of lags.
        Each column corresponds to one lag, each row to one coefficient.

    Returns
    -------
    V : np.ndarray, shape (K, K)
        Eigenvectors in coefficient space. Column i is the i-th principal
        component direction.
    A : np.ndarray, shape (K, T)
        Scores over lags: A = V.T @ X, where
            X = coeffs - mean(coeffs, axis=1, keepdims=True)
        Row i contains the time course (over lags) of component i.
    eigvals : np.ndarray, shape (K,)
        Eigenvalues (variances) for each component, sorted descending.

    Notes
    -----
    - This is standard covariance-based PCA:
          C = X X^T / (T - 1),
      with eigen-decomposition C v_i = eigvals[i] v_i.
    """
    coeffs = np.asarray(coeffs, float)

    if coeffs.ndim != 2:
        raise ValueError(
            f"pca_over_lags expects a 2D array (K, T), got shape {coeffs.shape}"
        )

    K, T = coeffs.shape
    if T < 1:
        # Degenerate case: no lags
        V = np.eye(K, dtype=float)
        A = np.zeros((K, 0), dtype=float)
        eigvals = np.zeros(K, dtype=float)
        return V, A, eigvals

    # Center across lags: remove mean per coefficient (row-wise)
    X = coeffs - coeffs.mean(axis=1, keepdims=True)

    if T == 1:
        # With a single sample, covariance degenerates: all variance is zero.
        C = np.zeros((K, K), dtype=float)
    else:
        C = X @ X.T / float(T - 1)

    # Symmetric covariance -> eigh
    eigvals, V = np.linalg.eigh(C)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # Scores: projection of centered data on eigenvectors
    A = V.T @ X  # shape (K, T)

    return V, A, eigvals


def select_r(eigvals: np.ndarray, var_thresh: float = 0.9, max_r: int = 6) -> int:
    """
    Select the number of principal components to keep.

    The rule is:
      - compute cumulative explained variance,
      - take the smallest r such that cumulative variance >= var_thresh,
      - cap r by max_r.

    Parameters
    ----------
    eigvals : np.ndarray, shape (K,)
        Eigenvalues from PCA, sorted in descending order.
    var_thresh : float, default 0.9
        Target fraction of explained variance (between 0 and 1).
    max_r : int, default 6
        Hard upper bound on the number of components.

    Returns
    -------
    r : int
        Number of components to use (1 <= r <= min(K, max_r)).

    Notes
    -----
    - If all eigenvalues are zero or negative (numerical edge case),
      we fall back to r = min(K, max_r).
    """
    eigvals = np.asarray(eigvals, float)

    if eigvals.ndim != 1:
        raise ValueError(
            f"select_r expects a 1D array of eigenvalues, got shape {eigvals.shape}"
        )

    K = eigvals.size
    if K == 0:
        return 0

    total = float(eigvals.sum())
    if total <= 0.0:
        # No meaningful variance; all components are equivalent.
        return min(max_r, K)

    cvar = np.cumsum(eigvals) / total
    # np.searchsorted returns index where var_thresh would be inserted
    # to keep order -> first index with cvar >= var_thresh
    idx = int(np.searchsorted(cvar, float(var_thresh)))
    r = idx + 1  # components are 1-based in this sense

    r = max(1, min(r, max_r, K))
    return r
