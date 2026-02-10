from typing import Tuple
import numpy as np


def pca_over_lags(coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs PCA on HCR coefficient trajectories.

    Implementation Details:
    1. Centering: Removes the mean per coefficient across all lags to ensure zero-mean distribution.
    2. Covariance: Computes the (K x K) covariance matrix from centered trajectories.
    3. Eigendecomposition: Uses np.linalg.eigh for stable decomposition of the symmetric matrix.
    4. Sorting: Orders components by descending eigenvalues (explained variance).

    Args:
        coeffs: np.ndarray - Input array of shape (K, T), where K is the number
                             of coefficients and T is the number of lags.

    Returns:
        V: np.ndarray - Eigenvectors (Principal Components) of shape (K, K).
        A: np.ndarray - Scores (projections) of shape (K, T).
        eigvals: np.ndarray - Eigenvalues representing variance per component.
    """
    coeffs = np.asarray(coeffs, float)
    if coeffs.ndim != 2:
        raise ValueError(f"pca_over_lags: expected (K, T) array, got {coeffs.shape}")

    K, T = coeffs.shape
    if T < 1:
        return np.eye(K), np.zeros((K, 0)), np.zeros(K)

    # Center data across lags
    X = coeffs - coeffs.mean(axis=1, keepdims=True)

    # Compute covariance and decompose
    C = (X @ X.T / float(T - 1)) if T > 1 else np.zeros((K, K))
    eigvals, V = np.linalg.eigh(C)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals, V = eigvals[idx], V[:, idx]

    # Project data to score space
    A = V.T @ X

    return V, A, eigvals


def select_r(eigvals: np.ndarray, var_thresh: float = 0.9, max_r: int = 6) -> int:
    """
    Determines the number of principal components to retain based on explained variance.

    Args:
        eigvals: np.ndarray - Sorted eigenvalues from PCA.
        var_thresh: float - Target fraction of cumulative explained variance (0-1).
        max_r: int - Hard upper bound on the number of components.

    Returns:
        r: int - Optimal number of components (1 <= r <= max_r).
    """
    eigvals = np.asarray(eigvals, float)
    if eigvals.size == 0 or np.sum(eigvals) <= 0:
        return min(max_r, eigvals.size)

    # Find smallest r satisfying the variance threshold
    cvar = np.cumsum(eigvals) / np.sum(eigvals)
    r = int(np.searchsorted(cvar, float(var_thresh))) + 1

    return max(1, min(r, max_r, eigvals.size))
