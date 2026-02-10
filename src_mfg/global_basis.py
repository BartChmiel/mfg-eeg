from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PCABasis:
    """
    Standardized PCA basis container for cross-phase connectivity comparison.
    """

    mean: np.ndarray  # (K,) - Mean coefficient vector
    V: np.ndarray  # (K, K) - Eigenvectors (columns)
    eigvals: np.ndarray  # (K,) - Explained variance
    lag_samples: np.ndarray  # (L,) - Lag grid in samples

    fs: int
    maxlag_ms: int
    lag_step_ms: int
    m: int
    mixed_only: bool

    chan_a: str
    chan_b: str
    mode: str
    epoch_len_s: float


def pca_from_second_moments(
    sum_x: np.ndarray, sum_xxT: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes PCA components from accumulated first and second moments.

    Implementation Details:
    1. Centering: Derives the global mean from sum_x and total sample count n.
    2. Covariance: Calculates the (K x K) covariance matrix using the formula:
       C = (ΣxxT - n*μμT) / (n - 1).
    3. Symmetry: Explicitly enforces C = 0.5 * (C + C.T) to mitigate numerical drift.
    4. Eigendecomposition: Returns sorted eigenvalues and corresponding eigenvectors.
    """
    sum_x = np.asarray(sum_x, float).ravel()
    sum_xxT = np.asarray(sum_xxT, float)
    K = sum_x.shape[0]

    if sum_xxT.shape != (K, K):
        raise ValueError(f"sum_xxT shape mismatch: expected ({K}, {K})")
    if n <= 0:
        raise ValueError("Sample count n must be positive")

    mean = sum_x / float(n)

    if n == 1:
        return mean, np.eye(K), np.zeros(K)

    # Calculate unbiased sample covariance
    C = (sum_xxT - float(n) * np.outer(mean, mean)) / float(n - 1)
    C = 0.5 * (C + C.T)

    eigvals, V = np.linalg.eigh(C)

    # Sort by descending variance
    idx = np.argsort(eigvals)[::-1]
    return mean, V[:, idx], eigvals[idx]


def project_coeffs(coeffs: np.ndarray, basis: PCABasis, r: int) -> np.ndarray:
    """
    Projects a coefficient trajectory (K, L) onto the top-r components of the basis.

    Args:
        coeffs: np.ndarray - Input HCR coefficients over lags.
        basis: PCABasis - The global PCA basis to project onto.
        r: int - Number of principal components to retain.

    Returns:
        np.ndarray - Projected scores of shape (r, L).
    """
    X_centered = np.asarray(coeffs, float) - basis.mean[:, None]
    scores = basis.V.T @ X_centered
    return scores[: int(r)]
