"""
Global PCA basis for comparing phases/windows for a fixed electrode pair.

We build PCA on many coefficient-vectors (e.g. HCR mixed-only vectors over lags
collected across multiple phase windows). This produces ONE basis that can be
used to project per-phase mean coefficient trajectories, making PC curves
comparable across phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PCABasis:
    mean: np.ndarray  # (K,)
    V: np.ndarray  # (K, K) columns = eigenvectors
    eigvals: np.ndarray  # (K,)
    lag_samples: np.ndarray  # (L,)

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
    Compute PCA from accumulated moments over samples x (shape (K,)).

    Inputs:
      sum_x   = Σ x
      sum_xxT = Σ x x^T
      n       = number of samples

    Returns:
      mean (K,), V (K,K), eigvals (K,)
    """
    sum_x = np.asarray(sum_x, float).ravel()
    sum_xxT = np.asarray(sum_xxT, float)

    K = sum_x.shape[0]
    if sum_xxT.shape != (K, K):
        raise ValueError(f"sum_xxT must be (K,K) with K={K}, got {sum_xxT.shape}")
    if n <= 0:
        raise ValueError("n must be positive")

    mean = sum_x / float(n)

    if n == 1:
        V = np.eye(K, dtype=float)
        eigvals = np.zeros(K, dtype=float)
        return mean, V, eigvals

    # Cov = (Σ xx^T - n μ μ^T) / (n-1)
    C = (sum_xxT - float(n) * np.outer(mean, mean)) / float(n - 1)
    C = 0.5 * (C + C.T)  # enforce symmetry

    eigvals, V = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]
    return mean, V, eigvals


def project_coeffs(coeffs: np.ndarray, basis: PCABasis, r: int) -> np.ndarray:
    """
    Project coefficient trajectory coeffs (K,L) onto basis -> (r,L).
    """
    coeffs = np.asarray(coeffs, float)
    X = coeffs - basis.mean[:, None]
    A = basis.V.T @ X
    return A[: int(r)]
