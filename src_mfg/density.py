from typing import Tuple
import numpy as np
from .basis import legendre_orthonormal


def coeffs_vec_to_matrix(c: np.ndarray, m: int) -> np.ndarray:
    """
    Reshapes a flattened (K,) coefficient vector back into an (m+1, m+1) matrix.
    """
    expected = (m + 1) ** 2
    if c.size != expected:
        raise ValueError(f"Shape mismatch: expected {expected}, got {c.size}")
    return c.reshape((m + 1, m + 1))


def _compute_grid_contribution(
    M: np.ndarray, m: int, grid_n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core vectorised reconstruction of the density contribution on a [0, 1] grid.
    """
    u = np.linspace(0.0, 1.0, grid_n)
    Fy = legendre_orthonormal(u, m)
    Fz = legendre_orthonormal(u, m)

    Ms = M.copy()
    Ms[0, :] = 0.0  # Remove marginal effects (degree 0)
    Ms[:, 0] = 0.0

    # Reconstruct: rho_contrib = Fy * M * Fz.T
    contrib = Fy @ Ms @ Fz.T
    return u, u, contrib


def reconstruct_density_model(
    M: np.ndarray, m: int, grid_n: int = 128, renormalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs the full joint density model: rho(y,z) ≈ 1 + contrib(y,z).
    """
    uy, uz, contrib = _compute_grid_contribution(M, m, grid_n)
    rho = 1.0 + contrib
    rho = np.maximum(rho, 0.0)  # Enforce non-negativity

    if renormalize:
        rho /= rho.mean() if rho.mean() > 0 else 1.0

    return uy, uz, rho
