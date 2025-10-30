import numpy as np
from .basis import legendre_orthonormal


def coeffs_vec_to_matrix(c: np.ndarray, m: int) -> np.ndarray:
    return np.asarray(c, float).reshape((m + 1, m + 1))


def _contrib_from_coeffs(
    M: np.ndarray, m: int, grid_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, grid_n)
    Fy = legendre_orthonormal(u, m)
    Fz = legendre_orthonormal(u, m)
    Ms = M.copy()
    # remove marginals (we modeled with subtract_marginals=True)
    Ms[0, :] = 0.0
    Ms[:, 0] = 0.0
    contrib = Fy @ Ms @ Fz.T
    return u, u, contrib


def reconstruct_density_contrib(
    M: np.ndarray, m: int, grid_n: int = 128, clip_q: float | None = None
):
    """Return only the contribution f(y) M f(z)^T centered around 0."""
    u, v, contrib = _contrib_from_coeffs(M, m, grid_n)
    if clip_q is not None:
        s = np.quantile(np.abs(contrib), clip_q)
        if s > 0:
            contrib = np.clip(contrib, -s, s)
    return u, v, contrib


def reconstruct_density_model(
    M: np.ndarray,
    m: int,
    grid_n: int = 128,
    renormalize: bool = True,
    clip_q: float | None = None,
):
    """Return full modeled density: rho ~= 1 + contribution, clipped to be non-negative and renormalized."""
    u, v, contrib = _contrib_from_coeffs(M, m, grid_n)
    rho = 1.0 + contrib
    rho = np.maximum(rho, 0.0)
    if renormalize:
        rho = rho / rho.mean()  # integral≈mean on uniform grid
    if clip_q is not None:
        lo, hi = np.quantile(rho, 1.0 - clip_q), np.quantile(rho, clip_q)
        rho = np.clip(rho, lo, hi)
    return u, v, rho
