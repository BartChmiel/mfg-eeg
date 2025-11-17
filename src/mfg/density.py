"""
Density reconstruction from HCR polynomial coefficients.

We work in the Legendre-orthonormal basis on [0, 1]:
    fy_j(y), fz_k(z),  j, k = 0..m

Joint density distortion is represented by a coefficient matrix M:

    contrib(y, z) = sum_{j,k} M[j,k] * fy_j(y) * fz_k(z)

In practice we:
- remove marginal contributions (j=0 or k=0),
- either return just the contribution (distortion),
- or return the full model: rho(y,z) ≈ 1 + contrib(y,z),
  clipped to be non-negative and optionally renormalized.
"""

from typing import Tuple

import numpy as np

from .basis import legendre_orthonormal


def coeffs_vec_to_matrix(c: np.ndarray, m: int) -> np.ndarray:
    """
    Reshape a flattened coefficient vector into (m+1, m+1) matrix.

    Parameters
    ----------
    c : np.ndarray, shape (K,)
        Flattened polynomial coefficients, usually K = (m+1)^2.
    m : int
        Maximum polynomial degree in each dimension.

    Returns
    -------
    M : np.ndarray, shape (m+1, m+1)
        Coefficient matrix with M[j, k] corresponding to fy_j * fz_k.
    """
    c = np.asarray(c, float).ravel()
    expected = (m + 1) * (m + 1)
    if c.size != expected:
        raise ValueError(
            f"coeffs_vec_to_matrix: expected {(m + 1)}^2={expected} elements "
            f"for m={m}, got {c.size}"
        )
    return c.reshape((m + 1, m + 1))


def _contrib_from_coeffs(
    M: np.ndarray, m: int, grid_n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute raw contribution f(y) M f(z)^T on a regular [0,1] grid.

    Parameters
    ----------
    M : np.ndarray, shape (m+1, m+1)
        Coefficient matrix for joint distortion.
    m : int
        Maximum polynomial degree in each dimension.
    grid_n : int
        Number of grid points per axis for y,z in [0, 1].

    Returns
    -------
    uy : np.ndarray, shape (grid_n,)
        Grid points for the first variable (y).
    uz : np.ndarray, shape (grid_n,)
        Grid points for the second variable (z).
    contrib : np.ndarray, shape (grid_n, grid_n)
        Distortion contribution evaluated on the grid.
    """
    M = np.asarray(M, float)
    if M.shape != (m + 1, m + 1):
        raise ValueError(
            f"_contrib_from_coeffs: expected M shape {(m + 1, m + 1)}, got {M.shape}"
        )

    u = np.linspace(0.0, 1.0, int(grid_n), endpoint=True)

    # Legendre basis on [0,1]: F(y) and F(z)
    Fy = legendre_orthonormal(u, m)  # (grid_n, m+1)
    Fz = legendre_orthonormal(u, m)  # (grid_n, m+1)

    Ms = M.copy()

    # Remove marginals: keep only j>0, k>0 (we already modeled
    # marginals separately / subtracted them at HCR stage).
    Ms[0, :] = 0.0
    Ms[:, 0] = 0.0

    # contrib(y,z) = Fy(y,:) @ Ms @ Fz(z,:)^T
    contrib = Fy @ Ms @ Fz.T  # (grid_n, grid_n)

    return u, u, contrib


def reconstruct_density_contrib(
    M: np.ndarray,
    m: int,
    grid_n: int = 128,
    clip_q: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct only the distortion contribution on a grid.

    This returns the centered contribution:
        contrib(y,z) = sum_{j>0,k>0} M[j,k] * fy_j(y) * fz_k(z)

    Parameters
    ----------
    M : np.ndarray, shape (m+1, m+1)
        Coefficient matrix (full, including marginals).
    m : int
        Maximum polynomial degree in each dimension.
    grid_n : int, default 128
        Grid resolution per axis.
    clip_q : float or None, default None
        If not None, clip contrib to symmetric quantile of |contrib|.
        Example: clip_q=0.99 -> clip at 99th percentile of |contrib|.

    Returns
    -------
    uy : np.ndarray, shape (grid_n,)
    uz : np.ndarray, shape (grid_n,)
    contrib : np.ndarray, shape (grid_n, grid_n)
        Distortion contribution (can be positive or negative).
    """
    uy, uz, contrib = _contrib_from_coeffs(M, m, grid_n)

    if clip_q is not None:
        clip_q = float(clip_q)
        if not (0.0 < clip_q <= 1.0):
            raise ValueError("clip_q must be in (0,1], got {clip_q}")
        s = float(np.quantile(np.abs(contrib), clip_q))
        if s > 0.0:
            contrib = np.clip(contrib, -s, s)

    return uy, uz, contrib


def reconstruct_density_model(
    M: np.ndarray,
    m: int,
    grid_n: int = 128,
    renormalize: bool = True,
    clip_q: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct full modeled joint density on a grid.

    We approximate:
        rho(y,z) ≈ 1 + contrib(y,z),
    where contrib comes from the HCR coefficients with marginal
    terms removed (j>0, k>0). We enforce non-negativity and
    optionally renormalize and clip.

    Parameters
    ----------
    M : np.ndarray, shape (m+1, m+1)
        Coefficient matrix (full, including marginals).
    m : int
        Maximum polynomial degree in each dimension.
    grid_n : int, default 128
        Grid resolution per axis.
    renormalize : bool, default True
        If True, divide rho by its mean so that average density ≈ 1
        on the uniform grid (approx. proper normalization).
    clip_q : float or None, default None
        If not None, clip rho to [q_lo, q_hi] where:
            q_lo = quantile(rho, 1-clip_q),
            q_hi = quantile(rho, clip_q).
        Example: clip_q=0.99 -> keep central 98% mass.

    Returns
    -------
    uy : np.ndarray, shape (grid_n,)
    uz : np.ndarray, shape (grid_n,)
    rho : np.ndarray, shape (grid_n, grid_n)
        Reconstructed density, rho >= 0.
    """
    uy, uz, contrib = _contrib_from_coeffs(M, m, grid_n)

    # Base uniform density 1 + distortion
    rho = 1.0 + contrib
    rho = np.maximum(rho, 0.0)

    if renormalize:
        mean_val = float(rho.mean() or 1.0)
        rho = rho / mean_val

    if clip_q is not None:
        clip_q = float(clip_q)
        if not (0.0 < clip_q <= 1.0):
            raise ValueError(f"clip_q must be in (0,1], got {clip_q}")
        q_lo = float(np.quantile(rho, 1.0 - clip_q))
        q_hi = float(np.quantile(rho, clip_q))
        rho = np.clip(rho, q_lo, q_hi)

    return uy, uz, rho
