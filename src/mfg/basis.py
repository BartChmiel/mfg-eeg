import numpy as np
from numpy.polynomial.legendre import legval


def legendre_orthonormal(x: np.ndarray, m: int) -> np.ndarray:
    """
    Orthonormal Legendre basis on [0, 1].

    We use the standard construction:
        f_j(x) = sqrt(2j+1) * P_j(2x - 1),   j = 0..m,
    where P_j is the j-th (non-normalized) Legendre polynomial on [-1, 1].

    This choice satisfies:
        ∫_0^1 f_j(x) f_k(x) dx = δ_{jk}.

    Parameters
    ----------
    x : array_like, shape (n,) or any shape broadcastable to 1D
        Points in [0, 1] at which the basis is evaluated.
    m : int
        Maximum degree of the Legendre basis (m >= 0).

    Returns
    -------
    F : np.ndarray, shape (n, m+1)
        Matrix of basis evaluations:
            F[i, j] = f_j(x[i]),  j = 0..m.
    """
    if m < 0:
        raise ValueError(f"legendre_orthonormal: m must be >= 0, got {m}")

    # Ensure 1D float array
    x = np.asarray(x, float).ravel()
    n = x.shape[0]

    # Map [0,1] → [-1,1] for Legendre polynomials
    x_scaled = 2.0 * x - 1.0

    F = np.empty((n, m + 1), dtype=float)

    # P_j(x_scaled) = legval(x_scaled, [0,...,0,1]) with j zeros then 1
    for j in range(m + 1):
        coeffs = [0.0] * j + [1.0]
        F[:, j] = np.sqrt(2 * j + 1) * legval(x_scaled, coeffs)

    return F
