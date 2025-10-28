import numpy as np
from numpy.polynomial.legendre import legval


def legendre_orthonormal(x: np.ndarray, m: int) -> np.ndarray:
    """
    Orthonormal Legendre basis functions on [0, 1]
    f_j(x) = sqrt(2j+1) * P_j(2x-1) for j=0..m
    -> integral_0^1 f_j(x) f_k(x) dx = delta_jk

    """

    x_scaled = 2.0 * x - 1.0
    n = x.shape[0]
    F = np.empty((n, m + 1), dtype=float)

    for j in range(m + 1):
        F[:, j] = np.sqrt(2 * j + 1) * legval(x_scaled, [0] * j + [1])

    return F
