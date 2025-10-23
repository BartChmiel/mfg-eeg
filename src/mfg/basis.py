import numpy as np
from numpy.polynomial.legendre import legval


def legendre_orthonormal(x: np.ndarray, m: int) -> np.ndarray:
    """
    Return matrix F of shape (len(x), m+1) with orthonormal Legendre basis on [0, 1].
    f_j(x) = sqrt((2j+1)/2) * P_j(2x - 1), where P_j is the j-th Legendre polynomial.

    """

    x_scaled = 2.0 * x - 1.0  # Scale x from [0, 1] to [-1, 1]
    n = x.shape[0]
    F = np.empty((n, m + 1), dtype=float)

    for j in range(m + 1):
        F[:, j] = np.sqrt((2 * j + 1) / 2) * legval(x_scaled, [0] * j + [1])

    return F
