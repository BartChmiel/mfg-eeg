import numpy as np


def legendre_orthonormal(x: np.ndarray, m: int) -> np.ndarray:
    """
    Computes an orthonormal Legendre polynomial basis on the interval [0, 1].

    Implementation Details:
    1. Input validation: Ensures m >= 0 and x contains only finite float values.
    2. Domain Mapping: Linearly maps x from [0, 1] to the standard Legendre domain [-1, 1].
    3. Recursion: Uses the three-term recurrence relation to compute polynomials P_0 through P_m.
    4. Orthonormality: Applies a sqrt(2j + 1) scaling factor for L2 orthonormality on [0, 1].

    Args:
        x: np.ndarray - Normalized input signal (expected range [0, 1]).
        m: int - Maximum polynomial degree.

    Returns:
        np.ndarray - Matrix of shape (n, m+1) containing basis function evaluations.
    """
    if m < 0:
        raise ValueError(f"legendre_orthonormal: m must be >= 0, got {m}")

    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return np.empty((0, m + 1), dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("legendre_orthonormal: input contains NaN/inf")

    # Map [0, 1] -> [-1, 1]
    u = 2.0 * x - 1.0
    n = x.size

    # Compute P_0..P_m via recurrence
    P = np.empty((n, m + 1), dtype=float)
    P[:, 0] = 1.0
    if m >= 1:
        P[:, 1] = u
    for k in range(1, m):
        P[:, k + 1] = ((2.0 * k + 1.0) * u * P[:, k] - k * P[:, k - 1]) / (k + 1.0)

    # Apply orthonormal scaling
    scales = np.sqrt(2.0 * np.arange(m + 1) + 1.0)
    return P * scales[None, :]
