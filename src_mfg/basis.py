import numpy as np


def legendre_orthonormal(x: np.ndarray, m: int) -> np.ndarray:
    """
    Orthonormal Legendre basis on [0, 1].

        f_j(x) = sqrt(2j+1) * P_j(2x - 1),   j = 0..m

    where P_j is the standard Legendre polynomial on [-1, 1].

    Returns F with shape (n, m+1), where F[i, j] = f_j(x[i]).
    Input is flattened to 1D.
    """
    if m < 0:
        raise ValueError(f"legendre_orthonormal: m must be >= 0, got {m}")

    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return np.empty((0, m + 1), dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("legendre_orthonormal: input contains NaN/inf")

    # Optional: enforce theory domain (usually satisfied because normalize_* clips to (0,1))
    # if np.min(x) < 0.0 or np.max(x) > 1.0:
    #     raise ValueError("legendre_orthonormal: x must be in [0,1]")

    u = 2.0 * x - 1.0  # map [0,1] -> [-1,1]
    n = x.size

    # Compute P_0..P_m via recurrence:
    # P_0(u)=1
    # P_1(u)=u
    # (k+1)P_{k+1}(u) = (2k+1)uP_k(u) - kP_{k-1}(u)
    P = np.empty((n, m + 1), dtype=float)
    P[:, 0] = 1.0
    if m >= 1:
        P[:, 1] = u
    for k in range(1, m):
        P[:, k + 1] = ((2.0 * k + 1.0) * u * P[:, k] - k * P[:, k - 1]) / (k + 1.0)

    # Orthonormal scaling on [0,1]
    scales = np.sqrt(2.0 * np.arange(m + 1) + 1.0)  # (m+1,)
    F = P * scales[None, :]
    return F
