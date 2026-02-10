import numpy as np


def pearson_over_lags(
    y: np.ndarray,
    z: np.ndarray,
    fs: int,
    maxlag_ms: int,
    step_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Pearson correlation between y(t) and z(t + lag) for non-negative lags.

    Implementation Details:
    1. Lag Grid: Calculates sample-based offsets from millisecond parameters and sampling frequency.
    2. Iterative Calculation: Shifts signal z relative to y for each lag in the grid.
    3. Standardization: Individually centers and scales segments at each lag to ensure valid correlation coefficients.
    4. Adaptive Termination: Ceases iteration if the remaining signal length is too short for meaningful analysis.

    Args:
        y: np.ndarray - Source signal (reason).
        z: np.ndarray - Destination signal (result).
        fs: int - Sampling frequency in Hz.
        maxlag_ms: int - Maximum lag to evaluate.
        step_ms: int - Resolution of the lag grid.

    Returns:
        corr: np.ndarray - Pearson correlation coefficients per lag.
        lag_samples: np.ndarray - Corresponding lag offsets in samples.
    """
    y, z = np.asarray(y, float), np.asarray(z, float)
    if y.shape != z.shape:
        raise ValueError(f"Shape mismatch: y {y.shape} and z {z.shape} must match")

    n = len(y)
    step_samples = max(1, int(round(step_ms * fs / 1000.0)))
    maxlag_samples = int(round(maxlag_ms * fs / 1000.0))
    lag_indices = np.arange(0, maxlag_samples + 1, step_samples, dtype=int)

    corr_list, lags_list = [], []

    for L in lag_indices:
        n_eff = n - L
        if n_eff <= 1:
            break

        # Extract and standardize overlapping segments
        a, b = y[:n_eff], z[L:]
        a_std = (a - a.mean()) / (a.std() + 1e-12)
        b_std = (b - b.mean()) / (b.std() + 1e-12)

        corr_list.append(float((a_std * b_std).mean()))
        lags_list.append(int(L))

    return np.asarray(corr_list, dtype=float), np.asarray(lags_list, dtype=int)
