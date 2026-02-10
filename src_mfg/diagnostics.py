"""
Diagnostic helpers for lag-based analysis (e.g. Pearson correlation over lags).
"""

import numpy as np


def pearson_over_lags(
    y: np.ndarray,
    z: np.ndarray,
    fs: int,
    maxlag_ms: int,
    step_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlation between y(t) and z(t + lag) for non-negative lags.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        First time series (reason).
    z : np.ndarray, shape (T,)
        Second time series (result). Must have the same length as y.
    fs : int
        Sampling frequency in Hz.
    maxlag_ms : int
        Maximum lag in milliseconds to consider (non-negative).
    step_ms : int
        Step in milliseconds between consecutive lags.

    Returns
    -------
    corr : np.ndarray, shape (L,)
        Pearson correlation values for each valid lag (L >= 0).
        corr[i] corresponds to lag_samples[i] samples.
    lag_samples : np.ndarray, shape (L,)
        Lags in samples (integer), non-negative and strictly increasing.

    Notes
    -----
    For each lag L >= 0 we compute correlation between y[:n-L] and z[L:].
    The loop stops once the effective number of pairs n-L becomes <= 1.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    if y.shape != z.shape:
        raise ValueError(
            f"y and z must have the same shape, got {y.shape} and {z.shape}"
        )

    n = len(y)

    # Candidate lags in samples, starting from 0 ms
    step_samples = max(1, int(round(step_ms * fs / 1000.0)))
    maxlag_samples = int(round(maxlag_ms * fs / 1000.0))
    lag_samples_full = np.arange(0, maxlag_samples + 1, step_samples, dtype=int)

    corr_list = []
    lags_list = []

    for L in lag_samples_full:
        n_eff = n - L
        if n_eff <= 1:
            # Not enough samples left to estimate correlation
            break

        a = y[:n_eff]
        b = z[L:]

        # Standardize each segment to zero mean and unit variance
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)

        corr_list.append(float((a * b).mean()))
        lags_list.append(int(L))

    corr = np.asarray(corr_list, dtype=float)
    lag_samples = np.asarray(lags_list, dtype=int)

    return corr, lag_samples
