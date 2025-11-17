"""
Normalization utilities for EEG time series.

This module provides three main 1D normalizations:

- normalize_gauss:
    x -> (x - mean) / std, then mapped by standard Gaussian CDF
    to approximately U(0,1). This corresponds to the basic

- normalize_edf:
    x -> empirical CDF via ranks, without assuming any parametric
    distribution. Also maps to approximately U(0,1).

- pnorm_student:
    time-series "p-normalization"
    Granger-causality section:
       * predict x from its AR(p) past,
       * take residuals,
       * scale them using an EMA-based variance estimate
         (with a specified half-life in seconds),
       * map residuals by Student-t CDF to U(0,1).
    The first `ar_order` samples are set to 0.5, as they lack
    enough past for AR prediction.
"""

from typing import Tuple

import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.tsatools import lagmat


def normalize_gauss(x: np.ndarray) -> np.ndarray:
    """
    Map a real-valued time series to approximately U(0,1) using a Gaussian CDF.

    For input x, we compute:
        z = (x - mean(x)) / std(x),
        u = Phi(z),
    where Phi is the CDF of N(0, 1).

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input time series.

    Returns
    -------
    u : np.ndarray, shape (T,)
        Normalized series, approximately uniform on (0,1) if x is close to Gaussian.
    """
    x = np.asarray(x, float)
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-12
    z = (x - mu) / sd
    return norm.cdf(z)


def normalize_edf(x: np.ndarray) -> np.ndarray:
    """
    Map a real-valued time series to U(0,1) via empirical CDF (rank transform).

    Each sample is replaced by its rank (1..n), then rescaled to (0,1).

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input time series.

    Returns
    -------
    u : np.ndarray, shape (T,)
        Empirical-CDF normalized series, approximately uniform on (0,1),
        invariant to any strictly monotone transform of x.
    """
    x = np.asarray(x, float)

    # Stable sort to make the transform deterministic in presence of ties
    order = np.argsort(x, kind="mergesort")

    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    # Slightly shrunk to avoid putting any sample exactly at 0 or 1
    return ranks / (len(x) + 1.0)


def pnorm_student(
    x: np.ndarray,
    fs: int,
    ar_order: int = 10,
    ema_half_life_s: float = 1.0,
    student_nu: int = 10,
) -> np.ndarray:
    """
    Predict x from its AR(p) past, scale residuals with EMA variance, then
    map by Student-t CDF to U(0,1).

    This is a simplified "p-normalization" step used for density-based
    Granger-like causality. Intuition:

        1. Fit AR(ar_order) with OLS on x[t] using its past values.
        2. Residuals res[t] = x[t] - x_hat[t] capture unpredictable part.
        3. Estimate a time-varying variance via exponential moving average
           of res^2, using a half-life in seconds (ema_half_life_s).
        4. Standardize residuals: z[t] = res[t] / s[t].
        5. Map z[t] by Student-t CDF with df = student_nu, giving u[t] in (0,1).

    The first `ar_order` samples are set to 0.5 because they do not have
    enough past for AR regression.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input time series.
    fs : int
        Sampling frequency in Hz.
    ar_order : int, default 10
        AR model order (number of past samples used).
    ema_half_life_s : float, default 1.0
        Half-life in seconds for the EMA of residual variance.
    student_nu : int, default 10
        Degrees of freedom for the Student-t distribution.

    Returns
    -------
    u : np.ndarray, shape (T,)
        P-normalized series, approximately i.i.d. U(0,1) under the model.
    """
    x = np.asarray(x, float)

    if x.ndim != 1:
        raise ValueError(f"pnorm_student expects 1D input, got shape {x.shape}")
    if len(x) <= ar_order:
        # Degenerate case: not enough data for AR order.
        # Fall back to a flat 0.5 sequence.
        return np.full_like(x, 0.5, dtype=float)

    # Build lag matrix: each row contains x[t-1], ..., x[t-p]
    X = lagmat(x, maxlag=ar_order, trim="both")  # shape (T - p, p)
    y = x[ar_order:]  # shape (T - p,)

    # Add intercept term
    X_design = np.c_[np.ones(len(X)), X]  # shape (T - p, p+1)

    # OLS fit
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    res = y - y_hat

    # EMA of residual variance with given half-life in seconds
    half_life = max(float(ema_half_life_s), 1e-6)
    alpha = 1.0 - np.exp(-np.log(2.0) / (half_life * float(fs)))

    s2 = np.empty_like(res)
    if len(res) > 0:
        s2[0] = float(res.var()) if res.var() > 0 else 1.0
    else:
        s2[0] = 1.0

    for i in range(1, len(res)):
        s2[i] = (1.0 - alpha) * s2[i - 1] + alpha * (res[i - 1] ** 2)

    s = np.sqrt(s2) + 1e-9

    # Map standardized residuals by Student-t CDF
    u_tail = t.cdf(res / s, df=int(student_nu))

    # Prepend 0.5 for the first ar_order samples (no AR prediction)
    u = np.r_[np.full(ar_order, 0.5), u_tail]
    return u
