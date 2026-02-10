"""
Normalization utilities for EEG time series.

Provides:
- normalize_gauss: Gaussian CDF mapping to (0,1)
- normalize_edf: empirical CDF (rank) mapping to (0,1)
- pnorm_student: AR(p) residual -> EMA variance -> Student-t CDF mapping to (0,1)

IMPORTANT:
We clip outputs away from exact 0 and 1 to avoid numerical issues in
Legendre basis evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t, rankdata
from statsmodels.tsa.tsatools import lagmat


def _clip01(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(u, float), eps, 1.0 - eps)


def normalize_gauss(x: np.ndarray) -> np.ndarray:
    """
    x -> z-score -> Gaussian CDF -> (0,1)
    """
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError(f"normalize_gauss expects 1D input, got shape {x.shape}")

    if not np.all(np.isfinite(x)):
        raise ValueError("normalize_gauss: input contains NaN/inf")

    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-12:
        return np.full_like(x, 0.5, dtype=float)

    z = (x - mu) / (sd + 1e-12)
    return _clip01(norm.cdf(z))


def normalize_edf(x: np.ndarray) -> np.ndarray:
    """
    x -> ranks -> (0,1) via empirical CDF.
    Uses average ranks for ties to avoid time-order artifacts.
    """
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError(f"normalize_edf expects 1D input, got shape {x.shape}")

    if not np.all(np.isfinite(x)):
        raise ValueError("normalize_edf: input contains NaN/inf")

    # ranks in {1..n}, average ranks for ties
    ranks = rankdata(x, method="average")
    u = ranks / (len(x) + 1.0)
    return _clip01(u)


def pnorm_student(
    x: np.ndarray,
    fs: int,
    ar_order: int = 10,
    ema_half_life_s: float = 1.0,
    student_nu: int = 10,
) -> np.ndarray:
    """
    AR(p) one-step prediction -> residuals -> EMA variance -> Student-t CDF -> (0,1).

    Notes:
    - First ar_order samples are set to 0.5 (no AR prediction possible).
    - EMA variance uses past residual energy (res[i-1]^2) to avoid look-ahead.
    """
    x = np.asarray(x, float)

    if x.ndim != 1:
        raise ValueError(f"pnorm_student expects 1D input, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("pnorm_student: input contains NaN/inf")
    if fs <= 0:
        raise ValueError(f"pnorm_student: fs must be positive, got {fs}")
    if ar_order < 1:
        return _clip01(normalize_edf(x))

    if len(x) <= ar_order:
        return np.full_like(x, 0.5, dtype=float)

    X = lagmat(x, maxlag=ar_order, trim="both")  # (T-p, p)
    y = x[ar_order:]  # (T-p,)

    X_design = np.c_[np.ones(len(X)), X]  # (T-p, p+1)

    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    res = y - y_hat

    half_life = max(float(ema_half_life_s), 1e-6)
    alpha = 1.0 - np.exp(-np.log(2.0) / (half_life * float(fs)))

    s2 = np.empty_like(res)
    v0 = float(np.var(res))
    s2[0] = v0 if v0 > 0 else 1.0

    for i in range(1, len(res)):
        s2[i] = (1.0 - alpha) * s2[i - 1] + alpha * (res[i - 1] ** 2)

    s = np.sqrt(s2) + 1e-9
    u_tail = t.cdf(res / s, df=int(student_nu))

    u = np.r_[np.full(ar_order, 0.5), u_tail]
    return _clip01(u)
