from __future__ import annotations
import numpy as np
from scipy.stats import norm, t, rankdata
from statsmodels.tsa.tsatools import lagmat


def _clip01(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Clips values to the interval [eps, 1-eps] to ensure numerical stability.
    """
    return np.clip(np.asarray(u, float), eps, 1.0 - eps)


def normalize_gauss(x: np.ndarray) -> np.ndarray:
    """
    Maps signal to (0, 1) using Z-score transformation and Gaussian CDF.
    """
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError(f"normalize_gauss expects 1D input, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("normalize_gauss: input contains NaN/inf")

    mu, sd = np.mean(x), np.std(x)
    if sd <= 1e-12:
        return np.full_like(x, 0.5, dtype=float)

    z = (x - mu) / (sd + 1e-12)
    return _clip01(norm.cdf(z))


def normalize_edf(x: np.ndarray) -> np.ndarray:
    """
    Maps signal to (0, 1) using Empirical Distribution Function (average ranks).
    """
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError(f"normalize_edf expects 1D input, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("normalize_edf: input contains NaN/inf")

    ranks = rankdata(x, method="average")
    return _clip01(ranks / (len(x) + 1.0))


def pnorm_student(
    x: np.ndarray,
    fs: int,
    ar_order: int = 10,
    ema_half_life_s: float = 1.0,
    student_nu: int = 10,
) -> np.ndarray:
    """
    Normalizes signal using AR-whitening, EMA variance, and Student-t CDF mapping.

    Processing Steps:
    1. AR(p) Modeling: Fits an Auto-Regressive model to capture temporal dependencies.
    2. Residual Extraction: Computes whitening residuals (prediction errors).
    3. EMA Variance: Estimates local residual energy using an Exponential Moving Average.
    4. Probabilistic Mapping: Maps standardized residuals to (0, 1) via Student-t CDF.
    """
    x = np.asarray(x, float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("pnorm_student: invalid 1D finite input required")
    if fs <= 0:
        raise ValueError(f"pnorm_student: fs must be positive, got {fs}")

    if ar_order < 1:
        return _clip01(normalize_edf(x))
    if len(x) <= ar_order:
        return np.full_like(x, 0.5, dtype=float)

    # AR(p) Least Squares Fit
    X_lags = lagmat(x, maxlag=ar_order, trim="both")
    y = x[ar_order:]
    X_design = np.c_[np.ones(len(X_lags)), X_lags]

    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    res = y - (X_design @ beta)

    # Dynamic variance estimation (EMA)
    alpha = 1.0 - np.exp(-np.log(2.0) / (max(ema_half_life_s, 1e-6) * float(fs)))
    s2 = np.empty_like(res)
    s2[0] = max(np.var(res), 1.0)

    for i in range(1, len(res)):
        s2[i] = (1.0 - alpha) * s2[i - 1] + alpha * (res[i - 1] ** 2)

    # Student-t CDF transformation
    u_tail = t.cdf(res / (np.sqrt(s2) + 1e-9), df=int(student_nu))
    return _clip01(np.r_[np.full(ar_order, 0.5), u_tail])
