import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.tsatools import lagmat


def normalize_gauss(x: np.ndarray) -> np.ndarray:
    """Map x to U(0,1) via Gaussian CDF of (x-mean)/stddev"""

    x = np.asarray(x, float)
    mu, sd = np.mean(x), np.std(x) + 1e-12
    return norm.cdf((x - mu) / sd)


def normalize_edf(x: np.ndarray) -> np.ndarray:
    """Map x to U(0,1) by empirical CDF (rank transform)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks / (len(x) + 1.0)


def pnorm_student(
    x: np.ndarray,
    fs: int,
    ar_order: int = 10,
    ema_half_life_s: float = 1.0,
    student_nu: int = 10,
) -> np.ndarray:
    """
    Predict x from its AR(p), take residuals, scale by EMA variance,
    then map by Student-t CDF -> U(0,1). First `ar_order` samples are 0.5.
    """
    x = np.asarray(x, float)

    X = lagmat(x, maxlag=ar_order, trim="both")  # shape (T-p, p)
    y = x[ar_order:]

    X1 = np.c_[np.ones(len(X)), X]

    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    res = y - yhat

    half_life = max(ema_half_life_s, 1e-6)
    alpha = 1 - np.exp(-np.log(2) / (half_life * fs))
    s2 = np.empty_like(res)
    s2[0] = res.var() if len(res) else 1.0
    for i in range(1, len(res)):
        s2[i] = (1 - alpha) * s2[i - 1] + alpha * res[i - 1] ** 2
    s = np.sqrt(s2) + 1e-9

    u = t.cdf(res / s, df=int(student_nu))

    return np.r_[np.full(ar_order, 0.5), u]
