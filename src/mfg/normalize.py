import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.tsatools import lagmat


def normalize_gauss(x: np.ndarray) -> np.ndarray:
    """Map x to U(0,1) via Gaussian CDF of (x-mean)/stddev"""

    x = np.asarray(x, float)
    mu, sd = np.mean(x), np.std(x) + 1e-12

    return norm.cdf((x - mu) / sd)


def pnorm_student(
    x: np.ndarray, fs: int, ar_order=10, ema_half_life_s: float = 1.0, student_nu=10
):
    """
    p-normalization of B: remove predictable mean/scale and map to U(0,1) via Student's t CDF.
    Simple AR(p) + EMA scale
    """

    x = np.asarray(x, float)

    X = lagmat(x, maxlag=ar_order, trim="both")
    y = x[ar_order:]

    Phi = np.c_[np.ones(len(X)), X]
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    pred = Phi @ w
    res = y - pred

    alpha = 1 - np.exp(-np.log(2) / (ema_half_life_s * fs))
    s2 = np.empty_like(res)
    s2[0] = np.var(res)

    for i in range(1, len(res)):
        s2[i] = (1 - alpha) * s2[i - 1] + alpha * res[i - 1] ** 2
    s = np.sqrt(s2) + 1e-9

    u = t.cdf(res / s, df=student_nu)

    u = np.r_[np.full(ar_order, 0.5), u]

    return u
