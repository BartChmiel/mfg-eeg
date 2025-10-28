import numpy as np


def gc_scalar(A_scores: np.ndarray) -> np.ndarray:
    """
    A_scores: shape (r, T) with scores a_v(delta)
    Returns the Granger causality scalar GC(delta) = sqrt(sum_v a_v(delta)^2) -> shape (T,)

    """
    return np.sqrt((A_scores**2).sum(axis=0))
