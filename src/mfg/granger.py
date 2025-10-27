import numpy as np


def gc_scalar(A_scores: np.ndarray):
    """
    A_scores: (r,T) -> GC(T) = ssqrt(sum_v a_v(delta)^2)

    """
    return np.sqrt(np.sum(A_scores**2).sum(axis=0))
