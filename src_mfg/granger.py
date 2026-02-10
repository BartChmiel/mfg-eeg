import numpy as np


def gc_scalar(scores: np.ndarray) -> np.ndarray:
    """
    Compute Granger-like scalar GC(lag) from PCA features.

    Parameters
    ----------
    scores : np.ndarray, shape (r, T)
        PCA feature time series a_v(lag) for v = 1..r.

    Returns
    -------
    gc : np.ndarray, shape (T,)
        GC(lag) = sqrt( sum_v a_v(lag)^2 ).
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2D array of shape (r, T)")
    return np.sqrt((scores**2).sum(axis=0))
