import numpy as np


def gc_scalar(scores: np.ndarray) -> np.ndarray:
    """
    Computes a Granger-like scalar connectivity metric from PCA feature scores.

    Implementation Details:
    1. Input Validation: Ensures scores are provided in a 2D (r, T) format,
       where r is the number of components and T is the number of lags.
    2. Vector Norm: Calculates the L2 norm across components for each lag:
       GC(lag) = sqrt( Σ a_v(lag)^2 ).

    Args:
        scores: np.ndarray - PCA projection scores of shape (r, T).

    Returns:
        gc: np.ndarray - Scalar connectivity curve of shape (T,).
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2:
        raise ValueError("gc_scalar: scores must be a 2D array of shape (r, T)")

    # Calculate Euclidean norm across the component axis (axis 0)
    return np.sqrt(np.sum(scores**2, axis=0))
