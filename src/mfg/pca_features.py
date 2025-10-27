import numpy as np


def pca_over_lags(coeffs: np.ndarray):
    """
    coeffs: (K,T) where K=(m+1)^2 and T=n_lags
    Return eigenvectors V (K,r), scores A (r,T), eigvals lamvda_ (r,)
    """

    X = coeffs - coeffs.mean(axis=1, keepdims=True)
    C = X @ X.T / (X.shape[1] - 1)
    eigvals, V = np.linalg.eigh(C)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]
    A = V.T @ X

    return V, A, eigvals


def select_r(eigvals, var_thresh=0.9, max_r=6):

    cvar = np.cumsum(eigvals) / eigvals.sum()
    r = min(max_r, int(np.searchsorted(cvar, var_thresh) + 1))

    return r
