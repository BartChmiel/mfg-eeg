from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Config:
    """
    Global immutable configuration for MFG EEG analysis.

    Attributes:
        fs: Sampling frequency in Hz.
        maxlag_ms: Maximum lag for HCR/Granger analysis.
        lag_step_ms: Resolution of the lag grid.
        m: Maximum degree for Legendre polynomial basis.
        subtract_marginals: Whether to center cross-moments by marginals.
        ar_order: Order of the Auto-Regressive model for whitening.
        ema_half_life_s: Half-life for Exponential Moving Average variance.
        student_nu: Degrees of freedom for Student-t CDF mapping.
        pca_var_thresh: Cumulative variance threshold for PCA component selection.
        pca_max_r: Hard upper limit on the number of retained PCA components.
    """

    fs: int = 500
    maxlag_ms: int = 1000
    lag_step_ms: int = 2
    m: int = 10
    subtract_marginals: bool = True
    ar_order: int = 10
    ema_half_life_s: float = 1.0
    student_nu: int = 10
    pca_var_thresh: float = 0.90
    pca_max_r: int = 4


def load_config(**overrides: Any) -> Config:
    """
    Factory function to initialize Config with optional parameter overrides.
    """
    base_params = asdict(Config())
    base_params.update(overrides)
    return Config(**base_params)
