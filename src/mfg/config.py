from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
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


def load_config(**overrides) -> Config:
    cfg = Config(**{**Config().__dict__, **overrides})
    return cfg
