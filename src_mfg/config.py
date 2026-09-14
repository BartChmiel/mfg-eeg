from dataclasses import dataclass, asdict, fields
from pathlib import Path
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


def load_config_yaml(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load Config from configs/default.yaml unless another path is given."""
    yaml_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    params = asdict(Config())
    if yaml_path.exists():
        import yaml

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            flat = _flatten_yaml_config(raw)
            allowed = {field.name for field in fields(Config)}
            params.update({key: flat[key] for key in flat if key in allowed})
    params.update(overrides)
    return Config(**params)


def _flatten_yaml_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map nested YAML keys to Config field names."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "pnorm" and isinstance(value, dict):
            if "ar_order" in value:
                out["ar_order"] = value["ar_order"]
            if "ema_half_life_s" in value:
                out["ema_half_life_s"] = value["ema_half_life_s"]
            if "student_nu" in value:
                out["student_nu"] = value["student_nu"]
            continue
        if key == "features_r":
            out["pca_max_r"] = value
            continue
        out[key] = value
    return out
