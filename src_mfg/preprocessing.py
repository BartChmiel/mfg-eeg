from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

PREPROCESSING_CHOICES = ("car_only", "bandpass_0_5_48", "ocular_proxy_regression")

CHANNEL_SETS: dict[str, list[str] | str | None] = {
    "all32": None,
    "motor_premotor": ["F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "no_fp1_fp2": "exclude_fp1_fp2",
    "occipital_visual": ["PO9", "O1", "Oz", "O2", "PO10"],
    "ocular_proxy_only": ["Fp1", "Fp2"],
}


def _safe_zscore(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def bandpass_0_5_48(X: np.ndarray, fs: int) -> np.ndarray:
    """Band-pass filter aligned with standard EEG cleaning (0.5-48 Hz at 500 Hz)."""
    if X.shape[0] < 16:
        return X
    high = min(48.0, 0.45 * float(fs))
    low = min(0.5, high * 0.5)
    sos = butter(4, [low, high], btype="bandpass", fs=float(fs), output="sos")
    padlen = min(max(X.shape[0] - 1, 0), 3 * (2 * len(sos) + 1))
    return sosfiltfilt(sos, X, axis=0, padlen=padlen)


def regress_ocular_proxies(
    X: np.ndarray,
    *,
    all_channels: Sequence[str],
    proxy_source: np.ndarray | None = None,
    proxy_names: Sequence[str] = ("Fp1", "Fp2"),
) -> np.ndarray:
    """Remove linear Fp1/Fp2 contribution (EOG-proxy regression without extra electrodes)."""
    names = [name for name in proxy_names if name in all_channels]
    if not names:
        return X
    source = proxy_source if proxy_source is not None else X
    proxy_idx = [all_channels.index(name) for name in names]
    proxies = _safe_zscore(source[:, proxy_idx])
    beta, *_ = np.linalg.lstsq(proxies, X, rcond=None)
    return X - proxies @ beta


def apply_preprocessing(
    X: np.ndarray,
    *,
    preprocess: str,
    fs: int,
    all_channels: Sequence[str],
    X_all: np.ndarray | None = None,
) -> np.ndarray:
    if preprocess not in PREPROCESSING_CHOICES:
        raise ValueError(f"Unknown preprocessing: {preprocess}")
    if preprocess == "car_only":
        return X
    if preprocess == "bandpass_0_5_48":
        return bandpass_0_5_48(X, fs)
    proxy_source = X_all if X_all is not None else X
    return regress_ocular_proxies(
        X,
        all_channels=all_channels,
        proxy_source=proxy_source,
    )


def resolve_channel_names(
    channel_set: str,
    all_channels: Sequence[str],
) -> list[str]:
    if channel_set not in CHANNEL_SETS:
        raise ValueError(f"Unknown channel set: {channel_set}")
    spec = CHANNEL_SETS[channel_set]
    if spec is None:
        return list(all_channels)
    if spec == "exclude_fp1_fp2":
        return [name for name in all_channels if name not in {"Fp1", "Fp2"}]
    available = set(all_channels)
    missing = [name for name in spec if name not in available]
    if missing:
        raise ValueError(f"Channel set {channel_set!r} is missing channels: {missing}")
    return list(spec)
