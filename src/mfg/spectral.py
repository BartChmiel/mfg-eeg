from typing import Tuple

import numpy as np


def fft_over_lags(
    curve: np.ndarray,
    lag_samples: np.ndarray,
    fs: float,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FFT of a scalar feature defined over a lag grid.

    Parameters
    ----------
    curve : np.ndarray, shape (T,)
        Scalar feature as a function of lag, e.g. PC1(lag) for A -> B.
    lag_samples : np.ndarray, shape (T,)
        Lags in samples corresponding to the entries of `curve`.
        Expected to be non-negative and monotonically increasing, but
        only the average step size is used.
    fs : float
        Sampling frequency of the underlying EEG signal in Hz.
        It is used to convert lag samples into seconds, which defines
        the effective sampling interval for FFT.
    detrend : bool, default True
        If True, subtract the mean of `curve` before FFT, removing
        the DC component.

    Returns
    -------
    freqs : np.ndarray, shape (T_fft,)
        Frequency grid in Hz for the lag-domain FFT.
    mag : np.ndarray, shape (T_fft,)
        Magnitude spectrum |FFT(curve)| at these frequencies.
    phase : np.ndarray, shape (T_fft,)
        Phase spectrum arg(FFT(curve)) in radians.

    Notes
    -----
    - This is an FFT over the lag axis, not over raw EEG time.
      It describes how the *shape of dependence vs lag* is modulated
      across different time scales.
    - We use rFFT because `curve` is real-valued, which returns
      non-negative frequencies only.
    """
    curve = np.asarray(curve, float)
    lag_samples = np.asarray(lag_samples, float)

    if curve.ndim != 1:
        raise ValueError(f"fft_over_lags expects 1D curve, got shape {curve.shape}")
    if lag_samples.ndim != 1 or lag_samples.shape[0] != curve.shape[0]:
        raise ValueError(
            "fft_over_lags expects lag_samples to be 1D and the same length "
            f"as curve, got lag_samples.shape={lag_samples.shape}"
        )

    if curve.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    # Effective sampling interval along the lag axis, in seconds.
    if lag_samples.size >= 2:
        # Average step in samples -> divide by fs to get seconds.
        step_samples = float(np.mean(np.diff(lag_samples)))
        dt = step_samples / float(fs) if step_samples > 0.0 else 1.0 / float(fs)
    else:
        dt = 1.0 / float(fs)

    x = curve.copy()
    if detrend:
        x -= float(x.mean())

    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt)
    mag = np.abs(spec)
    phase = np.angle(spec)

    return freqs, mag, phase
