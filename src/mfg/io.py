"""
I/O utilities for EEG data used in multi-feature Granger-like analysis.

This module provides:
- unified loaders for CSV EEG files and 'mfgin' MATLAB ROI files,
- helpers for extracting continuous signals, per-trial windows,
- a small NPZ saver utility.
"""

from typing import List, Any

import numpy as np
import pandas as pd
import scipy.io as sio

from .config import Config


def load_pair_for_analysis(
    path: str,
    chan_a: str,
    chan_b: str,
    *,
    roi_win_samples: np.ndarray | None = None,
    fs_fallback: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """
    Unified loader of a pair of signals (A, B) from CSV or mfgin .mat.

    Parameters
    ----------
    path : str
        Path to EEG file: CSV or MATLAB .mat with 'mfgin' struct.
    chan_a : str
        Name of the first channel (reason A).
    chan_b : str
        Name of the second channel (result B).
    roi_win_samples : np.ndarray or None, optional
        1-based sample indices WITHIN A TRIAL to keep for ROI .mat files.
        If None and `path` is MAT, the default is the first 64 samples
        (≈0.5 s for fs=128 Hz).
        Ignored for CSV.
    fs_fallback : int or None, optional
        Sampling frequency used if it cannot be read from the file
        (only relevant for CSV). If None, uses Config().fs.

    Returns
    -------
    a_1d : np.ndarray, shape (T_total,)
        Continuous signal for channel A. For CSV: the entire signal.
        For MAT: all selected samples from all trials concatenated in
        correct time order: trial 0, trial 1, ...
    b_1d : np.ndarray, shape (T_total,)
        Continuous signal for channel B, same convention as `a_1d`.
    fs : int
        Sampling frequency in Hz.
    meta : dict
        Dictionary with metadata, including at least:
          - "path": str
          - "ext": "csv" or "mat"
          - "type": "csv" or "roi"
          - "chan_a", "chan_b": str
          - "channel_names": list[str]
          - "n_trials": int
          - "trial_len": int
          - "fs_source": "Config.fs" or "fs_fallback" (for CSV)
        For MAT ("roi") there are additional fields:
          - "a_trials": np.ndarray, shape (n_trials, W)
          - "b_trials": np.ndarray, shape (n_trials, W)
          - "roi_win_samples_1based": list[int]
    """
    ext = path.lower().rsplit(".", 1)[-1]
    meta: dict[str, Any] = {
        "path": path,
        "ext": ext,
        "chan_a": chan_a,
        "chan_b": chan_b,
    }

    # ---------- CSV: single long continuous signal ----------
    if ext == "csv":
        X, ch_names = load_eeg_csv(path, [chan_a, chan_b])
        a = np.asarray(X[:, 0], float)
        b = np.asarray(X[:, 1], float)

        if fs_fallback is None:
            fs = Config().fs
            meta["fs_source"] = "Config.fs"
        else:
            fs = int(fs_fallback)
            meta["fs_source"] = "fs_fallback"

        meta.update(
            {
                "type": "csv",
                "channel_names": ch_names,
                "n_trials": 1,
                "trial_len": int(a.size),
            }
        )
        return a, b, fs, meta

    # ---------- mfgin .mat: ROI with multiple trials ----------
    if ext == "mat":
        # Default ROI window: first 64 samples of each trial (1-based),
        if roi_win_samples is None:
            win_samples = np.arange(1, 65, dtype=int)
        else:
            win_samples = np.asarray(roi_win_samples, dtype=int)

        # X: (T_total, C), T_total = n_trials * W
        X, names, fs, W, n_trials = load_mfgin_mat(path, win_samples=win_samples)

        meta.update(
            {
                "type": "roi",
                "channel_names": names,
                "n_trials": int(n_trials),
                "trial_len": int(W),
                "roi_win_samples_1based": win_samples.tolist(),
            }
        )

        try:
            idx_a = names.index(chan_a)
        except ValueError as exc:
            raise ValueError(
                f"Channel {chan_a!r} not found in ROI file {path!r}; "
                f"available: {names}"
            ) from exc

        try:
            idx_b = names.index(chan_b)
        except ValueError as exc:
            raise ValueError(
                f"Channel {chan_b!r} not found in ROI file {path!r}; "
                f"available: {names}"
            ) from exc

        # Continuous 1D signals in correct time order:
        # trial 0 samples, then trial 1, etc.
        a_1d = np.asarray(X[:, idx_a], float)
        b_1d = np.asarray(X[:, idx_b], float)

        if a_1d.size != n_trials * W or b_1d.size != n_trials * W:
            raise ValueError(
                f"Inconsistent dimensions for ROI file {path!r}: "
                f"expected n_trials * trial_len = {n_trials} * {W} = {n_trials * W}, "
                f"got a_1d.size={a_1d.size}, b_1d.size={b_1d.size}"
            )

        # Per-trial representation expected by mfg_pair_per_trial:
        # shape (n_trials, W)
        a_trials = a_1d.reshape(n_trials, W)
        b_trials = b_1d.reshape(n_trials, W)

        meta["a_trials"] = a_trials
        meta["b_trials"] = b_trials

        return a_1d, b_1d, int(fs), meta

    raise ValueError(f"Unsupported EEG file extension in path: {path!r}")


def load_eeg_csv(
    path: str, channels: List[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Load an EEG CSV file with channel columns.

    Parameters
    ----------
    path : str
        Path to CSV file.
    channels : list[str] or None
        If provided, selects given columns in this order.
        If None, all numeric columns are used.

    Returns
    -------
    X : np.ndarray, shape (T, C)
        Time samples in rows, channels in columns.
    channel_names : list[str]
        Names of the returned columns.
    """
    df = pd.read_csv(path)
    if channels is None:
        candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        X = df[candidates].to_numpy(dtype=float)
        return X, candidates

    X = df[channels].to_numpy(dtype=float)
    return X, channels


def load_mfgin_mat(path: str, win_samples=None):
    """
    Generic loader for MATLAB ROI files with struct 'mfgin'.

    Parameters
    ----------
    path : str
        Path to .mat file containing 'mfgin' struct.
    win_samples : sequence of int or None
        1-based sample indices WITHIN A TRIAL to keep.
        If None, the entire trial length is used.

    Returns
    -------
    X : np.ndarray, shape (T, C)
        Continuous signal: time in rows, channels in columns.
        It is constructed by concatenating all trials one after another.
    names : list[str]
        Channel names: ROI_01, ROI_02, ...
    fs : int
        Sampling frequency (Hz).
    trial_len : int
        Length of the kept window in samples (per trial).
    n_trials : int
        Number of trials.
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "mfgin" not in mat:
        raise KeyError("MAT file does not contain 'mfgin' struct named 'mfgin'")

    S = mat["mfgin"]

    D = np.asarray(S.data, float)  # shape (C, total_samples)
    chans = int(S.chans)  # e.g. 5
    L_total = int(S.len)  # e.g. 129
    fs = int(S.fsamp)  # e.g. 128

    total_samples = D.shape[1]
    if total_samples % L_total != 0:
        raise ValueError(
            f"Data length {total_samples} is not a multiple of trial length {L_total}"
        )
    n_trials = total_samples // L_total

    if chans != D.shape[0]:
        raise ValueError(f"Mismatch between chans={chans} and data shape {D.shape}")

    X_full = D.reshape(chans, n_trials, L_total)  # (C, n_trials, L_total)

    if win_samples is None:
        idx = np.arange(L_total, dtype=int)
    else:
        idx = np.asarray(win_samples, int) - 1  # convert 1-based to 0-based
        idx = idx[(idx >= 0) & (idx < L_total)]
        if idx.size == 0:
            raise ValueError("win_samples produced no valid indices inside a trial")

    W = int(idx.size)

    # Keep only selected window within each trial
    X_trials = X_full[:, :, idx]  # (C, n_trials, W)

    # Build continuous (T, C): trial 0, trial 1, ...
    X = np.transpose(X_trials, (1, 2, 0)).reshape(n_trials * W, chans)

    names = [f"ROI_{i + 1:02d}" for i in range(chans)]

    return X, names, fs, W, n_trials


def load_roi_mat(path: str):
    """
    Convenience wrapper for ROI .mat files.

    Returns
    -------
    X_cont : np.ndarray, shape (C, T)
        Continuous signal: channels x time. Time is obtained by concatenating
        all trials one after another.
    fs : int
        Sampling frequency (Hz).
    X_trials : np.ndarray, shape (C, W, n_trials)
        Fixed-length windows (default: first 64 samples of each trial).
        Axis 0: channels, axis 1: samples within trial, axis 2: trials.
    """
    # Default 1..64 in MATLAB 1-based convention
    win_samples = np.arange(1, 65, dtype=int)

    X, names, fs, W, n_trials = load_mfgin_mat(path, win_samples=win_samples)

    C = len(names)
    T = X.shape[0]  # T = W * n_trials

    # X is (T, C) -> convert to (C, T)
    X_cont = X.T.reshape(C, T)

    # Recover (C, W, n_trials) from continuous representation
    X_trials = X_cont.reshape(C, n_trials, W).transpose(0, 2, 1)

    return X_cont, fs, X_trials


def load_eeg(
    path: str,
    channels: list[str] | None = None,
    win_samples=None,
    fs_fallback: int | None = None,
) -> tuple[np.ndarray, list[str], int]:
    """
    Load EEG data from either CSV or mfgin .mat file.

    Parameters
    ----------
    path : str
        Path to CSV or MAT file.
    channels : list[str] or None
        For CSV only: list of channel names to load. If None, all numeric
        columns are used.
    win_samples : sequence of int or None
        For MAT only: 1-based sample indices within trial to keep.
        If None, the entire trial is used.
    fs_fallback : int or None
        For CSV only: sampling frequency if not specified elsewhere.
        If None, Config().fs is used.

    Returns
    -------
    X : np.ndarray, shape (T, C)
        Time samples in rows, channels in columns.
    channels : list[str]
        Names of the returned channels.
    fs : int
        Sampling frequency.
    """
    ext = path.lower().rsplit(".", 1)[-1]

    if ext == "csv":
        X, ch = load_eeg_csv(path, channels)
        if fs_fallback is None:
            fs_fallback = Config().fs
        return X, ch, int(fs_fallback)

    if ext == "mat":
        X, ch, fs, _, _ = load_mfgin_mat(path, win_samples=win_samples)
        return X, ch, fs

    raise ValueError(f"Unsupported EEG file extension for path: {path!r}")


def save_npz(path: str, **arrays) -> None:
    """
    Save multiple arrays to a compressed .npz file.

    Parameters
    ----------
    path : str
        Target file path.
    arrays : dict
        Named NumPy arrays to save as keyword arguments.
    """
    np.savez_compressed(path, **arrays)
