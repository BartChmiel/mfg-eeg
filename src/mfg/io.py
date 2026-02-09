"""
I/O utilities for EEG data used in multi-feature Granger-like analysis.

This module focuses on Kaggle Grasp-and-Lift EEG:
- data+events alignment by 'id'
- robust reconstruction of grasp cycles from Kaggle event windows
- fixed-length phase window bounds + slicing helpers
- simple CSV loader + NPZ saver
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


# =============================================================================
# Basic CSV loader
# =============================================================================


def load_eeg_csv(
    path: str, channels: List[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Load an EEG CSV file with channel columns.

    If channels is None -> loads all numeric columns.
    If channels is provided -> loads selected columns in given order.
    """
    df = pd.read_csv(path)

    if channels is None:
        candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        X = df[candidates].to_numpy(dtype=float)
        return X, candidates

    X = df[channels].to_numpy(dtype=float)
    return X, channels


def save_npz(path: str, **arrays) -> None:
    """Save multiple arrays to a compressed .npz file."""
    np.savez_compressed(path, **arrays)


# =============================================================================
# Kaggle Grasp-and-Lift helpers (data + events)
# =============================================================================

KAGGLE_EVENT_COLS = [
    "HandStart",
    "FirstDigitTouch",
    "BothStartLoadPhase",
    "LiftOff",
    "Replace",
    "BothReleased",
]

KAGGLE_PHASES: list[tuple[str, str]] = [
    ("HandStart", "FirstDigitTouch"),
    ("FirstDigitTouch", "BothStartLoadPhase"),
    ("BothStartLoadPhase", "LiftOff"),
    ("LiftOff", "Replace"),
    ("Replace", "BothReleased"),
]


def load_kaggle_data_with_events(
    data_csv_path: str,
    events_csv_path: str,
    channels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load Kaggle EEG series (data + events) aligned by 'id'.

    Returns
    -------
    ids : (T,)
    X : (T, C) float
    E : (T, 6) int8  in order KAGGLE_EVENT_COLS
    channel_names : list[str]
    event_names : list[str]
    """
    if channels is None:
        df_data = pd.read_csv(data_csv_path)
        if "id" not in df_data.columns:
            raise ValueError(f"Expected 'id' column in {data_csv_path!r}")
        channel_names = [
            c
            for c in df_data.columns
            if c != "id" and pd.api.types.is_numeric_dtype(df_data[c])
        ]
    else:
        channel_names = list(channels)
        usecols = ["id"] + channel_names
        df_data = pd.read_csv(data_csv_path, usecols=usecols)

    usecols_e = ["id"] + KAGGLE_EVENT_COLS
    df_ev = pd.read_csv(events_csv_path, usecols=usecols_e)

    if "id" not in df_data.columns or "id" not in df_ev.columns:
        raise ValueError("Both files must contain 'id' column.")

    # Align by id (events can be safely reindexed to data ids)
    df_data = df_data.set_index("id")
    df_ev = df_ev.set_index("id").reindex(df_data.index)

    if df_ev.isnull().any().any():
        missing = df_ev[df_ev.isnull().any(axis=1)].index[:5].tolist()
        raise ValueError(
            "Events could not be aligned to data by id. "
            f"Example missing ids: {missing}"
        )

    ids = df_data.index.to_numpy()
    X = df_data[channel_names].to_numpy(dtype=float)
    E = df_ev[KAGGLE_EVENT_COLS].to_numpy(dtype=np.int8)

    X -= np.mean(X, axis=1, keepdims=True)

    return ids, X, E, channel_names, list(KAGGLE_EVENT_COLS)


def _blocks_to_centers(mask_01: np.ndarray) -> np.ndarray:
    """
    Kaggle labels are 1 within a time window around the event => contiguous blocks of 1s.
    Convert each block into a single timestamp: the block center index.
    """
    mask = np.asarray(mask_01, dtype=np.int8) == 1
    if mask.size == 0:
        return np.array([], dtype=np.int64)

    # Rising/falling edges of the boolean mask
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1  # inclusive end

    centers = ((starts + ends) // 2).astype(np.int64)
    return centers


def _event_centers_from_E(E: np.ndarray) -> Dict[str, np.ndarray]:
    """Return centers per event column as sorted int arrays."""
    if E.ndim != 2 or E.shape[1] != len(KAGGLE_EVENT_COLS):
        raise ValueError("E must be (T, 6) in KAGGLE_EVENT_COLS order.")
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(KAGGLE_EVENT_COLS):
        out[name] = _blocks_to_centers(E[:, i])
    return out


def extract_grasp_cycles(
    E: np.ndarray,
    fs: int,
    max_cycle_s: float = 5.0,
) -> list[dict[str, int]]:
    """
    Reconstruct grasp cycles from Kaggle frame-wise event windows.

    Key design:
    - Use *centers of 1-blocks* (neutral timestamp inside Kaggle ±150ms window).
    - Anchor each cycle at HandStart center.
    - Hard boundary: next HandStart center defines the end of current cycle search.
    - Also enforce max_cycle_s to reject weird long matches.

    Returns list of dicts: {event_name: frame_index}
    """
    centers = _event_centers_from_E(E)
    hs = centers["HandStart"]
    if hs.size == 0:
        return []

    max_gap = int(round(max_cycle_s * fs))
    T = int(E.shape[0])

    cycles: list[dict[str, int]] = []

    for k, hs_t in enumerate(hs):
        hs_t = int(hs_t)
        next_hs = int(hs[k + 1]) if (k + 1) < hs.size else T

        times: dict[str, int] = {"HandStart": hs_t}
        cur = hs_t

        ok = True
        for name in KAGGLE_EVENT_COLS[1:]:
            arr = centers[name]
            j = int(np.searchsorted(arr, cur + 1))
            if j >= arr.size:
                ok = False
                break

            t = int(arr[j])
            # Must belong to this cycle (before next HandStart)
            if t >= next_hs:
                ok = False
                break

            times[name] = t
            cur = t

        if not ok:
            continue

        # Optional duration sanity
        if (times["BothReleased"] - times["HandStart"]) > max_gap:
            continue

        cycles.append(times)

    return cycles


def phase_window_bounds(
    T: int,
    t_start: int,
    t_end: int,
    fs: int,
    epoch_len_s: float,
) -> tuple[int, int] | None:
    """
    Return (a,b) bounds of a fixed-length window centered in [t_start, t_end).

    Guarantees b-a == round(epoch_len_s * fs).
    Returns None if window would go outside [0, T).
    """
    if t_end <= t_start:
        return None

    L = int(round(float(epoch_len_s) * float(fs)))
    if L <= 0:
        return None

    center = (int(t_start) + int(t_end)) // 2
    a = int(center - (L // 2))
    b = int(a + L)

    if a < 0 or b > int(T):
        return None

    return a, b


def slice_phase_window(
    X: np.ndarray,
    t_start: int,
    t_end: int,
    fs: int,
    epoch_len_s: float = 2.0,
    *,
    align: str = "center",  # "center" | "start" | "end"
    clip_to_phase: bool = False,
) -> np.ndarray:
    """
    Extract a window for a phase. By default returns a fixed-length window.

    align="center": window centered in [t_start, t_end)
    align="start" : window starts at t_start
    align="end"   : window ends at t_end

    If clip_to_phase=True, the window is forced to stay within [t_start, t_end).
    WARNING: this may shorten the window and should NOT be used for PCA/basis comparability.
    """
    T = int(X.shape[0])
    C = int(X.shape[1])
    if t_end <= t_start:
        return np.empty((0, C), dtype=float)

    L = int(round(float(epoch_len_s) * float(fs)))
    if L <= 0:
        return np.empty((0, C), dtype=float)

    align = str(align).lower().strip()
    if align not in ("center", "start", "end"):
        raise ValueError(f"slice_phase_window: invalid align={align!r}")

    if align == "center":
        center = (int(t_start) + int(t_end)) // 2
        a = int(center - (L // 2))
        b = int(a + L)
    elif align == "start":
        a = int(t_start)
        b = int(a + L)
    else:  # "end"
        b = int(t_end)
        a = int(b - L)

    if clip_to_phase:
        a = max(a, int(t_start))
        b = min(b, int(t_end))

    # Clip to signal bounds
    a = max(a, 0)
    b = min(b, T)

    if b <= a:
        return np.empty((0, C), dtype=float)

    return np.asarray(X[a:b], float)
