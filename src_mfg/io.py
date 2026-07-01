from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

# Standard event columns and phases for Kaggle Grasp-and-Lift task
KAGGLE_EVENT_COLS = [
    "HandStart",
    "FirstDigitTouch",
    "BothStartLoadPhase",
    "LiftOff",
    "Replace",
    "BothReleased",
]
KAGGLE_PHASES = [
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
    Loads and aligns EEG data and event labels. Applies Common Average Reference (CAR).
    """
    df_data = pd.read_csv(data_csv_path)
    df_ev = pd.read_csv(events_csv_path)

    if "id" not in df_data.columns or "id" not in df_ev.columns:
        raise ValueError(
            "Data and event files must contain an 'id' column for alignment."
        )

    df_data = df_data.set_index("id")
    df_ev = df_ev.set_index("id").reindex(df_data.index)

    if df_ev.isnull().any().any():
        raise ValueError("Event alignment failed; check for missing IDs in event CSV.")

    channel_names = (
        list(channels)
        if channels
        else [c for c in df_data.columns if pd.api.types.is_numeric_dtype(df_data[c])]
    )
    X = df_data[channel_names].to_numpy(dtype=float)
    E = df_ev[KAGGLE_EVENT_COLS].to_numpy(dtype=np.int8)

    # Apply Common Average Reference (CAR) to reduce global noise
    X -= np.mean(X, axis=1, keepdims=True)

    return df_data.index.to_numpy(), X, E, channel_names, KAGGLE_EVENT_COLS


def load_kaggle_data_only(
    data_csv_path: str,
    channels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads Kaggle EEG data without event labels. Applies Common Average Reference.
    """
    df_data = pd.read_csv(data_csv_path)
    if "id" not in df_data.columns:
        raise ValueError("Data file must contain an 'id' column.")

    df_data = df_data.set_index("id")
    channel_names = (
        list(channels)
        if channels
        else [c for c in df_data.columns if pd.api.types.is_numeric_dtype(df_data[c])]
    )
    X = df_data[channel_names].to_numpy(dtype=float)
    X -= np.mean(X, axis=1, keepdims=True)

    return df_data.index.to_numpy(), X, channel_names


def _blocks_to_centers(mask_01: np.ndarray) -> np.ndarray:
    """
    Identifies contiguous blocks of active events and returns their center indices.
    """
    mask = np.asarray(mask_01, dtype=np.int8) == 1
    if mask.size == 0:
        return np.array([], dtype=np.int64)

    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return ((starts + ends) // 2).astype(np.int64)


def extract_grasp_cycles(
    E: np.ndarray, fs: int, max_cycle_s: float = 5.0
) -> list[dict[str, int]]:
    """
    Reconstructs individual grasp cycles by identifying sequential event centers.
    """
    centers = {
        name: _blocks_to_centers(E[:, i]) for i, name in enumerate(KAGGLE_EVENT_COLS)
    }
    hs = centers["HandStart"]
    max_gap = int(round(max_cycle_s * fs))
    cycles = []

    for k, hs_t in enumerate(hs):
        next_hs = int(hs[k + 1]) if (k + 1) < hs.size else E.shape[0]
        times, cur, ok = {"HandStart": int(hs_t)}, int(hs_t), True

        for name in KAGGLE_EVENT_COLS[1:]:
            arr = centers[name]
            idx = int(np.searchsorted(arr, cur + 1))
            if idx >= arr.size or arr[idx] >= next_hs:
                ok = False
                break
            times[name] = int(arr[idx])
            cur = int(arr[idx])

        if ok and (times["BothReleased"] - times["HandStart"]) <= max_gap:
            cycles.append(times)
    return cycles


def phase_window_bounds(
    T: int, t_start: int, t_end: int, fs: int, epoch_len_s: float
) -> tuple[int, int] | None:
    """
    Calculates fixed-length window bounds centered between two timestamps.
    """
    L = int(round(float(epoch_len_s) * float(fs)))
    center = (int(t_start) + int(t_end)) // 2
    a, b = int(center - (L // 2)), int(center - (L // 2) + L)
    return (a, b) if a >= 0 and b <= T else None


def save_npz(path: str, **arrays) -> None:
    """Saves arrays to a compressed .npz archive."""
    np.savez_compressed(path, **arrays)
