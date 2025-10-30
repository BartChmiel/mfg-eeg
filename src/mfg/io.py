import numpy as np
import pandas as pd
from typing import List, Tuple


def load_eeg_csv(
    path: str, channels: List[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Load EEG CSV with channel columns; retrurnc (T,C), channel_names
    """
    df = pd.read_csv(path)
    if channels is None:
        candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        X = df[candidates].to_numpy(dtype=float)
        return X, candidates
    X = df[channels].to_numpy(dtype=float)
    return X, channels


def save_npz(path: str, **arrays) -> None:
    """
    Save multiple arrays to npz file
    """
    np.savez_compressed(path, **arrays)
