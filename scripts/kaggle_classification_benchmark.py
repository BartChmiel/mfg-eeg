from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from src_mfg.basis import legendre_orthonormal
from src_mfg.config import load_config_yaml
from src_mfg.io import KAGGLE_EVENT_COLS, load_kaggle_data_only, load_kaggle_data_with_events
from src_mfg.normalize import normalize_edf, normalize_gauss, pnorm_student
from src_mfg.preprocessing import (
    CHANNEL_SETS,
    PREPROCESSING_CHOICES,
    apply_preprocessing,
    resolve_channel_names,
)


_RX = re.compile(
    r"subj(?P<subj>\d+)_series(?P<series>\d+)_data\.csv$",
    re.IGNORECASE,
)

EVENT_PHASE_MAP = {
    "HandStart": ["HandStart__FirstDigitTouch"],
    "FirstDigitTouch": [
        "HandStart__FirstDigitTouch",
        "FirstDigitTouch__BothStartLoadPhase",
    ],
    "BothStartLoadPhase": [
        "FirstDigitTouch__BothStartLoadPhase",
        "BothStartLoadPhase__LiftOff",
    ],
    "LiftOff": ["BothStartLoadPhase__LiftOff", "LiftOff__Replace"],
    "Replace": ["LiftOff__Replace", "Replace__BothReleased"],
    "BothReleased": ["Replace__BothReleased"],
}

CLASSIFIER_CHOICES = ("sgd_logistic", "extra_trees")
FEATURE_SET_CHOICES = ("baseline", "mfg", "combined")
SUBMISSION_FEATURE_SET_CHOICES = (*FEATURE_SET_CHOICES, "fusion")
BASELINE_CONTEXT_CHOICES = ("causal", "centered")


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    dst: str
    lag_ms: int
    rank: int
    phase: str = ""


@dataclass
class FeatureBlock:
    baseline: np.ndarray
    mfg: np.ndarray
    labels: np.ndarray
    sample_index: np.ndarray


def _parse_int_list(values: Sequence[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            if token:
                out.append(int(token))
    return out or None


def _parse_int_values(values: Sequence[str] | None, default: Sequence[int]) -> list[int]:
    parsed = _parse_int_list(values)
    return list(default) if parsed is None else parsed


def _parse_subj_series(path: Path) -> tuple[int | None, int | None]:
    match = _RX.search(path.name)
    if match is None:
        return None, None
    return int(match.group("subj")), int(match.group("series"))


def _events_path_for_data(data_path: Path) -> Path:
    return data_path.with_name(data_path.name.replace("_data.csv", "_events.csv"))


def _iter_data_files(root: Path) -> list[Path]:
    train_dir = root / "train" if (root / "train").is_dir() else root
    return sorted(train_dir.glob("subj*_series*_data.csv"), key=_file_sort_key)


def _file_sort_key(path: Path) -> tuple[int, int, str]:
    subj, se = _parse_subj_series(path)
    return (
        subj if subj is not None else 10_000,
        se if se is not None else 10_000,
        path.name,
    )


def _select_files(
    files: Sequence[Path],
    *,
    subjects: Sequence[int] | None,
    series: Sequence[int] | None,
    max_files: int | None,
    require_events: bool = True,
) -> list[Path]:
    selected: list[Path] = []
    for path in files:
        subj, se = _parse_subj_series(path)
        if subj is None or se is None:
            continue
        if subjects is not None and subj not in subjects:
            continue
        if series is not None and se not in series:
            continue
        if require_events and not _events_path_for_data(path).exists():
            continue
        selected.append(path)
        if max_files is not None and len(selected) >= max_files:
            break
    return selected


def _read_data_channel_names(data_path: Path) -> list[str]:
    header = pd.read_csv(data_path, nrows=0)
    return [name for name in header.columns if name != "id"]


def _select_channel_set(all_channels: Sequence[str], channel_set: str) -> list[str]:
    return resolve_channel_names(channel_set, all_channels)


def _channel_indices(all_channels: Sequence[str], selected_channels: Sequence[str]) -> list[int]:
    index = {name: idx for idx, name in enumerate(all_channels)}
    return [index[name] for name in selected_channels]


def _safe_zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)


def _prepare_kaggle_data_with_events(
    data_path: Path,
    *,
    channel_set: str,
    preprocess: str,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    ids, X_all, E, all_channels, event_cols = load_kaggle_data_with_events(
        str(data_path),
        str(_events_path_for_data(data_path)),
    )
    selected_channels = _select_channel_set(all_channels, channel_set)
    X = X_all[:, _channel_indices(all_channels, selected_channels)]
    X = apply_preprocessing(
        X,
        preprocess=preprocess,
        fs=fs,
        X_all=X_all,
        all_channels=all_channels,
    )
    return ids, X, E, selected_channels, event_cols


def _prepare_kaggle_data_only(
    data_path: Path,
    *,
    channel_set: str,
    preprocess: str,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids, X_all, all_channels = load_kaggle_data_only(str(data_path))
    selected_channels = _select_channel_set(all_channels, channel_set)
    X = X_all[:, _channel_indices(all_channels, selected_channels)]
    X = apply_preprocessing(
        X,
        preprocess=preprocess,
        fs=fs,
        X_all=X_all,
        all_channels=all_channels,
    )
    return ids, X, selected_channels


def _basis_mixed(u_1d: np.ndarray, m: int) -> np.ndarray:
    return legendre_orthonormal(u_1d, m)[:, 1:]


def _ema_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(arr, float)
    if arr.shape[0] == 0:
        return arr.copy()
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    one_minus = 1.0 - float(alpha)
    for idx in range(1, arr.shape[0]):
        out[idx] = float(alpha) * arr[idx] + one_minus * out[idx - 1]
    return out


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    x2 = np.asarray(x, float) ** 2
    csum = np.cumsum(np.vstack([np.zeros((1, x2.shape[1])), x2]), axis=0)
    starts = np.maximum(np.arange(x2.shape[0]) + 1 - window, 0)
    ends = np.arange(x2.shape[0]) + 1
    counts = (ends - starts).astype(float)[:, None]
    mean_x2 = (csum[ends] - csum[starts]) / counts
    return np.sqrt(mean_x2 + 1e-12)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    x = np.asarray(x, float)
    csum = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    starts = np.maximum(np.arange(x.shape[0]) + 1 - window, 0)
    ends = np.arange(x.shape[0]) + 1
    counts = (ends - starts).astype(float)[:, None]
    return (csum[ends] - csum[starts]) / counts


def _centered_window_bounds(n_samples: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(int(window), 1)
    left = window // 2
    right = window - left
    centers = np.arange(n_samples)
    starts = np.maximum(centers - left, 0)
    ends = np.minimum(centers + right, n_samples)
    return starts, ends


def _centered_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    x = np.asarray(x, float)
    starts, ends = _centered_window_bounds(x.shape[0], window)
    csum = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    counts = (ends - starts).astype(float)[:, None]
    return (csum[ends] - csum[starts]) / counts


def _centered_rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    x2 = np.asarray(x, float) ** 2
    starts, ends = _centered_window_bounds(x2.shape[0], window)
    csum = np.cumsum(np.vstack([np.zeros((1, x2.shape[1])), x2]), axis=0)
    counts = (ends - starts).astype(float)[:, None]
    mean_x2 = (csum[ends] - csum[starts]) / counts
    return np.sqrt(mean_x2 + 1e-12)


def _lagged_matrix(x: np.ndarray, lag_samples: int) -> np.ndarray:
    if lag_samples <= 0:
        return x
    out = np.zeros_like(x)
    out[lag_samples:] = x[:-lag_samples]
    return out


def _lead_matrix(x: np.ndarray, lead_samples: int) -> np.ndarray:
    if lead_samples <= 0:
        return x
    out = np.zeros_like(x)
    out[:-lead_samples] = x[lead_samples:]
    return out


def _make_baseline_features(
    X: np.ndarray,
    fs: int,
    *,
    windows_ms: Sequence[int],
    lags_ms: Sequence[int],
    context: str,
) -> np.ndarray:
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    z = (X - mu) / sd
    dz = np.vstack([np.zeros((1, z.shape[1])), np.diff(z, axis=0)])
    blocks = [z, dz]

    for lag_ms in lags_ms:
        lag_samples = int(round(float(lag_ms) * fs / 1000.0))
        if lag_samples > 0:
            blocks.append(_lagged_matrix(z, lag_samples))
            if context == "centered":
                blocks.append(_lead_matrix(z, lag_samples))

    for window_ms in windows_ms:
        window = max(1, int(round(float(window_ms) * fs / 1000.0)))
        blocks.append(_rolling_mean(z, window))
        blocks.append(_rolling_rms(z, window))
        if context == "centered":
            blocks.append(_centered_rolling_mean(z, window))
            blocks.append(_centered_rolling_rms(z, window))

    return np.hstack(blocks).astype(np.float32)


def _normalize_for_mfg(
    X: np.ndarray,
    *,
    mode: str,
    fs: int,
    ar_order: int,
    ema_half_life_s: float,
    student_nu: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_dim, c_dim = X.shape
    Y = np.empty((t_dim, c_dim), dtype=float)
    Z = np.empty((t_dim, c_dim), dtype=float)

    if mode == "corr":
        for c_idx in range(c_dim):
            u = normalize_gauss(X[:, c_idx])
            Y[:, c_idx] = u
            Z[:, c_idx] = u
        return Y, Z

    for c_idx in range(c_dim):
        Y[:, c_idx] = normalize_edf(X[:, c_idx])
        Z[:, c_idx] = pnorm_student(
            X[:, c_idx],
            fs,
            ar_order=ar_order,
            ema_half_life_s=ema_half_life_s,
            student_nu=student_nu,
        )
    return Y, Z


def _shift_source(Fy_src: np.ndarray, lag_samples: int) -> np.ndarray:
    if lag_samples <= 0:
        return Fy_src
    shifted = np.zeros_like(Fy_src)
    shifted[lag_samples:] = Fy_src[:-lag_samples]
    return shifted


def _make_mfg_features(
    X: np.ndarray,
    ch_names: Sequence[str],
    edges: Sequence[EdgeSpec],
    *,
    mode: str,
    fs: int,
    m: int,
    ema_half_life_s: float,
    ar_order: int,
    student_nu: int,
    feature_mode: str,
) -> np.ndarray:
    if not edges:
        return np.zeros((X.shape[0], 0), dtype=np.float32)

    Y, Z = _normalize_for_mfg(
        X,
        mode=mode,
        fs=fs,
        ar_order=ar_order,
        ema_half_life_s=ema_half_life_s,
        student_nu=student_nu,
    )
    Fy = np.stack([_basis_mixed(Y[:, idx], m) for idx in range(Y.shape[1])], axis=1)
    Fz = np.stack([_basis_mixed(Z[:, idx], m) for idx in range(Z.shape[1])], axis=1)

    ch_to_idx = {name: idx for idx, name in enumerate(ch_names)}
    alpha = 1.0 - np.exp(-np.log(2.0) / (max(ema_half_life_s, 1e-6) * float(fs)))
    features: list[np.ndarray] = []

    for edge in edges:
        if edge.src not in ch_to_idx or edge.dst not in ch_to_idx:
            continue
        src_idx = ch_to_idx[edge.src]
        dst_idx = ch_to_idx[edge.dst]
        lag_samples = max(0, int(round(edge.lag_ms * fs / 1000.0)))

        source = _shift_source(Fy[:, src_idx, :], lag_samples)
        target = Fz[:, dst_idx, :]
        cross = source[:, :, None] * target[:, None, :]
        coeff = _ema_filter(cross, alpha)
        coeff -= _ema_filter(source, alpha)[:, :, None] * _ema_filter(target, alpha)[:, None, :]
        energy = np.sqrt(np.sum(coeff * coeff, axis=(1, 2)))
        features.append(np.log1p(energy).astype(np.float32))
        if feature_mode == "expanded":
            flat = coeff.reshape(coeff.shape[0], -1)
            signed_mean = flat.mean(axis=1)
            max_abs = np.max(np.abs(flat), axis=1)
            features.append(signed_mean.astype(np.float32))
            features.append(np.log1p(max_abs).astype(np.float32))
        elif feature_mode != "energy":
            raise ValueError(f"Unknown MFG feature mode: {feature_mode}")

    if not features:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    return np.column_stack(features).astype(np.float32)


def _sample_indices(
    labels: np.ndarray,
    *,
    stride: int,
    max_samples: int | None,
    keep_all_positive: bool,
) -> np.ndarray:
    base = np.arange(0, labels.shape[0], max(int(stride), 1), dtype=np.int64)
    if not keep_all_positive:
        if max_samples is None or base.size <= max_samples:
            return base
        return base[np.linspace(0, base.size - 1, max_samples).round().astype(int)]

    positive = np.flatnonzero(np.any(labels > 0, axis=1)).astype(np.int64)
    positive_set = set(map(int, positive))
    negative = np.array([idx for idx in base if int(idx) not in positive_set], dtype=np.int64)
    if max_samples is None:
        return np.sort(np.r_[positive, negative].astype(np.int64))

    if positive.size + negative.size <= max_samples:
        return np.sort(np.r_[positive, negative].astype(np.int64))

    min_negative = min(negative.size, max(1, int(round(max_samples * 0.25)))) if negative.size else 0
    positive_budget = max(max_samples - min_negative, 0)
    if positive.size > positive_budget and positive_budget > 0:
        positive = positive[np.linspace(0, positive.size - 1, positive_budget).round().astype(int)]
    need = max_samples - positive.size
    if negative.size > need and need > 0:
        negative = negative[np.linspace(0, negative.size - 1, need).round().astype(int)]
    elif need <= 0:
        negative = negative[:0]
    return np.sort(np.r_[positive, negative].astype(np.int64))


def _shift_labels(labels: np.ndarray, shift_samples: int) -> np.ndarray:
    labels = np.asarray(labels)
    shift = int(shift_samples)
    if shift == 0:
        return labels.copy()
    out = np.zeros_like(labels)
    if abs(shift) >= labels.shape[0]:
        return out
    if shift > 0:
        out[shift:] = labels[:-shift]
    else:
        out[:shift] = labels[-shift:]
    return out


def _cache_path(
    cache_dir: Path | None,
    data_path: Path,
    *,
    params: dict[str, Any],
) -> Path | None:
    if cache_dir is None:
        return None
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return cache_dir / f"{data_path.stem}_{digest}.npz"


def _load_feature_block(
    data_path: Path,
    *,
    edges: Sequence[EdgeSpec],
    mode: str,
    fs: int,
    m: int,
    ema_half_life_s: float,
    ar_order: int,
    student_nu: int,
    baseline_windows_ms: Sequence[int],
    baseline_lags_ms: Sequence[int],
    baseline_context: str,
    mfg_feature_mode: str,
    channel_set: str,
    preprocess: str,
    label_shift_samples: int,
    stride: int,
    max_samples: int | None,
    keep_all_positive: bool,
    cache_dir: Path | None,
) -> FeatureBlock:
    cache_params = {
        "path": str(data_path.resolve()),
        "mtime": data_path.stat().st_mtime,
        "events_mtime": _events_path_for_data(data_path).stat().st_mtime,
        "edges": [edge.__dict__ for edge in edges],
        "mode": mode,
        "fs": fs,
        "m": m,
        "ema_half_life_s": ema_half_life_s,
        "ar_order": ar_order,
        "student_nu": student_nu,
        "baseline_windows_ms": list(map(int, baseline_windows_ms)),
        "baseline_lags_ms": list(map(int, baseline_lags_ms)),
        "baseline_context": baseline_context,
        "mfg_feature_mode": mfg_feature_mode,
        "channel_set": channel_set,
        "preprocess": preprocess,
        "label_shift_samples": int(label_shift_samples),
        "stride": stride,
        "max_samples": max_samples,
        "keep_all_positive": keep_all_positive,
        "sampling_version": 2,
    }
    path = _cache_path(cache_dir, data_path, params=cache_params)
    if path is not None and path.exists():
        cached = np.load(path)
        sample_index = cached["sample_index"] if "sample_index" in cached else np.arange(cached["labels"].shape[0])
        return FeatureBlock(cached["baseline"], cached["mfg"], cached["labels"], sample_index)

    _, X, E, ch_names, _ = _prepare_kaggle_data_with_events(
        data_path,
        channel_set=channel_set,
        preprocess=preprocess,
        fs=fs,
    )
    labels_full = _shift_labels(E, int(label_shift_samples))
    idx = _sample_indices(
        labels_full,
        stride=stride,
        max_samples=max_samples,
        keep_all_positive=keep_all_positive,
    )
    baseline = _make_baseline_features(
        X,
        fs,
        windows_ms=baseline_windows_ms,
        lags_ms=baseline_lags_ms,
        context=baseline_context,
    )[idx]
    mfg = _make_mfg_features(
        X,
        ch_names,
        edges,
        mode=mode,
        fs=fs,
        m=m,
        ema_half_life_s=ema_half_life_s,
        ar_order=ar_order,
        student_nu=student_nu,
        feature_mode=mfg_feature_mode,
    )[idx]
    labels = labels_full[idx].astype(np.int8)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            baseline=baseline,
            mfg=mfg,
            labels=labels,
            sample_index=idx.astype(np.int64),
        )

    return FeatureBlock(baseline, mfg, labels, idx.astype(np.int64))


def _make_prediction_features(
    data_path: Path,
    *,
    edges: Sequence[EdgeSpec],
    mode: str,
    fs: int,
    m: int,
    ema_half_life_s: float,
    ar_order: int,
    student_nu: int,
    baseline_windows_ms: Sequence[int],
    baseline_lags_ms: Sequence[int],
    baseline_context: str,
    mfg_feature_mode: str,
    channel_set: str,
    preprocess: str,
) -> tuple[np.ndarray, FeatureBlock]:
    ids, X, ch_names = _prepare_kaggle_data_only(
        data_path,
        channel_set=channel_set,
        preprocess=preprocess,
        fs=fs,
    )
    baseline = _make_baseline_features(
        X,
        fs,
        windows_ms=baseline_windows_ms,
        lags_ms=baseline_lags_ms,
        context=baseline_context,
    )
    mfg = _make_mfg_features(
        X,
        ch_names,
        edges,
        mode=mode,
        fs=fs,
        m=m,
        ema_half_life_s=ema_half_life_s,
        ar_order=ar_order,
        student_nu=student_nu,
        feature_mode=mfg_feature_mode,
    )
    labels = np.zeros((X.shape[0], len(KAGGLE_EVENT_COLS)), dtype=np.int8)
    return ids, FeatureBlock(
        baseline=baseline,
        mfg=mfg,
        labels=labels,
        sample_index=np.arange(X.shape[0], dtype=np.int64),
    )


def _feature_matrix(block: FeatureBlock, feature_set: str) -> np.ndarray:
    mfg = block.mfg
    if mfg.shape[1] == 0:
        mfg = np.zeros((block.baseline.shape[0], 1), dtype=np.float32)
    if feature_set == "baseline":
        return block.baseline
    if feature_set == "mfg":
        return mfg
    if feature_set == "combined":
        return np.hstack([block.baseline, mfg]).astype(np.float32)
    raise ValueError(f"Unknown feature set: {feature_set}")


def _sort_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "stability_fraction" in df.columns:
        sort_cols = ["stability_fraction"]
        ascending = [False]
        if "min_q_value" in df.columns:
            sort_cols.append("min_q_value")
            ascending.append(True)
        return df.sort_values(sort_cols, ascending=ascending)

    if "q_value" in df.columns:
        if "significant" in df.columns:
            df = df[df["significant"].astype(bool)]
        return df.sort_values(["q_value"], ascending=True)

    return df


def _edges_from_rows(
    rows: pd.DataFrame,
    *,
    max_edges: int,
    seen: set[tuple[str, str, int]],
    edges: list[EdgeSpec],
) -> None:
    for _, row in rows.iterrows():
        key = (str(row["src"]), str(row["dst"]), int(row["lag_ms"]))
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            EdgeSpec(
                src=key[0],
                dst=key[1],
                lag_ms=key[2],
                rank=len(edges) + 1,
                phase=str(row.get("phase", "")),
            )
        )
        if len(edges) >= int(max_edges):
            break


def _load_edges(
    path: Path,
    *,
    max_edges: int,
    edge_selection: str = "global",
    allowed_channels: Sequence[str] | None = None,
) -> list[EdgeSpec]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"src", "dst", "lag_ms"}
    if not required.issubset(df.columns):
        raise ValueError(f"Edge table must contain columns: {sorted(required)}")
    if allowed_channels is not None:
        allowed = set(allowed_channels)
        df = df[df["src"].isin(allowed) & df["dst"].isin(allowed)]

    df = _sort_edge_table(df)
    edges: list[EdgeSpec] = []
    seen: set[tuple[str, str, int]] = set()

    if edge_selection == "global" or "phase" not in df.columns:
        _edges_from_rows(df, max_edges=max_edges, seen=seen, edges=edges)
        return edges

    if edge_selection != "event_union":
        raise ValueError(f"Unknown edge selection: {edge_selection}")

    for event_name in KAGGLE_EVENT_COLS:
        phases = EVENT_PHASE_MAP.get(event_name, [])
        event_rows = df[df["phase"].isin(phases)] if phases else df.iloc[0:0]
        before = len(edges)
        _edges_from_rows(
            event_rows,
            max_edges=before + int(max_edges),
            seen=seen,
            edges=edges,
        )

    return edges


def _fit_scalers(
    files: Sequence[Path],
    *,
    feature_sets: Sequence[str],
    block_kwargs: dict[str, Any],
) -> tuple[dict[str, StandardScaler], np.ndarray, int]:
    scalers = {name: StandardScaler() for name in feature_sets}
    label_counts = np.zeros(len(KAGGLE_EVENT_COLS), dtype=np.int64)
    n_samples = 0

    for path in files:
        block = _load_feature_block(path, **block_kwargs)
        label_counts += block.labels.sum(axis=0).astype(np.int64)
        n_samples += int(block.labels.shape[0])
        for name, scaler in scalers.items():
            X_feat = _feature_matrix(block, name)
            if X_feat.shape[1] > 0:
                scaler.partial_fit(X_feat)

    return scalers, label_counts, n_samples


def _make_sgd_models(feature_sets: Sequence[str], *, alpha: float, random_state: int) -> dict[str, list[Any]]:
    return {
        name: [
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=float(alpha),
                random_state=int(random_state) + idx,
                max_iter=1,
                tol=None,
                learning_rate="optimal",
            )
            for idx in range(len(KAGGLE_EVENT_COLS))
        ]
        for name in feature_sets
    }


def _make_extra_trees_models(
    feature_sets: Sequence[str],
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> dict[str, list[Any]]:
    return {
        name: [
            ExtraTreesClassifier(
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                min_samples_leaf=int(min_samples_leaf),
                random_state=int(random_state) + idx,
                n_jobs=1,
            )
            for idx in range(len(KAGGLE_EVENT_COLS))
        ]
        for name in feature_sets
    }


def _label_weights(labels: np.ndarray, label_counts: np.ndarray, n_samples: int) -> np.ndarray:
    weights = np.ones_like(labels, dtype=float)
    for idx in range(labels.shape[1]):
        pos = max(float(label_counts[idx]), 1.0)
        neg = max(float(n_samples - label_counts[idx]), 1.0)
        weights[:, idx] = np.where(labels[:, idx] > 0, n_samples / (2.0 * pos), n_samples / (2.0 * neg))
    return weights


def _train_sgd_models(
    models: dict[str, list[Any]],
    scalers: dict[str, StandardScaler],
    train_files: Sequence[Path],
    *,
    feature_sets: Sequence[str],
    epochs: int,
    balance_classes: bool,
    label_counts: np.ndarray,
    n_train_samples: int,
    block_kwargs: dict[str, Any],
) -> None:
    classes = np.array([0, 1], dtype=np.int8)
    initialized = {name: [False] * len(KAGGLE_EVENT_COLS) for name in feature_sets}

    for _epoch in range(int(epochs)):
        for path in train_files:
            block = _load_feature_block(path, **block_kwargs)
            weights = (
                _label_weights(block.labels, label_counts, n_train_samples)
                if balance_classes
                else np.ones_like(block.labels, dtype=float)
            )
            for name in feature_sets:
                X_scaled = scalers[name].transform(_feature_matrix(block, name))
                for label_idx, model in enumerate(models[name]):
                    kwargs: dict[str, Any] = {"sample_weight": weights[:, label_idx]}
                    if not initialized[name][label_idx]:
                        kwargs["classes"] = classes
                        initialized[name][label_idx] = True
                    model.partial_fit(X_scaled, block.labels[:, label_idx], **kwargs)


def _train_extra_trees_models(
    models: dict[str, list[Any]],
    scalers: dict[str, StandardScaler],
    train_files: Sequence[Path],
    *,
    feature_sets: Sequence[str],
    balance_classes: bool,
    label_counts: np.ndarray,
    n_train_samples: int,
    block_kwargs: dict[str, Any],
) -> None:
    X_parts: dict[str, list[np.ndarray]] = {name: [] for name in feature_sets}
    y_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    for path in train_files:
        block = _load_feature_block(path, **block_kwargs)
        y_parts.append(block.labels)
        weights = (
            _label_weights(block.labels, label_counts, n_train_samples)
            if balance_classes
            else np.ones_like(block.labels, dtype=float)
        )
        weight_parts.append(weights)
        for name in feature_sets:
            X_parts[name].append(scalers[name].transform(_feature_matrix(block, name)).astype(np.float32))

    y = np.vstack(y_parts)
    weights = np.vstack(weight_parts)
    for name in feature_sets:
        X = np.vstack(X_parts[name])
        for label_idx, model in enumerate(models[name]):
            model.fit(X, y[:, label_idx], sample_weight=weights[:, label_idx])


def _fit_model_bundle(
    files: Sequence[Path],
    *,
    feature_sets: Sequence[str],
    block_kwargs: dict[str, Any],
    classifier: str,
    alpha: float,
    random_state: int,
    epochs: int,
    balance_classes: bool,
    tree_estimators: int,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int,
) -> tuple[dict[str, StandardScaler], dict[str, list[Any]], np.ndarray, int]:
    scalers, label_counts, n_samples = _fit_scalers(
        files,
        feature_sets=feature_sets,
        block_kwargs=block_kwargs,
    )
    if classifier == "sgd_logistic":
        models = _make_sgd_models(
            feature_sets,
            alpha=alpha,
            random_state=random_state,
        )
        _train_sgd_models(
            models,
            scalers,
            files,
            feature_sets=feature_sets,
            epochs=epochs,
            balance_classes=balance_classes,
            label_counts=label_counts,
            n_train_samples=n_samples,
            block_kwargs=block_kwargs,
        )
    elif classifier == "extra_trees":
        models = _make_extra_trees_models(
            feature_sets,
            random_state=random_state,
            n_estimators=tree_estimators,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
        )
        _train_extra_trees_models(
            models,
            scalers,
            files,
            feature_sets=feature_sets,
            balance_classes=balance_classes,
            label_counts=label_counts,
            n_train_samples=n_samples,
            block_kwargs=block_kwargs,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return scalers, models, label_counts, n_samples


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _predict_positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if positive.size:
        return proba[:, int(positive[0])]
    return np.zeros(X.shape[0], dtype=float)


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _smooth_probability_scores(
    scores: np.ndarray,
    window_samples: int,
    sample_index: np.ndarray | None = None,
) -> np.ndarray:
    window = int(window_samples)
    if window <= 1:
        return np.asarray(scores, float)
    values = np.asarray(scores, float)
    if sample_index is None:
        return np.clip(_centered_rolling_mean(values, window), 0.0, 1.0)

    index = np.asarray(sample_index, dtype=np.int64)
    if index.shape[0] != values.shape[0]:
        raise ValueError("sample_index length must match probability rows.")
    left = window // 2
    right = window - left
    starts = np.searchsorted(index, index - left, side="left")
    ends = np.searchsorted(index, index + right, side="right")
    csum = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), values]), axis=0)
    counts = np.maximum((ends - starts).astype(float), 1.0)[:, None]
    return np.clip((csum[ends] - csum[starts]) / counts, 0.0, 1.0)


def _append_event_metric_rows(
    rows: list[dict[str, Any]],
    *,
    feature_set: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    for idx, event_name in enumerate(KAGGLE_EVENT_COLS):
        rows.append(
            {
                "feature_set": feature_set,
                "event": event_name,
                "roc_auc": _safe_auc(y_true[:, idx], y_score[:, idx]),
                "average_precision": _safe_ap(y_true[:, idx], y_score[:, idx]),
                "n_positive": int(y_true[:, idx].sum()),
                "n_negative": int(y_true.shape[0] - y_true[:, idx].sum()),
            }
        )


def _fuse_probability_scores(
    baseline_scores: np.ndarray,
    mfg_scores: np.ndarray,
    fusion_weight: float,
) -> np.ndarray:
    weight = float(fusion_weight)
    return np.clip((1.0 - weight) * baseline_scores + weight * mfg_scores, 0.0, 1.0)


def _mean_column_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values = [_safe_auc(y_true[:, idx], y_score[:, idx]) for idx in range(y_true.shape[1])]
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _evaluate_models(
    models: dict[str, list[Any]],
    scalers: dict[str, StandardScaler],
    val_files: Sequence[Path],
    *,
    feature_sets: Sequence[str],
    block_kwargs: dict[str, Any],
    fusion_weight: float | None = None,
    fusion_weights: Sequence[float] | None = None,
    smooth_proba_samples: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], int, float | None]:
    labels_by_set: dict[str, list[np.ndarray]] = {name: [] for name in feature_sets}
    scores_by_set: dict[str, list[np.ndarray]] = {name: [] for name in feature_sets}
    n_val_samples = 0

    for path in val_files:
        block = _load_feature_block(path, **block_kwargs)
        n_val_samples += int(block.labels.shape[0])
        for name in feature_sets:
            X_scaled = scalers[name].transform(_feature_matrix(block, name))
            score = np.column_stack(
                [_predict_positive_proba(model, X_scaled) for model in models[name]]
            )
            score = _smooth_probability_scores(score, smooth_proba_samples, block.sample_index)
            labels_by_set[name].append(block.labels)
            scores_by_set[name].append(score)

    rows: list[dict[str, Any]] = []
    label_arrays: dict[str, np.ndarray] = {}
    score_arrays: dict[str, np.ndarray] = {}
    for name in feature_sets:
        y_true = np.vstack(labels_by_set[name])
        y_score = np.vstack(scores_by_set[name])
        label_arrays[name] = y_true
        score_arrays[name] = y_score
        _append_event_metric_rows(rows, feature_set=name, y_true=y_true, y_score=y_score)

    selected_fusion_weight: float | None = None
    fusion_grid: list[float] = []
    if fusion_weights:
        fusion_grid = [float(weight) for weight in fusion_weights]
    elif fusion_weight is not None:
        fusion_grid = [float(fusion_weight)]

    if fusion_grid:
        if "baseline" not in score_arrays or "mfg" not in score_arrays:
            raise ValueError("Fusion requires both baseline and mfg in --feature-sets.")
        y_true = label_arrays["baseline"]
        candidates: list[tuple[float, float, np.ndarray]] = []
        for weight in fusion_grid:
            y_candidate = _fuse_probability_scores(score_arrays["baseline"], score_arrays["mfg"], weight)
            candidates.append((_mean_column_auc(y_true, y_candidate), weight, y_candidate))
        _best_auc, selected_fusion_weight, y_score = max(candidates, key=lambda item: item[0])
        score_arrays["fusion"] = y_score
        _append_event_metric_rows(rows, feature_set="fusion", y_true=y_true, y_score=y_score)

    return rows, score_arrays, n_val_samples, selected_fusion_weight


def _predict_block_probabilities(
    *,
    models: dict[str, list[Any]],
    scalers: dict[str, StandardScaler],
    block: FeatureBlock,
    feature_set: str,
) -> np.ndarray:
    X_scaled = scalers[feature_set].transform(_feature_matrix(block, feature_set))
    return np.column_stack(
        [_predict_positive_proba(model, X_scaled) for model in models[feature_set]]
    )


def _write_submission(
    *,
    models: dict[str, list[Any]],
    scalers: dict[str, StandardScaler],
    test_files: Sequence[Path],
    feature_set: str,
    fusion_weight: float | None,
    smooth_proba_samples: int,
    out_path: Path,
    block_kwargs: dict[str, Any],
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    first_chunk = True

    for path in test_files:
        ids, block = _make_prediction_features(path, **block_kwargs)
        if feature_set == "fusion":
            if fusion_weight is None:
                raise ValueError("--submission-feature-set fusion requires --fusion-weight or --fusion-weights.")
            baseline_probs = _predict_block_probabilities(
                models=models,
                scalers=scalers,
                block=block,
                feature_set="baseline",
            )
            mfg_probs = _predict_block_probabilities(
                models=models,
                scalers=scalers,
                block=block,
                feature_set="mfg",
            )
            probs = _fuse_probability_scores(baseline_probs, mfg_probs, fusion_weight)
        else:
            probs = _predict_block_probabilities(
                models=models,
                scalers=scalers,
                block=block,
                feature_set=feature_set,
            )
        probs = _smooth_probability_scores(probs, smooth_proba_samples, block.sample_index)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        df = pd.DataFrame(probs, columns=KAGGLE_EVENT_COLS)
        df.insert(0, "id", ids)
        df.to_csv(
            out_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            float_format="%.6g",
        )
        rows_written += int(df.shape[0])
        first_chunk = False

    return rows_written


def _zip_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname=src.name)


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config_yaml()
    fs = int(args.fs or cfg.fs)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = _iter_data_files(root)
    subjects = _parse_int_list(args.subjects)
    train_series = _parse_int_list(args.train_series)
    val_series = _parse_int_list(args.val_series)
    train_files = _select_files(
        all_files,
        subjects=subjects,
        series=train_series,
        max_files=args.max_train_files,
        require_events=True,
    )
    val_files = _select_files(
        all_files,
        subjects=subjects,
        series=val_series,
        max_files=args.max_val_files,
        require_events=True,
    )
    if not train_files or not val_files:
        raise ValueError("Training and validation file selections must both be non-empty.")

    channel_names = _select_channel_set(_read_data_channel_names(train_files[0]), args.channel_set)
    edge_table = Path(args.edge_table)
    edges = _load_edges(
        edge_table,
        max_edges=args.max_edges,
        edge_selection=args.edge_selection,
        allowed_channels=channel_names,
    )
    if not edges and args.fallback_edge_table:
        edges = _load_edges(
            Path(args.fallback_edge_table),
            max_edges=args.max_edges,
            edge_selection=args.edge_selection,
            allowed_channels=channel_names,
        )
    feature_sets = list(args.feature_sets)
    fusion_weight = args.fusion_weight
    fusion_weights = [float(weight) for weight in (args.fusion_weights or [])]
    if fusion_weight is not None and not 0.0 <= float(fusion_weight) <= 1.0:
        raise ValueError("--fusion-weight must be between 0 and 1.")
    if any(not 0.0 <= weight <= 1.0 for weight in fusion_weights):
        raise ValueError("--fusion-weights entries must be between 0 and 1.")
    if fusion_weight is not None and fusion_weights:
        raise ValueError("Use either --fusion-weight or --fusion-weights, not both.")
    if fusion_weight is not None or fusion_weights or args.submission_feature_set == "fusion":
        missing = {"baseline", "mfg"} - set(feature_sets)
        if missing:
            raise ValueError(
                "Fusion requires baseline and mfg in --feature-sets; "
                f"missing: {sorted(missing)}."
            )

    needs_mfg = "mfg" in feature_sets or "combined" in feature_sets or fusion_weight is not None or bool(fusion_weights)
    edge_source_exists = edge_table.exists() or bool(args.fallback_edge_table and Path(args.fallback_edge_table).exists())
    if not edges and needs_mfg and not edge_source_exists:
        raise ValueError("No MFG edge table found. Provide --edge-table or run meta/sensitivity analysis first.")
    edge_warning = None
    if not edges and needs_mfg:
        edge_warning = (
            f"No MFG edges remained for channel_set={args.channel_set}; "
            "MFG-only features use an intercept-only control column."
        )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    baseline_windows_ms = _parse_int_values(args.baseline_windows_ms, [100, 200, 500])
    baseline_lags_ms = _parse_int_values(args.baseline_lags_ms, [50, 100, 200])
    label_shift_samples = int(round(float(args.label_shift_ms) * float(fs) / 1000.0))
    smooth_proba_samples = int(round(float(args.smooth_proba_ms) * float(fs) / 1000.0))
    block_train_kwargs = {
        "edges": edges,
        "mode": args.mode,
        "fs": fs,
        "m": args.m,
        "ema_half_life_s": args.ema_half_life_s,
        "ar_order": args.ar_order,
        "student_nu": args.student_nu,
        "baseline_windows_ms": baseline_windows_ms,
        "baseline_lags_ms": baseline_lags_ms,
        "baseline_context": args.baseline_context,
        "mfg_feature_mode": args.mfg_feature_mode,
        "channel_set": args.channel_set,
        "preprocess": args.preprocess,
        "label_shift_samples": label_shift_samples,
        "stride": args.stride,
        "max_samples": args.max_train_samples_per_file,
        "keep_all_positive": True,
        "cache_dir": cache_dir,
    }
    block_val_kwargs = dict(block_train_kwargs)
    block_val_kwargs["max_samples"] = args.max_val_samples_per_file
    block_val_kwargs["keep_all_positive"] = bool(args.val_keep_all_positive)

    scalers, models, label_counts, n_train_samples = _fit_model_bundle(
        train_files,
        feature_sets=feature_sets,
        block_kwargs=block_train_kwargs,
        classifier=args.classifier,
        alpha=args.alpha,
        random_state=args.random_state,
        epochs=args.epochs,
        balance_classes=args.balance_classes,
        tree_estimators=args.tree_estimators,
        tree_max_depth=args.tree_max_depth,
        tree_min_samples_leaf=args.tree_min_samples_leaf,
    )
    rows, _scores, n_val_samples, selected_fusion_weight = _evaluate_models(
        models,
        scalers,
        val_files,
        feature_sets=feature_sets,
        block_kwargs=block_val_kwargs,
        fusion_weight=fusion_weight,
        fusion_weights=fusion_weights,
        smooth_proba_samples=smooth_proba_samples,
    )
    effective_fusion_weight = selected_fusion_weight if selected_fusion_weight is not None else fusion_weight

    test_root = Path(args.test_root) if args.test_root else None
    test_files: list[Path] = []
    submission_rows = 0
    submission_path = Path(args.submission_out)
    submission_zip_path = Path(args.submission_zip_out)
    submission_warning: str | None = None
    sample_submission_rows = _count_csv_rows(Path(args.sample_submission)) if args.sample_submission else None
    if test_root is not None:
        test_files = _select_files(
            _iter_data_files(test_root),
            subjects=subjects,
            series=_parse_int_list(args.test_series),
            max_files=args.max_test_files,
            require_events=False,
        )
        if test_files:
            if args.submission_feature_set == "fusion":
                if effective_fusion_weight is None:
                    raise ValueError("--submission-feature-set fusion requires --fusion-weight or --fusion-weights.")
            elif args.submission_feature_set not in feature_sets:
                raise ValueError("--submission-feature-set must be included in --feature-sets.")
            submission_models = models
            submission_scalers = scalers
            submission_train_files = list(train_files)
            if args.refit_for_submission:
                seen_paths = {path.resolve() for path in train_files}
                submission_train_files = list(train_files)
                for path in val_files:
                    if path.resolve() not in seen_paths:
                        submission_train_files.append(path)
                        seen_paths.add(path.resolve())
                refit_kwargs = dict(block_train_kwargs)
                refit_kwargs["max_samples"] = args.max_submission_train_samples_per_file
                submission_scalers, submission_models, _refit_counts, _refit_n = _fit_model_bundle(
                    submission_train_files,
                    feature_sets=feature_sets,
                    block_kwargs=refit_kwargs,
                    classifier=args.classifier,
                    alpha=args.alpha,
                    random_state=args.random_state + 10_000,
                    epochs=args.epochs,
                    balance_classes=args.balance_classes,
                    tree_estimators=args.tree_estimators,
                    tree_max_depth=args.tree_max_depth,
                    tree_min_samples_leaf=args.tree_min_samples_leaf,
                )
            prediction_kwargs = {
                "edges": edges,
                "mode": args.mode,
                "fs": fs,
                "m": args.m,
                "ema_half_life_s": args.ema_half_life_s,
                "ar_order": args.ar_order,
                "student_nu": args.student_nu,
                "baseline_windows_ms": baseline_windows_ms,
                "baseline_lags_ms": baseline_lags_ms,
                "baseline_context": args.baseline_context,
                "mfg_feature_mode": args.mfg_feature_mode,
                "channel_set": args.channel_set,
                "preprocess": args.preprocess,
            }
            submission_rows = _write_submission(
                models=submission_models,
                scalers=submission_scalers,
                test_files=test_files,
                feature_set=args.submission_feature_set,
                fusion_weight=effective_fusion_weight,
                smooth_proba_samples=smooth_proba_samples,
                out_path=submission_path,
                block_kwargs=prediction_kwargs,
            )
            if sample_submission_rows is not None and submission_rows != sample_submission_rows:
                submission_warning = (
                    f"Submission row count ({submission_rows}) differs from sample_submission "
                    f"row count ({sample_submission_rows})."
                )
            if args.zip_submission:
                _zip_file(submission_path, submission_zip_path)
        elif test_root.exists():
            submission_warning = f"No test data files found in {test_root}."
        else:
            submission_warning = f"Test root does not exist: {test_root}."

    event_df = pd.DataFrame(rows)
    event_df.insert(0, "preprocess", args.preprocess)
    event_df.insert(0, "channel_set", args.channel_set)
    comparison_rows = []
    for name, group in event_df.groupby("feature_set", sort=False):
        comparison_rows.append(
            {
                "channel_set": args.channel_set,
                "preprocess": args.preprocess,
                "label_shift_ms": float(args.label_shift_ms),
                "feature_set": name,
                "mean_roc_auc": float(np.nanmean(group["roc_auc"].to_numpy(float))),
                "mean_average_precision": float(np.nanmean(group["average_precision"].to_numpy(float))),
                "n_train_samples": int(n_train_samples),
                "n_val_samples": int(n_val_samples),
                "train_files": len(train_files),
                "val_files": len(val_files),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    edge_df = pd.DataFrame([edge.__dict__ for edge in edges])
    event_df.to_csv(out_dir / "auc_by_event.csv", index=False)
    comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)
    edge_df.to_csv(out_dir / "mfg_edges_used.csv", index=False)

    summary = {
        "metric": "mean column-wise ROC-AUC across the six Kaggle event labels",
        "generated_outputs": {
            "model_comparison": str(out_dir / "model_comparison.csv"),
            "auc_by_event": str(out_dir / "auc_by_event.csv"),
            "mfg_edges_used": str(out_dir / "mfg_edges_used.csv"),
            "submission": str(submission_path) if submission_rows else None,
            "submission_zip": str(submission_zip_path) if submission_rows and args.zip_submission else None,
        },
        "parameters": {
            "root": str(root),
            "edge_table": str(edge_table),
            "feature_sets": feature_sets,
            "mode": args.mode,
            "fs": fs,
            "m": args.m,
            "stride": args.stride,
            "train_series": train_series,
            "val_series": val_series,
            "subjects": subjects,
            "classifier": args.classifier,
            "epochs": args.epochs,
            "alpha": args.alpha,
            "tree_estimators": args.tree_estimators,
            "tree_max_depth": args.tree_max_depth,
            "tree_min_samples_leaf": args.tree_min_samples_leaf,
            "fusion_weight": fusion_weight,
            "fusion_weights": fusion_weights,
            "selected_fusion_weight": effective_fusion_weight,
            "balance_classes": bool(args.balance_classes),
            "max_edges": args.max_edges,
            "edge_selection": args.edge_selection,
            "edges_used": len(edges),
            "channel_set": args.channel_set,
            "channels": channel_names,
            "preprocess": args.preprocess,
            "label_shift_ms": float(args.label_shift_ms),
            "label_shift_samples": label_shift_samples,
            "baseline_windows_ms": baseline_windows_ms,
            "baseline_lags_ms": baseline_lags_ms,
            "baseline_context": args.baseline_context,
            "smooth_proba_ms": float(args.smooth_proba_ms),
            "smooth_proba_samples": smooth_proba_samples,
            "val_keep_all_positive": bool(args.val_keep_all_positive),
            "mfg_feature_mode": args.mfg_feature_mode,
            "test_root": str(test_root) if test_root else None,
            "submission_feature_set": args.submission_feature_set,
            "sample_submission": args.sample_submission,
            "zip_submission": bool(args.zip_submission),
            "refit_for_submission": bool(args.refit_for_submission),
            "max_submission_train_samples_per_file": args.max_submission_train_samples_per_file,
        },
        "files": {
            "train": [str(path) for path in train_files],
            "validation": [str(path) for path in val_files],
            "test": [str(path) for path in test_files],
            "submission_train": [str(path) for path in submission_train_files] if test_files and args.refit_for_submission else None,
        },
        "submission": {
            "path": str(submission_path) if submission_rows else None,
            "zip_path": str(submission_zip_path) if submission_rows and args.zip_submission else None,
            "rows": int(submission_rows),
            "sample_submission_rows": sample_submission_rows,
            "feature_set": args.submission_feature_set,
            "warning": submission_warning,
        },
        "warnings": [warning for warning in [edge_warning, submission_warning] if warning],
        "scores": comparison_rows,
    }
    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(out_dir / "benchmark_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Classification Benchmark\n\n")
        handle.write("Metric: mean column-wise ROC-AUC across the six event labels.\n\n")
        handle.write(f"Classifier: `{args.classifier}`.\n\n")
        handle.write(f"Baseline context: `{args.baseline_context}`.\n\n")
        if effective_fusion_weight is not None:
            handle.write(
                "Fusion: `fusion = (1 - w) * baseline + w * mfg`, "
                f"`w = {float(effective_fusion_weight):.3f}`.\n\n"
            )
            if fusion_weights:
                handle.write(
                    "Fusion grid: "
                    + ", ".join(f"`{weight:.3f}`" for weight in fusion_weights)
                    + ".\n\n"
                )
        if smooth_proba_samples > 1:
            handle.write(
                f"Probability smoothing: `{args.smooth_proba_ms}` ms "
                f"({smooth_proba_samples} samples).\n\n"
            )
        if args.val_keep_all_positive:
            handle.write("Validation sampling keeps event-positive samples before filling negatives.\n\n")
        handle.write(f"Channel set: `{args.channel_set}`; preprocessing: `{args.preprocess}`.\n\n")
        if label_shift_samples:
            handle.write(
                f"Label shift control: `{args.label_shift_ms}` ms ({label_shift_samples} samples).\n\n"
            )
        handle.write(f"Edge selection: `{args.edge_selection}`; MFG edges used: {len(edges)}.\n\n")
        if edge_warning:
            handle.write(f"Warning: {edge_warning}\n\n")
        handle.write("| feature_set | mean_roc_auc | mean_average_precision | n_train_samples | n_val_samples |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in comparison_rows:
            handle.write(
                f"| {row['feature_set']} | {row['mean_roc_auc']:.6f} | "
                f"{row['mean_average_precision']:.6f} | {row['n_train_samples']} | "
                f"{row['n_val_samples']} |\n"
            )
        handle.write("\n\n")
        handle.write(
            "Feature sets: baseline EEG, MFG edge dynamics, combined features, "
            "and optional late fusion of baseline and MFG predictions.\n"
        )
        if submission_rows:
            handle.write(f"\nKaggle-style submission rows written: {submission_rows}.\n")
            handle.write(f"Submission file: `{submission_path}`.\n")
            if args.zip_submission:
                handle.write(f"Compressed submission: `{submission_zip_path}`.\n")
        if submission_warning:
            handle.write(f"\nSubmission warning: {submission_warning}\n")

    print(comparison_df.to_string(index=False))
    if submission_rows:
        print(f"\nSaved Kaggle-style submission: {submission_path} ({submission_rows} rows)")
        if args.zip_submission:
            print(f"Saved compressed submission: {submission_zip_path}")
    if submission_warning:
        print(f"\nSubmission warning: {submission_warning}")
    print(f"\nSaved benchmark outputs to: {out_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kaggle Grasp-and-Lift classification benchmark with MFG feature ablation."
    )
    parser.add_argument("--root", default="data/grasp-and-lift-eeg-detection/train")
    parser.add_argument("--out-dir", default="out/classification_benchmark")
    parser.add_argument("--edge-table", default="out/article_meta_sensitivity/edge_stability.csv")
    parser.add_argument("--fallback-edge-table", default="out/article_meta_kaggle/meta_edges.csv")
    parser.add_argument("--feature-sets", nargs="+", default=["baseline", "mfg", "combined"], choices=FEATURE_SET_CHOICES)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--train-series", nargs="*", default=["1", "2", "3", "4", "5", "6", "7"])
    parser.add_argument("--val-series", nargs="*", default=["8"])
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--max-train-samples-per-file", type=int, default=None)
    parser.add_argument("--max-val-samples-per-file", type=int, default=None)
    parser.add_argument("--val-keep-all-positive", action="store_true")
    parser.add_argument("--test-root", default="data/grasp-and-lift-eeg-detection/test")
    parser.add_argument("--test-series", nargs="*")
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--submission-out", default="out/classification_benchmark/submission.csv")
    parser.add_argument("--submission-feature-set", choices=SUBMISSION_FEATURE_SET_CHOICES, default="combined")
    parser.add_argument("--sample-submission", default="data/grasp-and-lift-eeg-detection/sample_submission.csv")
    parser.add_argument("--zip-submission", action="store_true")
    parser.add_argument("--submission-zip-out", default="out/classification_benchmark/submission.zip")
    parser.add_argument("--refit-for-submission", action="store_true")
    parser.add_argument("--max-submission-train-samples-per-file", type=int, default=None)
    parser.add_argument("--cache-dir", default="out/classification_benchmark/cache")
    parser.add_argument("--max-edges", type=int, default=24)
    parser.add_argument("--edge-selection", choices=["global", "event_union"], default="global")
    parser.add_argument("--channel-set", choices=sorted(CHANNEL_SETS), default="all32")
    parser.add_argument("--preprocess", choices=PREPROCESSING_CHOICES, default="car_only")
    parser.add_argument("--label-shift-ms", type=float, default=0.0)
    parser.add_argument("--mode", choices=["gc", "corr"], default="gc")
    parser.add_argument("--fs", type=int, default=None)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--ema-half-life-s", type=float, default=0.1)
    parser.add_argument("--ar-order", type=int, default=10)
    parser.add_argument("--student-nu", type=int, default=10)
    parser.add_argument("--baseline-windows-ms", nargs="*", default=["100", "200", "500"])
    parser.add_argument("--baseline-lags-ms", nargs="*", default=["50", "100", "200"])
    parser.add_argument("--baseline-context", choices=BASELINE_CONTEXT_CHOICES, default="causal")
    parser.add_argument("--mfg-feature-mode", choices=["energy", "expanded"], default="expanded")
    parser.add_argument(
        "--fusion-weight",
        type=float,
        default=None,
        help="Optional late-fusion weight: fusion=(1-w)*baseline+w*mfg.",
    )
    parser.add_argument(
        "--fusion-weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional validation grid for selecting the best late-fusion weight after one model fit.",
    )
    parser.add_argument(
        "--smooth-proba-ms",
        type=float,
        default=0.0,
        help="Centered probability smoothing window for validation and submission.",
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--classifier", choices=CLASSIFIER_CHOICES, default="sgd_logistic")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--tree-estimators", type=int, default=120)
    parser.add_argument("--tree-max-depth", type=int, default=None)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--balance-classes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
