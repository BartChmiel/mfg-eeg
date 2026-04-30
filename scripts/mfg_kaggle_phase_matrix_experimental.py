from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src_mfg.basis import legendre_orthonormal
from src_mfg.config import load_config
from src_mfg.io import KAGGLE_EVENT_COLS, extract_grasp_cycles, load_kaggle_data_with_events
from src_mfg.normalize import normalize_edf, normalize_gauss, pnorm_student

DEFAULT_ANCHOR_EVENTS = ["HandStart", "LiftOff"]

_RX = re.compile(
    r"subj(?P<subj>\d+)_series(?P<series>\d+)_data\.csv$",
    re.IGNORECASE,
)


def _parse_subj_series(p: Path) -> Tuple[Optional[int], Optional[int]]:
    match = _RX.search(p.name)
    if not match:
        return None, None
    return int(match.group("subj")), int(match.group("series"))


def _iter_train_data_files(root: Path) -> List[Path]:
    train_dir = root / "train" if (root / "train").is_dir() else root
    return sorted(train_dir.glob("subj*_series*_data.csv"))


def _events_path_for_data(data_path: Path) -> Path:
    return data_path.with_name(data_path.name.replace("_data.csv", "_events.csv"))


def _basis_mixed(u_1d: np.ndarray, m: int) -> np.ndarray:
    basis = legendre_orthonormal(u_1d, m)
    return basis[:, 1:]


def _filter_files(
    files: List[Path],
    subjects: Optional[List[int]],
    series: Optional[List[int]],
    max_files: Optional[int],
) -> List[Path]:
    out: List[Path] = []
    for path in files:
        subj, se = _parse_subj_series(path)
        if subj is None or se is None:
            continue
        if subjects is not None and subj not in subjects:
            continue
        if series is not None and se not in series:
            continue
        out.append(path)
        if max_files is not None and len(out) >= max_files:
            break
    return out


def _ema_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"EMA alpha must lie in (0, 1], got {alpha}")

    arr = np.asarray(arr, float)
    if arr.shape[0] == 0:
        return arr.copy()

    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    one_minus_alpha = 1.0 - float(alpha)
    for t in range(1, arr.shape[0]):
        out[t] = float(alpha) * arr[t] + one_minus_alpha * out[t - 1]
    return out


def _align_basis_for_lag(
    Fy_full: np.ndarray,
    Fz_full: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if lag_samples < 0:
        raise ValueError(f"lag_samples must be >= 0, got {lag_samples}")

    total = Fy_full.shape[0]
    n_eff = total - lag_samples
    if n_eff <= 5:
        return None

    if lag_samples == 0:
        return Fy_full, Fz_full

    return Fy_full[:n_eff, :, :], Fz_full[lag_samples:, :, :]


def _compute_dynamic_coeffs_ema(
    Fy: np.ndarray,
    Fz: np.ndarray,
    alpha: float,
    subtract_marginals: bool,
) -> np.ndarray:
    """
    Build a dynamic connectivity tensor with an EMA instead of a hard window mean.

    The lag convention matches the batch script:
      source at time t, target at time t + lag.
    """
    cross = np.einsum("tcm,tdn->tcdmn", Fy, Fz)
    cross_ema = _ema_filter(cross, alpha)

    if subtract_marginals:
        my = _ema_filter(Fy, alpha)
        mz = _ema_filter(Fz, alpha)
        cross_ema = cross_ema - my[:, :, None, :, None] * mz[:, None, :, None, :]

    t_dim, c_dim, _, m_y, m_z = cross_ema.shape
    return cross_ema.reshape(t_dim, c_dim, c_dim, m_y * m_z)


def _robust_vmax(mat: np.ndarray) -> float:
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return 1.0
    return float(np.quantile(vals, 0.98))


def _save_npz_payload(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _save_top_edges(
    path: str,
    mat: np.ndarray,
    ch_names: Sequence[str],
    topk: int,
) -> list[tuple[int, int]]:
    arr = np.array(mat, float)
    np.fill_diagonal(arr, -np.inf)
    flat_idx = np.argsort(arr.ravel())[::-1]

    top_pairs: list[tuple[int, int]] = []
    with open(path, "a", encoding="utf-8") as handle:
        written = 0
        for idx in flat_idx:
            val = float(arr.ravel()[idx])
            if not np.isfinite(val):
                continue
            src = idx // arr.shape[1]
            dst = idx % arr.shape[1]
            handle.write(
                f"{written + 1:03d}. {ch_names[src]} -> {ch_names[dst]} : {val:.6f}\n"
            )
            top_pairs.append((int(src), int(dst)))
            written += 1
            if written >= int(topk):
                break
    return top_pairs


def _normalize_series(
    X: np.ndarray,
    mode: str,
    fs: int,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    t_dim, c_dim = X.shape
    Y = np.empty((t_dim, c_dim), dtype=float)
    Z = np.empty((t_dim, c_dim), dtype=float)

    if mode == "corr":
        for c in range(c_dim):
            u = normalize_gauss(X[:, c])
            Y[:, c] = u
            Z[:, c] = u
        return Y, Z

    for c in range(c_dim):
        Y[:, c] = normalize_edf(X[:, c])
        Z[:, c] = pnorm_student(
            X[:, c],
            fs,
            cfg.ar_order,
            cfg.ema_half_life_s,
            cfg.student_nu,
        )
    return Y, Z


def _summarize_scores(scores_t: np.ndarray, summary: str) -> np.ndarray:
    if summary == "final":
        return np.array(scores_t[-1], float)
    if summary == "mean":
        return np.mean(scores_t, axis=0)
    if summary == "max":
        return np.max(scores_t, axis=0)
    raise ValueError(f"Unknown summary: {summary}")


def _plot_matrix(
    *,
    out_path: Path,
    mat: np.ndarray,
    ch_names: Sequence[str],
    title: str,
    cbar_label: str,
) -> None:
    plt.figure(figsize=(11.5, 10))
    vmax = _robust_vmax(mat)
    im = plt.imshow(mat, origin="lower", vmin=0.0, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=8)
    plt.yticks(range(len(ch_names)), ch_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_edge_traces(
    *,
    out_path: Path,
    scores_t: np.ndarray,
    time_axis: np.ndarray,
    top_pairs: Sequence[tuple[int, int]],
    ch_names: Sequence[str],
    trace_topk: int,
    title: str,
) -> None:
    if not top_pairs:
        return

    plt.figure(figsize=(10, 6))
    for src_idx, dst_idx in top_pairs[: int(trace_topk)]:
        label = f"{ch_names[src_idx]} -> {ch_names[dst_idx]}"
        plt.plot(time_axis, scores_t[:, src_idx, dst_idx], label=label, linewidth=2)

    plt.axvline(x=0.0, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Time before event (s)")
    plt.ylabel("Coupling strength (EMA energy)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _run_group(
    *,
    group_name: str,
    files: List[Path],
    out_dir: Path,
    mode: str,
    anchor_events: Sequence[str],
    pre_s: float,
    burn_in_s: float,
    ema_half_life_s: float,
    max_cycle_s: float,
    min_cycles: int,
    m: int,
    lags_ms: List[int],
    metric: str,
    summary: str,
    topk: int,
    trace_topk: int,
    save_npz: bool,
) -> None:
    if metric != "energy":
        raise ValueError("Pre-event EMA analysis currently supports only metric=energy.")

    cfg = load_config()
    fs = int(cfg.fs)

    if not files:
        print(f"[WARN] group={group_name}: no files -> skipping")
        return

    first_data = files[0]
    first_events = _events_path_for_data(first_data)
    if not first_events.exists():
        print(f"[WARN] group={group_name}: missing events for first file -> skipping")
        return

    _, X0, _, ch_names0, _ = load_kaggle_data_with_events(
        str(first_data),
        str(first_events),
    )
    c_dim = X0.shape[1]

    lag_samples = [int(round(ms * fs / 1000.0)) for ms in lags_ms]
    max_lag = max(lag_samples) if lag_samples else 0
    k_dim = m * m
    t_eval = int(round(pre_s * fs))
    t_burn = int(round(burn_in_s * fs))
    t_context = t_burn + t_eval + max_lag

    if t_eval <= 5:
        raise ValueError("--pre-s is too short for a stable dynamic estimate.")
    if t_context <= max_lag + 5:
        raise ValueError("Not enough context to evaluate the requested lags.")

    alpha = 1.0 - np.exp(
        -np.log(2.0) / (max(float(ema_half_life_s), 1e-6) * float(fs))
    )
    subtract_marginals = True

    p_dim = len(anchor_events)
    l_dim = len(lag_samples)
    coeffs_sum_t = np.zeros(
        (p_dim, l_dim, t_eval, c_dim, c_dim, k_dim),
        dtype=np.float32,
    )
    n_windows = np.zeros((p_dim, l_dim), dtype=np.int32)

    n_files_used = 0
    n_cycles_total = 0
    n_anchor_windows_total = 0
    manifest: dict[str, Any] = {
        "group": group_name,
        "mode": mode,
        "anchor_events": list(anchor_events),
        "pre_s": float(pre_s),
        "burn_in_s": float(burn_in_s),
        "ema_half_life_s": float(ema_half_life_s),
        "max_cycle_s": float(max_cycle_s),
        "min_cycles": int(min_cycles),
        "m": int(m),
        "lags_ms": list(map(int, lags_ms)),
        "metric": metric,
        "summary": summary,
        "topk": int(topk),
        "trace_topk": int(trace_topk),
        "save_npz": bool(save_npz),
        "config": asdict(cfg),
        "input_files": [path.name for path in files],
        "used_files": [],
        "events": [],
    }

    for data_path in files:
        events_path = _events_path_for_data(data_path)
        if not events_path.exists():
            print(f"[WARN] Missing events for {data_path.name} -> skipping")
            continue

        _, X, E, ch_names, _ = load_kaggle_data_with_events(
            str(data_path),
            str(events_path),
        )
        if ch_names != ch_names0:
            idx_map = []
            ok = True
            for name in ch_names0:
                if name not in ch_names:
                    ok = False
                    break
                idx_map.append(ch_names.index(name))
            if not ok:
                raise RuntimeError(
                    f"Channel mismatch in {data_path.name}. Cannot align to reference order."
                )
            X = X[:, idx_map]
            ch_names = list(ch_names0)

        Y_all, Z_all = _normalize_series(X, mode=mode, fs=fs, cfg=cfg)
        cycles = extract_grasp_cycles(E, fs=fs, max_cycle_s=float(max_cycle_s))
        if len(cycles) < int(min_cycles):
            print(
                f"[WARN] {data_path.name}: cycles={len(cycles)} < min_cycles={min_cycles}"
            )
            continue

        n_files_used += 1
        n_cycles_total += int(len(cycles))
        manifest["used_files"].append(data_path.name)

        for cyc in cycles:
            for pi, event_name in enumerate(anchor_events):
                t_event = int(cyc[event_name])
                start_idx = t_event - t_context
                if start_idx < 0 or t_event > X.shape[0]:
                    continue

                Yr = Y_all[start_idx:t_event, :]
                Zr = Z_all[start_idx:t_event, :]
                if Yr.shape[0] != t_context or Zr.shape[0] != t_context:
                    continue

                Fy_full = np.empty((t_context, c_dim, m), dtype=float)
                Fz_full = np.empty((t_context, c_dim, m), dtype=float)
                for c in range(c_dim):
                    Fy_full[:, c, :] = _basis_mixed(Yr[:, c], m)
                    Fz_full[:, c, :] = _basis_mixed(Zr[:, c], m)

                used_any_lag = False
                for li, lag in enumerate(lag_samples):
                    aligned = _align_basis_for_lag(Fy_full, Fz_full, lag)
                    if aligned is None:
                        continue

                    Fy, Fz = aligned
                    if Fy.shape[0] < t_eval + max(1, t_burn):
                        continue

                    M_ema = _compute_dynamic_coeffs_ema(
                        Fy,
                        Fz,
                        alpha,
                        subtract_marginals=subtract_marginals,
                    )
                    M_eval = M_ema[-t_eval:, :, :, :]
                    coeffs_sum_t[pi, li] += M_eval.astype(np.float32, copy=False)
                    n_windows[pi, li] += 1
                    used_any_lag = True

                if used_any_lag:
                    n_anchor_windows_total += 1

    group_out = out_dir / group_name
    group_out.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(-t_eval + 1, 1, dtype=float) / float(fs)

    for pi, event_name in enumerate(anchor_events):
        for li, lag_ms in enumerate(lags_ms):
            if n_windows[pi, li] <= 0:
                continue

            coeffs_mean_t = coeffs_sum_t[pi, li] / float(n_windows[pi, li])
            scores_t = np.sqrt(np.sum(coeffs_mean_t**2, axis=-1))
            mat_summary = _summarize_scores(scores_t, summary)
            mat_disp = np.array(mat_summary, float)
            np.fill_diagonal(mat_disp, 0.0)

            common_name = (
                f"{event_name}_{mode}_m{m}_lag{lag_ms}ms_{metric}_{summary}"
            )
            top_edges_txt = group_out / f"event_top_edges_{common_name}.txt"
            heatmap_png = group_out / f"event_matrix_{common_name}.png"
            traces_png = group_out / f"event_traces_{common_name}.png"
            event_manifest: dict[str, Any] = {
                "anchor_event": event_name,
                "lag_ms": int(lag_ms),
                "windows_used": int(n_windows[pi, li]),
                "outputs": [],
            }

            with open(top_edges_txt, "w", encoding="utf-8") as handle:
                handle.write(f"anchor_event={event_name}\n")
                handle.write(
                    f"group={group_name} mode={mode} m={m} metric={metric} summary={summary}\n"
                )
                handle.write(
                    f"pre_s={pre_s} burn_in_s={burn_in_s} ema_half_life_s={ema_half_life_s} lag_ms={lag_ms}\n"
                )
                handle.write(
                    f"files={n_files_used} cycles_total={n_cycles_total} anchor_windows_total={n_anchor_windows_total} windows_used={int(n_windows[pi, li])}\n"
                )
                handle.write(f"config={asdict(cfg)}\n\n")
            top_pairs = _save_top_edges(
                str(top_edges_txt),
                mat_disp,
                ch_names0,
                topk=topk,
            )

            title = (
                f"{event_name} | group={group_name} | mode={mode} | m={m} | "
                f"lag={lag_ms} ms | summary={summary} | files={n_files_used}"
            )
            _plot_matrix(
                out_path=heatmap_png,
                mat=mat_disp,
                ch_names=ch_names0,
                title=title,
                cbar_label=f"EMA {metric}",
            )
            _plot_edge_traces(
                out_path=traces_png,
                scores_t=scores_t,
                time_axis=time_axis,
                top_pairs=top_pairs,
                ch_names=ch_names0,
                trace_topk=trace_topk,
                title=(
                    f"EMA trajectories before {event_name} | lag={lag_ms} ms | "
                    f"group={group_name}"
                ),
            )
            event_manifest["outputs"].extend(
                [heatmap_png.name, top_edges_txt.name, traces_png.name]
            )

            if save_npz:
                out_npz = group_out / f"event_data_{common_name}.npz"
                _save_npz_payload(
                    out_npz,
                    scores_t=np.asarray(scores_t, dtype=np.float32),
                    mat_summary=np.asarray(mat_summary, dtype=np.float32),
                    time_s=np.asarray(time_axis, dtype=np.float32),
                    ch_names=np.asarray(ch_names0, dtype=str),
                    top_pairs=np.asarray(top_pairs, dtype=np.int32),
                    group=np.array(group_name),
                    anchor_event=np.array(event_name),
                    mode=np.array(mode),
                    metric=np.array(metric),
                    summary=np.array(summary),
                    lag_ms=np.array(int(lag_ms)),
                    pre_s=np.array(float(pre_s)),
                    burn_in_s=np.array(float(burn_in_s)),
                    ema_half_life_s=np.array(float(ema_half_life_s)),
                    m=np.array(int(m)),
                    files_used=np.array(int(n_files_used)),
                    cycles_total=np.array(int(n_cycles_total)),
                    anchor_windows_total=np.array(int(n_anchor_windows_total)),
                    windows_used=np.array(int(n_windows[pi, li])),
                )
                event_manifest["outputs"].append(out_npz.name)
                print("Saved:", str(out_npz))

            manifest["events"].append(event_manifest)

            print("Saved:", str(top_edges_txt))
            print("Saved:", str(heatmap_png))
            print("Saved:", str(traces_png))

    manifest["files_requested"] = len(files)
    manifest["files_used"] = int(n_files_used)
    manifest["cycles_total"] = int(n_cycles_total)
    manifest["anchor_windows_total"] = int(n_anchor_windows_total)
    _write_json(group_out / "run_manifest.json", manifest)
    print("Saved:", str(group_out / "run_manifest.json"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Pre-event EEG connectivity analysis with EMA dynamics. "
            "Use this for readiness / decision-state questions rather than replacing "
            "the phase-to-phase analysis."
        )
    )
    ap.add_argument("--root", required=True, help="Dataset root or train/ directory.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--mode", choices=["corr", "gc"], default="gc")

    ap.add_argument(
        "--anchor-events",
        nargs="+",
        default=DEFAULT_ANCHOR_EVENTS,
        choices=KAGGLE_EVENT_COLS,
        help="Events used as anchors for pre-event windows.",
    )
    ap.add_argument(
        "--pre-s",
        type=float,
        default=0.5,
        help="Evaluation window length before the anchor event in seconds.",
    )
    ap.add_argument(
        "--burn-in-s",
        type=float,
        default=0.5,
        help="Extra context before the evaluation window to stabilize the EMA.",
    )
    ap.add_argument(
        "--ema-half-life-s",
        type=float,
        default=0.1,
        help="EMA half-life in seconds.",
    )
    ap.add_argument("--max-cycle-s", type=float, default=6.0)
    ap.add_argument("--min-cycles", type=int, default=10)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument(
        "--lags-ms",
        type=int,
        nargs="+",
        default=[0, 50, 100, 150, 200],
    )
    ap.add_argument(
        "--summary",
        choices=["final", "mean", "max"],
        default="final",
        help="How to summarize the EMA trajectory into a matrix for ranking.",
    )
    ap.add_argument("--metric", choices=["energy"], default="energy")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--trace-topk", type=int, default=5)
    ap.add_argument(
        "--save-npz",
        action="store_true",
        help="Save machine-readable .npz payloads alongside plots and text files.",
    )
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--series", type=int, nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--group-by", choices=["all", "subject"], default="all")
    args = ap.parse_args()

    if args.pre_s <= 0:
        raise ValueError("--pre-s must be positive.")
    if args.burn_in_s < 0:
        raise ValueError("--burn-in-s must be non-negative.")
    if args.ema_half_life_s <= 0:
        raise ValueError("--ema-half-life-s must be positive.")
    if not args.anchor_events:
        raise ValueError("At least one anchor event is required.")

    if max(args.lags_ms) > int(round(args.pre_s * 1000.0)):
        print(
            "[WARN] max lag exceeds pre-event window length. This is allowed, "
            "but interpretation is easier when pre_s >= maxlag."
        )

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_all = _iter_train_data_files(root)
    files = _filter_files(files_all, args.subjects, args.series, args.max_files)
    if not files:
        raise RuntimeError("No matching train data files found.")

    if args.group_by == "all":
        _run_group(
            group_name="ALL",
            files=files,
            out_dir=out_dir,
            mode=args.mode,
            anchor_events=args.anchor_events,
            pre_s=float(args.pre_s),
            burn_in_s=float(args.burn_in_s),
            ema_half_life_s=float(args.ema_half_life_s),
            max_cycle_s=float(args.max_cycle_s),
            min_cycles=int(args.min_cycles),
            m=int(args.m),
            lags_ms=list(args.lags_ms),
            metric=str(args.metric),
            summary=str(args.summary),
            topk=int(args.topk),
            trace_topk=int(args.trace_topk),
            save_npz=bool(args.save_npz),
        )
        return

    groups: dict[str, list[Path]] = {}
    for path in files:
        subj, _series = _parse_subj_series(path)
        if subj is None:
            continue
        key = f"subj{subj:02d}"
        groups.setdefault(key, []).append(path)

    for key in sorted(groups.keys()):
        _run_group(
            group_name=key,
            files=groups[key],
            out_dir=out_dir,
            mode=args.mode,
            anchor_events=args.anchor_events,
            pre_s=float(args.pre_s),
            burn_in_s=float(args.burn_in_s),
            ema_half_life_s=float(args.ema_half_life_s),
            max_cycle_s=float(args.max_cycle_s),
            min_cycles=int(args.min_cycles),
            m=int(args.m),
            lags_ms=list(args.lags_ms),
            metric=str(args.metric),
            summary=str(args.summary),
            topk=int(args.topk),
            trace_topk=int(args.trace_topk),
            save_npz=bool(args.save_npz),
        )


if __name__ == "__main__":
    main()
