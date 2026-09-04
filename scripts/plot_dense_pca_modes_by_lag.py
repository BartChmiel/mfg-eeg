from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import TwoSlopeNorm


DEFAULT_SCORES = (
    "out/pc_structure_diagnostics/preprocessing_comparison/"
    "bandpass_0_5_48/pooled/ALL/"
    "phasegrid_data_FirstDigitTouch__BothStartLoadPhase_"
    "gc_m4_pca_r3_lags0-50-100-150-200.npz"
)
DEFAULT_BASIS = "out/basis/basis_allpairs_gc_m4_bandpass.npz"
DEFAULT_OUT_BASE = "out/article_package/figures/article_dense_pca_modes_by_lag"

EXPECTED_PHASE = "FirstDigitTouch__BothStartLoadPhase"
EXPECTED_PREPROCESS = "bandpass_0_5_48"
EXPECTED_MODE = "gc"
EXPECTED_FS = 500
EXPECTED_M = 4
EXPECTED_PCA_R = 3
EXPECTED_LAGS_MS = [0, 50, 100, 150, 200]
EXPECTED_CHANNELS = 32


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "axes.linewidth": 0.45,
            "figure.dpi": 180,
            "savefig.dpi": 420,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else value


def _load_scores(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as data:
        required = {
            "scores",
            "lags_ms",
            "ch_names",
            "phase",
            "group",
            "mode",
            "m",
            "pca_r",
            "files_used",
            "cycles_total",
            "windows_total",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")

        scores = np.asarray(data["scores"], dtype=float)
        lags_ms = [int(value) for value in data["lags_ms"].tolist()]
        ch_names = [str(value) for value in data["ch_names"].tolist()]
        metadata = {
            "phase": str(_scalar(data["phase"])),
            "group": str(_scalar(data["group"])),
            "mode": str(_scalar(data["mode"])),
            "m": int(_scalar(data["m"])),
            "pca_r": int(_scalar(data["pca_r"])),
        }

    expected_shape = (
        EXPECTED_PCA_R,
        len(EXPECTED_LAGS_MS),
        EXPECTED_CHANNELS,
        EXPECTED_CHANNELS,
    )
    if scores.shape != expected_shape:
        raise ValueError(f"Expected scores shape {expected_shape}; got {scores.shape}")
    if lags_ms != EXPECTED_LAGS_MS:
        raise ValueError(f"Expected lags {EXPECTED_LAGS_MS}; got {lags_ms}")
    if len(ch_names) != EXPECTED_CHANNELS or len(set(ch_names)) != EXPECTED_CHANNELS:
        raise ValueError(f"Expected {EXPECTED_CHANNELS} unique channel names")

    expected_metadata = {
        "phase": EXPECTED_PHASE,
        "group": "ALL",
        "mode": EXPECTED_MODE,
        "m": EXPECTED_M,
        "pca_r": EXPECTED_PCA_R,
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise ValueError(f"Expected {key}={expected}; got {metadata[key]}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite score found in {path}")
    return scores, ch_names


def _validate_basis(path: Path, ch_names: list[str]) -> None:
    with np.load(path, allow_pickle=True) as data:
        required = {
            "U",
            "mean",
            "ch_names",
            "fs",
            "lags_ms",
            "lag_samples",
            "m",
            "mode",
            "preprocess",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")

        basis_channels = [str(value) for value in data["ch_names"].tolist()]
        observed = {
            "fs": int(_scalar(data["fs"])),
            "lags_ms": [int(value) for value in data["lags_ms"].tolist()],
            "lag_samples": [int(value) for value in data["lag_samples"].tolist()],
            "m": int(_scalar(data["m"])),
            "mode": str(_scalar(data["mode"])),
            "preprocess": str(_scalar(data["preprocess"])),
            "U_shape": list(np.asarray(data["U"]).shape),
            "mean_shape": list(np.asarray(data["mean"]).shape),
        }

    if basis_channels != ch_names:
        raise ValueError("Score payload and PCA basis channel orders differ")
    expected = {
        "fs": EXPECTED_FS,
        "lags_ms": EXPECTED_LAGS_MS,
        "lag_samples": [
            int(round(lag_ms * EXPECTED_FS / 1000.0))
            for lag_ms in EXPECTED_LAGS_MS
        ],
        "m": EXPECTED_M,
        "mode": EXPECTED_MODE,
        "preprocess": EXPECTED_PREPROCESS,
        "U_shape": [
            EXPECTED_CHANNELS,
            EXPECTED_CHANNELS,
            EXPECTED_PCA_R,
            EXPECTED_M**2,
        ],
        "mean_shape": [EXPECTED_CHANNELS, EXPECTED_CHANNELS, EXPECTED_M**2],
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"Expected basis {key}={value}; got {observed[key]}")


def _mask_diagonal(scores: np.ndarray) -> np.ndarray:
    matrices = np.array(scores, dtype=float, copy=True)
    diagonal = np.arange(matrices.shape[-1])
    matrices[:, :, diagonal, diagonal] = np.nan
    return matrices


def _scale_maxima(matrices: np.ndarray, strategy: str) -> list[float]:
    if strategy == "global":
        vmax = float(np.nanmax(np.abs(matrices)))
        maxima = [vmax] * matrices.shape[0]
    elif strategy == "per-pc":
        maxima = [
            float(np.nanmax(np.abs(matrices[pc_index])))
            for pc_index in range(matrices.shape[0])
        ]
    else:
        raise ValueError(f"Unknown scale strategy: {strategy}")
    if any(not np.isfinite(value) or value <= 0.0 for value in maxima):
        raise ValueError("Color-scale maxima must be positive and finite")
    return maxima


def build_figure(
    *,
    scores_path: Path | str = DEFAULT_SCORES,
    basis_path: Path | str = DEFAULT_BASIS,
    out_base: Path | str = DEFAULT_OUT_BASE,
    scale_strategy: str = "per-pc",
) -> list[Path]:
    scores_path = Path(scores_path)
    basis_path = Path(basis_path)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    scores, ch_names = _load_scores(scores_path)
    _validate_basis(basis_path, ch_names)
    matrices = _mask_diagonal(scores)
    scale_maxima = _scale_maxima(matrices, scale_strategy)

    plt = _load_pyplot()
    fig, axes = plt.subplots(
        EXPECTED_PCA_R,
        len(EXPECTED_LAGS_MS),
        figsize=(7.16, 4.65),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.885,
        top=0.945,
        bottom=0.155,
        wspace=0.055,
        hspace=0.085,
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    tick_positions = np.arange(0, EXPECTED_CHANNELS, 2)
    tick_labels = [ch_names[index] for index in tick_positions]
    row_images = []

    for pc_index in range(EXPECTED_PCA_R):
        norm = TwoSlopeNorm(
            vmin=-scale_maxima[pc_index],
            vcenter=0.0,
            vmax=scale_maxima[pc_index],
        )
        image = None
        for lag_index, lag_ms in enumerate(EXPECTED_LAGS_MS):
            axis = axes[pc_index, lag_index]
            image = axis.imshow(
                matrices[pc_index, lag_index],
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_xlim(-0.5, EXPECTED_CHANNELS - 0.5)
            axis.set_ylim(EXPECTED_CHANNELS - 0.5, -0.5)
            axis.set_xticks(tick_positions)
            axis.set_yticks(tick_positions)
            axis.tick_params(axis="both", length=1.0, width=0.3, pad=0.45)

            if pc_index == 0:
                axis.set_title(f"{lag_ms} ms", fontsize=7.2, pad=2.5)
            if pc_index == EXPECTED_PCA_R - 1:
                axis.set_xticklabels(
                    tick_labels,
                    rotation=90,
                    fontsize=4.1,
                    ha="center",
                )
            else:
                axis.tick_params(axis="x", labelbottom=False)
            if lag_index == 0:
                axis.set_yticklabels(tick_labels, fontsize=4.1)
                axis.set_ylabel(f"PC{pc_index + 1}", fontsize=7.0, labelpad=14.0)
            else:
                axis.tick_params(axis="y", labelleft=False)

        if image is None:
            raise RuntimeError(f"No matrices plotted for PC{pc_index + 1}")
        row_images.append(image)

    if scale_strategy == "global":
        color_axis = fig.add_axes([0.905, 0.205, 0.014, 0.67])
        colorbar = fig.colorbar(row_images[0], cax=color_axis)
        colorbar.set_label("Signed pair-specific PCA score", fontsize=6.0, labelpad=3.0)
        colorbar.ax.tick_params(labelsize=5.1, length=1.5, width=0.35)
        colorbar.outline.set_linewidth(0.4)
    else:
        row_height = (0.945 - 0.155) / EXPECTED_PCA_R
        for pc_index, image in enumerate(row_images):
            y = 0.155 + (EXPECTED_PCA_R - 1 - pc_index) * row_height
            color_axis = fig.add_axes([0.905, y + 0.08 * row_height, 0.012, 0.78 * row_height])
            colorbar = fig.colorbar(image, cax=color_axis)
            colorbar.set_label(
                f"Signed PC{pc_index + 1} score",
                fontsize=5.1,
                labelpad=2.5,
            )
            colorbar.ax.tick_params(labelsize=4.6, length=1.3, width=0.35)
            colorbar.outline.set_linewidth(0.4)
        fig.text(
            0.925,
            0.965,
            "Separate scales by PC row",
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="semibold",
        )

    fig.text(0.49, 0.035, "Target sensor", ha="center", va="center", fontsize=7.0)
    fig.text(
        0.018,
        0.55,
        "Source sensor",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.0,
    )

    outputs = [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]
    fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        outputs[0],
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Title": "Dense pair-specific PCA modes by lag",
            "Author": "mfg-eeg",
            "Creator": "scripts/plot_dense_pca_modes_by_lag.py",
            "Producer": "Matplotlib",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed pair-specific PC1-PC3 score matrices across five lags "
            "for the final band-pass analysis."
        )
    )
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--basis", default=DEFAULT_BASIS)
    parser.add_argument("--out-base", default=DEFAULT_OUT_BASE)
    parser.add_argument(
        "--scale-strategy",
        choices=["global", "per-pc"],
        default="per-pc",
    )
    args = parser.parse_args()
    outputs = build_figure(
        scores_path=args.scores,
        basis_path=args.basis,
        out_base=args.out_base,
        scale_strategy=args.scale_strategy,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
