import numpy as np
import matplotlib.pyplot as plt


def plot_gc_matrix(
    gc_mat: np.ndarray,
    ch_names: list[str],
    title: str,
    outpath: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    """
    Generates a square heatmap for global Granger-like connectivity matrices.
    """
    gc_mat = np.asarray(gc_mat, float)
    n = gc_mat.shape[0]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        gc_mat, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap
    )

    plt.colorbar(im).set_label("GC Intensity")
    plt.xticks(range(n), ch_names, rotation=90, fontsize=8)
    plt.yticks(range(n), ch_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_pc_features(
    lags: np.ndarray,
    scores: np.ndarray,
    title: str,
    outpath: str | None = None,
    gc: np.ndarray | None = None,
    lags_in_ms: bool = True,
    normalize_each: bool = True,
) -> None:
    """
    Plots PCA feature trajectories over time lags with an optional GC scalar overlay.
    """
    scores = np.atleast_2d(np.asarray(scores, float))
    x = np.asarray(lags, float)
    r, T = scores.shape

    Y = scores.copy()
    if normalize_each:
        for i in range(r):
            denom = np.max(np.abs(Y[i])) or 1.0
            Y[i] /= denom

    plt.figure(figsize=(10, 4))
    for i in range(min(3, r)):
        plt.plot(x, Y[i], label=f"PC{i+1}", linewidth=1.4)

    if gc is not None:
        gc_norm = gc / (np.max(gc) or 1.0)
        plt.plot(x, gc_norm, "k", label="GC (Scalar)", linewidth=2.0)

    plt.axhline(0.0, color="k", lw=0.7, alpha=0.7)
    plt.xlabel("Lag [ms]" if lags_in_ms else "Lag [s]")
    plt.ylabel("Projection / GC (Normalized)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=160)
        plt.close()
    else:
        plt.show()


def plot_granger_dicc(
    t: np.ndarray,
    scores: np.ndarray,
    title: str,
    outpath: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Standard DICC-style plot for the top three PCA components (Direct, Indirect, Common Cause).
    """
    scores = np.atleast_2d(np.asarray(scores, float))
    x, (r, T) = np.asarray(t, float), scores.shape

    labels = ["Direct", "Indirect", "Common Cause"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(4.5, 3))
    for i in range(min(3, r)):
        plt.plot(x, scores[i], label=labels[i], linewidth=1.4, color=colors[i])

    plt.xlabel("Lag [s]")
    plt.ylabel("Projection Score")
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()
