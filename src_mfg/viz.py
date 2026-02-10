import numpy as np
import matplotlib.pyplot as plt


def plot_features(
    ms: np.ndarray, scores: np.ndarray, gc: np.ndarray, title: str, outpath: str
) -> None:
    """
    Plot PC features and GC as functions of lag in milliseconds.

    Parameters
    ----------
    ms : array_like, shape (T,)
        Lag values in milliseconds.
    scores : array_like, shape (r, T)
        Feature curves (e.g. PCA projections) for r components.
    gc : array_like, shape (T,)
        Granger-like scalar curve evaluated on the same lag grid.
    title : str
        Plot title.
    outpath : str
        Path to the output PNG file.
    """
    ms = np.asarray(ms, float)
    scores = np.asarray(scores, float)
    gc = np.asarray(gc, float)

    if scores.ndim == 1:
        scores = scores[None, :]
    r, T = scores.shape
    assert ms.shape[0] == T, "plot_features: ms and scores must match in length"
    assert gc.shape[0] == T, "plot_features: gc and scores must match in length"

    plt.figure(figsize=(10, 4))
    for i in range(r):
        plt.plot(ms, scores[i], label=f"PC{i+1}", linewidth=1.0)
    plt.plot(ms, gc, label="GC(lag)", linewidth=2.0, color="k")

    plt.axvline(0.0, color="k", lw=0.5)
    plt.axhline(0.0, color="k", lw=0.5)
    plt.xlabel("lag [ms]")
    plt.ylabel("projection / GC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_gc_matrix(
    gc_mat: np.ndarray,
    ch_names: list[str],
    title: str,
    outpath: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    gc_mat = np.asarray(gc_mat, float)
    n = gc_mat.shape[0]
    assert gc_mat.shape == (n, n), "plot_gc_matrix: gc_mat must be square"
    assert len(ch_names) == n, "plot_gc_matrix: ch_names length must match gc_mat"

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        gc_mat,
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("GC summary")

    plt.xticks(range(n), ch_names, rotation=90, fontsize=8)
    plt.yticks(range(n), ch_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_pc_features(
    lags,
    scores: np.ndarray,
    title: str,
    outpath: str | None = None,
    gc: np.ndarray | None = None,
    lags_in_ms: bool = True,
    normalize_each: bool = True,
) -> None:
    """
    Plot PC features vs lag, with optional GC overlay.

    Parameters
    ----------
    lags : array_like, shape (T,)
        Lag grid in milliseconds or seconds (see lags_in_ms).
    scores : array_like, shape (r, T)
        Feature curves (e.g. PCA components).
    title : str
        Plot title.
    outpath : str or None
        If given, save to this path. Otherwise show interactively.
    gc : array_like, shape (T,), optional
        Optional scalar GC curve to overlay (normalized to [0,1]).
    lags_in_ms : bool, default True
        If True, x-axis is interpreted as milliseconds. Otherwise as seconds.
    normalize_each : bool, default True
        If True, each feature is divided by its max absolute value.
    """
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[None, :]
    r, T = scores.shape

    x = np.asarray(lags, float)
    assert x.shape[0] == T, "plot_pc_features: lags and scores must match in length"

    xlabel = "lag [ms]" if lags_in_ms else "lag [s]"

    Y = scores.copy()
    if normalize_each:
        for i in range(r):
            m = float(np.max(np.abs(Y[i])) or 1.0)
            Y[i] /= m

    plt.figure(figsize=(10, 4))
    for i in range(min(3, r)):
        plt.plot(x, Y[i], label=f"PC{i+1}", linewidth=1.4)

    if gc is not None:
        gc = np.asarray(gc, float)
        assert gc.shape[0] == T, "plot_pc_features: gc must match lags length"
        gc_norm = gc / (np.max(gc) or 1.0)
        plt.plot(x, gc_norm, "k", label="GC", linewidth=2.0)

    plt.axhline(0.0, color="k", lw=0.7, alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("projection / GC (norm.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=160)
        plt.close()
    else:
        plt.show()


def plot_granger_simple(
    lag_samples: np.ndarray,
    scores: np.ndarray,
    fs: float,
    title: str,
    outpath: str | None = None,
) -> None:
    """
    Simple PC1-PC3 vs lag >= 0, x-axis in seconds.

    Parameters
    ----------
    lag_samples : array_like, shape (T,)
        Lag offsets in samples (>= 0).
    scores : array_like, shape (r, T) or (T,)
        PCA scores over lags (A_ab or subset).
    fs : float
        Sampling frequency in Hz (used to convert samples to seconds).
    title : str
        Plot title.
    outpath : str or None
        If given, save to this path. Otherwise show interactively.
    """
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[None, :]
    r, T = scores.shape

    x = np.asarray(lag_samples, float) / float(fs)
    assert x.shape[0] == T, "plot_granger_simple: lag_samples and scores must match"

    plt.figure(figsize=(10, 4))
    for i in range(min(3, r)):
        plt.plot(x, scores[i], label=f"PC{i+1}", linewidth=1.4)

    plt.xlabel("lag [s]")
    plt.ylabel("projection")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if outpath is not None:
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
    xlabel: str = "lag [s]",
) -> None:
    """
    Granger-like curves for the first three PCA features (DICC-style).

    Parameters
    ----------
    t : array_like, shape (T,)
        Time / lag axis in seconds (typically lag >= 0).
    scores : array_like, shape (r, T) or (T,)
        Feature curves over time; first three are plotted if available.
    title : str
        Plot title.
    outpath : str or None
        If given, save to this path. Otherwise show interactively.
    ylim : tuple(float, float) or None
        Optional y-limits for the main axis.
    xlabel : str, default "lag [s]"
        Label for the x-axis (no special symbols by default).
    """
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[None, :]
    r, T = scores.shape

    x = np.asarray(t, float)
    assert x.shape[0] == T, "plot_granger_dicc: time axis and scores must match"

    labels = ["direct", "indirect", "common cause"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(4.5, 3))
    for i in range(min(3, r)):
        plt.plot(x, scores[i], label=labels[i], linewidth=1.4, color=colors[i])

    plt.xlabel(xlabel)
    plt.ylabel("projection")
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()
