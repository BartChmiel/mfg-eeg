import numpy as np
import matplotlib.pyplot as plt


def plot_features(
    ms: np.ndarray, scores: np.ndarray, gc: np.ndarray, title: str, outpath: str
):
    """
    ms: lags in milliseconds (T,)
    scores: (r, T)
    gc: (T,)
    """
    plt.figure(figsize=(10, 4))
    for i in range(scores.shape[0]):
        plt.plot(ms, scores[i], label=f"PC{i+1}", linewidth=1)
    plt.plot(ms, gc, label="GC(delta)", linewidth=2)
    plt.axvline(0, color="k", lw=0.5)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("lag [ms]")
    plt.ylabel("scores / GC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_gc_matrix(gc_mat: np.ndarray, ch_names: list[str], title: str, outpath: str):
    plt.figure(figsize=(7, 6))
    plt.imshow(gc_mat, origin="lower", interpolation="nearest")
    plt.colorbar(label="GC(delta)")
    plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=8)
    plt.yticks(range(len(ch_names)), ch_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
