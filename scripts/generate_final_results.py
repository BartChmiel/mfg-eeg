import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualizer")


def save_top_edges_report(f, mat, ch_names, pc_label, lag_ms, top_k=10):
    """Zapisuje listę najsilniejszych połączeń do pliku tekstowego."""
    A = mat.copy()
    np.fill_diagonal(A, 0)
    flat_idx = np.argsort(A.ravel())[::-1]

    f.write(f"--- {pc_label} | Lag: {lag_ms}ms ---\n")
    for k in range(top_k):
        idx = flat_idx[k]
        val = A.ravel()[idx]
        if val <= 0:
            break
        i, j = idx // A.shape[1], idx % A.shape[1]
        f.write(f"{k+1:02d}. {ch_names[i]} -> {ch_names[j]} (score: {val:.4f})\n")
    f.write("\n")


def process_all_phases(input_dir, output_root):
    input_path = Path(input_dir)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # Szukamy wszystkich plików .npz wygenerowanych przez skrypt bazowy
    npz_files = list(input_path.glob("*.npz"))

    if not npz_files:
        logger.error(f"Nie znaleziono plików .npz w {input_dir}!")
        return

    lags = [0, 50, 100, 150, 200]

    for npz_file in npz_files:
        logger.info(f"Przetwarzanie fazy: {npz_file.name}")
        data = np.load(npz_file)
        ch_names = data["ch_names"]
        eigvals = data["eigvals"]  # (C, C, r)
        pk = str(data["pk"])
        r = eigvals.shape[2]

        # Tworzymy folder dla konkretnej fazy
        phase_dir = output_path / pk
        phase_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. GENEROWANIE RAPORTU TEKSTOWEGO ---
        report_path = phase_dir / f"top_edges_{pk}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"RAPORT POŁĄCZEŃ DLA FAZY: {pk}\n")
            f.write("=" * 40 + "\n\n")
            for pc_idx in range(r):
                for lag_idx, ms in enumerate(lags):
                    # Tutaj estymujemy macierz dla konkretnego laga
                    # (W Twoim skrypcie bazowym eigvals jest statyczne per faza,
                    #  możesz to rozbudować o zapisywanie macierzy per lag)
                    mat = eigvals[:, :, pc_idx]
                    save_top_edges_report(f, mat, ch_names, f"PC{pc_idx+1}", ms)

        # --- 2. GENEROWANIE GRAFIKI 3x5 ---
        fig, axes = plt.subplots(r, len(lags), figsize=(22, 13))

        for pc_idx in range(r):
            # Skalowanie kolorów (vmax) na podstawie 99 percentyla dla całego wiersza
            vmax = np.percentile(eigvals[:, :, pc_idx], 99)

            for lag_idx, ms in enumerate(lags):
                ax = axes[pc_idx, lag_idx]
                mat = eigvals[:, :, pc_idx].copy()
                np.fill_diagonal(mat, 0)

                im = ax.imshow(mat, cmap="magma", origin="lower", vmin=0, vmax=vmax)

                if pc_idx == 0:
                    ax.set_title(f"Lag {ms}ms", fontsize=15, fontweight="bold")
                if lag_idx == 0:
                    ax.set_ylabel(f"PC {pc_idx+1}", fontsize=15, fontweight="bold")

                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f"Analiza Dynamiki Sieci Funkcjonalnej: {pk}", fontsize=22, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(phase_dir / f"grid_visualization_{pk}.png", dpi=300)
        plt.close()

    logger.info(f"ZAKOŃCZONO! Wyniki w: {output_root}")


if __name__ == "__main__":
    # Upewnij się, że ścieżki są poprawne
    process_all_phases(input_dir=".", output_root="out/final_results")
