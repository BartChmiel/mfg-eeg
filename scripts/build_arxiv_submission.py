from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("article_dense_pca_modes_by_lag.pdf", "top_edges_replication.png",
           "classification_timing_control.png")
TABLES = ("primary_edge_sensitivity.tex",)


def build_submission(output: Path, *, package: Path, source: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mfg_arxiv_") as temporary:
        stage = Path(temporary)
        (stage / "figures").mkdir()
        (stage / "tables").mkdir()
        shutil.copy2(source, stage / "eeg_mfg_article.tex")
        shutil.copy2(source.with_name("references.bib"), stage / "references.bib")
        for name in FIGURES:
            shutil.copy2(package / "figures" / name, stage / "figures" / name)
        for name in TABLES:
            shutil.copy2(package / "tables" / name, stage / "tables" / name)
        commands = [
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eeg_mfg_article.tex"],
            ["bibtex", "eeg_mfg_article"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eeg_mfg_article.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eeg_mfg_article.tex"],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=stage, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{' '.join(command)} failed:\n{result.stdout[-6000:]}\n{result.stderr}")
        log = (stage / "eeg_mfg_article.log").read_text(encoding="utf-8", errors="replace")
        if "undefined" in log or "Overfull" in log:
            raise RuntimeError("Unresolved references or overflowing content in the submission PDF")
        archive = output / "arxiv-source.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in ["eeg_mfg_article.tex", "references.bib", "eeg_mfg_article.bbl"]:
                bundle.write(stage / name, name)
            for name in FIGURES:
                bundle.write(stage / "figures" / name, f"figures/{name}")
            for name in TABLES:
                bundle.write(stage / "tables" / name, f"tables/{name}")
        shutil.copy2(stage / "eeg_mfg_article.pdf", output / "eeg_mfg_article.pdf")
        shutil.copy2(stage / "eeg_mfg_article.log", output / "build.log")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and locally compile a self-contained arXiv source archive.")
    parser.add_argument("--output", type=Path, default=ROOT / "out/arxiv_submission")
    parser.add_argument("--package", type=Path, default=ROOT / "out/article_package")
    parser.add_argument("--source", type=Path, default=ROOT / "docs/eeg_mfg_article.tex")
    args = parser.parse_args()
    print(build_submission(args.output, package=args.package, source=args.source))


if __name__ == "__main__":
    main()
