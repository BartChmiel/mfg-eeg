# MFG EEG Grasp-and-Lift Analysis

Event-locked analysis of the public Kaggle Grasp-and-Lift EEG Detection dataset
using multi-feature Granger-style (MFG) lagged dependence. The pipeline aligns
32-channel recordings to movement phases, estimates Legendre mixed moments,
and uses pair-specific PCA to summarize dependence across participants and lags.
A secondary classifier experiment compares MFG features with direct EEG features.

Authors: Bartlomiej Chmiel, Adrian Przybysz, Jarek Duda.

## Main findings

- Ten band-pass edge instances are significant in the fixed reference setting
  (top-k 10, minimum recurrence four, null inflation ten, FDR alpha 0.05).
- The strongest non-zero-lag Fp2 -> Fp1 findings at 50 and 200 ms recur in 10/12
  and 9/12 participants and are retained in 24/27 and 21/27 sensitivity settings.
  No band-pass instance survives all 27 settings.
- The classifier reaches mean best-augmented ROC-AUC 0.8698, with a matched
  aligned-minus-shifted difference of about +0.0014. This increment is descriptive
  because candidate and model selection were not fully nested.

The findings concern sensor-level dependence, not anatomical connectivity.
Phase-dependent ocular activity could explain the frontopolar effects and has
not been ruled out. Shared-source contributions also remain possible.
See the [article](docs/eeg_mfg_article.pdf) and
[evidence summary](out/article_package/article_summary.md) for the full results.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`.
PDF generation also requires PDFLaTeX and BibTeX on `PATH`.

## Reproducing the article

From the repository root:

```powershell
python -m scripts.reproduce_article --source snapshots
```

This rebuilds both preprocessing comparisons, the 27-setting sensitivity grids,
tables, figures, PDF, and arXiv source archive in `out/reproduced_article/`.
It uses archived rankings and retains the reported classifier results; the
original EEG recordings are not needed. Add `--no-pdf` to run without LaTeX.

For a full raw-data run, place the Kaggle training CSVs in
`data/grasp-and-lift-eeg-detection/train/`, then run:

```powershell
python -m scripts.reproduce_article --source raw --output out/raw_reproduction
```

This also refits PCA bases and reruns the aligned/shifted classifiers. It is
substantially more expensive; PCA and subject stages support resuming.
See [input provenance](out/article_package/reproduction/README.md) and the
[technical reference](docs/METHODOLOGY.md) for details.

To compile and package the tracked manuscript without recalculating results:

```powershell
python -m scripts.build_arxiv_submission
```

The PDF and self-contained source ZIP are written to `out/arxiv_submission/`.

## Repository structure

- `docs/`: manuscript source/PDF, bibliography, and methodology.
- `src_mfg/`: preprocessing, normalization, basis, montage, configuration, and I/O.
- `scripts/`: analysis, reproduction, packaging, and a private legacy GUI (`launch_gui.bat`).
- `out/article_package/`: article tables, figures, numerical inputs, and reports.
- `tests/`: regression tests.

The GUI is an experimental helper, outside the publication workflow. Its package
output defaults to `out/gui_runs/article_package/`, and GUI writes to
`out/article_package/` are blocked. Use `scripts.reproduce_article` for the article.

## Tests

```powershell
python -m compileall src_mfg scripts tests
python -m unittest discover -s tests -v
```

## Article

*Event-Locked Multi-Feature Directed Dependence in Grasp-and-Lift EEG*,
Bartlomiej Chmiel, Adrian Przybysz, and Jarek Duda.
[PDF](docs/eeg_mfg_article.pdf) | [LaTeX source](docs/eeg_mfg_article.tex)
