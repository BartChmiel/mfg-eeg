# MFG EEG Grasp-and-Lift Analysis

This repository analyzes the public Kaggle Grasp-and-Lift EEG Detection dataset
with event-locked, multi-feature lagged dependence. The main result is a
cross-subject sensor-level connectivity analysis; classification is a secondary
check of whether MFG edge features add information beyond direct EEG features.

Authors: Adrian Przybysz, Bartlomiej Chmiel, Jarek Duda.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the Kaggle files in `data/grasp-and-lift-eeg-detection/train/` and
`data/grasp-and-lift-eeg-detection/test/`. Raw data and exploratory outputs are
ignored by Git.

## Repository Contents

- `docs/` contains the final article PDF/source, bibliography, and technical
  method notes.
- `src_mfg/` contains reusable preprocessing, normalization, basis, montage,
  configuration, and IO code.
- `scripts/` contains the article pipeline and reproducibility utilities.
- `out/article_package/` contains the final evidence bundle used by the article:
  compact tables, selected figures, reports, and provenance metadata.
- `tests/` contains regression checks for the pipeline components.

Raw Kaggle data, local virtual environments, temporary logs, caches, private
transfer bundles, and exploratory outputs are intentionally ignored.

## Run

Run the cleaned article workflow:

```powershell
python -m scripts.run_resumable_article_pipeline
```

The workflow checkpoints the PCA basis after each file, skips completed subject
runs with matching manifests, retries failed steps, and rebuilds the final
evidence bundle. It can be restarted with the same command after an interruption.

To rebuild only the bundle from existing outputs, run:

```powershell
python -m scripts.build_article_package --help
```

The lightweight GUI entry point `launch_gui.bat` is kept for local workflow
execution, but the command-line pipeline above is the canonical reproduction
path.

## Outputs

- Article source and PDF: `docs/eeg_mfg_article.tex`, `docs/eeg_mfg_article.pdf`
- Technical method notes: `docs/METHODOLOGY.md`
- Final evidence bundle: `out/article_package/`

The filtered analysis leaves five edge instances stable across the full
sensitivity grid. Two pass the distance/lag/asymmetry screen, both involving
Fp2--Fp1, so the paper reports sensor dependence rather than anatomical
causality. The matched classifier timing-control gain is about `+0.0014` AUC.

## Checks

```powershell
python -m compileall src_mfg scripts tests
python -m unittest discover -s tests -v
```
