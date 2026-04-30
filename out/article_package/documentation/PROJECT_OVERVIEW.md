# Project Overview

## Scope

The repository contains the final Kaggle Grasp-and-Lift EEG analysis pipeline.
Non-core scripts, exploratory outputs, old reports, cache files, and alternate
dataset paths are stored under `legacy/`.

## Active Structure

```text
configs/      configuration defaults
data/         active Kaggle data
docs/         project documentation
out/          final result folders
scripts/      executable pipeline modules
src_mfg/      reusable analysis code
tests/        automated tests
legacy/       archived non-core material
```

## Active Outputs

```text
out/basis/
out/phase_mats_pca_by_subject/
out/experimental_kaggle_by_subject/
out/article_meta_kaggle/
out/article_meta_sensitivity/
out/article_package/
```

## Pipeline

```text
Build PCA Basis
Phase Analysis
Pre-Event EMA
Meta Analysis
Sensitivity Analysis
Article Package
```

## Core Scripts

```text
scripts/build_kaggle_basis_allpairs_batch.py
scripts/mfg_kaggle_phase_matrix_batch.py
scripts/mfg_kaggle_phase_matrix_experimental.py
scripts/meta_analysis.py
scripts/meta_sensitivity.py
scripts/build_article_package.py
scripts/gui_app.py
```

## Validation

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall src_mfg scripts tests
```

## Reporting Standard

A result is suitable for article use when it:

- is reproducible across subjects,
- survives the selected FDR setting,
- remains stable in the sensitivity grid,
- has a clear phase and lag interpretation,
- is available in machine-readable form,
- has matching provenance in output metadata.

