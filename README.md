# MFG EEG: Kaggle Grasp-and-Lift Article Pipeline

This project analyzes the Kaggle Grasp-and-Lift EEG Detection dataset with an
article-focused pipeline for event-locked directed lagged functional
connectivity.

The project is intentionally centered on Kaggle data. Older exploratory paths
can stay in the repository, but the GUI and recommended workflow focus on the
results intended for the article.

## What It Does

The pipeline estimates reproducible directed lagged coupling patterns in EEG:

```text
Kaggle CSV data
-> event alignment and common average reference
-> grasp-cycle reconstruction
-> phase windows and pre-event windows
-> marginal normalization
-> Legendre mixed-moment features
-> all-pairs PCA projection
-> subject-level top directed edges
-> cross-subject meta-analysis with binomial tests and FDR
-> sensitivity grid over meta-analysis parameters
-> article package with tables, summaries, and provenance
```

Use this wording in the article:

```text
directional lagged innovation-coupling / directed functional connectivity
```

Avoid claiming direct anatomical causality. The `gc` mode is not classical VAR
Granger causality; it is an innovation-transformed directional lagged
dependence measure.

## Setup

Recommended on Windows PowerShell:

```powershell
uv venv --python 3.12
.venv\Scripts\activate
pip install -r .\requirements.txt
```

Standard `venv` also works:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r .\requirements.txt
```

## Data Layout

Place the Kaggle train files here:

```text
data/grasp-and-lift-eeg-detection/train/
  subj01_series01_data.csv
  subj01_series01_events.csv
  subj01_series02_data.csv
  subj01_series02_events.csv
  ...
```

The pipeline expects matching `_data.csv` and `_events.csv` files.

Default sampling rate in code: `500 Hz`.

## Clean Repository Layout

Core article files are kept in the root project:

```text
configs/      runtime defaults
data/         active Kaggle data only
docs/         methodology and project overview
out/          final article outputs and source result folders
scripts/      current article pipeline scripts
src_mfg/      reusable core code
tests/        automated checks for the current pipeline
legacy/       old scripts, WAY paths, smoke runs, archives, and non-core outputs
```

Anything not needed for the article path should live under `legacy/`.

## GUI: Recommended Usage

Start the graphical interface:

```powershell
launch_gui.bat
```

or:

```powershell
python -m scripts.gui_app
```

In the GUI, use `Run Full Article Pipeline`. It runs:

```text
1. Build PCA Basis
2. Phase Analysis
3. Pre-Event EMA
4. Meta Analysis
5. Sensitivity Analysis
6. Article Package
```

The GUI also provides:

- Recommended Kaggle article presets.
- Setup/results readiness check.
- File and folder pickers.
- Step-by-step status during full pipeline runs.
- Live logs.
- Stop button.
- Results browser with quick views for article package, meta tables, sensitivity,
  phase results, and pre-event results.
- Preview for `.png`, `.txt`, `.csv`, `.json`, `.md`, and `.npz`.
- One final article package folder with tables and provenance.

## Main CLI Pipeline

You can use the GUI for normal work. The CLI commands below are the same
article path, useful for reproducibility and debugging.

### 1. Build PCA Basis

```powershell
python -m scripts.build_kaggle_basis_allpairs_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --out out/basis_allpairs.npz `
  --mode gc `
  --epoch-len-s 2.0 `
  --m 4 `
  --lags-ms 0 50 100 150 200 `
  --pca-r 3
```

Output:

```text
out/basis_allpairs.npz
```

### 2. Phase Analysis

```powershell
python -m scripts.mfg_kaggle_phase_matrix_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --out out/phase_mats_pca_by_subject `
  --mode gc `
  --basis-allpairs out/basis_allpairs.npz `
  --pca-r 3 `
  --epoch-len-s 2.0 `
  --m 4 `
  --lags-ms 0 50 100 150 200 `
  --topk 40 `
  --save-npz `
  --group-by subject
```

Outputs per subject:

```text
phasegrid_*.png
phase_top_edges_*.txt
phasegrid_data_*.npz
run_manifest.json
```

### 3. Pre-Event EMA

This analysis checks what builds before key movement events. The default
article window is `[-0.5 s, 0]` before the anchor event, with `0.5 s` burn-in.

```powershell
python -m scripts.mfg_kaggle_phase_matrix_experimental `
  --root data/grasp-and-lift-eeg-detection/train `
  --out out/experimental_kaggle_by_subject `
  --mode gc `
  --anchor-events HandStart LiftOff `
  --m 4 `
  --lags-ms 0 50 100 150 200 `
  --pre-s 0.5 `
  --burn-in-s 0.5 `
  --ema-half-life-s 0.1 `
  --summary final `
  --save-npz `
  --group-by subject
```

Outputs per subject:

```text
event_matrix_*.png
event_top_edges_*.txt
event_traces_*.png
event_data_*.npz
run_manifest.json
```

### 4. Meta-Analysis

```powershell
python -m scripts.meta_analysis `
  --dir out/phase_mats_pca_by_subject `
  --topk 10 `
  --min-subjects 4 `
  --mode gc `
  --use-fdr `
  --significant-only `
  --out-dir out/article_meta_kaggle
```

Outputs:

```text
out/article_meta_kaggle/meta_report.txt
out/article_meta_kaggle/meta_edges.csv
out/article_meta_kaggle/meta_scenarios.csv
out/article_meta_kaggle/meta_report.json
```

### 5. Sensitivity Analysis

This checks whether key edges remain significant when meta-analysis settings
change. Use it to avoid parameter-picked article claims.

```powershell
python -m scripts.meta_sensitivity `
  --dir out/phase_mats_pca_by_subject `
  --out-dir out/article_meta_sensitivity `
  --topk 5 10 15 `
  --min-subjects 3 4 5 `
  --p0-inflate 5 10 15 `
  --mode gc `
  --use-fdr
```

Outputs:

```text
out/article_meta_sensitivity/sensitivity_summary.md
out/article_meta_sensitivity/sensitivity_edges.csv
out/article_meta_sensitivity/edge_stability.csv
out/article_meta_sensitivity/sensitivity_manifest.json
```

### 6. Article Package

```powershell
python -m scripts.build_article_package `
  --phase-dir out/phase_mats_pca_by_subject `
  --pre-event-dir out/experimental_kaggle_by_subject `
  --meta-dir out/article_meta_kaggle `
  --sensitivity-dir out/article_meta_sensitivity `
  --out out/article_package
```

Outputs:

```text
out/article_package/article_summary.md
out/article_package/package_manifest.json
out/article_package/tables/top_scenarios.csv
out/article_package/tables/top_edges.csv
out/article_package/raw_meta_exports/
out/article_package/sensitivity/
out/article_package/documentation/
```

## Key Parameters

- `--mode gc`: recommended article mode; directional lagged innovation-coupling.
- `--mode corr`: symmetric marginal-normalized correlation-like baseline.
- `--m`: number of non-constant Legendre basis terms.
- `--lags-ms`: target lag grid in milliseconds.
- `--epoch-len-s`: fixed phase-window length.
- `--basis-allpairs`: enables PCA projection using the learned basis.
- `--pca-r`: number of PCA components to export.
- `--topk`: number of directed edges saved per subject/scenario.
- `--use-fdr`: applies BH-FDR in meta-analysis.
- `--significant-only`: exports only significant meta-analysis edges.

## Documentation

For article writing, use:

- `docs/METHODOLOGY.md`
- `docs/PROJECT_OVERVIEW.md`

These documents define the active method, assumptions, outputs, and validation
workflow.

## Tests

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall src_mfg scripts tests
```
