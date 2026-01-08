## Setup

### 1) Create virtual environment (Python 3.12)
```bash
uv venv --python 3.12
```

### 2) Activate
**Windows (PowerShell / cmd):**
```bash
.venv`Scripts`activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Kaggle - workflow 
### A) Build shared PCA basis (single file)
Build a basis for a *fixed* channel pair (A->B), mode and epoch length.
This basis can later be reused to make PC curves comparable across phases.

```bash
python -m scripts.build_kaggle_basis `
  --data data/grasp-and-lift-eeg-detection/train/subj1_series1_data.csv `
  --events data/grasp-and-lift-eeg-detection/train/subj1_series1_events.csv `
  --chan-a Fp1 --chan-b Fp2 `
  --mode gc `
  --epoch-len-s 2.0 `
  --m 4 `
  --out out/basis/basis_Fp1_Fp2_gc_e2s_m4.npz
```

### A2) Build shared PCA basis 
More stable than single-file basis. Works only on TRAIN split (needs *_events.csv).

```bash
python -m scripts.build_kaggle_basis_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --chan-a Fp1 --chan-b Fp2 `
  --mode gc `
  --epoch-len-s 2.0 `
  --m 4 `
  --out out/basis/basis_Fp1_Fp2_gc_e2s_m4_train_batch.npz
```

Optional filters:
- `--subjects 1 2 3`
- `--series 1 2 3 4 5 6 7 8`
- `--max-files 10`
- `--dry-run`

---

### B) Phase pair curves (uses the shared PCA basis)
Projects per-phase mean HCR trajectory onto the shared basis and plots Pearson + PC1..PC3 vs lag.

```bash
python -m scripts.mfg_kaggle_phase_pair `
  --data data/grasp-and-lift-eeg-detection/train/subj1_series1_data.csv `
  --events data/grasp-and-lift-eeg-detection/train/subj1_series1_events.csv `
  --chan-a Fp1 --chan-b Fp2 `
  --mode gc `
  --epoch-len-s 2.0 `
  --m 4 `
  --basis out/basis/basis_Fp1_Fp2_gc_e2s_m4_train_batch.npz `
  --out out/phase_pair/subj1_series1_Fp1_Fp2_gc_e2s_m4
```

---

### C) Phase matrices (single file)
Computes per-phase directed matrices (C×C) summarizing dependency strength across all channels,
using a chosen metric aggregated over selected lags.

```bash
python -m scripts.mfg_kaggle_phase_matrix `
  --data data/grasp-and-lift-eeg-detection/train/subj1_series1_data.csv `
  --events data/grasp-and-lift-eeg-detection/train/subj1_series1_events.csv `
  --mode gc `
  --epoch-len-s 2.0 `
  --lags-ms 50 100 150 200 `
  --m 4 `
  --metric neglog10p `
  --summary mean `
  --out out/phase_mats_single
```

- `--metric`: `energy` | `chi2` | `neglog10p`
- `--summary`: `mean` | `max`

Outputs:
- heatmap PNG per phase,
- top edges TXT per phase.

---

### D) Phase matrices (batch across MANY train files) - recommended
Aggregates phase matrices across many files.
Useful for repeatability and stable “global” phase structure.

**Global aggregate (ALL files):**
```bash
python -m scripts.mfg_kaggle_phase_matrix_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --out out/phase_mats_batch `
  --mode gc `
  --epoch-len-s 2.0 `
  --lags-ms 50 100 150 200 `
  --m 4 `
  --metric neglog10p `
  --summary mean `
  --group-by all
```

**Group by subject (separate outputs per subject):**
```bash
python -m scripts.mfg_kaggle_phase_matrix_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --out out/phase_mats_by_subject `
  --mode gc `
  --epoch-len-s 2.0 `
  --lags-ms 50 100 150 200 `
  --m 4 `
  --metric neglog10p `
  --summary mean `
  --group-by subject
```

Optional filters:
- `--subjects 1 2 3`
- `--series 1 2 3 4 5 6 7 8`
- `--max-files 10`

---

## Legacy pipeline 

These scripts run analysis on:
- a full Kaggle series treated as one long trial (no phase slicing),
- or ROI .mat trial-based data.

### Pairwise analysis (CSV or MAT)
```bash
python -m scripts.mfg_pair --file <path> --chan-a <A> --chan-b <B> --mode gc --out <dir>
```

Example (Kaggle CSV):
```bash
python -m scripts.mfg_pair `
  --file data/grasp-and-lift-eeg-detection/train/subj10_series1_data.csv `
  --chan-a Fp1 `
  --chan-b Fp2 `
  --mode gc `
  --out out/legacy_pair/csv_run
```

Example (ROI .mat):
```bash
python -m scripts.mfg_pair `
  --file data/mat/A_01_47_FLA_CONG_mfgin.mat `
  --chan-a ROI_01 `
  --chan-b ROI_02 `
  --mode gc `
  --out out/legacy_pair/mat_run
```

### Batch legacy (all *_data.csv under root)
```bash
python -m scripts.mfg_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --chan-a Fp1 `
  --chan-b Fp2 `
  --mode gc `
  --out-root out/legacy_batch/csv_run
```

Limit files:
```bash
python -m scripts.mfg_batch `
  --root data/grasp-and-lift-eeg-detection/train `
  --chan-a Fp1 `
  --chan-b Fp2 `
  --mode gc `
  --max-files 5 `
  --out-root out/legacy_batch/csv_run
```

---

## Notes / consistency requirements

- Kaggle phase scripts require TRAIN split (needs `*_events.csv`).
- For phase comparability, keep these consistent between basis build and phase_pair:
  - `--mode` (corr/gc)
  - `--epoch-len-s`
  - `--m`
  - `config.fs` and `config.lag_step_ms`
- `m=4` is typically used with `mixed_only=True` internally (K = m*m = 16 features per lag).
