Markdown

# MFG EEG Connectivity Analysis System

Advanced framework for functional brain connectivity analysis based on **Model-Free Granger (MFG)** causality and **Hierarchical Correlation Reconstruction (HCR)**. Developed specifically for high-density EEG data (e.g., Kaggle Grasp-and-Lift dataset).

## Core Methodology
The system utilizes an orthonormal **Legendre polynomial basis** to capture non-linear interactions between neural signals. By projecting time-lagged signals into this polynomial space, the framework extracts higher-order statistical dependencies that traditional linear correlation methods (like Pearson) fail to detect.



## Project Structure

### 1. Library Modules (`src_mfg/`)
* **`basis.py`**: Implementation of stable Bonnet’s recursion for orthonormal Legendre polynomials on [0, 1].
* **`normalize.py`**: Advanced signal conditioning including **AR-whitened Student-t** CDF mapping to eliminate autocorrelation bias.
* **`hcr.py`**: Core engine for computing polynomial cross-moments across temporal lags.
* **`pca_features.py`**: Dimensionality reduction and PCA feature extraction for coefficient trajectories.
* **`global_basis.py`**: Management of unified PCA bases to ensure cross-subject and cross-phase comparability.
* **`density.py`**: Tools for 2D joint probability density function (PDF) reconstruction from HCR coefficients.
* **`io.py`**: Optimized Kaggle dataset loaders with Common Average Reference (CAR) and grasp cycle reconstruction logic.
* **`viz.py`**: Standardized connectivity heatmaps and DICC-style plotting.
* **`config.py`**: Centralized hyperparameter management.

### 2. Execution Scripts (`scripts/`)
* **`build_kaggle_basis_allpairs_batch.py`**: Pre-computes global PCA bases for all electrode pairs ($C \times C$).
* **`mfg_kaggle_phase_matrix_batch.py`**: Generates phase-specific connectivity matrices and statistical significance reports ($-\log_{10}p$, $\chi^2$).
* **`meta_analysis_v2.py`**: Automated neurobiological interpreter with binomial consistency testing and Directionality Index (DI) calculation.

## Usage Pipeline

### Step 1: Build Global Basis
Generate the statistical reference for all channel interactions (run once):
```bash
python scripts/build_kaggle_basis_allpairs_batch.py --root ./data --out ./models/global_basis.npz --mode gc

Step 2: Extract Connectivity Matrices

Analyze specific movement phases (e.g., LiftOff, HandStart) using the pre-computed basis:
Bash

python scripts/mfg_kaggle_phase_matrix_batch.py --root ./data --basis-allpairs ./models/global_basis.npz --out ./results --group-by subject

Step 3: Meta-Analysis & Interpretation

Statistically aggregate results across subjects and identify dominant brain regions:
Bash

python scripts/meta_analysis_v2.py --results-dir ./results --show-all-scenarios

Statistical Features

    Whitening: AR(p) residual extraction to prevent "spurious connectivity" from signal self-prediction.

    Stationarity: Window-centered epoching (2.0s default) for robust phase-locked analysis.

    Significance: Binomial p-values for inter-subject consistency and χ2 tests for interaction strength.

    Directionality: Calculation of the Directionality Index (DI) to distinguish between information sources and sinks.