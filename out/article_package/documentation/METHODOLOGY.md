# Methodology

This project analyzes the Kaggle Grasp-and-Lift EEG Detection dataset with an
event-locked, multi-feature Granger-style (MFG) dependence pipeline.

The method estimates directed lagged dependence between EEG channels by:

```text
EEG and event CSV files
-> event alignment
-> common average reference (CAR)
-> grasp-cycle reconstruction
-> phase and pre-event windows
-> marginal normalization
-> Legendre mixed-moment features
-> pair-specific PCA dependence modes
-> subject-level edge rankings
-> cross-subject meta-analysis
```

## Dataset

The active dataset is the Kaggle Grasp-and-Lift EEG Detection training split.
The expected sampling rate is `500 Hz`.

Each recording uses paired files:

```text
subjX_seriesY_data.csv
subjX_seriesY_events.csv
```

The event columns are:

```text
HandStart
FirstDigitTouch
BothStartLoadPhase
LiftOff
Replace
BothReleased
```

The analysis uses the adjacent task phases:

```text
HandStart -> FirstDigitTouch
FirstDigitTouch -> BothStartLoadPhase
BothStartLoadPhase -> LiftOff
LiftOff -> Replace
Replace -> BothReleased
```

## Preprocessing

Data and event rows are aligned by sample ID. EEG channels are re-referenced
with common average reference (CAR):

```text
X_c(t) = X_c(t) - mean_j X_j(t)
```

Grasp cycles are reconstructed from event blocks. A cycle starts at
`HandStart` and follows the next valid event blocks until `BothReleased`, with
an upper duration limit.

The article compares CAR-only recordings with fourth-order, zero-phase
0.5-48 Hz band-pass filtering after CAR. Each phase uses a two-second window
around its center, with a separately fitted PCA basis for each preprocessing
variant.

## Normalization

The method maps channel values to approximately uniform variables on `(0, 1)`.
This separates marginal amplitude distribution from dependence structure.

Correlation mode:

```text
U_c(t) = Phi((X_c(t) - mean(X_c)) / sd(X_c))
```

Directional mode:

```text
source: empirical CDF transform
target: AR residual -> EMA variance -> Student-t CDF
```

Directional mode estimates lagged innovation coupling at the sensor level.

## Mixed-Moment Features

For a normalized value `u`, the pipeline evaluates shifted orthonormal Legendre
polynomials:

```text
phi_k(u) = sqrt(2k + 1) P_k(2u - 1),  k = 0, ..., m
```

The constant term is excluded from dependence scoring. For source channel `i`,
target channel `j`, and lag `ell`:

```text
F_i,a(t) = phi_a(Y_i(t))
G_j,b(t) = phi_b(Z_j(t + ell))
```

The centered mixed moment is:

```text
M_ijab(ell) =
  mean_t F_i,a(t) G_j,b(t)
  - mean_t F_i,a(t) mean_t G_j,b(t)
```

The lag convention is:

```text
source at t
target at t + ell
```

## PCA Dependence Modes

For each ordered channel pair, mixed-moment vectors are accumulated across
windows and lags. A pair-specific covariance matrix is decomposed with PCA.

For component `k`:

```text
s_ij,k(ell) = u_ij,k^T (m_ij(ell) - mu_ij)
```

The article presets use:

```text
m = 4
lags_ms = 0 50 100 150 200
pca_r = 3
```

## Pre-Event EMA

The optional pre-event analysis estimates dynamic coupling before selected event anchors.
The default window is:

```text
[-0.5 s, 0]
```

with an additional `0.5 s` burn-in for EMA stabilization.

The dynamic coefficient is:

```text
C_t = EMA(F_t G_t^T) - EMA(F_t) EMA(G_t)^T
```

The default summary is `final`, i.e. the value closest to event onset.

## Meta-Analysis

For every subject and scenario `(phase, PC, lag)`, the pipeline stores top
directed edges ranked by absolute score. Meta-analysis counts how often each
ordered pair appears across subjects; score signs do not affect this ranking.

For edge `i -> j`:

```text
K_ij = number of subjects containing the edge in top-k
N = number of subjects available for the scenario
```

The null model is:

```text
K_ij ~ Binomial(N, p0)
```

Default null probability:

```text
p0 = p0_inflate * topk / (C * (C - 1))
```

The fixed reference setting uses top-k 10, minimum recurrence four, null
inflation ten, and alpha 0.05. Global BH spans 74,400 hypotheses
(`32 * 31` ordered pairs, five phases, three PCs, five lags), including K=0
hypotheses with p=1. Minimum recurrence is applied after correction, so it does
not change q-values. Calibration of the binomial working null, including effects
of non-uniform rankings and shared fitted bases, requires empirical validation.

## Sensitivity Analysis

The sensitivity grid repeats meta-analysis over:

```text
topk = 5 10 15
min_subjects = 3 4 5
p0_inflate = 5 10 15
```

The 27 configurations are robustness checks using the same participants;
full stability is their strict intersection. The union counts instances
significant in at least one setting and is not a separately FDR-controlled
discovery set. Significance in the fixed reference setting and retention across
the grid are reported separately in `tables/primary_edge_sensitivity.csv` within
the article package.

## Volume-Conduction Control

The control (`scripts.volume_conduction_control`) screens the fully stable
intersection using an approximate standard 10-20 montage (`src_mfg.montage`)
and three edge properties:

```text
spatial    geodesic inter-electrode distance vs a random-pair null
lag        zero-lag edges are more compatible with instantaneous field spread
symmetry   reproducible reverse edges increase concern about a shared source
```

The spatial enrichment is a one-sided permutation test: the mean edge distance is
compared against the mean distance of equally many directed pairs drawn uniformly
from the montage. A significantly shorter observed distance flags short-range
leakage.

The field `vc_robust` identifies edges that are
simultaneously long-range, non-zero-lag, and directionally asymmetric:

```text
vc_robust = (distance > short_range_cm) AND (lag_ms > 0) AND (reverse edge not reproducible)
```

The article uses a 4.5 cm distance threshold and 20,000 permutations with seed
20240617. This heuristic cannot rule out shared-source, ocular, reference, or
filtering effects; an empty candidate set leaves spatial enrichment unassessed.

## Classification Benchmark

The classification stage compares feature sets for sample-level detection of
the six Kaggle event labels.

The benchmark trains three feature sets:

```text
baseline = normalized EEG, temporal differences, lagged samples, rolling means and RMS
mfg      = dynamic EMA features on selected candidate directed edges
combined = baseline + mfg
```

The reported `expanded` mode includes log1p energy, signed coefficient mean,
and log1p maximum absolute coefficient for each edge.

It can also evaluate late fusion of two separately trained predictors:

```text
p_fusion(y = 1 | x) = (1 - w) p_baseline(y = 1 | x) + w p_mfg(y = 1 | x)
```

The `best_assisted` output is the row-wise better of `combined` and `fusion`.

The reported classifier uses regularized logistic SGD with three training passes.
The software also supports `extra_trees` with the same feature sets.

MFG candidate edges can be selected globally or by event-union. Global selection
uses one highest-ranked candidate set. Event-union selects candidates from
phases adjacent to each event and de-duplicates their union; its edge count
parameter is a per-event phase cap. `mfg_edges_used.csv` records the resulting
feature graph.

The reported experiment uses a frozen CAR-only candidate graph selected from the
full labelled collection, including validation series. Candidate selection and
row-wise augmented-model selection were not fully nested, so the incremental
result is descriptive rather than an unbiased estimate of generalization.

Available control options are:

```text
channel_set = all32 | motor_premotor | no_fp1_fp2 | occipital_visual | ocular_proxy_only
preprocess  = car_only | bandpass_0_5_48 | ocular_proxy_regression
label_shift_ms = 0 for aligned labels, non-zero for negative timing controls
```

`motor_premotor` selects motor/premotor electrodes, `no_fp1_fp2` excludes frontal
eye-proxy channels, and `ocular_proxy_only` isolates those channels as a confound
check. `ocular_proxy_regression` removes linear components explained by Fp1/Fp2;
it is not full EOG artefact rejection.

The article reports `all32` with band-pass preprocessing and aligned versus
500 ms shifted-label controls. Non-zero `label_shift_ms` misaligns the events
from the EEG samples; these control rows are reported separately.

Timing controls match the split, channel set, preprocessing, edge selection,
and model settings, then subtract shifted-label lift from aligned-label lift.
Positive shifted-control lift can reflect slow task structure or serial dependence.

For a positive lag `ell`, the classifier uses the lag convention in a
sample-aligned form:

```text
feature at t uses source at t - ell and target at t
```

The reported score is mean ROC-AUC across the six event labels.

For fast validation runs, the classifier may evaluate a strided subset of the
series. When `--val-keep-all-positive` is enabled, event-positive samples are
kept before the remaining budget is filled with strided negatives. This avoids
losing short event intervals through coarse subsampling. Probability smoothing,
if enabled, is computed with the original sample indices rather than compressed
validation row numbers.

Participant-specific validation is available through
`scripts.kaggle_subject_classification`. Training and held-out series belong to
the same participant. The classification sweep repeats the ablation across
these splits and feature settings.

If Kaggle test files are available, the trained model can also export
`submission.csv`. This file contains one probability per sample and event
column. It is not scored locally because test labels are hidden.

Lift is computed for each augmented variant:

```text
lift_combined = ROC_AUC(combined) - ROC_AUC(baseline)
lift_fusion   = ROC_AUC(fusion) - ROC_AUC(baseline)
```

Positive lift means a higher measured ROC-AUC than the direct EEG baseline for
the same validation split.

## Reproduction

Run from the repository root:

```text
python -m scripts.reproduce_article --source snapshots
python -m scripts.reproduce_article --source raw --output out/raw_reproduction
```

Snapshot mode rebuilds statistics, tables, figures, and the article using archived
rankings and retained classifier results. Raw mode also recomputes the EEG
features, PCA bases, and classifier runs. Both use the same candidate table.
Inputs and SHA256 manifests are described in the article package's
`reproduction/README.md`.

## Interpretation Scope

Direction and lag describe sensor-level dependence. Anatomical interpretation
requires source-level analysis and independent validation.
