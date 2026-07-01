# Methodology

This project analyzes the Kaggle Grasp-and-Lift EEG Detection dataset with an
event-locked, multi-feature dependence pipeline.

The method estimates directed lagged dependence between EEG channels by:

```text
EEG and event CSV files
-> event alignment
-> common average reference
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
with common average reference:

```text
X_c(t) = X_c(t) - mean_j X_j(t)
```

Grasp cycles are reconstructed from event blocks. A cycle starts at
`HandStart` and follows the next valid event blocks until `BothReleased`, with
an upper duration limit.

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

The pre-event analysis estimates dynamic coupling before selected event anchors.
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
directed edges. Meta-analysis counts how often each edge appears across
subjects.

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

BH-FDR correction is available across tested edges.

## Sensitivity Analysis

The sensitivity grid repeats meta-analysis over:

```text
topk = 5 10 15
min_subjects = 3 4 5
p0_inflate = 5 10 15
```

Stable edges are those that remain significant across multiple settings.

## Volume-Conduction Control

Sensor-level dependence can arise from volume conduction and shared-reference
leakage. This alternative is especially relevant here because the strongest
reproducible edges in the initial analysis connect physically adjacent posterior
electrodes (for example `PO9 -> O1`, `PO10 -> O2`).

The control (`scripts.volume_conduction_control`) annotates each reproducible /
stable edge using an approximate standard 10-20 montage (`src_mfg.montage`) and
summarizes three edge properties:

```text
spatial    geodesic inter-electrode distance vs a random-pair null
lag        zero-lag edges are more compatible with instantaneous field spread
symmetry   reproducible reverse edges increase concern about a shared source
```

The spatial enrichment is a one-sided permutation test: the mean edge distance is
compared against the mean distance of equally many directed pairs drawn uniformly
from the montage. A significantly shorter observed distance flags short-range
leakage.

The implementation retains the field name `vc_robust` for edges that are
simultaneously long-range, non-zero-lag, and directionally asymmetric:

```text
vc_robust = (distance > short_range_cm) AND (lag_ms > 0) AND (reverse edge not reproducible)
```

This is a heuristic screen, not proof of cortical connectivity. Non-zero lag and
asymmetry reduce concern about simple zero-lag mixing but do not remove common
sources, reference effects, or filtering artefacts. Edge maps dominated by
adjacency should therefore remain sensor-level observations. Stronger follow-up
would use a surface Laplacian/current-source-density transform or estimators such
as imaginary coherence or the weighted phase-lag index.

## Classification Benchmark

The classification stage evaluates whether the discovered dependence features
help sample-level detection of the six Kaggle event labels. It is an ablation,
not a replacement for the connectivity analysis.

The benchmark trains three feature sets:

```text
baseline = normalized EEG, temporal difference, rolling RMS
mfg      = dynamic EMA energy on selected stable directed edges
combined = baseline + mfg
```

It can also evaluate late fusion of two separately trained predictors:

```text
p_fusion(y = 1 | x) = (1 - w) p_baseline(y = 1 | x) + w p_mfg(y = 1 | x)
```

This is a score-level ablation. It tests whether the MFG predictor contributes
useful probability information without forcing the final classifier to learn in
one concatenated feature space.

The default fast classifier is regularized logistic SGD. A stronger nonlinear
`extra_trees` option is available for final classifier tuning; it uses the same
feature sets and therefore keeps the ablation interpretation unchanged.

Stable MFG edges can be selected in two ways. The global setting uses the
highest-ranked reproducible edges as one common feature set. The event-union
setting selects stable edges from the movement phases adjacent to each event and
uses their de-duplicated union. The second option is more aligned with the
phase-locked research question, while the global option remains a conservative
control. In the event-union setting, the edge count parameter is a per-event
phase cap; the exported `mfg_edges_used.csv` gives the actual de-duplicated
feature graph.

The benchmark records two additional controls for artefact interpretation:

```text
channel_set = all32 | motor_premotor | no_fp1_fp2 | occipital_visual | ocular_proxy_only
preprocess  = car_only | bandpass_0_5_48 | ocular_proxy_regression
label_shift_ms = 0 for aligned labels, non-zero for negative timing controls
```

`motor_premotor` focuses on electrodes expected to be more relevant for motor
planning and execution. `no_fp1_fp2` tests whether the result survives removing
the strongest frontal eye-proxy channels. `ocular_proxy_only` is a negative
control: strong performance there should be interpreted as possible eye/gaze
confounding. `ocular_proxy_regression` removes linear components explained by
`Fp1/Fp2`; it is a proxy control, not full EOG artefact rejection.
Non-zero `label_shift_ms` intentionally misaligns the events from the EEG
samples. Such rows are negative controls and are reported separately from
useful prediction rows, even if a model obtains a high score.

Timing controls are interpreted by matched comparison: for the same split,
channel set, preprocessing, edge selection, and model settings, the aligned
`label_shift_ms = 0` lift should exceed the shifted-label lift. A shifted-label
win indicates that the apparent improvement may reflect slow task structure,
serial dependence, or artefact timing rather than event-locked information.

For a positive lag `ell`, the classifier uses the lag convention in a
sample-aligned form:

```text
feature at t uses source at t - ell and target at t
```

This avoids using future target samples when the feature is interpreted as a
time-local predictor. The reported score is mean ROC-AUC across the six event
labels, matching the competition-style multi-label objective.

For fast validation runs, the classifier may evaluate a strided subset of the
series. When `--val-keep-all-positive` is enabled, event-positive samples are
kept before the remaining budget is filled with strided negatives. This avoids
losing short event intervals through coarse subsampling. Probability smoothing,
if enabled, is computed with the original sample indices rather than compressed
validation row numbers.

Participant-specific validation is available through
`scripts.kaggle_subject_classification`. This is the preferred Kaggle-facing
classifier check because train and test series are organized by the same
participants. Hyperparameter grids such as `--alpha-values 0.1 0.3 1.0` should
be interpreted through held-out subject/series splits and matched timing
controls, not through a single best row.

If Kaggle test files are available, the trained model can also export
`submission.csv`. This file contains one probability per sample and event
column. It is not scored locally because test labels are hidden.

The classification sweep repeats the ablation across multiple splits and
feature settings. The reported lift is computed for each assisted variant:

```text
lift_combined = ROC_AUC(combined) - ROC_AUC(baseline)
lift_fusion   = ROC_AUC(fusion) - ROC_AUC(baseline)
```

Positive lift means that the MFG-derived features add predictive information
beyond the direct EEG baseline for the same validation split. If multiple
assisted variants are run, model selection must use held-out validation and the
selected variant must still pass timing and artefact controls.

## Interpretation Scope

The pipeline estimates event-locked, cross-subject reproducible directed
lagged innovation-coupling patterns in the Grasp-and-Lift EEG task. These are
sensor-level statistical dependencies. Anatomical interpretation requires
source localization, stronger volume-conduction controls, and independent
validation.
