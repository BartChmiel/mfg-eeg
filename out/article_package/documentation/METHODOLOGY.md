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
subjXX_seriesYY_data.csv
subjXX_seriesYY_events.csv
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

The directional mode should be interpreted as lagged innovation coupling, not
as proof of anatomical causality.

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

## Interpretation

Supported statement:

```text
The pipeline identifies event-locked, cross-subject reproducible directed
lagged innovation-coupling patterns in the Grasp-and-Lift EEG task.
```

Avoid:

```text
The pipeline proves anatomical causality.
```

