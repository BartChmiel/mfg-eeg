# Kaggle Article Package

Generated at UTC: 2026-04-25T00:01:23+00:00

## Purpose

This package collects the final outputs from the Grasp-and-Lift EEG pipeline: phase-wise directed lagged coupling, pre-event EMA dynamics, cross-subject meta-analysis, sensitivity analysis, and provenance.

## Inputs

- Phase directory: `out\phase_mats_pca_by_subject`
- Pre-event directory: `out\experimental_kaggle_by_subject`
- Meta-analysis directory: `out\article_meta_kaggle`
- Sensitivity directory: `out\article_meta_sensitivity`
- Run manifests indexed: 0
- Figure candidates indexed: 132
- Sensitivity files copied: 4

## Top Scenarios

| phase | pc | lag_ms | subjects | reported_edges | dominant_flow | process |
| --- | --- | --- | --- | --- | --- | --- |
| LiftOff__Replace | pc2 | 0 | 12/12 | 16 | O_Vis->O_Vis | VISUAL RECURRENCE / BINDING |
| LiftOff__Replace | pc3 | 0 | 12/12 | 14 | P_Space->P_Space | SPATIAL INTEGRATION |
| HandStart__FirstDigitTouch | pc2 | 50 | 12/12 | 13 | O_Vis->O_Vis | VISUAL RECURRENCE / BINDING |
| BothStartLoadPhase__LiftOff | pc3 | 0 | 12/12 | 13 | P_Space->P_Space | SPATIAL INTEGRATION |
| FirstDigitTouch__BothStartLoadPhase | pc3 | 0 | 12/12 | 13 | P_Space->P_Space | SPATIAL INTEGRATION |
| LiftOff__Replace | pc3 | 200 | 12/12 | 13 | P_Space->P_Space | SPATIAL INTEGRATION |
| LiftOff__Replace | pc3 | 50 | 12/12 | 13 | O_Vis->F_Motor | VISUO-MOTOR TRANSFORMATION |
| LiftOff__Replace | pc2 | 100 | 12/12 | 12 | O_Vis->O_Vis | VISUAL RECURRENCE / BINDING |
| BothStartLoadPhase__LiftOff | pc2 | 0 | 12/12 | 12 | O_Vis->O_Vis | VISUAL RECURRENCE / BINDING |
| BothStartLoadPhase__LiftOff | pc1 | 200 | 12/12 | 12 | O_Vis->O_Vis | VISUAL RECURRENCE / BINDING |

## Top Replicated Edges

| phase | pc | lag_ms | edge | k/N | q_or_p | significant |
| --- | --- | --- | --- | --- | --- | --- |
| BothStartLoadPhase__LiftOff | pc1 | 50 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 50 | PO9->O1 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 100 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 100 | PO9->O1 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 150 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 150 | PO9->O1 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 200 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc1 | 200 | PO9->O1 | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc2 | 50 | O2->Oz | 12/12 | 8.716741550770279e-12 | True |
| BothStartLoadPhase__LiftOff | pc2 | 50 | Oz->O2 | 12/12 | 8.716741550770279e-12 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 50 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 50 | PO9->O1 | 12/12 | 8.716741550770279e-12 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 100 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 100 | PO9->O1 | 12/12 | 8.716741550770279e-12 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 150 | PO10->O2 | 12/12 | 8.716741550770279e-12 | True |

## Interpretation

The primary supported result is reproducible, phase-locked, directed lagged dependence between EEG channels after marginal normalization and innovation-style whitening. The meta-analysis evaluates whether the same edges recur across subjects under a binomial null with optional BH-FDR.

The results should not be described as direct proof of anatomical causality. Recommended terms are directional lagged innovation-coupling, directed functional connectivity, or candidate information flow. Interpretation should refer to replication, event phase, lag, and region-level convergence.

## Files To Use In The Article

- `tables/top_scenarios.csv`: compact table of strongest phase/PC/lag scenarios.
- `tables/top_edges.csv`: compact table of most replicated directed edges.
- `article_summary.md`: summary of the final output package.
- `package_manifest.json`: provenance index for inputs, manifests, and figures.
- `raw_meta_exports/`: exact meta-analysis exports copied from the source run.
- `sensitivity/`: optional robustness-grid exports, if provided.
- `documentation/`: methodology and project overview copied from the repo.

## Warnings

- No run_manifest.json files were found. Re-run the current pipeline before final submission to capture full provenance.
