# Article Evidence Bundle

Generated at UTC: 2026-09-13T16:15:15+00:00

This directory contains the results used in the Grasp-and-Lift EEG article: phase-wise directed lagged dependence, cross-subject meta-analysis, sensitivity checks, classifier controls, and provenance.

Input paths and the file inventory are recorded in [package_manifest.json](package_manifest.json).

## Top Scenarios

| phase | pc | lag_ms | subjects | reported_edges | dominant_flow |
| --- | --- | --- | --- | --- | --- |
| LiftOff__Replace | pc1 | 200 | 12/12 | 2 | Frontal->Frontocentral |
| Replace__BothReleased | pc2 | 0 | 12/12 | 2 | Frontal->Frontal |
| LiftOff__Replace | pc1 | 50 | 12/12 | 1 | Parietal->Parietal |
| BothStartLoadPhase__LiftOff | pc1 | 200 | 12/12 | 1 | Frontal->Frontocentral |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | 12/12 | 1 | Frontal->Frontocentral |
| Replace__BothReleased | pc2 | 50 | 12/12 | 1 | Frontal->Frontal |
| Replace__BothReleased | pc1 | 200 | 12/12 | 1 | Frontal->Frontocentral |
| Replace__BothReleased | pc2 | 200 | 12/12 | 1 | Frontal->Frontal |

## Top Replicated Edges

| phase | pc | lag_ms | edge | k/N | q_or_p | significant |
| --- | --- | --- | --- | --- | --- | --- |
| Replace__BothReleased | pc2 | 50 | Fp2->Fp1 | 10/12 | 0.0004390872806518259 | True |
| BothStartLoadPhase__LiftOff | pc1 | 200 | F7->FC5 | 9/12 | 0.001890204401765346 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | F7->FC5 | 9/12 | 0.001890204401765346 | True |
| LiftOff__Replace | pc1 | 50 | Pz->CP2 | 9/12 | 0.001890204401765346 | True |
| Replace__BothReleased | pc2 | 0 | Fp1->Fp2 | 9/12 | 0.001890204401765346 | True |
| Replace__BothReleased | pc2 | 0 | Fp2->Fp1 | 9/12 | 0.001890204401765346 | True |
| Replace__BothReleased | pc2 | 200 | Fp2->Fp1 | 9/12 | 0.001890204401765346 | True |
| LiftOff__Replace | pc1 | 200 | F8->FC6 | 8/12 | 0.026997376549368873 | True |
| LiftOff__Replace | pc1 | 200 | TP10->T8 | 8/12 | 0.026997376549368873 | True |
| Replace__BothReleased | pc1 | 200 | Fp2->F4 | 8/12 | 0.026997376549368873 | True |

## Classifier Evidence Snapshot

- Subject-aware timing control: 288 matched rows, aligned lift +0.0026, shifted lift +0.0012, delta +0.0014, aligned wins 175/288.
- Best timing-control group: alpha=0.2, fusion=0.25, baseline AUC 0.8609, assisted AUC 0.8634, lift +0.0025, matched delta +0.0020.
- Subject-level robustness: 7/12 subjects have positive mean delta; subject-mean delta +0.0014; bootstrap 95% CI [+0.0000, +0.0028]; one-sided sign-test p=0.3872.
- The classifier increment is descriptive; selection and validation are documented in [Methodology](documentation/METHODOLOGY.md#classification-benchmark).

## Sensitivity Evidence Snapshot

- Sensitivity grid: 62 instances significant in at least one setting; 0 remain significant in all grid settings.
- Fully stable region flow: none (0 edge instances).
- Per-instance retention is listed in [primary_edge_sensitivity.csv](tables/primary_edge_sensitivity.csv).

## Volume-Conduction Snapshot

- The secondary screen has zero fully stable candidates; spatial enrichment is unassessed.

## Interpretation

The reported discoveries are significant in the fixed reference setting. They describe sensor-level lagged dependence, not anatomical pathways.

## Package Contents

- `tables/top_scenarios.csv`: compact table of strongest phase/PC/lag scenarios.
- `tables/top_edges.csv`: compact table of most replicated directed edges.
- `tables/primary_edge_sensitivity.csv`: reference-setting instances, q-values, and sensitivity retention.
- `tables/classifier_evidence.csv`: compact uncertainty summary for the subject-aware timing control.
- `tables/sensitivity_evidence.csv`: compact sensitivity-grid stability summary.
- `tables/preprocessing_comparison.csv`: CAR-only versus band-pass robustness comparison, when supplied.
- `package_manifest.json`: provenance index for inputs, manifests, and figures.
- `raw_meta_exports/`: exact meta-analysis exports copied from the source run.
- `sensitivity/`: robustness-grid exports, when supplied.
- `classification_subject/`: participant-specific classifier results and timing controls, when supplied.
- `volume_conduction/`: distance/lag/asymmetry screen, when supplied.
- `documentation/`: article PDF, references, and the technical methodology note.

## Reproduction Inputs

- [reproduction/README.md](reproduction/README.md): ranking snapshots, checksums, and classifier candidate provenance.
- `comparison/car_only/`: CAR-only statistics and sensitivity analysis.
