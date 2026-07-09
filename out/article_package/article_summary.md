# Article Evidence Bundle

Generated at UTC: 2026-07-09T14:54:24+00:00

## Purpose

This directory contains the results used in the Grasp-and-Lift EEG article: phase-wise directed lagged dependence, cross-subject meta-analysis, sensitivity checks, classifier controls, and provenance.

## Inputs

- Phase directory: `out\phase_mats_pca_bandpass_by_subject`
- Pre-event directory: `out\experimental_kaggle_by_subject`
- Meta-analysis directory: `out\article_meta_kaggle_bandpass`
- Sensitivity directory: `out\article_meta_sensitivity_bandpass`
- Classification directory: `out\classification_benchmark`
- Classification sweep directory: `out\classification_sweep`
- Classification controls directory: `out\classification_controls`
- Subject classification directory: `out\classification_subject_epochs3`
- Volume-conduction directory: `out\article_volume_conduction_bandpass`
- Run manifests indexed: 12
- Figure candidates indexed: 132
- Article figures generated: 3
- Sensitivity files copied: 4
- Classification files copied: 5
- Classification sweep files copied: 4
- Classification controls files copied: 4
- Subject classification files copied: 7
- Volume-conduction files copied: 4
- Evidence table files written: 3

## Top Scenarios

| phase | pc | lag_ms | subjects | reported_edges | dominant_flow |
| --- | --- | --- | --- | --- | --- |
| LiftOff__Replace | pc1 | 50 | 12/12 | 11 | Parietal->Parietal |
| FirstDigitTouch__BothStartLoadPhase | pc2 | 200 | 12/12 | 10 | Frontal->Parietal |
| LiftOff__Replace | pc3 | 50 | 12/12 | 9 | Parietal->Occipital |
| BothStartLoadPhase__LiftOff | pc2 | 200 | 12/12 | 8 | Frontal->Parietal |
| BothStartLoadPhase__LiftOff | pc1 | 50 | 12/12 | 8 | Parietal->Parietal |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 50 | 12/12 | 8 | Parietal->Parietal |
| HandStart__FirstDigitTouch | pc2 | 200 | 12/12 | 7 | Frontal->Parietal |
| LiftOff__Replace | pc1 | 200 | 12/12 | 7 | Parietal->Parietal |
| Replace__BothReleased | pc1 | 0 | 12/12 | 7 | Frontal->Occipital |
| Replace__BothReleased | pc2 | 150 | 12/12 | 7 | Frontal->Frontal |

## Top Replicated Edges

| phase | pc | lag_ms | edge | k/N | q_or_p | significant |
| --- | --- | --- | --- | --- | --- | --- |
| Replace__BothReleased | pc2 | 50 | Fp2->Fp1 | 10/12 | 1.5639533517840572e-06 | True |
| BothStartLoadPhase__LiftOff | pc1 | 200 | F7->FC5 | 9/12 | 6.732582882631943e-06 | True |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | F7->FC5 | 9/12 | 6.732582882631943e-06 | True |
| LiftOff__Replace | pc1 | 50 | Pz->CP2 | 9/12 | 6.732582882631943e-06 | True |
| Replace__BothReleased | pc2 | 0 | Fp1->Fp2 | 9/12 | 6.732582882631943e-06 | True |
| Replace__BothReleased | pc2 | 0 | Fp2->Fp1 | 9/12 | 6.732582882631943e-06 | True |
| Replace__BothReleased | pc2 | 200 | Fp2->Fp1 | 9/12 | 6.732582882631943e-06 | True |
| LiftOff__Replace | pc1 | 200 | F8->FC6 | 8/12 | 9.616001055890796e-05 | True |
| LiftOff__Replace | pc1 | 200 | TP10->T8 | 8/12 | 9.616001055890796e-05 | True |
| Replace__BothReleased | pc1 | 200 | Fp2->F4 | 8/12 | 9.616001055890796e-05 | True |
| HandStart__FirstDigitTouch | pc2 | 200 | Fp2->Fp1 | 7/12 | 0.0008758090532656227 | True |
| LiftOff__Replace | pc1 | 50 | P4->CP2 | 7/12 | 0.0008758090532656227 | True |
| LiftOff__Replace | pc1 | 50 | Pz->CP1 | 7/12 | 0.0008758090532656227 | True |
| Replace__BothReleased | pc2 | 50 | Fp1->Fp2 | 7/12 | 0.0008758090532656227 | True |
| Replace__BothReleased | pc2 | 150 | Fp1->Fp2 | 7/12 | 0.0008758090532656227 | True |

## Classifier Evidence Snapshot

- Subject-aware timing control: 288 matched rows, aligned lift +0.0026, shifted lift +0.0012, delta +0.0014, aligned wins 175/288.
- Best subject-aware group: alpha=0.2, fusion=0.25, baseline AUC 0.8609, assisted AUC 0.8634, lift +0.0025, matched delta +0.0020.
- Subject-level robustness: 7/12 subjects have positive mean delta; subject-mean delta +0.0014; bootstrap 95% CI [+0.0000, +0.0028]; one-sided sign-test p=0.3872.
- Classifier interpretation: positive but modest subject-aware incremental value.

## Sensitivity Evidence Snapshot

- Sensitivity grid: 766 unique stable edges across the sensitivity grid; 5 remain significant in all grid settings.
- Fully stable region flow: Frontal->Frontal (4 edge instances).

## Volume-Conduction Snapshot

- Distance/lag/asymmetry screen: 2 / 5 edges pass (40.0%).
- Short-range fraction among fully stable edges: 20.0% (permutation p shorter than chance: 0.00039998000099995).

## Interpretation

The central result is reproducible, phase-locked, directed lagged dependence between EEG channels after marginal normalization and innovation-style whitening. The meta-analysis evaluates whether the same edges recur across subjects under a binomial null with optional BH-FDR.

The results are sensor-level statistical dependencies rather than anatomical pathways. We describe them as directional lagged sensor dependence and interpret only their replication, event phase, lag, and sensor-group distribution.

## Files To Use In The Article

- `tables/top_scenarios.csv`: compact table of strongest phase/PC/lag scenarios.
- `tables/top_edges.csv`: compact table of most replicated directed edges.
- `tables/classifier_evidence.csv`: compact uncertainty summary for the subject-aware timing control.
- `tables/sensitivity_evidence.csv`: compact sensitivity-grid stability summary.
- `tables/preprocessing_comparison.csv`: compact CAR-only versus cleaned-run robustness comparison, if requested.
- `article_summary.md`: summary of the final evidence.
- `package_manifest.json`: provenance index for inputs, manifests, and figures.
- `raw_meta_exports/`: exact meta-analysis exports copied from the source run.
- `sensitivity/`: optional robustness-grid exports, if provided.
- `classification/`: optional Kaggle-style classifier ablation exports, if provided.
- `classification_sweep/`: optional classification-grid robustness exports, if provided.
- `classification_controls/`: optional artefact and channel-set control exports, if provided.
- `classification_subject/`: optional participant-specific classifier validation exports, if provided.
- `volume_conduction/`: distance/lag/asymmetry screening report and edge table, if provided.
- `documentation/`: article PDF, references, and the technical methodology note.

## Warnings

- No packaging warnings.
