# Volume-Conduction Control

Source edge table: `out/article_package/comparison/car_only\sensitivity\edge_stability.csv`
Reproducible edge instances analysed: 244

## Purpose

Summarizes whether reproducible directed edges are short-range, zero-lag, or bidirectional. These features can be consistent with volume conduction or shared-reference effects; this heuristic screen does not identify genuine cortical coupling.

## Spatial enrichment (are edges just nearest neighbours?)

- Observed mean inter-electrode distance: 3.945 cm
- Null mean distance (random directed pairs): 14.051 cm
- One-sided permutation p (edges shorter than chance): 4.999750012499375e-05
- z-score vs null: -24.568

## Lag structure

- Zero-lag edge instances: 17 / 244
- Non-zero-lag edge instances: 227 (93.0%)

## Directional asymmetry

- Asymmetric edge instances (reverse not reproducible): 178 (73.0%)

## Short-range fraction

- Short-range (<= 4.5 cm): 174 (71.3%)
- Long-range: 70

## Distance/lag/asymmetry screen

Edges that are long-range AND non-zero-lag AND asymmetric: 70 (28.7%).

Region flow of the screen-passing subset:

| flow | count |
| --- | --- |
| Occipital->Occipital | 37 |
| Parietal->Parietal | 13 |
| Occipital->Frontocentral | 6 |
| Parietal->Occipital | 6 |
| Parietal->Frontocentral | 5 |
| Frontocentral->Frontocentral | 2 |
| Frontal->Frontal | 1 |

Top screen-passing edges (by distance):

| phase | pc | lag_ms | edge | distance_cm |
| --- | --- | --- | --- | --- |
| HandStart__FirstDigitTouch | pc3 | 50 | O1->FC1 | 17.658 |
| LiftOff__Replace | pc3 | 50 | O1->FC1 | 17.658 |
| BothStartLoadPhase__LiftOff | pc3 | 200 | Oz->FC5 | 17.398 |
| FirstDigitTouch__BothStartLoadPhase | pc3 | 50 | Oz->FC5 | 17.398 |
| FirstDigitTouch__BothStartLoadPhase | pc3 | 200 | Oz->FC5 | 17.398 |
| HandStart__FirstDigitTouch | pc3 | 50 | O2->Cz | 14.441 |
| LiftOff__Replace | pc3 | 200 | P4->CP6 | 6.149 |
| Replace__BothReleased | pc1 | 100 | Fp2->Fp1 | 5.804 |
| HandStart__FirstDigitTouch | pc3 | 200 | P7->O1 | 5.758 |
| LiftOff__Replace | pc3 | 50 | P7->O1 | 5.758 |
| LiftOff__Replace | pc3 | 100 | P7->O1 | 5.758 |
| LiftOff__Replace | pc3 | 150 | P7->O1 | 5.758 |
| LiftOff__Replace | pc3 | 200 | P7->O1 | 5.758 |
| Replace__BothReleased | pc3 | 200 | P7->O1 | 5.758 |
| BothStartLoadPhase__LiftOff | pc1 | 50 | PO10->Oz | 5.45 |

## Verdict

- WARNING: reproducible edges are significantly shorter-range than chance (permutation p=5e-05); short-range volume conduction is a real concern.
- A majority of reproducible edges are short-range (<= 4.5 cm): interpret raw edge maps cautiously.
- 70 edge instance(s) pass all three screening criteria (long-range AND non-zero lag AND directionally asymmetric). They still require sensor-level interpretation and are not proof of cortical connectivity.
