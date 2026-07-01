# Volume-Conduction Control

Source edge table: `out\article_meta_sensitivity_bandpass\edge_stability.csv`
Reproducible edge instances analysed: 5

## Purpose

Summarizes whether reproducible directed edges are short-range, zero-lag, or bidirectional. These features can be consistent with volume conduction or shared-reference effects; this heuristic screen does not identify genuine cortical coupling.

## Spatial enrichment (are edges just nearest neighbours?)

- Observed mean inter-electrode distance: 5.385 cm
- Null mean distance (random directed pairs): 14.057 cm
- One-sided permutation p (edges shorter than chance): 0.00039998000099995
- z-score vs null: -3.016

## Lag structure

- Zero-lag edge instances: 2 / 5
- Non-zero-lag edge instances: 3 (60.0%)

## Directional asymmetry

- Asymmetric edge instances (reverse not reproducible): 3 (60.0%)

## Short-range fraction

- Short-range (<= 4.5 cm): 1 (20.0%)
- Long-range: 4

## Distance/lag/asymmetry screen

Edges that are long-range AND non-zero-lag AND asymmetric: 2 (40.0%).

Region flow of the screen-passing subset:

| flow | count |
| --- | --- |
| Frontal->Frontal | 2 |

Top screen-passing edges (by distance):

| phase | pc | lag_ms | edge | distance_cm |
| --- | --- | --- | --- | --- |
| Replace__BothReleased | pc2 | 50 | Fp2->Fp1 | 5.804 |
| Replace__BothReleased | pc2 | 200 | Fp2->Fp1 | 5.804 |

## Verdict

- WARNING: reproducible edges are significantly shorter-range than chance (permutation p=0.0004); short-range volume conduction is a real concern.
- 2 edge instance(s) pass all three screening criteria (long-range AND non-zero lag AND directionally asymmetric). They still require sensor-level interpretation and are not proof of cortical connectivity.
