# Meta-Analysis Sensitivity Report

Input directory: `out\phase_mats_pca_by_subject`
Requested parameter configurations: 27
Successful configurations: 27
Significant edge observations: 17372
Unique stable edges: 1323

## Most Stable Edges

| phase | pc | lag_ms | edge | configs | stability | max_k | best_q_or_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BothStartLoadPhase__LiftOff | pc1 | 50 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc1 | 50 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc1 | 100 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc1 | 150 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc1 | 150 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc1 | 200 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc1 | 200 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| BothStartLoadPhase__LiftOff | pc2 | 50 | O2->Oz | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 50 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 50 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 100 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 150 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 150 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| HandStart__FirstDigitTouch | pc1 | 50 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| HandStart__FirstDigitTouch | pc1 | 50 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| HandStart__FirstDigitTouch | pc1 | 100 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| HandStart__FirstDigitTouch | pc1 | 100 | PO9->O1 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |
| HandStart__FirstDigitTouch | pc1 | 150 | PO10->O2 | 27/27 | 1.0 | 12 | 4.625026347489317e-19 |

## Interpretation

Edges that remain significant across multiple top-k, minimum-subject, and p0-inflation settings are stronger candidates for article claims. This analysis does not replace the main FDR table; it checks whether the conclusions are parameter-stable.

## Warnings

- No sensitivity warnings.
