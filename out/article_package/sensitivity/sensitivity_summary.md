# Meta-Analysis Sensitivity Report

Input directory: `out\phase_mats_pca_bandpass_by_subject`
Requested parameter configurations: 27
Successful configurations: 27
Significant edge observations: 4288
Unique stable edges: 766

## Most Stable Edges

| phase | pc | lag_ms | edge | configs | stability | max_k | best_q_or_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Replace__BothReleased | pc2 | 50 | Fp2->Fp1 | 27/27 | 1.0 | 10 | 1.1721155763334154e-13 |
| Replace__BothReleased | pc2 | 0 | Fp1->Fp2 | 27/27 | 1.0 | 10 | 4.414103477523732e-10 |
| Replace__BothReleased | pc2 | 200 | Fp2->Fp1 | 27/27 | 1.0 | 9 | 4.414103477523732e-10 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | F7->FC5 | 27/27 | 1.0 | 9 | 6.200376927702528e-09 |
| Replace__BothReleased | pc2 | 0 | Fp2->Fp1 | 27/27 | 1.0 | 9 | 6.200376927702528e-09 |
| LiftOff__Replace | pc1 | 200 | F8->FC6 | 26/27 | 0.9629629629629629 | 8 | 1.1762870294953607e-08 |
| Replace__BothReleased | pc1 | 200 | Fp2->F4 | 26/27 | 0.9629629629629629 | 8 | 1.8541277913284212e-07 |
| HandStart__FirstDigitTouch | pc2 | 200 | Fp2->Fp1 | 26/27 | 0.9629629629629629 | 8 | 2.4922105647274695e-07 |
| Replace__BothReleased | pc2 | 50 | Fp1->Fp2 | 26/27 | 0.9629629629629629 | 8 | 2.4922105647274695e-07 |
| BothStartLoadPhase__LiftOff | pc1 | 200 | F8->FC6 | 25/27 | 0.9259259259259259 | 8 | 2.4922105647274695e-07 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | F8->FC6 | 25/27 | 0.9259259259259259 | 8 | 2.4922105647274695e-07 |
| Replace__BothReleased | pc2 | 200 | Fp1->Fp2 | 25/27 | 0.9259259259259259 | 7 | 1.1762870294953607e-08 |
| Replace__BothReleased | pc2 | 150 | Fp1->Fp2 | 25/27 | 0.9259259259259259 | 7 | 2.4922105647274695e-07 |
| LiftOff__Replace | pc1 | 50 | Pz->CP2 | 24/27 | 0.8888888888888888 | 11 | 1.216035901020426e-09 |
| BothStartLoadPhase__LiftOff | pc1 | 200 | F7->FC5 | 24/27 | 0.8888888888888888 | 9 | 6.200376927702528e-09 |
| BothStartLoadPhase__LiftOff | pc2 | 200 | Fp2->Fp1 | 24/27 | 0.8888888888888888 | 7 | 2.4922105647274695e-07 |
| HandStart__FirstDigitTouch | pc2 | 0 | FC1->Fp1 | 24/27 | 0.8888888888888888 | 7 | 6.941643366319013e-06 |
| LiftOff__Replace | pc1 | 200 | F7->FC5 | 24/27 | 0.8888888888888888 | 7 | 6.941643366319013e-06 |
| LiftOff__Replace | pc1 | 50 | Pz->CP1 | 23/27 | 0.8518518518518519 | 8 | 3.5272891748982306e-06 |
| Replace__BothReleased | pc2 | 100 | Fp2->Fp1 | 23/27 | 0.8518518518518519 | 6 | 2.4922105647274695e-07 |

## Interpretation

Edges that remain significant across multiple top-k, minimum-subject, and p0-inflation settings are stronger candidates for article claims. This analysis does not replace the main FDR table; it checks whether the conclusions are parameter-stable.

## Warnings

- No sensitivity warnings.
