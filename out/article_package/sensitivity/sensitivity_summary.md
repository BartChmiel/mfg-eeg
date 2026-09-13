# Meta-Analysis Sensitivity Report

Input directory: `out\article_package\reproduction\bandpass_0_5_48_rankings.csv.gz`
Requested parameter configurations: 27
Successful configurations: 27
Significant edge observations: 414
Instances significant in at least one setting: 62

## Most Frequently Retained Instances

| phase | pc | lag_ms | edge | configs | stability | max_k | best_q_or_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Replace__BothReleased | pc2 | 50 | Fp2->Fp1 | 24/27 | 0.8888888888888888 | 10 | 4.844744382178117e-10 |
| Replace__BothReleased | pc2 | 0 | Fp1->Fp2 | 21/27 | 0.7777777777777778 | 10 | 1.8244961040431427e-06 |
| Replace__BothReleased | pc2 | 200 | Fp2->Fp1 | 21/27 | 0.7777777777777778 | 9 | 1.8244961040431427e-06 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | F7->FC5 | 18/27 | 0.6666666666666666 | 9 | 4.271370772417297e-06 |
| Replace__BothReleased | pc2 | 0 | Fp2->Fp1 | 18/27 | 0.6666666666666666 | 9 | 4.271370772417297e-06 |
| LiftOff__Replace | pc1 | 200 | F8->FC6 | 15/27 | 0.5555555555555556 | 8 | 4.861986388580824e-05 |
| LiftOff__Replace | pc1 | 50 | Pz->CP2 | 12/27 | 0.4444444444444444 | 11 | 3.8336047049118517e-07 |
| BothStartLoadPhase__LiftOff | pc1 | 200 | F7->FC5 | 12/27 | 0.4444444444444444 | 9 | 4.271370772417297e-06 |
| LiftOff__Replace | pc1 | 200 | TP10->T8 | 12/27 | 0.4444444444444444 | 9 | 0.00011883806212569653 |
| Replace__BothReleased | pc1 | 200 | Fp2->F4 | 12/27 | 0.4444444444444444 | 8 | 0.00012772880340262456 |
| Replace__BothReleased | pc2 | 200 | Fp1->Fp2 | 12/27 | 0.4444444444444444 | 7 | 4.861986388580824e-05 |
| Replace__BothReleased | pc1 | 0 | Fp1->PO9 | 9/27 | 0.3333333333333333 | 10 | 6.497139204868006e-06 |
| BothStartLoadPhase__LiftOff | pc1 | 200 | F8->FC6 | 9/27 | 0.3333333333333333 | 8 | 0.0010301137000873542 |
| FirstDigitTouch__BothStartLoadPhase | pc1 | 200 | F8->FC6 | 9/27 | 0.3333333333333333 | 8 | 0.0010301137000873542 |
| HandStart__FirstDigitTouch | pc2 | 200 | Fp2->Fp1 | 9/27 | 0.3333333333333333 | 8 | 0.0010301137000873542 |
| Replace__BothReleased | pc2 | 50 | Fp1->Fp2 | 9/27 | 0.3333333333333333 | 8 | 0.0010301137000873542 |
| BothStartLoadPhase__LiftOff | pc2 | 200 | Fp2->Fp1 | 9/27 | 0.3333333333333333 | 7 | 0.0010301137000873542 |
| Replace__BothReleased | pc2 | 150 | Fp1->Fp2 | 9/27 | 0.3333333333333333 | 7 | 0.0010301137000873542 |
| HandStart__FirstDigitTouch | pc2 | 0 | FC1->Fp1 | 9/27 | 0.3333333333333333 | 7 | 0.011827721705814124 |
| LiftOff__Replace | pc1 | 200 | F7->FC5 | 9/27 | 0.3333333333333333 | 7 | 0.011827721705814124 |

## Columns

`configs` counts retention across the grid; `best_q_or_p` is the minimum across those settings. Reference-setting q-values are reported in the meta-analysis edge table.

## Warnings

- No sensitivity warnings.
