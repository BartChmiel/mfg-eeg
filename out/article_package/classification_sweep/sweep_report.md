# Classification Sweep

Question: do MFG-derived features improve sample-level event classification beyond a baseline EEG predictor?

## Summary

- Completed configurations: 8
- Combined > baseline win rate: 0.500
- Mean combined-baseline ROC-AUC lift: -0.001466
- Best lift: 0.004384 in config 5

## Top Configurations

| config | split | mode | m | edges | stride | mfg_mode | baseline | mfg | combined | lift |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 5 | 1-2-3-4-5-6-7|8 | corr | 2 | 8 | 25 | expanded | 0.606719 | 0.517776 | 0.611103 | 0.004384 |
| 8 | 1-2-3-4-5-6-7|8 | gc | 2 | 24 | 25 | expanded | 0.606719 | 0.521692 | 0.609261 | 0.002542 |
| 1 | 1-2-3-4-5-6|7 | corr | 2 | 8 | 25 | expanded | 0.605467 | 0.535091 | 0.607521 | 0.002054 |
| 2 | 1-2-3-4-5-6|7 | corr | 2 | 24 | 25 | expanded | 0.605467 | 0.526934 | 0.606531 | 0.001065 |
| 4 | 1-2-3-4-5-6|7 | gc | 2 | 24 | 25 | expanded | 0.605467 | 0.510089 | 0.601235 | -0.004232 |
| 7 | 1-2-3-4-5-6-7|8 | gc | 2 | 8 | 25 | expanded | 0.606719 | 0.514426 | 0.601163 | -0.005556 |
| 3 | 1-2-3-4-5-6|7 | gc | 2 | 8 | 25 | expanded | 0.605467 | 0.511935 | 0.599678 | -0.005789 |
| 6 | 1-2-3-4-5-6-7|8 | corr | 2 | 24 | 25 | expanded | 0.606719 | 0.535158 | 0.600526 | -0.006193 |

## Mean Event-Level Lift

| event | mean combined-baseline lift |
|---|---:|
| BothStartLoadPhase | 0.009520 |
| FirstDigitTouch | 0.009438 |
| BothReleased | -0.003442 |
| HandStart | -0.004912 |
| LiftOff | -0.005121 |
| Replace | -0.014277 |
