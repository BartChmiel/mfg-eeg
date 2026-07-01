# Subject-Aware Timing Control

Purpose: compare aligned classifier lift against a matched 500 ms shifted-label control.

## Summary

- Matched rows: 288
- Mean aligned lift: 0.002601
- Mean shifted lift: 0.001195
- Mean aligned-minus-shifted delta: 0.001406
- Aligned wins: 175/288

## Best Hyperparameter Group

- Alpha: `0.2`
- Fusion: `0.25`
- Mean delta: `0.002002`

## Hyperparameter Groups

| alpha | fusion | classifier | channels | preprocess | matches | aligned_lift | shifted_lift | delta | aligned_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0.25 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002506 | 0.000504 | 0.002002 | 14 |
| 0.3 | 0.25 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002720 | 0.000770 | 0.001950 | 15 |
| 0.1 | 0.25 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002032 | 0.000230 | 0.001801 | 15 |
| 0.2 | 0.2 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002710 | 0.001161 | 0.001549 | 14 |
| 0.3 | 0.2 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002910 | 0.001366 | 0.001544 | 15 |
| 0.1 | 0.2 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002371 | 0.000835 | 0.001535 | 15 |
| 0.1 | 0.15 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002444 | 0.001091 | 0.001353 | 15 |
| 0.2 | 0.15 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002734 | 0.001564 | 0.001171 | 14 |
| 0.3 | 0.15 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002951 | 0.001812 | 0.001139 | 15 |
| 0.1 | 0.1 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002278 | 0.001143 | 0.001135 | 16 |
| 0.2 | 0.1 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002638 | 0.001788 | 0.000849 | 13 |
| 0.3 | 0.1 | sgd_logistic | all32 | bandpass_0_5_48 | 24 | 0.002924 | 0.002078 | 0.000845 | 14 |

## Top Matched Rows

| subject | split | alpha | fusion | aligned_lift | shifted_lift | delta | assisted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 1-2-3-4-5-6-7/8 | 0.3 | 0.25 | 0.004914 | -0.008945 | 0.013859 | combined |
| 9 | 1-2-3-4-5-6-7/8 | 0.2 | 0.25 | 0.003921 | -0.008457 | 0.012378 | combined |
| 9 | 1-2-3-4-5-6-7/8 | 0.3 | 0.2 | 0.004914 | -0.006518 | 0.011431 | combined |
| 9 | 1-2-3-4-5-6-7/8 | 0.2 | 0.2 | 0.003921 | -0.006182 | 0.010103 | combined |
| 9 | 1-2-3-4-5-6-7/8 | 0.3 | 0.15 | 0.004914 | -0.004343 | 0.009257 | combined |
| 6 | 1-2-3-4-5-6-7/8 | 0.1 | 0.25 | 0.006983 | -0.002086 | 0.009069 | combined |
| 11 | 1-2-3-4-5-6/7 | 0.1 | 0.25 | 0.008586 | -0.000443 | 0.009029 | combined |
| 3 | 1-2-3-4-5-6/7 | 0.3 | 0.25 | 0.006405 | -0.002411 | 0.008816 | combined |
| 6 | 1-2-3-4-5-6-7/8 | 0.2 | 0.25 | 0.006923 | -0.001881 | 0.008804 | combined |
| 9 | 1-2-3-4-5-6-7/8 | 0.1 | 0.25 | 0.000770 | -0.007820 | 0.008589 | combined |
| 3 | 1-2-3-4-5-6/7 | 0.3 | 0.2 | 0.006405 | -0.002171 | 0.008577 | combined |
| 6 | 1-2-3-4-5-6/7 | 0.3 | 0.25 | 0.002895 | -0.005461 | 0.008356 | fusion |
| 11 | 1-2-3-4-5-6/7 | 0.3 | 0.25 | 0.007455 | -0.000811 | 0.008266 | combined |
| 11 | 1-2-3-4-5-6/7 | 0.2 | 0.25 | 0.007319 | -0.000819 | 0.008138 | combined |
| 6 | 1-2-3-4-5-6-7/8 | 0.3 | 0.25 | 0.006181 | -0.001950 | 0.008131 | combined |
| 9 | 1-2-3-4-5-6-7/8 | 0.2 | 0.15 | 0.003921 | -0.004158 | 0.008079 | combined |
