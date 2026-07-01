# Classification Controls

Purpose: test whether MFG-assisted classification survives channel-set and preprocessing controls.

## Summary

- Requested configurations: 64
- Completed configurations: 64
- Failed configurations: 0
- Best lift: 0.051679
- Best classifier: `sgd_logistic`
- Best channel set: `all32`
- Best preprocessing: `bandpass_0_5_48`
- Best label shift: `0.0` ms

## Control Groups

| classifier | channel_set | preprocess | label_shift_ms | edge_selection | configs | mean_lift | best_lift | mean_combined_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_logistic | no_fp1_fp2 | car_only | 500.000000 | event_union | 4 | 0.031270 | 0.048051 | 0.671372 |
| sgd_logistic | all32 | bandpass_0_5_48 | 0.000000 | event_union | 4 | 0.019773 | 0.051679 | 0.692296 |
| sgd_logistic | motor_premotor | car_only | 500.000000 | event_union | 4 | 0.018618 | 0.034558 | 0.584608 |
| sgd_logistic | motor_premotor | bandpass_0_5_48 | 0.000000 | event_union | 4 | 0.013739 | 0.038555 | 0.632693 |
| sgd_logistic | no_fp1_fp2 | car_only | 0.000000 | event_union | 4 | 0.011456 | 0.016342 | 0.666516 |
| sgd_logistic | ocular_proxy_only | car_only | 500.000000 | event_union | 4 | 0.003906 | 0.007859 | 0.656787 |
| sgd_logistic | all32 | car_only | 0.000000 | event_union | 4 | 0.001165 | 0.029875 | 0.688566 |
| sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | 0.000000 | event_union | 4 | -0.000575 | 0.015931 | 0.621131 |
| sgd_logistic | no_fp1_fp2 | bandpass_0_5_48 | 500.000000 | event_union | 4 | -0.001034 | 0.011952 | 0.680508 |
| sgd_logistic | no_fp1_fp2 | bandpass_0_5_48 | 0.000000 | event_union | 4 | -0.004505 | 0.008991 | 0.659763 |
| sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | 500.000000 | event_union | 4 | -0.005180 | 0.011774 | 0.606011 |
| sgd_logistic | all32 | bandpass_0_5_48 | 500.000000 | event_union | 4 | -0.008900 | 0.000174 | 0.691459 |
| sgd_logistic | ocular_proxy_only | car_only | 0.000000 | event_union | 4 | -0.009804 | 0.002410 | 0.678982 |
| sgd_logistic | motor_premotor | bandpass_0_5_48 | 500.000000 | event_union | 4 | -0.013241 | 0.010976 | 0.623056 |
| sgd_logistic | all32 | car_only | 500.000000 | event_union | 4 | -0.013754 | -0.011339 | 0.667856 |
| sgd_logistic | motor_premotor | car_only | 0.000000 | event_union | 4 | -0.029352 | 0.000554 | 0.616009 |

## Top Configurations

| config | classifier | channel_set | preprocess | label_shift_ms | edge_selection | mode | edge_cap | edges_used | baseline | combined | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | sgd_logistic | all32 | bandpass_0_5_48 | 0.000000 | event_union | corr | 8 | 48 | 0.666732 | 0.718411 | 0.051679 |
| 54 | sgd_logistic | no_fp1_fp2 | car_only | 500.000000 | event_union | corr | 8 | 48 | 0.653453 | 0.701505 | 0.048051 |
| 22 | sgd_logistic | no_fp1_fp2 | car_only | 500.000000 | event_union | corr | 8 | 48 | 0.626751 | 0.667214 | 0.040464 |
| 11 | sgd_logistic | motor_premotor | bandpass_0_5_48 | 0.000000 | event_union | corr | 4 | 20 | 0.612303 | 0.650858 | 0.038555 |
| 10 | sgd_logistic | motor_premotor | car_only | 500.000000 | event_union | corr | 4 | 20 | 0.557714 | 0.592272 | 0.034558 |
| 42 | sgd_logistic | motor_premotor | car_only | 500.000000 | event_union | corr | 4 | 20 | 0.574267 | 0.605115 | 0.030848 |
| 17 | sgd_logistic | all32 | car_only | 0.000000 | event_union | corr | 8 | 48 | 0.689233 | 0.719109 | 0.029875 |
| 38 | sgd_logistic | no_fp1_fp2 | car_only | 500.000000 | event_union | corr | 4 | 24 | 0.653453 | 0.682635 | 0.029182 |
| 51 | sgd_logistic | all32 | bandpass_0_5_48 | 0.000000 | event_union | corr | 8 | 48 | 0.678313 | 0.705062 | 0.026749 |
| 3 | sgd_logistic | all32 | bandpass_0_5_48 | 0.000000 | event_union | corr | 4 | 24 | 0.666732 | 0.688738 | 0.022006 |
| 5 | sgd_logistic | no_fp1_fp2 | car_only | 0.000000 | event_union | corr | 4 | 24 | 0.659801 | 0.676143 | 0.016342 |
| 15 | sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | 0.000000 | event_union | corr | 4 | 8 | 0.588873 | 0.604804 | 0.015931 |

## Matched Timing Control

Positive `mean_delta` means the aligned labels produced more combined-vs-baseline lift than the shifted-label control under the same split and model settings.

| shift_ms | classifier | channel_set | preprocess | edge_selection | matches | mean_aligned_lift | mean_shifted_lift | mean_delta | aligned_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500.000000 | sgd_logistic | all32 | bandpass_0_5_48 | event_union | 4 | 0.019773 | -0.008900 | 0.028674 | 3 |
| 500.000000 | sgd_logistic | motor_premotor | bandpass_0_5_48 | event_union | 4 | 0.013739 | -0.013241 | 0.026980 | 4 |
| 500.000000 | sgd_logistic | all32 | car_only | event_union | 4 | 0.001165 | -0.013754 | 0.014919 | 3 |
| 500.000000 | sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | event_union | 4 | -0.000575 | -0.005180 | 0.004605 | 4 |
| 500.000000 | sgd_logistic | no_fp1_fp2 | bandpass_0_5_48 | event_union | 4 | -0.004505 | -0.001034 | -0.003470 | 2 |
| 500.000000 | sgd_logistic | ocular_proxy_only | car_only | event_union | 4 | -0.009804 | 0.003906 | -0.013710 | 0 |
| 500.000000 | sgd_logistic | no_fp1_fp2 | car_only | event_union | 4 | 0.011456 | 0.031270 | -0.019814 | 1 |
| 500.000000 | sgd_logistic | motor_premotor | car_only | event_union | 4 | -0.029352 | 0.018618 | -0.047970 | 0 |

## Event-Level Lift

| event | classifier | channel_set | preprocess | label_shift_ms | mean_lift |
| --- | --- | --- | --- | --- | --- |
| FirstDigitTouch | sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | 0.000000 | 0.151746 |
| BothReleased | sgd_logistic | ocular_proxy_only | car_only | 0.000000 | 0.150123 |
| BothStartLoadPhase | sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | 0.000000 | 0.083640 |
| BothReleased | sgd_logistic | ocular_proxy_only | car_only | 500.000000 | 0.077589 |
| HandStart | sgd_logistic | motor_premotor | car_only | 500.000000 | 0.075417 |
| FirstDigitTouch | sgd_logistic | all32 | bandpass_0_5_48 | 0.000000 | 0.069009 |
| BothStartLoadPhase | sgd_logistic | all32 | bandpass_0_5_48 | 0.000000 | 0.056996 |
| BothReleased | sgd_logistic | no_fp1_fp2 | car_only | 500.000000 | 0.052370 |
| FirstDigitTouch | sgd_logistic | all32 | car_only | 0.000000 | 0.050379 |
| FirstDigitTouch | sgd_logistic | ocular_proxy_only | car_only | 0.000000 | 0.048067 |
| BothReleased | sgd_logistic | no_fp1_fp2 | bandpass_0_5_48 | 500.000000 | 0.047338 |
| LiftOff | sgd_logistic | motor_premotor | bandpass_0_5_48 | 0.000000 | 0.047058 |
| BothReleased | sgd_logistic | all32 | bandpass_0_5_48 | 500.000000 | 0.045711 |
| FirstDigitTouch | sgd_logistic | ocular_proxy_only | bandpass_0_5_48 | 500.000000 | 0.044363 |
| Replace | sgd_logistic | motor_premotor | car_only | 500.000000 | 0.042948 |
| FirstDigitTouch | sgd_logistic | no_fp1_fp2 | bandpass_0_5_48 | 0.000000 | 0.042238 |
| LiftOff | sgd_logistic | no_fp1_fp2 | car_only | 500.000000 | 0.040559 |
| HandStart | sgd_logistic | motor_premotor | bandpass_0_5_48 | 0.000000 | 0.040006 |

## Interpretation Criteria

`ocular_proxy_only` is a confound-control setting. Strong performance there indicates possible gaze or blink information. `no_fp1_fp2` and `motor_premotor` are stricter channel controls for brain-focused interpretation. Non-zero `label_shift_ms` rows are negative timing controls and are interpreted separately from aligned prediction.
