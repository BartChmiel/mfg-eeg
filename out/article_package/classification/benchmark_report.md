# Classification Benchmark

Metric: mean column-wise ROC-AUC across the six event labels.

| feature_set | mean_roc_auc | mean_average_precision | n_train_samples | n_val_samples |
|---|---:|---:|---:|---:|
| baseline | 0.556541 | 0.026816 | 14400 | 14400 |
| mfg | 0.505518 | 0.022087 | 14400 | 14400 |
| combined | 0.563729 | 0.027662 | 14400 | 14400 |


Feature sets: baseline EEG, MFG edge dynamics, and combined features.

Kaggle-style submission rows written: 3144171.
Submission file: `out\classification_benchmark\submission.csv`.
Compressed submission: `out\classification_benchmark\submission.zip`.
