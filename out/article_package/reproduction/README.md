# Reproduction inputs

The archived rankings retain the top 15 ordered pairs for each of 12 subjects,
five phases, three PCA components, and five lags:

- `bandpass_0_5_48_rankings.csv.gz`: CAR followed by 0.5-48 Hz band-pass filtering.
- `car_only_rankings.csv.gz`: CAR-only recordings, using the July 21 all-pairs
  PCA basis rebuild. Each preprocessing variant uses its own fitted basis.

The adjacent `.manifest.json` files identify source score payloads by SHA256.
`inputs_manifest.json` provides checksums for the archived inputs and retained
classifier outputs; reproduction verifies them before calculating results.

`classifier_candidates.csv` preserves the CAR-only edge ordering used by the
reported classifier. It was extracted from the stability table in commit
`f129a71`, separately from the CAR-only ranking snapshot above. The classifier
reuses this table rather than selecting a graph from the current meta-analysis.
The experiment design is described in [Methodology](../documentation/METHODOLOGY.md#classification-benchmark).

`dense_scores.npz` and `dense_basis.npz` regenerate Figure 1.
`article_values.json` contains the expected manuscript counts and frontal-edge
sensitivity counts checked before PDF compilation.

From the repository root:

```text
python -m scripts.reproduce_article --source snapshots
```

The fixed reference setting uses top-k 10, minimum recurrence four, null
inflation ten, and alpha 0.05. Table I joins reference-setting q-values with
retention across the 27 sensitivity configurations. Outputs are written to
`out/reproduced_article/`; the archived inputs remain unchanged.

With the Kaggle training CSVs available, use `--source raw` to recompute the EEG
and classifier stages as well. Installation and paths are in the
[repository README](../../../README.md).
