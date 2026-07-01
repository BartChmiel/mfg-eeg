import csv
import contextlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_classification_benchmark import (  # noqa: E402
    _load_edges,
    _sample_indices,
    _smooth_probability_scores,
    build_parser,
    run_benchmark,
)
from src_mfg.io import KAGGLE_EVENT_COLS  # noqa: E402


class ClassificationBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "out" / "test_classification_benchmark"
        shutil.rmtree(self.root, ignore_errors=True)
        self.train_dir = self.root / "train"
        self.out_dir = self.root / "benchmark"
        self.train_dir.mkdir(parents=True)
        self.edge_table = self.root / "edge_stability.csv"
        self._write_edge_table()
        self._write_series(1)
        self._write_series(2)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_edge_table(self) -> None:
        rows = [
            {
                "src": "C3",
                "dst": "C4",
                "lag_ms": "50",
                "stability_fraction": "1.0",
                "min_q_value": "0.001",
            },
            {
                "src": "Fp1",
                "dst": "Fp2",
                "lag_ms": "0",
                "stability_fraction": "0.9",
                "min_q_value": "0.01",
            },
        ]
        with open(self.edge_table, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_series(self, series: int) -> None:
        n_samples = 240
        channels = ["Fp1", "Fp2", "C3", "C4"]
        data_path = self.train_dir / f"subj1_series{series}_data.csv"
        event_path = self.train_dir / f"subj1_series{series}_events.csv"

        with open(data_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", *channels])
            for t in range(n_samples):
                signal = 1.0 if any(abs(t - (30 + 30 * idx)) <= 4 for idx in range(6)) else 0.0
                writer.writerow(
                    [
                        f"subj1_series{series}_{t}",
                        t % 11,
                        (2 * t + series) % 13,
                        signal + (t % 7) * 0.1,
                        signal * 0.5 + (t % 5) * 0.1,
                    ]
                )

        with open(event_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", *KAGGLE_EVENT_COLS])
            for t in range(n_samples):
                labels = [1 if abs(t - (30 + 30 * idx)) <= 4 else 0 for idx in range(6)]
                writer.writerow([f"subj1_series{series}_{t}", *labels])

    def _write_test_series(self) -> None:
        data_path = self.root / "test" / "subj1_series3_data.csv"
        data_path.parent.mkdir(parents=True)
        channels = ["Fp1", "Fp2", "C3", "C4"]
        with open(data_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", *channels])
            for t in range(40):
                writer.writerow(
                    [
                        f"subj1_series3_{t}",
                        t % 11,
                        (2 * t) % 13,
                        (t % 7) * 0.1,
                        (t % 5) * 0.1,
                    ]
                )

    def _write_sample_submission(self) -> Path:
        path = self.root / "sample_submission.csv"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", *KAGGLE_EVENT_COLS])
            for t in range(40):
                writer.writerow([f"subj1_series3_{t}", *([0] * len(KAGGLE_EVENT_COLS))])
        return path

    def test_benchmark_writes_comparison_outputs(self) -> None:
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "mfg",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--stride",
                "3",
                "--max-edges",
                "2",
                "--mode",
                "corr",
                "--m",
                "2",
                "--epochs",
                "1",
                "--max-train-samples-per-file",
                "80",
                "--max-val-samples-per-file",
                "80",
                "--cache-dir",
                str(self.out_dir / "cache"),
                "--test-root",
                "",
                "--sample-submission",
                "",
                "--val-keep-all-positive",
                "--balance-classes",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        self.assertEqual(len(summary["scores"]), 3)
        self.assertTrue((self.out_dir / "model_comparison.csv").exists())
        self.assertTrue((self.out_dir / "auc_by_event.csv").exists())
        self.assertTrue((self.out_dir / "mfg_edges_used.csv").exists())
        with open(self.out_dir / "benchmark_summary.json", "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["parameters"]["feature_sets"], ["baseline", "mfg", "combined"])
        self.assertEqual(saved["parameters"]["classifier"], "sgd_logistic")
        self.assertEqual(saved["parameters"]["channel_set"], "all32")
        self.assertEqual(saved["parameters"]["preprocess"], "car_only")
        self.assertEqual(saved["parameters"]["label_shift_ms"], 0.0)
        self.assertTrue(saved["parameters"]["val_keep_all_positive"])

    def test_sampling_can_keep_event_positive_samples_between_stride_points(self) -> None:
        labels = np.zeros((12, 2), dtype=np.int8)
        labels[3, 0] = 1
        labels[7, 1] = 1

        indices = _sample_indices(labels, stride=5, max_samples=None, keep_all_positive=True)

        self.assertEqual(indices.tolist(), [0, 3, 5, 7, 10])

    def test_probability_smoothing_uses_original_sample_index(self) -> None:
        scores = np.array([[0.0], [1.0], [0.0]])
        sample_index = np.array([0, 50, 200])

        smoothed = _smooth_probability_scores(scores, 101, sample_index)

        self.assertTrue(np.allclose(smoothed[:, 0], [0.5, 0.5, 0.0]))

    def test_extra_trees_classifier_is_recorded(self) -> None:
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--classifier",
                "extra_trees",
                "--tree-estimators",
                "5",
                "--tree-min-samples-leaf",
                "1",
                "--stride",
                "5",
                "--max-edges",
                "2",
                "--mode",
                "corr",
                "--m",
                "2",
                "--max-train-samples-per-file",
                "80",
                "--max-val-samples-per-file",
                "80",
                "--cache-dir",
                str(self.out_dir / "cache_extra_trees"),
                "--test-root",
                "",
                "--sample-submission",
                "",
                "--balance-classes",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        self.assertEqual(summary["parameters"]["classifier"], "extra_trees")
        self.assertEqual(summary["parameters"]["tree_estimators"], 5)
        self.assertEqual(len(summary["scores"]), 2)

    def test_fusion_weight_adds_blended_scores(self) -> None:
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "mfg",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--fusion-weight",
                "0.25",
                "--stride",
                "4",
                "--max-edges",
                "2",
                "--mode",
                "corr",
                "--m",
                "2",
                "--epochs",
                "1",
                "--max-train-samples-per-file",
                "80",
                "--max-val-samples-per-file",
                "80",
                "--cache-dir",
                str(self.out_dir / "cache_fusion"),
                "--test-root",
                "",
                "--sample-submission",
                "",
                "--balance-classes",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        score_names = {row["feature_set"] for row in summary["scores"]}
        self.assertIn("fusion", score_names)
        self.assertEqual(summary["parameters"]["fusion_weight"], 0.25)
        with open(self.out_dir / "model_comparison.csv", "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("fusion", {row["feature_set"] for row in rows})

    def test_fusion_weight_grid_selects_blend_after_single_fit(self) -> None:
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "mfg",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--fusion-weights",
                "0.05",
                "0.25",
                "0.5",
                "--stride",
                "4",
                "--max-edges",
                "2",
                "--mode",
                "corr",
                "--m",
                "2",
                "--epochs",
                "1",
                "--max-train-samples-per-file",
                "80",
                "--max-val-samples-per-file",
                "80",
                "--cache-dir",
                str(self.out_dir / "cache_fusion_grid"),
                "--test-root",
                "",
                "--sample-submission",
                "",
                "--balance-classes",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        score_names = {row["feature_set"] for row in summary["scores"]}
        self.assertIn("fusion", score_names)
        self.assertIsNone(summary["parameters"]["fusion_weight"])
        self.assertEqual(summary["parameters"]["fusion_weights"], [0.05, 0.25, 0.5])
        self.assertIn(summary["parameters"]["selected_fusion_weight"], [0.05, 0.25, 0.5])

    def test_benchmark_writes_submission_when_test_data_exists(self) -> None:
        self._write_test_series()
        sample_submission = self._write_sample_submission()
        submission_path = self.out_dir / "submission.csv"
        submission_zip = self.out_dir / "submission.zip"
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--test-root",
                str(self.root / "test"),
                "--submission-out",
                str(submission_path),
                "--submission-zip-out",
                str(submission_zip),
                "--submission-feature-set",
                "combined",
                "--sample-submission",
                str(sample_submission),
                "--stride",
                "4",
                "--max-edges",
                "2",
                "--mode",
                "corr",
                "--m",
                "2",
                "--epochs",
                "1",
                "--max-train-samples-per-file",
                "80",
                "--max-val-samples-per-file",
                "80",
                "--cache-dir",
                str(self.out_dir / "cache"),
                "--zip-submission",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        self.assertEqual(summary["submission"]["rows"], 40)
        self.assertTrue(submission_path.exists())
        self.assertTrue(submission_zip.exists())
        with open(submission_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 40)
        self.assertEqual(["id", *KAGGLE_EVENT_COLS], list(rows[0].keys()))

    def test_event_union_edge_selection_uses_adjacent_phases(self) -> None:
        edge_path = self.root / "phase_edges.csv"
        rows = [
            {
                "phase": "HandStart__FirstDigitTouch",
                "src": "C3",
                "dst": "C4",
                "lag_ms": "50",
                "stability_fraction": "1.0",
                "min_q_value": "0.001",
            },
            {
                "phase": "FirstDigitTouch__BothStartLoadPhase",
                "src": "Fp1",
                "dst": "Fp2",
                "lag_ms": "100",
                "stability_fraction": "0.9",
                "min_q_value": "0.002",
            },
            {
                "phase": "UnrelatedPhase",
                "src": "P3",
                "dst": "P4",
                "lag_ms": "150",
                "stability_fraction": "0.8",
                "min_q_value": "0.003",
            },
        ]
        with open(edge_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        edges = _load_edges(edge_path, max_edges=1, edge_selection="event_union")
        keys = {(edge.src, edge.dst, edge.lag_ms) for edge in edges}

        self.assertIn(("C3", "C4", 50), keys)
        self.assertIn(("Fp1", "Fp2", 100), keys)
        self.assertNotIn(("P3", "P4", 150), keys)

    def test_channel_set_filters_edges_and_records_preprocessing(self) -> None:
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "mfg",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--channel-set",
                "no_fp1_fp2",
                "--preprocess",
                "bandpass_0_5_48",
                "--stride",
                "5",
                "--max-edges",
                "4",
                "--mode",
                "corr",
                "--m",
                "2",
                "--epochs",
                "1",
                "--max-train-samples-per-file",
                "60",
                "--max-val-samples-per-file",
                "60",
                "--cache-dir",
                str(self.out_dir / "cache_channel"),
                "--test-root",
                "",
                "--sample-submission",
                "",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        self.assertEqual(summary["parameters"]["channel_set"], "no_fp1_fp2")
        self.assertEqual(summary["parameters"]["preprocess"], "bandpass_0_5_48")
        self.assertNotIn("Fp1", summary["parameters"]["channels"])
        self.assertNotIn("Fp2", summary["parameters"]["channels"])
        self.assertEqual(summary["parameters"]["edges_used"], 1)

    def test_label_shift_is_recorded_in_outputs(self) -> None:
        args = build_parser().parse_args(
            [
                "--root",
                str(self.train_dir),
                "--out-dir",
                str(self.out_dir),
                "--edge-table",
                str(self.edge_table),
                "--feature-sets",
                "baseline",
                "combined",
                "--train-series",
                "1",
                "--val-series",
                "2",
                "--label-shift-ms",
                "20",
                "--stride",
                "4",
                "--max-edges",
                "2",
                "--mode",
                "corr",
                "--m",
                "2",
                "--epochs",
                "1",
                "--max-train-samples-per-file",
                "60",
                "--max-val-samples-per-file",
                "60",
                "--cache-dir",
                str(self.out_dir / "cache_shift"),
                "--test-root",
                "",
                "--sample-submission",
                "",
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_benchmark(args)

        self.assertEqual(summary["parameters"]["label_shift_ms"], 20.0)
        self.assertEqual(summary["parameters"]["label_shift_samples"], 10)


if __name__ == "__main__":
    unittest.main()
