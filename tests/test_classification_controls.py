import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_classification_controls import (  # noqa: E402
    _build_sweep_args,
    _write_ablation_report,
    build_parser_controls,
)


class ClassificationControlsTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutil.rmtree(REPO_ROOT / "out" / "test_classification_controls", ignore_errors=True)

    def test_build_sweep_args_preserves_control_grid(self) -> None:
        args = build_parser_controls().parse_args(
            [
                "--out-dir",
                "out/test_classification_controls",
                "--splits",
                "1 2|3",
                "--channel-sets",
                "all32",
                "motor_premotor",
                "--preprocessings",
                "car_only",
                "--label-shift-ms-values",
                "0",
                "500",
                "--edge-selections",
                "event_union",
                "--classifiers",
                "sgd_logistic",
                "extra_trees",
                "--max-train-samples-per-file",
                "100",
                "--max-val-samples-per-file",
                "100",
            ]
        )

        sweep_args = _build_sweep_args(args)

        self.assertEqual(sweep_args.out_dir, "out/test_classification_controls")
        self.assertEqual(sweep_args.splits, ["1 2|3"])
        self.assertEqual(sweep_args.channel_sets, ["all32", "motor_premotor"])
        self.assertEqual(sweep_args.preprocessings, ["car_only"])
        self.assertEqual(sweep_args.label_shift_ms_values, [0.0, 500.0])
        self.assertEqual(sweep_args.edge_selections, ["event_union"])
        self.assertEqual(sweep_args.classifiers, ["sgd_logistic", "extra_trees"])

    def test_write_ablation_report_summarizes_controls(self) -> None:
        out_dir = REPO_ROOT / "out" / "test_classification_controls"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "ablation_report.md"
        rows = pd.DataFrame(
            [
                {
                    "config_id": 1,
                    "split": "1-2|3",
                    "channel_set": "all32",
                    "preprocess": "car_only",
                    "edge_selection": "event_union",
                    "mode": "corr",
                    "m": 2,
                    "max_edges": 4,
                    "stride": 100,
                    "mfg_feature_mode": "expanded",
                    "label_shift_ms": 0.0,
                    "edges_used": 12,
                    "baseline_auc": 0.60,
                    "combined_auc": 0.62,
                    "combined_minus_baseline": 0.02,
                },
                {
                    "config_id": 2,
                    "split": "1-2|3",
                    "channel_set": "all32",
                    "preprocess": "car_only",
                    "edge_selection": "event_union",
                    "mode": "corr",
                    "m": 2,
                    "max_edges": 4,
                    "stride": 100,
                    "mfg_feature_mode": "expanded",
                    "label_shift_ms": 500.0,
                    "edges_used": 12,
                    "baseline_auc": 0.60,
                    "combined_auc": 0.61,
                    "combined_minus_baseline": 0.01,
                },
                {
                    "config_id": 3,
                    "split": "1-2|3",
                    "channel_set": "ocular_proxy_only",
                    "preprocess": "car_only",
                    "edge_selection": "event_union",
                    "mode": "corr",
                    "m": 2,
                    "max_edges": 4,
                    "stride": 100,
                    "mfg_feature_mode": "expanded",
                    "label_shift_ms": 0.0,
                    "edges_used": 2,
                    "baseline_auc": 0.70,
                    "combined_auc": 0.69,
                    "combined_minus_baseline": -0.01,
                },
            ]
        )
        event_rows = pd.DataFrame(
            [
                {
                    "event": "HandStart",
                    "channel_set": "all32",
                    "preprocess": "car_only",
                    "label_shift_ms": 0.0,
                    "combined_minus_baseline": 0.04,
                }
            ]
        )

        summary = _write_ablation_report(
            out_path=report,
            rows=rows,
            event_rows=event_rows,
            summary={"configs_requested": 3, "configs_failed": 0},
        )

        text = report.read_text(encoding="utf-8")
        self.assertEqual(summary["configs_completed"], 3)
        self.assertEqual(summary["matched_timing_rows"], 1)
        self.assertIn("Classification Controls", text)
        self.assertIn("ocular_proxy_only", text)
        self.assertIn("Matched Timing Control", text)
        self.assertIn("0.010000", text)
        self.assertIn("Interpretation Criteria", text)


if __name__ == "__main__":
    unittest.main()
