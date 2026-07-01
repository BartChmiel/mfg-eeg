import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_classification_sweep import _write_markdown, build_configs, build_parser_sweep  # noqa: E402


class ClassificationSweepTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutil.rmtree(REPO_ROOT / "out" / "test_classification_sweep", ignore_errors=True)

    def test_build_configs_expands_grid_and_splits(self) -> None:
        args = build_parser_sweep().parse_args(
            [
                "--splits",
                "1 2|3",
                "--modes",
                "corr",
                "--m-values",
                "2",
                "--max-edges",
                "4",
                "8",
                "--edge-selections",
                "global",
                "event_union",
                "--channel-sets",
                "all32",
                "no_fp1_fp2",
                "--preprocessings",
                "car_only",
                "--label-shift-ms-values",
                "0",
                "500",
                "--strides",
                "10",
                "--mfg-feature-modes",
                "energy",
            ]
        )

        configs = build_configs(args)

        self.assertEqual(len(configs), 16)
        self.assertEqual(configs[0]["train_series"], [1, 2])
        self.assertEqual(configs[0]["val_series"], [3])
        self.assertEqual(configs[0]["max_edges"], 4)
        self.assertEqual(configs[0]["edge_selection"], "global")
        self.assertEqual(configs[0]["classifier"], "sgd_logistic")
        self.assertEqual(configs[0]["channel_set"], "all32")
        self.assertEqual(configs[0]["label_shift_ms"], 0.0)
        self.assertEqual(configs[1]["label_shift_ms"], 500.0)
        self.assertEqual(configs[2]["channel_set"], "no_fp1_fp2")
        self.assertEqual(configs[4]["edge_selection"], "event_union")
        self.assertEqual(configs[8]["max_edges"], 8)

    def test_build_configs_expands_fusion_weights(self) -> None:
        args = build_parser_sweep().parse_args(
            [
                "--splits",
                "1|2",
                "--modes",
                "corr",
                "--m-values",
                "2",
                "--max-edges",
                "4",
                "--edge-selections",
                "global",
                "--channel-sets",
                "all32",
                "--preprocessings",
                "car_only",
                "--label-shift-ms-values",
                "0",
                "--strides",
                "10",
                "--mfg-feature-modes",
                "energy",
                "--fusion-weights",
                "0.1",
                "0.25",
            ]
        )

        configs = build_configs(args)

        self.assertEqual(len(configs), 2)
        self.assertEqual([config["fusion_weight"] for config in configs], [0.1, 0.25])

    def test_write_markdown_reports_edge_cap_and_used_edges(self) -> None:
        out_dir = REPO_ROOT / "out" / "test_classification_sweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "sweep_report.md"

        _write_markdown(
            report_path,
            [
                {
                    "config_id": 1,
                    "split": "1-2|3",
                    "classifier": "sgd_logistic",
                    "channel_set": "motor_premotor",
                    "preprocess": "bandpass_0_5_48",
                    "edge_selection": "event_union",
                    "mode": "corr",
                    "m": 2,
                    "max_edges": 4,
                    "edges_used": 12,
                    "stride": 25,
                    "mfg_feature_mode": "expanded",
                    "baseline_auc": 0.6,
                    "mfg_auc": 0.55,
                    "combined_auc": 0.63,
                    "combined_minus_baseline": 0.03,
                }
            ],
            [],
        )

        text = report_path.read_text(encoding="utf-8")
        self.assertIn("edge selection", text)
        self.assertIn("edge cap", text)
        self.assertIn("label_shift_ms", text)
        self.assertIn("preprocess", text)
        self.assertIn("| 1 | 1-2|3 | sgd_logistic | motor_premotor | bandpass_0_5_48 | causal | 0.0 | 0.0 | none | event_union | corr | 2 | 4 | 12 |", text)


if __name__ == "__main__":
    unittest.main()
