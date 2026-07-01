import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_subject_classification import _write_report, build_parser_subject  # noqa: E402


class SubjectClassificationTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutil.rmtree(REPO_ROOT / "out" / "test_subject_classification", ignore_errors=True)

    def test_parser_accepts_alpha_grid(self) -> None:
        args = build_parser_subject().parse_args(["--alpha-values", "0.1", "0.3", "1.0"])

        self.assertEqual(args.alpha_values, [0.1, 0.3, 1.0])

    def test_parser_accepts_fusion_grid(self) -> None:
        args = build_parser_subject().parse_args(["--fusion-weights", "0.05", "0.1", "0.25"])

        self.assertEqual(args.fusion_weights, [0.05, 0.1, 0.25])

    def test_parser_accepts_fixed_fusion_sweep(self) -> None:
        args = build_parser_subject().parse_args(["--fusion-weight-values", "0.1", "0.25", "0.4"])

        self.assertEqual(args.fusion_weight_values, [0.1, 0.25, 0.4])

    def test_report_summarizes_hyperparameters(self) -> None:
        out_dir = REPO_ROOT / "out" / "test_subject_classification"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "subject_report.md"

        _write_report(
            report,
            [
                {
                    "subject": 1,
                    "split": "1-6|7",
                    "alpha": 0.3,
                    "fusion_weight": 0.25,
                    "baseline_auc": 0.8,
                    "mfg_auc": 0.7,
                    "combined_auc": 0.82,
                    "fusion_auc": 0.83,
                    "best_assisted_feature_set": "fusion",
                    "best_assisted_auc": 0.83,
                    "best_assisted_minus_baseline": 0.03,
                },
                {
                    "subject": 2,
                    "split": "1-6|7",
                    "alpha": 1.0,
                    "fusion_weight": 0.25,
                    "baseline_auc": 0.78,
                    "mfg_auc": 0.69,
                    "combined_auc": 0.79,
                    "fusion_auc": 0.80,
                    "best_assisted_feature_set": "fusion",
                    "best_assisted_auc": 0.80,
                    "best_assisted_minus_baseline": 0.02,
                },
            ],
        )

        text = report.read_text(encoding="utf-8")
        self.assertIn("Hyperparameter Summary", text)
        self.assertIn("| 0.3 | 0.25 |", text)
        self.assertIn("| 1 | 1-6|7 | 0.3 | 0.25 |", text)


if __name__ == "__main__":
    unittest.main()
