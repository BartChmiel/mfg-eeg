import csv
import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_subject_classification_timing import compare_subject_timing  # noqa: E402


class SubjectTimingCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "out" / "test_subject_timing_compare"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.aligned = self.root / "aligned.csv"
        self.shifted = self.root / "shifted.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _row(
        self,
        lift: float,
        *,
        alpha: float = 0.3,
        fusion_weight: float = 0.25,
        fusion_grid: str | None = None,
    ) -> dict[str, object]:
        row = {
            "subject": 1,
            "split": "1-6|7",
            "classifier": "sgd_logistic",
            "channel_set": "all32",
            "preprocess": "bandpass_0_5_48",
            "baseline_context": "causal",
            "smooth_proba_ms": 0.0,
            "val_keep_all_positive": True,
            "edge_selection": "event_union",
            "mode": "corr",
            "m": 2,
            "max_edges": 8,
            "alpha": alpha,
            "fusion_weight": fusion_weight,
            "stride": 100,
            "mfg_feature_mode": "expanded",
            "baseline_auc": 0.8,
            "best_assisted_auc": 0.8 + lift,
            "best_assisted_feature_set": "fusion",
            "best_assisted_minus_baseline": lift,
        }
        if fusion_grid is not None:
            row["fusion_grid"] = fusion_grid
        return row

    def _write(self, path: Path, lift: float, *, fusion_weight: float = 0.25, fusion_grid: str | None = None) -> None:
        row = self._row(lift, fusion_weight=fusion_weight, fusion_grid=fusion_grid)
        self._write_rows(path, [row])

    def _write_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_compare_subject_timing_computes_matched_delta(self) -> None:
        self._write(self.aligned, 0.03)
        self._write(self.shifted, 0.01)

        matched, grouped = compare_subject_timing(self.aligned, self.shifted)

        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(float(matched.iloc[0]["aligned_minus_shifted"]), 0.02)
        self.assertAlmostEqual(float(grouped.iloc[0]["mean_delta"]), 0.02)

    def test_compare_subject_timing_matches_grid_not_selected_weight(self) -> None:
        self._write(self.aligned, 0.03, fusion_weight=0.1, fusion_grid="0.05 0.1 0.25")
        self._write(self.shifted, 0.01, fusion_weight=0.25, fusion_grid="0.05 0.1 0.25")

        matched, grouped = compare_subject_timing(self.aligned, self.shifted)

        self.assertEqual(len(matched), 1)
        self.assertIn("fusion_grid", matched.columns)
        self.assertNotIn("fusion_weight", grouped.columns)
        self.assertAlmostEqual(float(matched.iloc[0]["aligned_minus_shifted"]), 0.02)

    def test_compare_subject_timing_prefers_positive_aligned_lift_groups(self) -> None:
        self._write_rows(
            self.aligned,
            [
                self._row(-0.01, alpha=0.1, fusion_weight=0.4),
                self._row(0.02, alpha=0.3, fusion_weight=0.15),
            ],
        )
        self._write_rows(
            self.shifted,
            [
                self._row(-0.04, alpha=0.1, fusion_weight=0.4),
                self._row(0.005, alpha=0.3, fusion_weight=0.15),
            ],
        )

        _matched, grouped = compare_subject_timing(self.aligned, self.shifted)

        self.assertAlmostEqual(float(grouped.iloc[0]["alpha"]), 0.3)
        self.assertTrue(bool(grouped.iloc[0]["summary_eligible"]))


if __name__ == "__main__":
    unittest.main()
