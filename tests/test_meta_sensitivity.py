import csv
import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.meta_sensitivity import build_sensitivity  # noqa: E402


class MetaSensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "out" / "test_meta_sensitivity"
        shutil.rmtree(self.root, ignore_errors=True)
        self.input_dir = self.root / "phase"
        self.out_dir = self.root / "sensitivity"
        for subject in ["subj01", "subj02", "subj03"]:
            subject_dir = self.input_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            (subject_dir / "phase_top_edges_A__B_gc_m4_pc1_lag50ms.txt").write_text(
                "\n".join(
                    [
                        "phase=A__B",
                        "group=" + subject,
                        "",
                        "001. PO10 -> O2 : 1.000000",
                        "002. Fz -> Cz : 0.100000",
                    ]
                ),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_sensitivity_counts_stable_edges_across_grid(self) -> None:
        payload = build_sensitivity(
            dir_root=self.input_dir,
            out_dir=self.out_dir,
            topk_values=[1, 2],
            min_subjects_values=[2, 3],
            p0_inflate_values=[1.0],
            mode="gc",
            channels=4,
            alpha=0.05,
            use_fdr=True,
            significant_only=True,
        )

        self.assertEqual(payload["configs_successful"], 4)
        self.assertTrue((self.out_dir / "edge_stability.csv").exists())
        self.assertTrue((self.out_dir / "sensitivity_summary.md").exists())
        summary = (self.out_dir / "sensitivity_summary.md").read_text(encoding="utf-8")
        self.assertIn("Instances significant in at least one setting:", summary)
        self.assertIn("Reference-setting q-values", summary)
        self.assertNotIn("stronger candidates for article claims", summary)

        with open(
            self.out_dir / "edge_stability.csv",
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["src"], "PO10")
        self.assertEqual(rows[0]["dst"], "O2")
        self.assertEqual(rows[0]["config_count"], "4")
        self.assertEqual(rows[0]["successful_config_count"], "4")


if __name__ == "__main__":
    unittest.main()
