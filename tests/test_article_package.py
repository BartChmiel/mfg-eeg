import csv
import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_article_package import build_package  # noqa: E402


class ArticlePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "out" / "test_article_package"
        shutil.rmtree(self.root, ignore_errors=True)
        self.phase_dir = self.root / "phase"
        self.pre_event_dir = self.root / "pre_event"
        self.meta_dir = self.root / "meta"
        self.out_dir = self.root / "package"
        self.meta_dir.mkdir(parents=True)
        self.phase_dir.mkdir(parents=True)
        self.pre_event_dir.mkdir(parents=True)

        (self.phase_dir / "subj01").mkdir()
        (self.phase_dir / "subj01" / "run_manifest.json").write_text(
            '{"group": "subj01", "mode": "gc", "files_used": 1, "cycles_total": 20}',
            encoding="utf-8",
        )
        (self.phase_dir / "subj01" / "phasegrid_demo.png").write_bytes(b"png")

        self._write_csv(
            self.meta_dir / "meta_scenarios.csv",
            [
                {
                    "phase": "A__B",
                    "pc": "pc1",
                    "lag_ms": "50",
                    "n_subjects": "12",
                    "total_subjects_in_dir": "12",
                    "candidate_edges": "8",
                    "reported_edges": "8",
                    "dominant_flow_src": "O_Vis",
                    "dominant_flow_dst": "F_Motor",
                    "dominant_flow_count": "4",
                    "dominant_flow_di": "1.0",
                    "process_label": "VISUO-MOTOR TRANSFORMATION",
                    "process_desc": "demo",
                },
                {
                    "phase": "C__D",
                    "pc": "pc2",
                    "lag_ms": "100",
                    "n_subjects": "12",
                    "total_subjects_in_dir": "12",
                    "candidate_edges": "2",
                    "reported_edges": "2",
                    "dominant_flow_src": "O_Vis",
                    "dominant_flow_dst": "O_Vis",
                    "dominant_flow_count": "1",
                    "dominant_flow_di": "0.0",
                    "process_label": "VISUAL RECURRENCE",
                    "process_desc": "demo",
                },
            ],
        )
        self._write_csv(
            self.meta_dir / "meta_edges.csv",
            [
                {
                    "phase": "A__B",
                    "pc": "pc1",
                    "lag_ms": "50",
                    "src": "PO10",
                    "dst": "O2",
                    "region_src": "O_Vis",
                    "region_dst": "O_Vis",
                    "k": "12",
                    "n_subjects": "12",
                    "total_subjects_in_dir": "12",
                    "p_value": "1e-12",
                    "q_value": "2e-12",
                    "significant": "True",
                },
                {
                    "phase": "A__B",
                    "pc": "pc1",
                    "lag_ms": "50",
                    "src": "Fz",
                    "dst": "Cz",
                    "region_src": "F_Exec",
                    "region_dst": "F_Motor",
                    "k": "4",
                    "n_subjects": "12",
                    "total_subjects_in_dir": "12",
                    "p_value": "0.04",
                    "q_value": "0.08",
                    "significant": "False",
                },
            ],
        )
        (self.meta_dir / "meta_report.txt").write_text("demo report", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_build_package_writes_article_tables_and_manifest(self) -> None:
        payload = build_package(
            phase_dir=self.phase_dir,
            pre_event_dir=self.pre_event_dir,
            meta_dir=self.meta_dir,
            out_dir=self.out_dir,
            top_scenarios=1,
            top_edges=1,
        )

        self.assertEqual(payload["counts"]["top_scenarios"], 1)
        self.assertEqual(payload["counts"]["top_edges"], 1)
        self.assertEqual(payload["counts"]["run_manifests"], 1)
        self.assertTrue((self.out_dir / "article_summary.md").exists())
        self.assertTrue((self.out_dir / "package_manifest.json").exists())
        self.assertTrue((self.out_dir / "tables" / "top_scenarios.csv").exists())
        self.assertTrue((self.out_dir / "tables" / "top_edges.csv").exists())
        self.assertTrue((self.out_dir / "raw_meta_exports" / "meta_report.txt").exists())

        with open(
            self.out_dir / "tables" / "top_edges.csv",
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["src"], "PO10")
        self.assertEqual(rows[0]["dst"], "O2")


if __name__ == "__main__":
    unittest.main()
