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
        self.classification_dir = self.root / "classification"
        self.classification_sweep_dir = self.root / "classification_sweep"
        self.classification_controls_dir = self.root / "classification_controls"
        self.classification_subject_dir = self.root / "classification_subject"
        self.sensitivity_dir = self.root / "sensitivity"
        self.out_dir = self.root / "package"
        self.meta_dir.mkdir(parents=True)
        self.phase_dir.mkdir(parents=True)
        self.pre_event_dir.mkdir(parents=True)
        self.classification_dir.mkdir(parents=True)
        self.classification_sweep_dir.mkdir(parents=True)
        self.classification_controls_dir.mkdir(parents=True)
        self.classification_subject_dir.mkdir(parents=True)
        self.sensitivity_dir.mkdir(parents=True)

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
                    "dominant_flow_src": "Occipital",
                    "dominant_flow_dst": "Frontocentral",
                    "dominant_flow_count": "4",
                    "dominant_flow_di": "1.0",
                    "process_label": "OCCIPITAL TO FRONTOCENTRAL SENSOR FLOW",
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
                    "dominant_flow_src": "Occipital",
                    "dominant_flow_dst": "Occipital",
                    "dominant_flow_count": "1",
                    "dominant_flow_di": "0.0",
                    "process_label": "WITHIN OCCIPITAL SENSOR GROUP",
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
                    "region_src": "Occipital",
                    "region_dst": "Occipital",
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
                    "region_src": "Frontal",
                    "region_dst": "Frontocentral",
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
        (self.classification_dir / "model_comparison.csv").write_text(
            "feature_set,mean_roc_auc\nbaseline,0.70\ncombined,0.76\n",
            encoding="utf-8",
        )
        (self.classification_dir / "benchmark_summary.json").write_text(
            '{"metric": "mean column-wise ROC-AUC"}',
            encoding="utf-8",
        )
        (self.classification_sweep_dir / "sweep_results.csv").write_text(
            "config_id,combined_minus_baseline\n1,0.01\n",
            encoding="utf-8",
        )
        (self.classification_sweep_dir / "sweep_report.md").write_text(
            "# sweep",
            encoding="utf-8",
        )
        (self.classification_controls_dir / "ablation_results.csv").write_text(
            "config_id,channel_set,combined_minus_baseline\n1,motor_premotor,0.01\n",
            encoding="utf-8",
        )
        (self.classification_controls_dir / "ablation_report.md").write_text(
            "# controls",
            encoding="utf-8",
        )
        (self.classification_controls_dir / "ablation_summary.json").write_text(
            (
                '{"sweep_summary": {"best": {'
                '"baseline_auc": 0.66, '
                '"combined_auc": 0.71, '
                '"combined_minus_baseline": 0.05'
                "}}}"
            ),
            encoding="utf-8",
        )
        (self.classification_subject_dir / "subject_results.csv").write_text(
            "subject,split,best_assisted_minus_baseline\n1,1-6|7,0.01\n",
            encoding="utf-8",
        )
        (self.classification_subject_dir / "subject_report.md").write_text(
            "# subject",
            encoding="utf-8",
        )
        (self.classification_subject_dir / "timing_control.md").write_text(
            "# timing",
            encoding="utf-8",
        )
        (self.classification_subject_dir / "timing_control.csv").write_text(
            (
                "subject,aligned_minus_shifted\n"
                "1,0.010\n"
                "1,0.004\n"
                "2,-0.002\n"
                "2,0.001\n"
            ),
            encoding="utf-8",
        )
        (self.classification_subject_dir / "timing_control_summary.json").write_text(
            (
                '{"matched_rows": 4, "mean_aligned_lift": 0.003, '
                '"mean_shifted_lift": 0.001, "mean_delta": 0.002, '
                '"aligned_wins": 3}'
            ),
            encoding="utf-8",
        )
        self._write_csv(
            self.sensitivity_dir / "edge_stability.csv",
            [
                {
                    "phase": "A__B",
                    "pc": "pc1",
                    "lag_ms": "50",
                    "src": "PO10",
                    "dst": "O2",
                    "region_src": "Occipital",
                    "region_dst": "Occipital",
                    "config_count": "27",
                    "successful_config_count": "27",
                    "stability_fraction": "1.0",
                }
            ],
        )

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
            classification_dir=self.classification_dir,
            classification_sweep_dir=self.classification_sweep_dir,
            classification_controls_dir=self.classification_controls_dir,
            classification_subject_dir=self.classification_subject_dir,
            sensitivity_dir=self.sensitivity_dir,
            out_dir=self.out_dir,
            top_scenarios=1,
            top_edges=1,
        )

        self.assertEqual(payload["counts"]["top_scenarios"], 1)
        self.assertEqual(payload["counts"]["top_edges"], 1)
        self.assertEqual(payload["counts"]["run_manifests"], 1)
        self.assertEqual(payload["counts"]["evidence_table_files"], 2)
        self.assertGreaterEqual(payload["counts"]["article_figures"], 1)
        self.assertTrue((self.out_dir / "article_summary.md").exists())
        self.assertTrue((self.out_dir / "package_manifest.json").exists())
        self.assertTrue((self.out_dir / "tables" / "top_scenarios.csv").exists())
        self.assertTrue((self.out_dir / "tables" / "top_edges.csv").exists())
        self.assertTrue((self.out_dir / "tables" / "classifier_evidence.csv").exists())
        self.assertTrue((self.out_dir / "tables" / "sensitivity_evidence.csv").exists())
        self.assertTrue((self.out_dir / "raw_meta_exports" / "meta_report.txt").exists())
        self.assertTrue((self.out_dir / "classification" / "model_comparison.csv").exists())
        self.assertTrue((self.out_dir / "classification_sweep" / "sweep_results.csv").exists())
        self.assertTrue((self.out_dir / "classification_controls" / "ablation_results.csv").exists())
        self.assertTrue((self.out_dir / "classification_subject" / "subject_results.csv").exists())
        self.assertTrue((self.out_dir / "classification_subject" / "timing_control.md").exists())
        self.assertTrue((self.out_dir / "documentation" / "eeg_mfg_article.pdf").exists())
        self.assertTrue((self.out_dir / "documentation" / "references.bib").exists())
        self.assertFalse((self.out_dir / "documentation" / "eeg_mfg_article.tex").exists())
        self.assertTrue((self.out_dir / "figures" / "top_edges_replication.png").exists())

        with open(
            self.out_dir / "tables" / "top_edges.csv",
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["src"], "PO10")
        self.assertEqual(rows[0]["dst"], "O2")

        summary = (self.out_dir / "article_summary.md").read_text(encoding="utf-8")
        self.assertIn("Classifier Evidence Snapshot", summary)
        self.assertIn("Sensitivity Evidence Snapshot", summary)
        self.assertNotIn("process", summary.split("## Top Replicated Edges", 1)[0].lower())


if __name__ == "__main__":
    unittest.main()
