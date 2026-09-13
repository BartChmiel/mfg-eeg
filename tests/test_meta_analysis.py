import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import false_discovery_control
from scripts.meta_analysis import RegionSummary, build_reports, fdr_bh, get_region  # noqa: E402


def _write_top_edges(path: Path, edges: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("phase=stub\n\n")
        for idx, (src, dst, val) in enumerate(edges, start=1):
            handle.write(f"{idx:03d}. {src} -> {dst} : {val:.6f}\n")


def _make_case_root(case_name: str) -> Path:
    root = REPO_ROOT / "out" / "test_meta_analysis" / case_name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


class MetaAnalysisTests(unittest.TestCase):
    def test_sparse_bh_equals_full_family(self) -> None:
        p = [0.0001, 0.003, 0.4]
        expected = false_discovery_control(p + [1.0] * 997)[:3]
        np.testing.assert_allclose(fdr_bh(p, total_tests=1000), expected)

    def test_reporting_threshold_does_not_change_q(self) -> None:
        root = _make_case_root("fdr_family")
        try:
            name = "phase_top_edges_A__B_gc_m4_pc1_lag50ms.txt"
            for subject in range(6):
                edges = [("Fp1", "Fp2", 1.0)]
                if subject < 2:
                    edges.append(("F3", "F4", 0.5))
                _write_top_edges(root / f"subj{subject:02d}" / name, edges)
            kwargs = dict(dir_root=str(root), topk=2, channels=32, p0=0.1,
                p0_inflate=1, mode_filter="gc", use_fdr=True, alpha=0.05,
                significant_only=True, dominant_flow_scope="reported", max_edges=3)
            low, _, _, meta = build_reports(min_subjects=1, **kwargs)
            high, _, _, _ = build_reports(min_subjects=3, **kwargs)
            self.assertEqual(high[0].q_value, low[0].q_value)
            self.assertEqual(meta["fdr_total_tests"], 992)
            self.assertEqual(meta["zero_recurrence_tests"], 990)
            self.assertAlmostEqual(high[0].q_value, 992 * 0.1**6)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_region_labels_are_anatomical_and_non_interpretive(self) -> None:
        self.assertEqual(get_region("Fp1"), "Frontal")
        self.assertEqual(get_region("C3"), "Frontocentral")
        label, description = RegionSummary.get_context("Frontal", "Parietal")
        self.assertEqual(label, "FRONTAL TO PARIETAL SENSOR FLOW")
        self.assertNotIn("executive", description.lower())

    def test_scenario_specific_subject_count_is_used(self) -> None:
        root = _make_case_root("scenario_subject_count")
        try:
            filename = "phase_top_edges_HandStart__FirstDigitTouch_gc_m4_pc1_lag50ms.txt"

            _write_top_edges(root / "subj01" / filename, [("Fp1", "Fp2", 1.0)])
            _write_top_edges(root / "subj02" / filename, [("Fp1", "Fp2", 1.0)])

            other_filename = "phase_top_edges_LiftOff__Replace_gc_m4_pc1_lag50ms.txt"
            _write_top_edges(root / "subj01" / other_filename, [("F3", "F4", 1.0)])
            _write_top_edges(root / "subj02" / other_filename, [("F3", "F4", 1.0)])
            _write_top_edges(root / "subj03" / other_filename, [("F3", "F4", 1.0)])

            edge_rows, scenario_rows, _text, metadata = build_reports(
                dir_root=str(root),
                topk=10,
                min_subjects=1,
                channels=32,
                p0=0.01,
                p0_inflate=10.0,
                mode_filter="gc",
                use_fdr=False,
                alpha=0.05,
                significant_only=False,
                dominant_flow_scope="reported",
                max_edges=3,
            )

            self.assertEqual(metadata["total_subjects"], 3)

            target_edge = next(
                row
                for row in edge_rows
                if row.phase == "HandStart__FirstDigitTouch" and row.src == "Fp1"
            )
            self.assertEqual(target_edge.n_subjects, 2)
            self.assertEqual(target_edge.total_subjects_in_dir, 3)

            target_scenario = next(
                row
                for row in scenario_rows
                if row.phase == "HandStart__FirstDigitTouch"
            )
            self.assertEqual(target_scenario.n_subjects, 2)
            self.assertEqual(target_scenario.total_subjects_in_dir, 3)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_significant_only_filters_console_report_candidates(self) -> None:
        root = _make_case_root("significant_only_filter")
        try:
            filename = "phase_top_edges_HandStart__FirstDigitTouch_gc_m4_pc1_lag50ms.txt"

            for subj in ["subj01", "subj02", "subj03"]:
                _write_top_edges(
                    root / subj / filename,
                    [("Fp1", "Fp2", 1.0), ("O1", "O2", 0.5)],
                )

            edge_rows, _scenario_rows, report_text, _metadata = build_reports(
                dir_root=str(root),
                topk=10,
                min_subjects=1,
                channels=32,
                p0=0.9,
                p0_inflate=10.0,
                mode_filter="gc",
                use_fdr=False,
                alpha=0.05,
                significant_only=True,
                dominant_flow_scope="reported",
                max_edges=3,
            )

            self.assertTrue(any(not row.significant for row in edge_rows))
            self.assertNotIn("HandStart__FirstDigitTouch | pc1 | lag=50ms", report_text)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
