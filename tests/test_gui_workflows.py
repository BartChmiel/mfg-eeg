import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_mfg.gui_workflows import WORKFLOW_MAP, build_command, parse_int_list  # noqa: E402
from src_mfg.gui_workflows import build_article_pipeline_inputs  # noqa: E402


class GuiWorkflowTests(unittest.TestCase):
    def test_article_workflows_define_recommended_presets(self) -> None:
        for key in [
            "build_basis",
            "batch_phase",
            "pre_event",
            "meta_analysis",
            "meta_sensitivity",
            "article_package",
        ]:
            self.assertIsNotNone(WORKFLOW_MAP[key].recommended_preset)
        self.assertNotIn("convert_way", WORKFLOW_MAP)
        self.assertNotIn("compare_meta", WORKFLOW_MAP)
        self.assertEqual(
            sorted(WORKFLOW_MAP),
            [
                "article_package",
                "batch_phase",
                "build_basis",
                "meta_analysis",
                "meta_sensitivity",
                "pre_event",
            ],
        )

    def test_article_pipeline_links_outputs_between_steps(self) -> None:
        steps = build_article_pipeline_inputs(
            dataset_root="data/grasp-and-lift-eeg-detection/train",
            out_root="out/article_run",
        )
        expected_basis = str(Path("out/article_run/basis_allpairs.npz"))
        expected_phase = str(Path("out/article_run/phase_mats_pca_by_subject"))
        expected_pre_event = str(Path("out/article_run/experimental_kaggle_by_subject"))
        expected_meta = str(Path("out/article_run/article_meta_kaggle"))
        expected_sensitivity = str(Path("out/article_run/article_meta_sensitivity"))

        self.assertEqual(
            [spec.key for spec, _ in steps],
            [
                "build_basis",
                "batch_phase",
                "pre_event",
                "meta_analysis",
                "meta_sensitivity",
                "article_package",
            ],
        )
        self.assertEqual(steps[0][1]["out"], expected_basis)
        self.assertEqual(
            steps[1][1]["basis_allpairs"],
            expected_basis,
        )
        self.assertEqual(
            steps[3][1]["dir"],
            expected_phase,
        )
        self.assertEqual(steps[4][1]["dir"], expected_phase)
        self.assertEqual(steps[4][1]["out_dir"], expected_sensitivity)
        self.assertEqual(steps[5][1]["phase_dir"], expected_phase)
        self.assertEqual(steps[5][1]["pre_event_dir"], expected_pre_event)
        self.assertEqual(steps[5][1]["meta_dir"], expected_meta)
        self.assertEqual(steps[5][1]["sensitivity_dir"], expected_sensitivity)

    def test_parse_int_list_accepts_spaces_and_commas(self) -> None:
        self.assertEqual(parse_int_list("0, 50 100;200"), [0, 50, 100, 200])

    def test_batch_command_omits_empty_optional_values(self) -> None:
        spec = WORKFLOW_MAP["batch_phase"]
        command, values = build_command(
            spec,
            {
                "root": "data/train",
                "out": "out/test",
                "mode": "gc",
                "epoch_len_s": "2.0",
                "max_cycle_s": "6.0",
                "min_cycles": "10",
                "m": "4",
                "lags_ms": "0 50 100",
                "summary": "mean",
                "metric": "energy",
                "basis_allpairs": "",
                "pca_r": "3",
                "subjects": "",
                "series": "",
                "max_files": "",
                "topk": "40",
                "save_npz": True,
                "group_by": "subject",
            },
            python_executable="python",
        )

        self.assertEqual(values["lags_ms"], [0, 50, 100])
        self.assertIn("--save-npz", command)
        self.assertNotIn("--basis-allpairs", command)
        self.assertNotIn("--subjects", command)
        self.assertNotIn("--series", command)

    def test_meta_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["meta_analysis"]
        preset = spec.presets["Kaggle article meta"]
        command, _values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertIn("--use-fdr", command)
        self.assertIn("--significant-only", command)
        self.assertIn("--out-dir", command)
        self.assertIn("out/article_meta_kaggle", command)

    def test_article_package_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["article_package"]
        preset = spec.presets["Kaggle article package"]
        command, _values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertIn("--phase-dir", command)
        self.assertIn("--pre-event-dir", command)
        self.assertIn("--meta-dir", command)
        self.assertIn("--sensitivity-dir", command)
        self.assertIn("--out", command)

    def test_meta_sensitivity_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["meta_sensitivity"]
        preset = spec.presets["Kaggle robustness grid"]
        command, _values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertIn("--topk", command)
        self.assertIn("--min-subjects", command)
        self.assertIn("--p0-inflate", command)
        self.assertIn("--use-fdr", command)

    def test_batch_workflow_marks_some_fields_as_advanced(self) -> None:
        spec = WORKFLOW_MAP["batch_phase"]
        advanced_keys = {field.key for field in spec.fields if field.advanced}
        self.assertIn("max_files", advanced_keys)
        self.assertIn("subjects", advanced_keys)
        self.assertNotIn("root", advanced_keys)


if __name__ == "__main__":
    unittest.main()
