import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_mfg.gui_workflows import WORKFLOW_MAP, build_command, parse_int_list  # noqa: E402
from src_mfg.gui_workflows import build_article_pipeline_inputs, get_initial_values  # noqa: E402


class GuiWorkflowTests(unittest.TestCase):
    def test_gui_rejects_protected_output_paths(self) -> None:
        destinations = [
            "out/article_package",
            "out/article_package/tables/overwrite.csv",
            "out/gui_runs/../article_package/overwrite.csv",
            str(REPO_ROOT / "out/article_package/overwrite.csv"),
        ]
        for spec in WORKFLOW_MAP.values():
            for field in spec.fields:
                if field.key not in {spec.output_key, "cache_dir"} and field.kind != "file_save":
                    continue
                for destination in destinations:
                    with self.subTest(workflow=spec.key, field=field.key, path=destination):
                        values = get_initial_values(spec)
                        values[field.key] = destination
                        with self.assertRaisesRegex(ValueError, "read-only in the GUI"):
                            build_command(spec, values, python_executable="python")

    def test_gui_defaults_and_experimental_pipeline_have_safe_outputs(self) -> None:
        for spec in WORKFLOW_MAP.values():
            build_command(spec, get_initial_values(spec), python_executable="python")
        steps = build_article_pipeline_inputs(dataset_root="data/train", out_root="out/gui_runs")
        for spec, values in steps:
            build_command(spec, values, python_executable="python")
        self.assertEqual(Path(steps[-1][1]["out"]), Path("out/gui_runs/article_package"))

    def test_gui_can_read_archived_inputs_and_write_to_sibling_directory(self) -> None:
        spec = WORKFLOW_MAP["article_package"]
        values = get_initial_values(spec)
        values["meta_dir"] = "out/article_package/raw_meta_exports"
        values["out"] = "out/article_package_copy"
        build_command(spec, values, python_executable="python")

    def test_article_workflows_define_recommended_presets(self) -> None:
        for key in [
            "build_basis",
            "batch_phase",
            "pre_event",
            "meta_analysis",
            "meta_sensitivity",
            "classification_benchmark",
            "classification_sweep",
            "classification_controls",
            "classification_subject",
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
                "classification_benchmark",
                "classification_controls",
                "classification_subject",
                "classification_sweep",
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
        expected_classification = str(Path("out/article_run/classification_benchmark"))
        expected_classification_sweep = str(Path("out/article_run/classification_sweep"))
        expected_classification_controls = str(Path("out/article_run/classification_controls"))
        expected_classification_subject = str(Path("out/article_run/classification_subject_epochs3"))

        self.assertEqual(
            [spec.key for spec, _ in steps],
            [
                "build_basis",
                "batch_phase",
                "pre_event",
                "meta_analysis",
                "meta_sensitivity",
                "classification_benchmark",
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
        self.assertEqual(steps[5][1]["out_dir"], expected_classification)
        self.assertEqual(steps[5][1]["edge_table"], str(Path("out/article_run/article_meta_sensitivity/edge_stability.csv")))
        self.assertEqual(steps[6][1]["phase_dir"], expected_phase)
        self.assertEqual(steps[6][1]["pre_event_dir"], expected_pre_event)
        self.assertEqual(steps[6][1]["meta_dir"], expected_meta)
        self.assertEqual(steps[6][1]["sensitivity_dir"], expected_sensitivity)
        self.assertEqual(steps[6][1]["classification_dir"], expected_classification)
        self.assertEqual(steps[6][1]["classification_sweep_dir"], expected_classification_sweep)
        self.assertEqual(steps[6][1]["classification_controls_dir"], expected_classification_controls)
        self.assertEqual(steps[6][1]["classification_subject_dir"], expected_classification_subject)

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

    def test_basis_command_accepts_preprocessing(self) -> None:
        spec = WORKFLOW_MAP["build_basis"]
        preset = dict(spec.presets["Kaggle article basis"])
        preset["preprocess"] = "bandpass_0_5_48"
        command, values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertIn("--preprocess", command)
        self.assertIn("bandpass_0_5_48", command)
        self.assertEqual(values["preprocess"], "bandpass_0_5_48")

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
        self.assertIn("--classification-dir", command)
        self.assertIn("--classification-sweep-dir", command)
        self.assertIn("--classification-controls-dir", command)
        self.assertIn("--classification-subject-dir", command)
        self.assertIn("--out", command)

    def test_classification_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["classification_benchmark"]
        preset = spec.presets["Kaggle MFG ablation"]
        command, values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertEqual(values["feature_sets"], ["baseline", "mfg", "combined"])
        self.assertIn("--feature-sets", command)
        self.assertIn("--balance-classes", command)
        self.assertIn("--zip-submission", command)
        self.assertIn("--refit-for-submission", command)
        self.assertIn("--edge-table", command)
        self.assertIn("--edge-selection", command)
        self.assertIn("--channel-set", command)
        self.assertIn("--preprocess", command)
        self.assertIn("--label-shift-ms", command)
        self.assertIn("--classifier", command)
        self.assertIn("--fusion-weight", command)
        self.assertIn("--baseline-context", command)
        self.assertIn("--smooth-proba-ms", command)
        self.assertIn("--val-keep-all-positive", command)
        self.assertEqual(values["edge_selection"], "event_union")
        self.assertEqual(values["channel_set"], "all32")
        self.assertEqual(values["preprocess"], "bandpass_0_5_48")
        self.assertEqual(values["classifier"], "extra_trees")
        self.assertEqual(values["label_shift_ms"], 0.0)
        self.assertEqual(values["baseline_context"], "causal")
        self.assertEqual(values["fusion_weight"], 0.25)
        self.assertEqual(values["smooth_proba_ms"], 0.0)
        self.assertTrue(values["val_keep_all_positive"])
        self.assertEqual(values["submission_feature_set"], "fusion")

    def test_classification_sweep_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["classification_sweep"]
        preset = spec.presets["Kaggle MFG sweep"]
        command, values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertEqual(values["modes"], ["corr", "gc"])
        self.assertIn("--splits", command)
        self.assertIn("--mfg-feature-modes", command)
        self.assertIn("--edge-selections", command)
        self.assertIn("--channel-sets", command)
        self.assertIn("--preprocessings", command)
        self.assertIn("--label-shift-ms-values", command)
        self.assertIn("--classifiers", command)
        self.assertIn("--fusion-weights", command)
        self.assertIn("--baseline-contexts", command)
        self.assertIn("--smooth-proba-ms-values", command)
        self.assertIn("--val-keep-all-positive", command)
        self.assertIn("--balance-classes", command)
        self.assertEqual(values["classifiers"], ["sgd_logistic", "extra_trees"])
        self.assertEqual(values["fusion_weights"], ["0.25"])
        self.assertEqual(values["baseline_contexts"], ["causal", "centered"])
        self.assertEqual(values["smooth_proba_ms_values"], ["0.0", "100.0"])
        self.assertTrue(values["val_keep_all_positive"])

    def test_classification_controls_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["classification_controls"]
        preset = spec.presets["Kaggle artefact controls"]
        command, values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertEqual(values["channel_sets"], ["all32", "no_fp1_fp2", "motor_premotor", "ocular_proxy_only"])
        self.assertIn("--channel-sets", command)
        self.assertIn("--preprocessings", command)
        self.assertIn("--label-shift-ms-values", command)
        self.assertIn("--edge-selections", command)
        self.assertIn("--classifiers", command)
        self.assertIn("--fusion-weights", command)
        self.assertIn("--baseline-contexts", command)
        self.assertIn("--smooth-proba-ms-values", command)
        self.assertIn("--val-keep-all-positive", command)
        self.assertIn("--balance-classes", command)
        self.assertEqual(values["fusion_weights"], ["0.25"])
        self.assertEqual(values["baseline_contexts"], ["causal", "centered"])
        self.assertEqual(values["smooth_proba_ms_values"], ["0.0", "100.0"])
        self.assertEqual(values["classifiers"], ["sgd_logistic"])
        self.assertTrue(values["val_keep_all_positive"])

    def test_classification_subject_preset_builds_expected_flags(self) -> None:
        spec = WORKFLOW_MAP["classification_subject"]
        preset = spec.presets["Kaggle subject-aware validation"]
        command, values = build_command(
            spec,
            preset,
            python_executable="python",
        )

        self.assertIn("--subjects", command)
        self.assertIn("--splits", command)
        self.assertIn("--fusion-weight-values", command)
        self.assertIn("--epochs", command)
        self.assertIn("--baseline-context", command)
        self.assertIn("--alpha-values", command)
        self.assertIn("--val-keep-all-positive", command)
        self.assertIn("--balance-classes", command)
        self.assertEqual(values["subjects"], list(range(1, 13)))
        self.assertEqual(values["baseline_context"], "causal")
        self.assertEqual(values["fusion_weight_values"], ["0.1", "0.15", "0.2", "0.25"])
        self.assertEqual(values["epochs"], 3)
        self.assertEqual(values["alpha_values"], ["0.1", "0.2", "0.3"])
        self.assertTrue(values["val_keep_all_positive"])

    def test_bool_field_can_emit_negative_flag(self) -> None:
        spec = WORKFLOW_MAP["classification_subject"]
        preset = dict(spec.presets["Kaggle subject-aware validation"])
        preset["val_keep_all_positive"] = False

        command, values = build_command(spec, preset, python_executable="python")

        self.assertFalse(values["val_keep_all_positive"])
        self.assertIn("--no-val-keep-all-positive", command)
        self.assertNotIn("--val-keep-all-positive", command)

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
