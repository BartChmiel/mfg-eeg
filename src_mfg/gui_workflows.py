from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


FieldKind = Literal[
    "text",
    "int",
    "float",
    "bool",
    "choice",
    "dir",
    "file_open",
    "file_save",
    "int_list",
    "str_list",
]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    flag: str
    kind: FieldKind
    default: Any = ""
    required: bool = False
    help_text: str = ""
    choices: tuple[str, ...] = ()
    advanced: bool = False
    false_flag: str = ""


@dataclass(frozen=True)
class WorkflowSpec:
    key: str
    title: str
    module: str
    description: str
    fields: tuple[FieldSpec, ...]
    output_key: str | None = None
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    recommended_preset: str | None = None


def _split_tokens(text: str) -> list[str]:
    cleaned = str(text or "").replace(",", " ").replace(";", " ")
    return [token for token in cleaned.split() if token]


def parse_int_list(text: str) -> list[int]:
    tokens = _split_tokens(text)
    if not tokens:
        return []
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Expected a list of integers, got: {text!r}") from exc


def parse_str_list(text: str) -> list[str]:
    return _split_tokens(text)


def stringify_value(kind: FieldKind, value: Any) -> str:
    if kind in {"int_list", "str_list"}:
        if value is None:
            return ""
        return " ".join(str(item) for item in value)
    if value is None:
        return ""
    if kind == "bool":
        return "1" if bool(value) else "0"
    return str(value)


def _normalize_scalar(field: FieldSpec, raw_value: Any) -> Any:
    if field.kind == "bool":
        return bool(raw_value)

    if field.kind == "int_list" and isinstance(raw_value, (list, tuple)):
        values = [int(item) for item in raw_value]
        if not values and field.required:
            raise ValueError(f"Field '{field.label}' requires at least one integer.")
        return values or None

    if field.kind == "str_list" and isinstance(raw_value, (list, tuple)):
        values = [str(item) for item in raw_value if str(item).strip()]
        if not values and field.required:
            raise ValueError(f"Field '{field.label}' requires at least one value.")
        return values or None

    text = "" if raw_value is None else str(raw_value).strip()
    if text == "":
        if field.required:
            raise ValueError(f"Field '{field.label}' is required.")
        return None

    if field.kind == "int":
        return int(text)
    if field.kind == "float":
        return float(text)
    if field.kind == "choice":
        if field.choices and text not in field.choices:
            raise ValueError(
                f"Field '{field.label}' must be one of: {', '.join(field.choices)}."
            )
        return text
    if field.kind == "int_list":
        values = parse_int_list(text)
        if not values and field.required:
            raise ValueError(f"Field '{field.label}' requires at least one integer.")
        return values or None
    if field.kind == "str_list":
        values = parse_str_list(text)
        if not values and field.required:
            raise ValueError(f"Field '{field.label}' requires at least one value.")
        return values or None
    return text


def normalize_values(spec: WorkflowSpec, raw_values: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in spec.fields:
        values[field.key] = _normalize_scalar(field, raw_values.get(field.key))
    return values


def build_command(
    spec: WorkflowSpec,
    raw_values: dict[str, Any],
    *,
    python_executable: str,
) -> tuple[list[str], dict[str, Any]]:
    values = normalize_values(spec, raw_values)
    repo_root = Path(__file__).resolve().parents[1]
    protected = (repo_root / "out/article_package").resolve()
    for field in spec.fields:
        is_output = field.key in {spec.output_key, "cache_dir"} or field.kind == "file_save"
        value = values[field.key]
        if is_output and value:
            destination = (repo_root / str(value)).resolve()
            if destination == protected or protected in destination.parents:
                raise ValueError(
                    "The final out/article_package archive is read-only in the GUI; "
                    "choose an output under out/gui_runs."
                )
    cmd = [python_executable, "-m", spec.module]

    for field in spec.fields:
        value = values[field.key]
        if field.kind == "bool":
            if value:
                cmd.append(field.flag)
            elif field.false_flag:
                cmd.append(field.false_flag)
            continue
        if value is None or value == "":
            continue
        if isinstance(value, list):
            cmd.append(field.flag)
            cmd.extend(str(item) for item in value)
            continue
        cmd.extend([field.flag, str(value)])

    return cmd, values


def get_initial_values(spec: WorkflowSpec) -> dict[str, Any]:
    values: dict[str, Any] = {}
    preset = (
        spec.presets.get(spec.recommended_preset, {})
        if spec.recommended_preset is not None
        else {}
    )
    for field in spec.fields:
        values[field.key] = preset.get(field.key, field.default)
    return values


WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        key="build_basis",
        title="1. Build PCA Basis",
        module="scripts.build_kaggle_basis_allpairs_batch",
        description="Build the global all-pairs PCA basis used by the Kaggle PCA phase analysis.",
        output_key="out",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out", "Output .npz", "--out", "file_save", "out/basis_allpairs.npz", True),
            FieldSpec("mode", "Mode", "--mode", "choice", "gc", choices=("gc", "corr")),
            FieldSpec("epoch_len_s", "Epoch length (s)", "--epoch-len-s", "float", 2.0),
            FieldSpec("max_cycle_s", "Max cycle (s)", "--max-cycle-s", "float", 6.0, advanced=True),
            FieldSpec("min_cycles", "Min cycles", "--min-cycles", "int", 10),
            FieldSpec("m", "Legendre m", "--m", "int", 4),
            FieldSpec("lags_ms", "Lags (ms)", "--lags-ms", "int_list", [0, 50, 100, 150, 200]),
            FieldSpec("pca_r", "PCA components", "--pca-r", "int", 3),
            FieldSpec(
                "preprocess",
                "Preprocessing",
                "--preprocess",
                "choice",
                "car_only",
                choices=("car_only", "bandpass_0_5_48", "ocular_proxy_regression"),
                advanced=True,
            ),
            FieldSpec("subjects", "Subjects", "--subjects", "int_list", "", advanced=True),
            FieldSpec("series", "Series", "--series", "int_list", "", advanced=True),
            FieldSpec("max_files", "Max files", "--max-files", "int", "", advanced=True),
            FieldSpec("dry_run", "Dry run only", "--dry-run", "bool", False, advanced=True),
        ),
        presets={
            "Kaggle article basis": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out": "out/basis_allpairs.npz",
                "mode": "gc",
                "epoch_len_s": 2.0,
                "max_cycle_s": 6.0,
                "min_cycles": 10,
                "m": 4,
                "lags_ms": [0, 50, 100, 150, 200],
                "pca_r": 3,
                "preprocess": "car_only",
            }
        },
        recommended_preset="Kaggle article basis",
    ),
    WorkflowSpec(
        key="batch_phase",
        title="2. Phase Analysis",
        module="scripts.mfg_kaggle_phase_matrix_batch",
        description=(
            "Standard within-phase connectivity analysis for the Grasp-and-Lift Kaggle dataset."
        ),
        output_key="out",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out", "Output directory", "--out", "dir", "out/phase_mats_pca_by_subject", True),
            FieldSpec("mode", "Mode", "--mode", "choice", "gc", choices=("gc", "corr")),
            FieldSpec("epoch_len_s", "Epoch length (s)", "--epoch-len-s", "float", 2.0),
            FieldSpec("max_cycle_s", "Max cycle (s)", "--max-cycle-s", "float", 6.0, advanced=True),
            FieldSpec("min_cycles", "Min cycles", "--min-cycles", "int", 10),
            FieldSpec("m", "Legendre m", "--m", "int", 4),
            FieldSpec("lags_ms", "Lags (ms)", "--lags-ms", "int_list", [0, 50, 100, 150, 200]),
            FieldSpec("summary", "Summary", "--summary", "choice", "mean", choices=("mean", "max")),
            FieldSpec("metric", "Metric", "--metric", "choice", "energy", choices=("energy", "chi2", "neglog10p")),
            FieldSpec("basis_allpairs", "PCA basis .npz", "--basis-allpairs", "file_open", ""),
            FieldSpec("pca_r", "PCA components", "--pca-r", "int", 3),
            FieldSpec("subjects", "Subjects", "--subjects", "int_list", "", advanced=True),
            FieldSpec("series", "Series", "--series", "int_list", "", advanced=True),
            FieldSpec("max_files", "Max files", "--max-files", "int", "", advanced=True),
            FieldSpec("topk", "Top-k edges", "--topk", "int", 40, advanced=True),
            FieldSpec("save_npz", "Save .npz exports", "--save-npz", "bool", True),
            FieldSpec("group_by", "Group by", "--group-by", "choice", "subject", choices=("all", "subject")),
            FieldSpec(
                "preprocess",
                "Preprocessing",
                "--preprocess",
                "choice",
                "car_only",
                choices=("car_only", "bandpass_0_5_48", "ocular_proxy_regression"),
                advanced=True,
            ),
        ),
        presets={
            "Kaggle standard": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out": "out/phase_mats_by_subject",
                "mode": "gc",
                "epoch_len_s": 2.0,
                "max_cycle_s": 6.0,
                "min_cycles": 10,
                "m": 4,
                "lags_ms": [0, 50, 100, 150, 200],
                "summary": "mean",
                "metric": "energy",
                "basis_allpairs": "",
                "pca_r": 3,
                "topk": 40,
                "save_npz": True,
                "group_by": "subject",
            },
            "Kaggle PCA article": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out": "out/phase_mats_pca_by_subject",
                "mode": "gc",
                "epoch_len_s": 2.0,
                "max_cycle_s": 6.0,
                "min_cycles": 10,
                "m": 4,
                "lags_ms": [0, 50, 100, 150, 200],
                "summary": "mean",
                "metric": "energy",
                "basis_allpairs": "out/basis_allpairs.npz",
                "pca_r": 3,
                "topk": 40,
                "save_npz": True,
                "group_by": "subject",
            },
        },
        recommended_preset="Kaggle PCA article",
    ),
    WorkflowSpec(
        key="pre_event",
        title="3. Pre-Event EMA",
        module="scripts.mfg_kaggle_phase_matrix_experimental",
        description=(
            "Pre-event EMA analysis for readiness and decision-state dynamics before anchor events."
        ),
        output_key="out",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out", "Output directory", "--out", "dir", "out/experimental_kaggle_by_subject", True),
            FieldSpec("mode", "Mode", "--mode", "choice", "gc", choices=("gc", "corr")),
            FieldSpec("anchor_events", "Anchor events", "--anchor-events", "str_list", ["HandStart", "LiftOff"]),
            FieldSpec("pre_s", "Pre-event window (s)", "--pre-s", "float", 0.5),
            FieldSpec("burn_in_s", "Burn-in (s)", "--burn-in-s", "float", 0.5),
            FieldSpec("ema_half_life_s", "EMA half-life (s)", "--ema-half-life-s", "float", 0.1),
            FieldSpec("max_cycle_s", "Max cycle (s)", "--max-cycle-s", "float", 6.0, advanced=True),
            FieldSpec("min_cycles", "Min cycles", "--min-cycles", "int", 10),
            FieldSpec("m", "Legendre m", "--m", "int", 4),
            FieldSpec("lags_ms", "Lags (ms)", "--lags-ms", "int_list", [0, 50, 100, 150, 200]),
            FieldSpec("summary", "Summary", "--summary", "choice", "final", choices=("final", "mean", "max")),
            FieldSpec("metric", "Metric", "--metric", "choice", "energy", choices=("energy",)),
            FieldSpec("topk", "Top-k edges", "--topk", "int", 20, advanced=True),
            FieldSpec("trace_topk", "Trace top-k", "--trace-topk", "int", 5, advanced=True),
            FieldSpec("save_npz", "Save .npz exports", "--save-npz", "bool", True),
            FieldSpec("subjects", "Subjects", "--subjects", "int_list", "", advanced=True),
            FieldSpec("series", "Series", "--series", "int_list", "", advanced=True),
            FieldSpec("max_files", "Max files", "--max-files", "int", "", advanced=True),
            FieldSpec("group_by", "Group by", "--group-by", "choice", "subject", choices=("all", "subject")),
        ),
        presets={
            "Kaggle readiness article": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out": "out/experimental_kaggle_by_subject",
                "mode": "gc",
                "anchor_events": ["HandStart", "LiftOff"],
                "pre_s": 0.5,
                "burn_in_s": 0.5,
                "ema_half_life_s": 0.1,
                "max_cycle_s": 6.0,
                "min_cycles": 10,
                "m": 4,
                "lags_ms": [0, 50, 100, 150, 200],
                "summary": "final",
                "metric": "energy",
                "topk": 20,
                "trace_topk": 5,
                "save_npz": True,
                "group_by": "subject",
            }
        },
        recommended_preset="Kaggle readiness article",
    ),
    WorkflowSpec(
        key="meta_analysis",
        title="4. Meta Analysis",
        module="scripts.meta_analysis",
        description=(
            "Aggregate per-subject top edges, test reproducibility, and export article-ready tables and JSON summaries."
        ),
        output_key="out_dir",
        fields=(
            FieldSpec("dir", "Input result directory", "--dir", "dir", "out/phase_mats_pca_by_subject", True),
            FieldSpec("topk", "Top-k per file", "--topk", "int", 10),
            FieldSpec("min_subjects", "Min subjects", "--min-subjects", "int", 4),
            FieldSpec("channels", "Channels", "--channels", "int", 32, advanced=True),
            FieldSpec("p0", "p0 override", "--p0", "float", "", advanced=True),
            FieldSpec("p0_inflate", "p0 inflate", "--p0-inflate", "float", 10.0, advanced=True),
            FieldSpec("mode", "Mode filter", "--mode", "choice", "gc", choices=("gc", "corr", "any")),
            FieldSpec("use_fdr", "Use FDR", "--use-fdr", "bool", True),
            FieldSpec("alpha", "Alpha", "--alpha", "float", 0.05),
            FieldSpec("max_edges", "Max edges in text report", "--max-edges", "int", 3, advanced=True),
            FieldSpec("quiet", "Quiet console", "--quiet", "bool", False, advanced=True),
            FieldSpec("significant_only", "Significant only", "--significant-only", "bool", True),
            FieldSpec("dominant_flow_scope", "Dominant flow scope", "--dominant-flow-scope", "choice", "reported", choices=("reported", "all"), advanced=True),
            FieldSpec("out_dir", "Output export directory", "--out-dir", "dir", "out/article_meta_kaggle"),
            FieldSpec("out_text", "Optional report path", "--out-text", "file_save", "", advanced=True),
            FieldSpec("out_csv", "Optional edge CSV path", "--out-csv", "file_save", "", advanced=True),
            FieldSpec("out_scenarios_csv", "Optional scenario CSV path", "--out-scenarios-csv", "file_save", "", advanced=True),
            FieldSpec("out_json", "Optional JSON path", "--out-json", "file_save", "", advanced=True),
        ),
        presets={
            "Kaggle article meta": {
                "dir": "out/phase_mats_pca_by_subject",
                "topk": 10,
                "min_subjects": 4,
                "channels": 32,
                "mode": "gc",
                "use_fdr": True,
                "alpha": 0.05,
                "max_edges": 3,
                "significant_only": True,
                "dominant_flow_scope": "reported",
                "out_dir": "out/article_meta_kaggle",
            }
        },
        recommended_preset="Kaggle article meta",
    ),
    WorkflowSpec(
        key="meta_sensitivity",
        title="5. Sensitivity Analysis",
        module="scripts.meta_sensitivity",
        description=(
            "Run a robustness grid over top-k, minimum-subject, and p0-inflation settings."
        ),
        output_key="out_dir",
        fields=(
            FieldSpec("dir", "Input result directory", "--dir", "dir", "out/phase_mats_pca_by_subject", True),
            FieldSpec("out_dir", "Sensitivity output", "--out-dir", "dir", "out/article_meta_sensitivity", True),
            FieldSpec("topk", "Top-k grid", "--topk", "int_list", [5, 10, 15]),
            FieldSpec("min_subjects", "Min-subject grid", "--min-subjects", "int_list", [3, 4, 5]),
            FieldSpec("p0_inflate", "p0 inflate grid", "--p0-inflate", "str_list", [5.0, 10.0, 15.0]),
            FieldSpec("mode", "Mode filter", "--mode", "choice", "gc", choices=("gc", "corr", "any")),
            FieldSpec("channels", "Channels", "--channels", "int", 32, advanced=True),
            FieldSpec("alpha", "Alpha", "--alpha", "float", 0.05),
            FieldSpec("use_fdr", "Use FDR", "--use-fdr", "bool", True),
            FieldSpec("include_nonsignificant", "Include non-significant", "--include-nonsignificant", "bool", False, advanced=True),
        ),
        presets={
            "Kaggle robustness grid": {
                "dir": "out/phase_mats_pca_by_subject",
                "out_dir": "out/article_meta_sensitivity",
                "topk": [5, 10, 15],
                "min_subjects": [3, 4, 5],
                "p0_inflate": [5.0, 10.0, 15.0],
                "mode": "gc",
                "channels": 32,
                "alpha": 0.05,
                "use_fdr": True,
                "include_nonsignificant": False,
            }
        },
        recommended_preset="Kaggle robustness grid",
    ),
    WorkflowSpec(
        key="classification_benchmark",
        title="6. Classification Benchmark",
        module="scripts.kaggle_classification_benchmark",
        description=(
            "Train a Kaggle-style event classifier and compare baseline EEG features against MFG-assisted features."
        ),
        output_key="out_dir",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out_dir", "Benchmark output", "--out-dir", "dir", "out/classification_benchmark", True),
            FieldSpec("edge_table", "MFG edge table", "--edge-table", "file_open", "out/article_meta_sensitivity/edge_stability.csv"),
            FieldSpec("fallback_edge_table", "Fallback edge table", "--fallback-edge-table", "file_open", "out/article_meta_kaggle/meta_edges.csv", advanced=True),
            FieldSpec("feature_sets", "Feature sets", "--feature-sets", "str_list", ["baseline", "mfg", "combined"]),
            FieldSpec("subjects", "Subjects", "--subjects", "int_list", "", advanced=True),
            FieldSpec("train_series", "Train series", "--train-series", "int_list", [1, 2, 3, 4, 5, 6, 7]),
            FieldSpec("val_series", "Validation series", "--val-series", "int_list", [8]),
            FieldSpec("max_train_files", "Max train files", "--max-train-files", "int", "", advanced=True),
            FieldSpec("max_val_files", "Max validation files", "--max-val-files", "int", "", advanced=True),
            FieldSpec("max_train_samples_per_file", "Max train samples/file", "--max-train-samples-per-file", "int", "", advanced=True),
            FieldSpec("max_val_samples_per_file", "Max validation samples/file", "--max-val-samples-per-file", "int", "", advanced=True),
            FieldSpec("val_keep_all_positive", "Keep validation events", "--val-keep-all-positive", "bool", True, advanced=True),
            FieldSpec("test_root", "Test dataset root", "--test-root", "dir", "data/grasp-and-lift-eeg-detection/test"),
            FieldSpec("test_series", "Test series", "--test-series", "int_list", "", advanced=True),
            FieldSpec("max_test_files", "Max test files", "--max-test-files", "int", "", advanced=True),
            FieldSpec("submission_out", "Submission CSV", "--submission-out", "file_save", "out/classification_benchmark/submission.csv"),
            FieldSpec("submission_feature_set", "Submission feature set", "--submission-feature-set", "choice", "combined", choices=("baseline", "mfg", "combined", "fusion")),
            FieldSpec("sample_submission", "Sample submission", "--sample-submission", "file_open", "data/grasp-and-lift-eeg-detection/sample_submission.csv", advanced=True),
            FieldSpec("zip_submission", "Create submission ZIP", "--zip-submission", "bool", True),
            FieldSpec("submission_zip_out", "Submission ZIP", "--submission-zip-out", "file_save", "out/classification_benchmark/submission.zip"),
            FieldSpec("refit_for_submission", "Refit train+validation for submission", "--refit-for-submission", "bool", True),
            FieldSpec("max_submission_train_samples_per_file", "Max submission train samples/file", "--max-submission-train-samples-per-file", "int", "", advanced=True),
            FieldSpec("cache_dir", "Feature cache", "--cache-dir", "dir", "out/classification_benchmark/cache", advanced=True),
            FieldSpec("max_edges", "MFG edges", "--max-edges", "int", 24),
            FieldSpec("edge_selection", "Edge selection", "--edge-selection", "choice", "global", choices=("global", "event_union")),
            FieldSpec("channel_set", "Channel set", "--channel-set", "choice", "all32", choices=("all32", "motor_premotor", "no_fp1_fp2", "occipital_visual", "ocular_proxy_only")),
            FieldSpec("preprocess", "Preprocessing", "--preprocess", "choice", "car_only", choices=("car_only", "bandpass_0_5_48", "ocular_proxy_regression")),
            FieldSpec("label_shift_ms", "Label shift control (ms)", "--label-shift-ms", "float", 0.0, advanced=True),
            FieldSpec("mode", "Mode", "--mode", "choice", "gc", choices=("gc", "corr")),
            FieldSpec("m", "Legendre m", "--m", "int", 4),
            FieldSpec("ema_half_life_s", "EMA half-life (s)", "--ema-half-life-s", "float", 0.1),
            FieldSpec("ar_order", "AR order", "--ar-order", "int", 10, advanced=True),
            FieldSpec("student_nu", "Student nu", "--student-nu", "int", 10, advanced=True),
            FieldSpec("baseline_windows_ms", "Baseline windows (ms)", "--baseline-windows-ms", "int_list", [100, 200, 500]),
            FieldSpec("baseline_lags_ms", "Baseline lags (ms)", "--baseline-lags-ms", "int_list", [50, 100, 200]),
            FieldSpec("baseline_context", "Baseline context", "--baseline-context", "choice", "causal", choices=("causal", "centered")),
            FieldSpec("mfg_feature_mode", "MFG feature mode", "--mfg-feature-mode", "choice", "expanded", choices=("energy", "expanded")),
            FieldSpec("fusion_weight", "Fusion weight", "--fusion-weight", "float", "", advanced=True),
            FieldSpec("smooth_proba_ms", "Smooth probabilities (ms)", "--smooth-proba-ms", "float", 0.0, advanced=True),
            FieldSpec("stride", "Sample stride", "--stride", "int", 10),
            FieldSpec("classifier", "Classifier", "--classifier", "choice", "sgd_logistic", choices=("sgd_logistic", "extra_trees")),
            FieldSpec("epochs", "Training epochs", "--epochs", "int", 1),
            FieldSpec("alpha", "SGD alpha", "--alpha", "float", 0.0001, advanced=True),
            FieldSpec("tree_estimators", "Tree estimators", "--tree-estimators", "int", 120, advanced=True),
            FieldSpec("tree_max_depth", "Tree max depth", "--tree-max-depth", "int", "", advanced=True),
            FieldSpec("tree_min_samples_leaf", "Tree min samples/leaf", "--tree-min-samples-leaf", "int", 5, advanced=True),
            FieldSpec("random_state", "Random seed", "--random-state", "int", 42, advanced=True),
            FieldSpec("balance_classes", "Balance event classes", "--balance-classes", "bool", True),
        ),
        presets={
            "Kaggle MFG ablation": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out_dir": "out/classification_benchmark",
                "edge_table": "out/article_meta_sensitivity/edge_stability.csv",
                "fallback_edge_table": "out/article_meta_kaggle/meta_edges.csv",
                "feature_sets": ["baseline", "mfg", "combined"],
                "train_series": [1, 2, 3, 4, 5, 6, 7],
                "val_series": [8],
                "val_keep_all_positive": True,
                "test_root": "data/grasp-and-lift-eeg-detection/test",
                "submission_out": "out/classification_benchmark/submission.csv",
                "submission_feature_set": "fusion",
                "sample_submission": "data/grasp-and-lift-eeg-detection/sample_submission.csv",
                "zip_submission": True,
                "submission_zip_out": "out/classification_benchmark/submission.zip",
                "refit_for_submission": True,
                "cache_dir": "out/classification_benchmark/cache",
                "max_edges": 24,
                "edge_selection": "event_union",
                "channel_set": "all32",
                "preprocess": "bandpass_0_5_48",
                "label_shift_ms": 0.0,
                "mode": "gc",
                "m": 4,
                "ema_half_life_s": 0.1,
                "ar_order": 10,
                "student_nu": 10,
                "baseline_windows_ms": [100, 200, 500],
                "baseline_lags_ms": [50, 100, 200],
                "baseline_context": "causal",
                "mfg_feature_mode": "expanded",
                "fusion_weight": 0.25,
                "smooth_proba_ms": 0.0,
                "stride": 10,
                "classifier": "extra_trees",
                "epochs": 1,
                "alpha": 0.0001,
                "tree_estimators": 120,
                "tree_max_depth": "",
                "tree_min_samples_leaf": 5,
                "random_state": 42,
                "balance_classes": True,
            }
        },
        recommended_preset="Kaggle MFG ablation",
    ),
    WorkflowSpec(
        key="classification_sweep",
        title="7. Classification Sweep",
        module="scripts.kaggle_classification_sweep",
        description=(
            "Run a grid of classification ablations and rank when MFG-assisted features improve the EEG baseline."
        ),
        output_key="out_dir",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out_dir", "Sweep output", "--out-dir", "dir", "out/classification_sweep", True),
            FieldSpec("edge_table", "MFG edge table", "--edge-table", "file_open", "out/article_meta_sensitivity/edge_stability.csv"),
            FieldSpec("fallback_edge_table", "Fallback edge table", "--fallback-edge-table", "file_open", "out/article_meta_kaggle/meta_edges.csv", advanced=True),
            FieldSpec("splits", "Train|validation splits", "--splits", "str_list", ["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"]),
            FieldSpec("modes", "Modes", "--modes", "str_list", ["corr", "gc"]),
            FieldSpec("m_values", "Legendre m grid", "--m-values", "int_list", [2, 4]),
            FieldSpec("max_edges", "MFG edge grid", "--max-edges", "int_list", [8, 16, 24]),
            FieldSpec("edge_selections", "Edge selection grid", "--edge-selections", "str_list", ["global", "event_union"]),
            FieldSpec("channel_sets", "Channel-set grid", "--channel-sets", "str_list", ["all32"]),
            FieldSpec("preprocessings", "Preprocessing grid", "--preprocessings", "str_list", ["car_only"]),
            FieldSpec("label_shift_ms_values", "Label-shift grid (ms)", "--label-shift-ms-values", "str_list", [0.0]),
            FieldSpec("strides", "Stride grid", "--strides", "int_list", [10, 25]),
            FieldSpec("mfg_feature_modes", "MFG feature modes", "--mfg-feature-modes", "str_list", ["energy", "expanded"]),
            FieldSpec("fusion_weights", "Fusion weights", "--fusion-weights", "str_list", "", advanced=True),
            FieldSpec("baseline_windows_ms", "Baseline windows (ms)", "--baseline-windows-ms", "int_list", [100, 200, 500]),
            FieldSpec("baseline_lags_ms", "Baseline lags (ms)", "--baseline-lags-ms", "int_list", [50, 100, 200]),
            FieldSpec("baseline_contexts", "Baseline contexts", "--baseline-contexts", "str_list", ["causal"]),
            FieldSpec("smooth_proba_ms_values", "Smoothing grid (ms)", "--smooth-proba-ms-values", "str_list", [0.0], advanced=True),
            FieldSpec("ema_half_life_s", "EMA half-life (s)", "--ema-half-life-s", "float", 0.1),
            FieldSpec("classifiers", "Classifiers", "--classifiers", "str_list", ["sgd_logistic"]),
            FieldSpec("epochs", "Epochs", "--epochs", "int", 1),
            FieldSpec("tree_estimators", "Tree estimators", "--tree-estimators", "int", 120, advanced=True),
            FieldSpec("tree_max_depth", "Tree max depth", "--tree-max-depth", "int", "", advanced=True),
            FieldSpec("tree_min_samples_leaf", "Tree min samples/leaf", "--tree-min-samples-leaf", "int", 5, advanced=True),
            FieldSpec("max_train_files", "Max train files", "--max-train-files", "int", "", advanced=True),
            FieldSpec("max_val_files", "Max validation files", "--max-val-files", "int", "", advanced=True),
            FieldSpec("max_train_samples_per_file", "Max train samples/file", "--max-train-samples-per-file", "int", 1200, advanced=True),
            FieldSpec("max_val_samples_per_file", "Max validation samples/file", "--max-val-samples-per-file", "int", 1200, advanced=True),
            FieldSpec("val_keep_all_positive", "Keep validation events", "--val-keep-all-positive", "bool", True, advanced=True, false_flag="--no-val-keep-all-positive"),
            FieldSpec("cache_dir", "Sweep cache", "--cache-dir", "dir", "out/classification_sweep/cache", advanced=True),
            FieldSpec("balance_classes", "Balance event classes", "--balance-classes", "bool", True),
            FieldSpec("quiet", "Quiet run logs", "--quiet", "bool", False, advanced=True),
        ),
        presets={
            "Kaggle MFG sweep": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out_dir": "out/classification_sweep",
                "edge_table": "out/article_meta_sensitivity/edge_stability.csv",
                "fallback_edge_table": "out/article_meta_kaggle/meta_edges.csv",
                "splits": ["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"],
                "modes": ["corr", "gc"],
                "m_values": [2, 4],
                "max_edges": [8, 16, 24],
                "edge_selections": ["global", "event_union"],
                "channel_sets": ["all32"],
                "preprocessings": ["car_only"],
                "label_shift_ms_values": [0.0],
                "strides": [10, 25],
                "mfg_feature_modes": ["energy", "expanded"],
                "fusion_weights": [0.25],
                "baseline_windows_ms": [100, 200, 500],
                "baseline_lags_ms": [50, 100, 200],
                "baseline_contexts": ["causal", "centered"],
                "smooth_proba_ms_values": [0.0, 100.0],
                "ema_half_life_s": 0.1,
                "classifiers": ["sgd_logistic", "extra_trees"],
                "epochs": 1,
                "tree_estimators": 120,
                "tree_max_depth": "",
                "tree_min_samples_leaf": 5,
                "max_train_samples_per_file": 1200,
                "max_val_samples_per_file": 1200,
                "val_keep_all_positive": True,
                "cache_dir": "out/classification_sweep/cache",
                "balance_classes": True,
            }
        },
        recommended_preset="Kaggle MFG sweep",
    ),
    WorkflowSpec(
        key="classification_controls",
        title="8. Classification Controls",
        module="scripts.kaggle_classification_controls",
        description=(
            "Run the artefact and channel-set control grid for article-facing classifier claims."
        ),
        output_key="out_dir",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out_dir", "Controls output", "--out-dir", "dir", "out/classification_controls", True),
            FieldSpec("edge_table", "MFG edge table", "--edge-table", "file_open", "out/article_meta_sensitivity/edge_stability.csv"),
            FieldSpec("fallback_edge_table", "Fallback edge table", "--fallback-edge-table", "file_open", "out/article_meta_kaggle/meta_edges.csv", advanced=True),
            FieldSpec("splits", "Train|validation splits", "--splits", "str_list", ["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"]),
            FieldSpec("modes", "Modes", "--modes", "str_list", ["corr"]),
            FieldSpec("m_values", "Legendre m grid", "--m-values", "int_list", [2]),
            FieldSpec("max_edges", "MFG edge grid", "--max-edges", "int_list", [4, 8]),
            FieldSpec("edge_selections", "Edge selection grid", "--edge-selections", "str_list", ["global", "event_union"]),
            FieldSpec("channel_sets", "Channel-set grid", "--channel-sets", "str_list", ["all32", "no_fp1_fp2", "motor_premotor", "ocular_proxy_only"]),
            FieldSpec("preprocessings", "Preprocessing grid", "--preprocessings", "str_list", ["car_only", "bandpass_0_5_48"]),
            FieldSpec("label_shift_ms_values", "Label-shift controls (ms)", "--label-shift-ms-values", "str_list", [0.0, 500.0]),
            FieldSpec("strides", "Stride grid", "--strides", "int_list", [25]),
            FieldSpec("mfg_feature_modes", "MFG feature modes", "--mfg-feature-modes", "str_list", ["expanded"]),
            FieldSpec("fusion_weights", "Fusion weights", "--fusion-weights", "str_list", [0.25], advanced=True),
            FieldSpec("baseline_contexts", "Baseline contexts", "--baseline-contexts", "str_list", ["causal", "centered"], advanced=True),
            FieldSpec("smooth_proba_ms_values", "Smoothing grid (ms)", "--smooth-proba-ms-values", "str_list", [0.0, 100.0], advanced=True),
            FieldSpec("classifiers", "Classifiers", "--classifiers", "str_list", ["sgd_logistic"]),
            FieldSpec("tree_estimators", "Tree estimators", "--tree-estimators", "int", 120, advanced=True),
            FieldSpec("tree_max_depth", "Tree max depth", "--tree-max-depth", "int", "", advanced=True),
            FieldSpec("tree_min_samples_leaf", "Tree min samples/leaf", "--tree-min-samples-leaf", "int", 5, advanced=True),
            FieldSpec("max_train_samples_per_file", "Max train samples/file", "--max-train-samples-per-file", "int", 1200, advanced=True),
            FieldSpec("max_val_samples_per_file", "Max validation samples/file", "--max-val-samples-per-file", "int", 1200, advanced=True),
            FieldSpec("val_keep_all_positive", "Keep validation events", "--val-keep-all-positive", "bool", True, advanced=True, false_flag="--no-val-keep-all-positive"),
            FieldSpec("cache_dir", "Controls cache", "--cache-dir", "dir", "out/classification_controls/cache", advanced=True),
            FieldSpec("balance_classes", "Balance event classes", "--balance-classes", "bool", True),
            FieldSpec("quiet", "Quiet run logs", "--quiet", "bool", False, advanced=True),
        ),
        presets={
            "Kaggle artefact controls": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out_dir": "out/classification_controls",
                "edge_table": "out/article_meta_sensitivity/edge_stability.csv",
                "fallback_edge_table": "out/article_meta_kaggle/meta_edges.csv",
                "splits": ["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"],
                "modes": ["corr"],
                "m_values": [2],
                "max_edges": [4, 8],
                "edge_selections": ["global", "event_union"],
                "channel_sets": ["all32", "no_fp1_fp2", "motor_premotor", "ocular_proxy_only"],
                "preprocessings": ["car_only", "bandpass_0_5_48"],
                "label_shift_ms_values": [0.0, 500.0],
                "strides": [25],
                "mfg_feature_modes": ["expanded"],
                "fusion_weights": [0.25],
                "baseline_contexts": ["causal", "centered"],
                "smooth_proba_ms_values": [0.0, 100.0],
                "classifiers": ["sgd_logistic"],
                "tree_estimators": 120,
                "tree_max_depth": "",
                "tree_min_samples_leaf": 5,
                "max_train_samples_per_file": 1200,
                "max_val_samples_per_file": 1200,
                "val_keep_all_positive": True,
                "cache_dir": "out/classification_controls/cache",
                "balance_classes": True,
            }
        },
        recommended_preset="Kaggle artefact controls",
    ),
    WorkflowSpec(
        key="classification_subject",
        title="9. Subject Classification",
        module="scripts.kaggle_subject_classification",
        description=(
            "Run participant-specific event decoders and aggregate MFG-assisted classification lift."
        ),
        output_key="out_dir",
        fields=(
            FieldSpec("root", "Dataset root", "--root", "dir", "data/grasp-and-lift-eeg-detection/train", True),
            FieldSpec("out_dir", "Subject output", "--out-dir", "dir", "out/classification_subject", True),
            FieldSpec("edge_table", "MFG edge table", "--edge-table", "file_open", "out/article_meta_sensitivity/edge_stability.csv"),
            FieldSpec("fallback_edge_table", "Fallback edge table", "--fallback-edge-table", "file_open", "out/article_meta_kaggle/meta_edges.csv", advanced=True),
            FieldSpec("subjects", "Subjects", "--subjects", "int_list", list(range(1, 13))),
            FieldSpec("splits", "Train|validation splits", "--splits", "str_list", ["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"]),
            FieldSpec("feature_sets", "Feature sets", "--feature-sets", "str_list", ["baseline", "mfg", "combined"]),
            FieldSpec("mode", "Mode", "--mode", "choice", "corr", choices=("corr", "gc")),
            FieldSpec("m", "Legendre m", "--m", "int", 2),
            FieldSpec("max_edges", "MFG edges", "--max-edges", "int", 8),
            FieldSpec("edge_selection", "Edge selection", "--edge-selection", "choice", "event_union", choices=("global", "event_union")),
            FieldSpec("channel_set", "Channel set", "--channel-set", "choice", "all32", choices=("all32", "motor_premotor", "no_fp1_fp2", "occipital_visual", "ocular_proxy_only")),
            FieldSpec("preprocess", "Preprocessing", "--preprocess", "choice", "bandpass_0_5_48", choices=("car_only", "bandpass_0_5_48", "ocular_proxy_regression")),
            FieldSpec("label_shift_ms", "Label shift control (ms)", "--label-shift-ms", "float", 0.0, advanced=True),
            FieldSpec("stride", "Sample stride", "--stride", "int", 100),
            FieldSpec("baseline_windows_ms", "Baseline windows (ms)", "--baseline-windows-ms", "int_list", [100, 200, 500]),
            FieldSpec("baseline_lags_ms", "Baseline lags (ms)", "--baseline-lags-ms", "int_list", [50, 100, 200]),
            FieldSpec("baseline_context", "Baseline context", "--baseline-context", "choice", "causal", choices=("causal", "centered")),
            FieldSpec("mfg_feature_mode", "MFG feature mode", "--mfg-feature-mode", "choice", "expanded", choices=("energy", "expanded")),
            FieldSpec("smooth_proba_ms", "Smooth probabilities (ms)", "--smooth-proba-ms", "float", 0.0, advanced=True),
            FieldSpec("fusion_weight", "Fusion weight", "--fusion-weight", "float", "", advanced=True),
            FieldSpec("fusion_weight_values", "Fixed fusion sweep", "--fusion-weight-values", "str_list", [0.1, 0.15, 0.2, 0.25]),
            FieldSpec("classifier", "Classifier", "--classifier", "choice", "sgd_logistic", choices=("sgd_logistic", "extra_trees")),
            FieldSpec("epochs", "Training epochs", "--epochs", "int", 3),
            FieldSpec("alpha", "SGD alpha", "--alpha", "float", 0.2, advanced=True),
            FieldSpec("alpha_values", "Alpha grid", "--alpha-values", "str_list", [0.1, 0.2, 0.3]),
            FieldSpec("max_train_samples_per_file", "Max train samples/file", "--max-train-samples-per-file", "int", 1200, advanced=True),
            FieldSpec("max_val_samples_per_file", "Max validation samples/file", "--max-val-samples-per-file", "int", 1200, advanced=True),
            FieldSpec("val_keep_all_positive", "Keep validation events", "--val-keep-all-positive", "bool", True, advanced=True, false_flag="--no-val-keep-all-positive"),
            FieldSpec("balance_classes", "Balance event classes", "--balance-classes", "bool", True),
            FieldSpec("quiet", "Quiet run logs", "--quiet", "bool", False, advanced=True),
        ),
        presets={
            "Kaggle subject-aware validation": {
                "root": "data/grasp-and-lift-eeg-detection/train",
                "out_dir": "out/classification_subject",
                "edge_table": "out/article_meta_sensitivity/edge_stability.csv",
                "fallback_edge_table": "out/article_meta_kaggle/meta_edges.csv",
                "subjects": list(range(1, 13)),
                "splits": ["1 2 3 4 5 6|7", "1 2 3 4 5 6 7|8"],
                "feature_sets": ["baseline", "mfg", "combined"],
                "mode": "corr",
                "m": 2,
                "max_edges": 8,
                "edge_selection": "event_union",
                "channel_set": "all32",
                "preprocess": "bandpass_0_5_48",
                "label_shift_ms": 0.0,
                "stride": 100,
                "baseline_windows_ms": [100, 200, 500],
                "baseline_lags_ms": [50, 100, 200],
                "baseline_context": "causal",
                "mfg_feature_mode": "expanded",
                "smooth_proba_ms": 0.0,
                "fusion_weight": "",
                "fusion_weight_values": [0.1, 0.15, 0.2, 0.25],
                "classifier": "sgd_logistic",
                "epochs": 3,
                "alpha": 0.2,
                "alpha_values": [0.1, 0.2, 0.3],
                "max_train_samples_per_file": 1200,
                "max_val_samples_per_file": 1200,
                "val_keep_all_positive": True,
                "balance_classes": True,
                "quiet": False,
            }
        },
        recommended_preset="Kaggle subject-aware validation",
    ),
    WorkflowSpec(
        key="article_package",
        title="10. Experimental Package",
        module="scripts.build_article_package",
        description=(
            "Collect article-facing tables, copied meta exports, figure indexes, and provenance into one package."
        ),
        output_key="out",
        fields=(
            FieldSpec("phase_dir", "Phase results", "--phase-dir", "dir", "out/phase_mats_pca_by_subject", True),
            FieldSpec("pre_event_dir", "Pre-event results", "--pre-event-dir", "dir", "out/experimental_kaggle_by_subject", True),
            FieldSpec("meta_dir", "Meta-analysis exports", "--meta-dir", "dir", "out/article_meta_kaggle", True),
            FieldSpec("sensitivity_dir", "Sensitivity exports", "--sensitivity-dir", "dir", "out/article_meta_sensitivity"),
            FieldSpec("classification_dir", "Classification benchmark", "--classification-dir", "dir", "out/classification_benchmark"),
            FieldSpec("classification_sweep_dir", "Classification sweep", "--classification-sweep-dir", "dir", "out/classification_sweep"),
            FieldSpec("classification_controls_dir", "Classification controls", "--classification-controls-dir", "dir", "out/classification_controls"),
            FieldSpec("classification_subject_dir", "Subject classification", "--classification-subject-dir", "dir", "out/classification_subject_epochs3"),
            FieldSpec("out", "Experimental package", "--out", "dir", "out/gui_runs/article_package", True),
            FieldSpec("top_scenarios", "Top scenarios", "--top-scenarios", "int", 20),
            FieldSpec("top_edges", "Top edges", "--top-edges", "int", 50),
        ),
        presets={
            "Kaggle article package": {
                "phase_dir": "out/phase_mats_pca_by_subject",
                "pre_event_dir": "out/experimental_kaggle_by_subject",
                "meta_dir": "out/article_meta_kaggle",
                "sensitivity_dir": "out/article_meta_sensitivity",
                "classification_dir": "out/classification_benchmark",
                "classification_sweep_dir": "out/classification_sweep",
                "classification_controls_dir": "out/classification_controls",
                "classification_subject_dir": "out/classification_subject_epochs3",
                "out": "out/gui_runs/article_package",
                "top_scenarios": 20,
                "top_edges": 50,
            }
        },
        recommended_preset="Kaggle article package",
    ),
)


WORKFLOW_MAP = {workflow.key: workflow for workflow in WORKFLOWS}


def build_article_pipeline_inputs(
    *,
    dataset_root: str,
    out_root: str,
) -> list[tuple[WorkflowSpec, dict[str, Any]]]:
    base = Path(out_root)
    basis_out = str(base / "basis_allpairs.npz")
    phase_out = str(base / "phase_mats_pca_by_subject")
    pre_event_out = str(base / "experimental_kaggle_by_subject")
    meta_out = str(base / "article_meta_kaggle")
    sensitivity_out = str(base / "article_meta_sensitivity")
    classification_out = str(base / "classification_benchmark")
    classification_sweep_out = str(base / "classification_sweep")
    classification_controls_out = str(base / "classification_controls")
    classification_subject_out = str(base / "classification_subject_epochs3")
    package_out = str(base / "article_package")

    basis_spec = WORKFLOW_MAP["build_basis"]
    basis_values = get_initial_values(basis_spec)
    basis_values.update(
        {
            "root": dataset_root,
            "out": basis_out,
        }
    )

    phase_spec = WORKFLOW_MAP["batch_phase"]
    phase_values = get_initial_values(phase_spec)
    phase_values.update(
        {
            "root": dataset_root,
            "out": phase_out,
            "basis_allpairs": basis_out,
        }
    )

    pre_spec = WORKFLOW_MAP["pre_event"]
    pre_values = get_initial_values(pre_spec)
    pre_values.update(
        {
            "root": dataset_root,
            "out": pre_event_out,
        }
    )

    meta_spec = WORKFLOW_MAP["meta_analysis"]
    meta_values = get_initial_values(meta_spec)
    meta_values.update(
        {
            "dir": phase_out,
            "out_dir": meta_out,
        }
    )

    sensitivity_spec = WORKFLOW_MAP["meta_sensitivity"]
    sensitivity_values = get_initial_values(sensitivity_spec)
    sensitivity_values.update(
        {
            "dir": phase_out,
            "out_dir": sensitivity_out,
        }
    )

    classification_spec = WORKFLOW_MAP["classification_benchmark"]
    classification_values = get_initial_values(classification_spec)
    classification_values.update(
        {
            "root": dataset_root,
            "out_dir": classification_out,
            "edge_table": str(base / "article_meta_sensitivity" / "edge_stability.csv"),
            "fallback_edge_table": str(base / "article_meta_kaggle" / "meta_edges.csv"),
            "submission_out": str(base / "classification_benchmark" / "submission.csv"),
            "sample_submission": "data/grasp-and-lift-eeg-detection/sample_submission.csv",
            "submission_zip_out": str(base / "classification_benchmark" / "submission.zip"),
            "cache_dir": str(base / "classification_benchmark" / "cache"),
        }
    )

    package_spec = WORKFLOW_MAP["article_package"]
    package_values = get_initial_values(package_spec)
    package_values.update(
        {
            "phase_dir": phase_out,
            "pre_event_dir": pre_event_out,
            "meta_dir": meta_out,
            "sensitivity_dir": sensitivity_out,
            "classification_dir": classification_out,
            "classification_sweep_dir": classification_sweep_out,
            "classification_controls_dir": classification_controls_out,
            "classification_subject_dir": classification_subject_out,
            "out": package_out,
        }
    )

    return [
        (basis_spec, basis_values),
        (phase_spec, phase_values),
        (pre_spec, pre_values),
        (meta_spec, meta_values),
        (sensitivity_spec, sensitivity_values),
        (classification_spec, classification_values),
        (package_spec, package_values),
    ]
