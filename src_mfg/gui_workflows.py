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

    text = str(raw_value or "").strip()
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
    cmd = [python_executable, "-m", spec.module]

    for field in spec.fields:
        value = values[field.key]
        if field.kind == "bool":
            if value:
                cmd.append(field.flag)
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
        key="article_package",
        title="6. Article Package",
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
            FieldSpec("out", "Article package", "--out", "dir", "out/article_package", True),
            FieldSpec("top_scenarios", "Top scenarios", "--top-scenarios", "int", 20),
            FieldSpec("top_edges", "Top edges", "--top-edges", "int", 50),
        ),
        presets={
            "Kaggle article package": {
                "phase_dir": "out/phase_mats_pca_by_subject",
                "pre_event_dir": "out/experimental_kaggle_by_subject",
                "meta_dir": "out/article_meta_kaggle",
                "sensitivity_dir": "out/article_meta_sensitivity",
                "out": "out/article_package",
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

    package_spec = WORKFLOW_MAP["article_package"]
    package_values = get_initial_values(package_spec)
    package_values.update(
        {
            "phase_dir": phase_out,
            "pre_event_dir": pre_event_out,
            "meta_dir": meta_out,
            "sensitivity_dir": sensitivity_out,
            "out": package_out,
        }
    )

    return [
        (basis_spec, basis_values),
        (phase_spec, phase_values),
        (pre_spec, pre_values),
        (meta_spec, meta_values),
        (sensitivity_spec, sensitivity_values),
        (package_spec, package_values),
    ]
