from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.build_article_package import build_package
from scripts.export_phase_rankings import export_rankings
from scripts.meta_analysis import build_reports, write_edge_csv, write_scenario_csv, write_json_report
from scripts.meta_sensitivity import build_sensitivity
from scripts.volume_conduction_control import run_control
from scripts.plot_dense_pca_modes_by_lag import build_figure
from scripts.build_arxiv_submission import build_submission


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = ROOT / "out/article_package"
PHASE_FILE = "phasegrid_data_FirstDigitTouch__BothStartLoadPhase_gc_m4_pca_r3_lags0-50-100-150-200.npz"


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True)


def statistics(rankings: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        input_name = rankings.relative_to(ROOT).as_posix()
    except ValueError:
        input_name = str(rankings)
    edges, scenarios, report, metadata = build_reports(dir_root=input_name,
        topk=10, min_subjects=4, channels=32, p0=None, p0_inflate=10,
        mode_filter="gc", use_fdr=True, alpha=.05, significant_only=True,
        dominant_flow_scope="reported", max_edges=3)
    write_edge_csv(str(output / "meta_edges.csv"), edges)
    write_scenario_csv(str(output / "meta_scenarios.csv"), scenarios)
    write_json_report(str(output / "meta_report.json"), metadata=metadata,
                      edge_rows=edges, scenario_rows=scenarios)
    (output / "meta_report.txt").write_text(report, encoding="utf-8")
    build_sensitivity(dir_root=input_name, out_dir=output / "sensitivity",
        topk_values=[5, 10, 15], min_subjects_values=[3, 4, 5], p0_inflate_values=[5, 10, 15])
    run_control(edge_table=str(output / "sensitivity/edge_stability.csv"),
        out_dir=str(output / "volume_conduction"), short_range_cm=4.5,
        min_stability=1.0, n_permutations=20000, seed=20240617)


def raw_variant(data: Path, work: Path, variant: str) -> tuple[Path, Path, Path]:
    target = work / variant
    basis = target / "basis.npz"
    subject = target / "subjects"
    run("scripts.run_resumable_article_pipeline", "--root", data, "--preprocess", variant,
        "--basis", basis, "--phase-dir", subject, "--meta-dir", target / "stats",
        "--sensitivity-dir", target / "stats/sensitivity", "--vc-dir", target / "stats/volume_conduction",
        "--log-dir", target / "logs", "--force-derived", "--skip-package")
    rankings = target / "rankings.csv.gz"
    export_rankings(subject, rankings)
    return rankings, basis, subject


def classifier_commands(data: Path, work: Path) -> list[tuple[str, list[str]]]:
    common = ["--root", str(data), "--edge-table", str(ARCHIVED / "reproduction/classifier_candidates.csv"),
        "--fallback-edge-table", "", "--epochs", "3", "--alpha-values", "0.1", "0.2", "0.3",
        "--fusion-weight-values", "0.1", "0.15", "0.2", "0.25", "--quiet"]
    return [("scripts.kaggle_subject_classification", common + ["--out-dir", str(work / name),
            "--label-shift-ms", str(shift)]) for name, shift in [("aligned", 0), ("shifted", 500)]]


def verify_inputs(archive: Path) -> None:
    manifest = json.loads((archive / "reproduction/inputs_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((archive / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Archived input checksum mismatch: {name}")


def validate_article_values(work: Path, output: Path) -> None:
    expected = json.loads((ARCHIVED / "reproduction/article_values.json").read_text(encoding="utf-8"))
    for variant in ["car_only", "bandpass_0_5_48"]:
        sensitivity = json.loads((work / variant / "stats/sensitivity/sensitivity_manifest.json").read_text())
        vc = json.loads((work / variant / "stats/volume_conduction/volume_conduction_summary.json").read_text())
        observed = [sensitivity["stable_edge_count"], vc["total_edges"], vc["vc_robust_edges"]]
        if observed != expected[variant]:
            raise RuntimeError(f"New {variant} results {observed} differ from manuscript values; review text before compiling")
    with (output / "tables/primary_edge_sensitivity.csv").open(newline="", encoding="utf-8") as handle:
        primary = list(csv.DictReader(handle))
    if len(primary) != expected["primary_bandpass_count"]:
        raise RuntimeError("Primary band-pass instance count differs from the manuscript")
    for endpoint in expected["frontopolar_sensitivity"]:
        matching = [row for row in primary if all(str(row[key]) == str(endpoint[key])
            for key in ("phase", "pc", "lag_ms", "src", "dst"))]
        if len(matching) != 1 or any(int(matching[0][key]) != endpoint[key]
            for key in ("k", "config_count", "successful_config_count")):
            raise RuntimeError(f"Frontopolar recurrence differs from the manuscript: {endpoint}")


def reproduce(source: str, data: Path, output: Path, *, compile_pdf: bool = True) -> None:
    if output.resolve() == ARCHIVED.resolve() or ARCHIVED.resolve() in output.resolve().parents:
        raise ValueError("Use a separate output directory; the archived evidence is an input")
    verify_inputs(ARCHIVED)
    output = Path(os.path.relpath(output.resolve(), ROOT))
    archived = Path(os.path.relpath(ARCHIVED, ROOT))
    work = output.parent / (output.name + "_work")
    work.mkdir(parents=True, exist_ok=True)
    rankings = {}
    for variant in ["car_only", "bandpass_0_5_48"]:
        if source == "raw":
            rankings[variant], basis, subjects = raw_variant(data, work, variant)
            if variant == "bandpass_0_5_48":
                run("scripts.mfg_kaggle_phase_matrix_batch", "--root", data,
                    "--out", work / "pooled", "--mode", "gc", "--m", "4", "--pca-r", "3",
                    "--lags-ms", "0", "50", "100", "150", "200", "--basis-allpairs", basis,
                    "--preprocess", variant, "--group-by", "all", "--save-npz")
                dense_scores = work / "pooled/ALL" / PHASE_FILE
                dense_basis = basis
        else:
            rankings[variant] = archived / "reproduction" / f"{variant}_rankings.csv.gz"
            subjects = rankings[variant]
            dense_scores = ARCHIVED / "reproduction/dense_scores.npz"
            dense_basis = ARCHIVED / "reproduction/dense_basis.npz"
        statistics(rankings[variant], work / variant / "stats")
    classifier = archived / "classification_subject"
    if source == "raw":
        for module, args in classifier_commands(data, work):
            run(module, *args)
        run("scripts.compare_subject_classification_timing", "--aligned", work / "aligned/subject_results.csv",
            "--shifted", work / "shifted/subject_results.csv", "--out-dir", work / "aligned")
        classifier = work / "aligned"
    current = work / "bandpass_0_5_48/stats"
    baseline = work / "car_only/stats"
    manifest = build_package(phase_dir=subjects, pre_event_dir=work / "no_pre_event_analysis",
        meta_dir=current, sensitivity_dir=current / "sensitivity",
        classification_subject_dir=classifier, volume_conduction_dir=current / "volume_conduction",
        comparison_baseline_label="CAR-only", comparison_baseline_sensitivity_dir=baseline / "sensitivity",
        comparison_baseline_volume_conduction_dir=baseline / "volume_conduction",
        comparison_current_label="bandpass_0_5_48", out_dir=output)
    build_figure(scores_path=dense_scores, basis_path=dense_basis,
        out_base=output / "figures/article_dense_pca_modes_by_lag", scale_strategy="per-pc")
    shutil.copytree(ARCHIVED / "reproduction", output / "reproduction", dirs_exist_ok=True)
    shutil.copytree(baseline, output / "comparison/car_only", dirs_exist_ok=True)
    manifest["reproduction"] = {"source": source, "classifier_retrained": source == "raw",
        "classifier_graph": "reproduction/classifier_candidates.csv; selected from the full labelled collection",
        "fdr_family": "all 74400 ordered-pair/scenario hypotheses before recurrence filtering"}
    manifest["article_figures"] = [f"figures/{name}" for name in ["article_dense_pca_modes_by_lag.pdf",
        "top_edges_replication.png", "classification_timing_control.png"]]
    manifest["outputs"]["article_figures"] = [str(output / name) for name in manifest["article_figures"]]
    manifest["outputs"]["primary_sensitivity_table"] = str(output / "tables/primary_edge_sensitivity.csv")
    manifest["outputs"]["car_only_statistics"] = str(output / "comparison/car_only")
    manifest["counts"]["article_figures"] = len(manifest["article_figures"])
    manifest["warnings"] = [warning for warning in manifest["warnings"] if not warning.startswith(
        ("No run_manifest.json", "No PNG figure candidates"))]
    summary_path = output / "article_summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    summary = "\n".join(line for line in summary.splitlines() if not line.startswith(
        ("- No run_manifest.json", "- No PNG figure candidates",
         "- `classification/`:", "- `classification_sweep/`:", "- `classification_controls/`:")))
    if not manifest["warnings"]:
        summary = summary.replace("## Warnings\n", "")
    summary = summary.rstrip() + "\n\n## Reproduction Inputs\n\n"
    summary += "- [reproduction/README.md](reproduction/README.md): ranking snapshots, checksums, and classifier candidate provenance.\n"
    summary += "- `comparison/car_only/`: CAR-only statistics and sensitivity analysis.\n"
    summary_path.write_text(summary, encoding="utf-8")
    (output / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_article_values(work, output)
    if compile_pdf:
        build_submission(output / "arxiv", package=output, source=ROOT / "docs/eeg_mfg_article.tex")
        shutil.copy2(output / "arxiv/eeg_mfg_article.pdf", output / "documentation/eeg_mfg_article.pdf")
    print(f"Reproduction complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild article statistics and figures from archived scores or raw EEG.")
    parser.add_argument("--source", choices=["snapshots", "raw"], default="snapshots")
    parser.add_argument("--data", type=Path, default=ROOT / "data/grasp-and-lift-eeg-detection/train")
    parser.add_argument("--output", type=Path, default=ROOT / "out/reproduced_article")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    reproduce(args.source, args.data, args.output, compile_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
