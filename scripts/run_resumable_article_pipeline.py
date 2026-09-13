from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path
    basis: Path
    phase_dir: Path
    meta_dir: Path
    sensitivity_dir: Path
    vc_dir: Path
    package_dir: Path
    pre_event_dir: Path
    classification_dir: Path
    classification_sweep_dir: Path
    classification_controls_dir: Path
    classification_subject_dir: Path
    log_dir: Path


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _exists(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def _run_with_retries(
    *,
    name: str,
    cmd: list[str],
    log_dir: Path,
    retries: int,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    last_rc = 1
    for attempt in range(1, retries + 2):
        log_path = log_dir / f"{name}.attempt{attempt}.log"
        print(f"[RUN] {name} attempt {attempt}/{retries + 1}", flush=True)
        print("      " + " ".join(cmd), flush=True)
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(cmd)}\n\n")
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        last_rc = int(proc.returncode)
        if last_rc == 0:
            print(f"[OK] {name} (log: {log_path})", flush=True)
            return
        print(f"[WARN] {name} failed with exit code {last_rc}; log: {log_path}", flush=True)
    raise RuntimeError(f"{name} failed after {retries + 1} attempt(s), last exit code {last_rc}")


def _subject_done(phase_dir: Path, subject: int, preprocess: str = "bandpass_0_5_48") -> bool:
    manifest = phase_dir / f"subj{subject:02d}" / "run_manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("used_files")) and data.get("preprocess") == preprocess and bool(data.get("save_npz"))


def _write_manifest(paths: Paths, *, preprocess: str, subjects: list[int]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "preprocess": preprocess,
        "subjects": subjects,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
    }
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    (paths.log_dir / "pipeline_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_pipeline(
    *,
    paths: Paths,
    subjects: list[int],
    preprocess: str,
    retries: int,
    force_basis: bool,
    force_phase: bool,
    force_derived: bool,
    skip_package: bool = False,
) -> None:
    py = sys.executable
    _write_manifest(paths, preprocess=preprocess, subjects=subjects)

    if force_basis or not _exists(paths.basis):
        _run_with_retries(
            name="01_basis",
            retries=retries,
            log_dir=paths.log_dir,
            cmd=[
                py,
                "-m",
                "scripts.build_kaggle_basis_allpairs_batch",
                "--root",
                str(paths.root),
                "--out",
                str(paths.basis),
                "--mode",
                "gc",
                "--m",
                "4",
                "--lags-ms",
                "0",
                "50",
                "100",
                "150",
                "200",
                "--pca-r",
                "3",
                "--preprocess",
                preprocess,
                "--resume",
                "--checkpoint-every-files",
                "1",
            ],
        )
    else:
        print(f"[SKIP] basis exists: {paths.basis}", flush=True)

    for subject in subjects:
        name = f"02_phase_subj{subject:02d}"
        if not force_phase and _subject_done(paths.phase_dir, subject, preprocess):
            print(f"[SKIP] {name} already has a matching run_manifest.json", flush=True)
            continue
        _run_with_retries(
            name=name,
            retries=retries,
            log_dir=paths.log_dir,
            cmd=[
                py,
                "-m",
                "scripts.mfg_kaggle_phase_matrix_batch",
                "--root",
                str(paths.root),
                "--out",
                str(paths.phase_dir),
                "--mode",
                "gc",
                "--m",
                "4",
                "--lags-ms",
                "0",
                "50",
                "100",
                "150",
                "200",
                "--basis-allpairs",
                str(paths.basis),
                "--pca-r",
                "3",
                "--group-by",
                "subject",
                "--save-npz",
                "--subjects",
                str(subject),
                "--preprocess",
                preprocess,
            ],
        )

    if force_derived or not _exists(paths.meta_dir / "meta_edges.csv"):
        _run_with_retries(
            name="03_meta",
            retries=retries,
            log_dir=paths.log_dir,
            cmd=[
                py,
                "-m",
                "scripts.meta_analysis",
                "--dir",
                str(paths.phase_dir),
                "--topk",
                "10",
                "--min-subjects",
                "4",
                "--channels",
                "32",
                "--mode",
                "gc",
                "--use-fdr",
                "--alpha",
                "0.05",
                "--max-edges",
                "3",
                "--significant-only",
                "--dominant-flow-scope",
                "reported",
                "--out-dir",
                str(paths.meta_dir),
                "--quiet",
            ],
        )
    else:
        print(f"[SKIP] meta exports exist: {paths.meta_dir}", flush=True)

    if force_derived or not _exists(paths.sensitivity_dir / "edge_stability.csv"):
        _run_with_retries(
            name="04_sensitivity",
            retries=retries,
            log_dir=paths.log_dir,
            cmd=[
                py,
                "-m",
                "scripts.meta_sensitivity",
                "--dir",
                str(paths.phase_dir),
                "--out-dir",
                str(paths.sensitivity_dir),
                "--topk",
                "5",
                "10",
                "15",
                "--min-subjects",
                "3",
                "4",
                "5",
                "--p0-inflate",
                "5",
                "10",
                "15",
                "--mode",
                "gc",
                "--channels",
                "32",
                "--alpha",
                "0.05",
                "--use-fdr",
            ],
        )
    else:
        print(f"[SKIP] sensitivity exports exist: {paths.sensitivity_dir}", flush=True)

    if force_derived or not _exists(paths.vc_dir / "volume_conduction_summary.json"):
        _run_with_retries(
            name="05_volume_conduction",
            retries=retries,
            log_dir=paths.log_dir,
            cmd=[
                py,
                "-m",
                "scripts.volume_conduction_control",
                "--edge-table",
                str(paths.sensitivity_dir / "edge_stability.csv"),
                "--out-dir",
                str(paths.vc_dir),
                "--quiet",
            ],
        )
    else:
        print(f"[SKIP] volume-conduction exports exist: {paths.vc_dir}", flush=True)

    if skip_package:
        return
    _run_with_retries(
        name="06_article_package",
        retries=retries,
        log_dir=paths.log_dir,
        cmd=[
            py,
            "-m",
            "scripts.build_article_package",
            "--phase-dir",
            str(paths.phase_dir),
            "--pre-event-dir",
            str(paths.pre_event_dir),
            "--meta-dir",
            str(paths.meta_dir),
            "--sensitivity-dir",
            str(paths.sensitivity_dir),
            "--classification-dir",
            str(paths.classification_dir),
            "--classification-sweep-dir",
            str(paths.classification_sweep_dir),
            "--classification-controls-dir",
            str(paths.classification_controls_dir),
            "--classification-subject-dir",
            str(paths.classification_subject_dir),
            "--volume-conduction-dir",
            str(paths.vc_dir),
            "--comparison-baseline-label",
            "CAR-only",
            "--comparison-baseline-sensitivity-dir",
            "out/article_meta_sensitivity",
            "--comparison-baseline-volume-conduction-dir",
            "out/article_volume_conduction",
            "--comparison-current-label",
            preprocess,
            "--out",
            str(paths.package_dir),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the article-facing cleaned pipeline with retries and resume points."
    )
    parser.add_argument("--root", default="data/grasp-and-lift-eeg-detection/train")
    parser.add_argument("--preprocess", default="bandpass_0_5_48", choices=("car_only", "bandpass_0_5_48"))
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force-basis", action="store_true")
    parser.add_argument("--force-phase", action="store_true")
    parser.add_argument("--force-derived", action="store_true")
    parser.add_argument("--basis", default="out/basis/basis_allpairs_gc_m4_bandpass.npz")
    parser.add_argument("--phase-dir", default="out/phase_mats_pca_bandpass_by_subject")
    parser.add_argument("--meta-dir", default="out/article_meta_kaggle_bandpass")
    parser.add_argument("--sensitivity-dir", default="out/article_meta_sensitivity_bandpass")
    parser.add_argument("--vc-dir", default="out/article_volume_conduction_bandpass")
    parser.add_argument("--package-dir", default="out/article_package")
    parser.add_argument("--log-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else Path("out/resumable_runs") / f"bandpass_{_now_id()}"
    paths = Paths(
        root=Path(args.root),
        basis=Path(args.basis),
        phase_dir=Path(args.phase_dir),
        meta_dir=Path(args.meta_dir),
        sensitivity_dir=Path(args.sensitivity_dir),
        vc_dir=Path(args.vc_dir),
        package_dir=Path(args.package_dir),
        pre_event_dir=Path("out/experimental_kaggle_by_subject"),
        classification_dir=Path("out/classification_benchmark"),
        classification_sweep_dir=Path("out/classification_sweep"),
        classification_controls_dir=Path("out/classification_controls"),
        classification_subject_dir=Path("out/classification_subject_epochs3"),
        log_dir=log_dir,
    )
    run_pipeline(
        paths=paths,
        subjects=list(args.subjects),
        preprocess=str(args.preprocess),
        retries=max(0, int(args.retries)),
        force_basis=bool(args.force_basis),
        force_phase=bool(args.force_phase),
        force_derived=bool(args.force_derived),
        skip_package=bool(args.skip_package),
    )
    print("[DONE] resumable article pipeline completed", flush=True)


if __name__ == "__main__":
    main()
