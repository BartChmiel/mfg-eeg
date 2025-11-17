"""
Batch runner for MFG/HCR analysis on multiple EEG files.

This script searches for EEG CSV files (e.g. Kaggle grasp-and-lift
`*_data.csv`) under a given root directory and, for each file, calls
`scripts.mfg_pair` as a separate CLI process.

Example:
    python -m scripts.mfg_batch \
        --root data/grasp-and-lift-eeg-detection/train \
        --chan-a Fp1 \
        --chan-b Fp2 \
        --mode gc \
        --out-root out/csv_run
"""

import os
import sys
import argparse
import subprocess
from typing import List


def find_eeg_csv_files(root: str) -> List[str]:
    """
    Recursively find all EEG CSV files under the given root.

    Current convention: files ending with '_data.csv'.
    """
    matches: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith("_data.csv"):
                full = os.path.join(dirpath, fname)
                matches.append(os.path.normpath(full))
    matches.sort()
    return matches


def main():
    ap = argparse.ArgumentParser(
        prog="mfg_batch",
        description="Batch runner for MFG pair analysis over many EEG CSV files.",
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory to search for EEG CSV files (e.g. train/).",
    )
    ap.add_argument(
        "--chan-a",
        required=True,
        help="Name of electrode/channel A (reason).",
    )
    ap.add_argument(
        "--chan-b",
        required=True,
        help="Name of electrode/channel B (result).",
    )
    ap.add_argument(
        "--mode",
        choices=["corr", "gc"],
        default="corr",
        help="Analysis mode: 'corr' (symmetric correlation) or 'gc' (Granger-like).",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Root output directory for all per-file results.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of files processed (for quick tests).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only print what would be executed, do not run mfg_pair.",
    )

    args = ap.parse_args()

    root = os.path.normpath(args.root)
    out_root = os.path.normpath(args.out_root)

    os.makedirs(out_root, exist_ok=True)

    files = find_eeg_csv_files(root)
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        print(f"[mfg_batch] No *_data.csv files found under: {root}")
        return

    print(f"[mfg_batch] Found {len(files)} EEG CSV files under: {root}")
    print(f"[mfg_batch] Output root: {out_root}")
    print(f"[mfg_batch] Mode: {args.mode}, chan-a={args.chan_a}, chan-b={args.chan_b}")
    if args.dry_run:
        print("[mfg_batch] DRY RUN enabled - commands will not be executed.")

    for idx, csv_path in enumerate(files, start=1):
        rel = os.path.relpath(csv_path, root)  # e.g. 'subj10_series1_data.csv'
        base_no_ext = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(out_root, base_no_ext)

        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "scripts.mfg_pair",
            "--file",
            csv_path,
            "--chan-a",
            args.chan_a,
            "--chan-b",
            args.chan_b,
            "--mode",
            args.mode,
            "--out",
            out_dir,
        ]

        print(f"[{idx}/{len(files)}] {rel}")
        print("  -> out:", out_dir)
        print("  -> cmd:", " ".join(cmd))

        if args.dry_run:
            continue

        try:
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                print(
                    f"  !! mfg_pair failed with return code {res.returncode} "
                    f"for file: {csv_path}"
                )
        except Exception as exc:
            print(f"  !! Exception while running mfg_pair for {csv_path}: {exc}")


if __name__ == "__main__":
    main()
