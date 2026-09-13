from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ["subject", "phase", "mode", "m", "pc", "lag_ms", "rank", "src", "dst", "score"]


def export_rankings(root: Path, output: Path, topk: int = 15) -> dict:
    files = sorted(root.glob("subj*/phasegrid_data_*.npz"))
    if not files:
        raise ValueError(f"No subject score payloads under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    row_count = 0
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for path in files:
            with np.load(path, allow_pickle=True) as payload:
                scores = np.asarray(payload["scores"], dtype=float)
                names = payload["ch_names"].tolist()
                phase = str(payload["phase"].item())
                mode = str(payload["mode"].item())
                degree = int(payload["m"].item())
                lags = payload["lags_ms"].tolist()
                subject = str(payload["group"].item())
            if not subject.startswith("subj") or subject != path.parent.name:
                raise ValueError(f"Unexpected subject in {path}")
            if not np.isfinite(scores).all() or topk > len(names) * (len(names) - 1):
                raise ValueError(f"Invalid scores or ranking length in {path}")
            for pc in range(scores.shape[0]):
                for li, lag in enumerate(lags):
                    matrix = scores[pc, li].copy()
                    np.fill_diagonal(matrix, -np.inf)
                    order = np.argsort(np.abs(matrix).ravel())[::-1]
                    order = [idx for idx in order if np.isfinite(matrix.ravel()[idx])][:topk]
                    for rank, idx in enumerate(order, 1):
                        i, j = divmod(idx, len(names))
                        writer.writerow(dict(zip(FIELDS, [subject, phase, mode, degree,
                            f"pc{pc + 1}", lag, rank, names[i], names[j], matrix[i, j]])))
                        row_count += 1
            sources.append({"file": path.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"topk": topk, "rows": row_count, "score_payloads": sources,
                "ranking": "descending absolute pair-specific PCA score; no diagonal",
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reproducible subject edge rankings from saved PCA scores.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=15)
    args = parser.parse_args()
    result = export_rankings(args.root, args.output, args.topk)
    print(f"Exported {result['rows']} rankings from {len(result['score_payloads'])} score payloads")


if __name__ == "__main__":
    main()
