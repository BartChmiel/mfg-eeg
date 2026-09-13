import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.volume_conduction_control import (  # noqa: E402
    annotate_edges,
    run_control,
    spatial_enrichment,
)
from src_mfg.montage import (  # noqa: E402
    GAL_CHANNELS,
    geodesic_distance_cm,
    has_channel,
)


class MontageTests(unittest.TestCase):
    def test_all_channels_present(self) -> None:
        self.assertEqual(len(GAL_CHANNELS), 32)
        for name in GAL_CHANNELS:
            self.assertTrue(has_channel(name))

    def test_adjacent_closer_than_contralateral(self) -> None:
        # Adjacent posterior electrodes must be much closer than a far frontal pair.
        adjacent = geodesic_distance_cm("PO9", "O1")
        contralateral = geodesic_distance_cm("PO9", "Fp2")
        self.assertLess(adjacent, contralateral)
        self.assertLess(adjacent, 5.0)

    def test_distance_is_symmetric(self) -> None:
        self.assertAlmostEqual(
            geodesic_distance_cm("F3", "P4"),
            geodesic_distance_cm("P4", "F3"),
            places=9,
        )


class VolumeConductionControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "out" / "test_volume_conduction"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.edge_table = self.root / "edge_stability.csv"
        self._write_edges()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_edges(self) -> None:
        # Two short-range zero-lag adjacent edges (volume-conduction-like) plus
        # one long-range, non-zero-lag, asymmetric edge (robust).
        rows = [
            {
                "phase": "P", "pc": "pc1", "lag_ms": 0,
                "src": "PO9", "dst": "O1",
                "region_src": "Occipital", "region_dst": "Occipital",
                "stability_fraction": 1.0,
            },
            {
                "phase": "P", "pc": "pc1", "lag_ms": 0,
                "src": "O1", "dst": "PO9",
                "region_src": "Occipital", "region_dst": "Occipital",
                "stability_fraction": 1.0,
            },
            {
                "phase": "P", "pc": "pc2", "lag_ms": 100,
                "src": "O2", "dst": "F3",
                "region_src": "Occipital", "region_dst": "Frontocentral",
                "stability_fraction": 1.0,
            },
        ]
        with open(self.edge_table, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_annotation_flags(self) -> None:
        with open(self.edge_table, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        annotated = annotate_edges(rows, short_range_cm=4.5)
        self.assertEqual(len(annotated), 3)

        by_edge = {(e.src, e.dst): e for e in annotated}
        # Adjacent zero-lag bidirectional edge -> not robust.
        po9_o1 = by_edge[("PO9", "O1")]
        self.assertTrue(po9_o1.short_range)
        self.assertTrue(po9_o1.zero_lag)
        self.assertTrue(po9_o1.reverse_present)
        self.assertFalse(po9_o1.asymmetric)
        self.assertFalse(po9_o1.vc_robust)

        # Long-range, lagged, one-directional edge -> robust.
        o2_f3 = by_edge[("O2", "F3")]
        self.assertFalse(o2_f3.short_range)
        self.assertFalse(o2_f3.zero_lag)
        self.assertFalse(o2_f3.reverse_present)
        self.assertTrue(o2_f3.asymmetric)
        self.assertTrue(o2_f3.vc_robust)

    def test_run_control_writes_outputs(self) -> None:
        out_dir = self.root / "vc"
        payload = run_control(
            edge_table=str(self.edge_table),
            out_dir=str(out_dir),
            short_range_cm=4.5,
            min_stability=1.0,
            n_permutations=500,
            seed=123,
        )
        self.assertEqual(payload["total_edges"], 3)
        self.assertEqual(payload["vc_robust_edges"], 1)
        self.assertTrue((out_dir / "vc_robust_edge_stability.csv").exists())
        self.assertTrue((out_dir / "volume_conduction_edges.csv").exists())
        self.assertTrue((out_dir / "volume_conduction_summary.json").exists())
        self.assertTrue((out_dir / "volume_conduction_report.md").exists())

        with open(out_dir / "volume_conduction_summary.json", "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertIn("spatial_enrichment", data)
        self.assertEqual(data["spatial_enrichment"]["n_edges"], 3)

    def test_enrichment_handles_empty(self) -> None:
        result = spatial_enrichment([], n_permutations=10, seed=1)
        self.assertEqual(result["n_edges"], 0)
        self.assertIsNone(result["permutation_p_shorter"])

    def test_control_exports_empty_screen_without_failure(self) -> None:
        result = run_control(edge_table=str(self.edge_table), out_dir=str(self.root / "empty"),
            short_range_cm=100.0, min_stability=1.0, n_permutations=10, seed=1)
        self.assertEqual(result["vc_robust_edges"], 0)
        with open(result["vc_robust_edge_table"], newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("src", reader.fieldnames)
            self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
