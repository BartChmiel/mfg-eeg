import tempfile
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.export_phase_rankings import export_rankings
from scripts.meta_analysis import collect_edge_presence
from scripts.reproduce_article import classifier_commands, verify_inputs


class ReproductionTests(unittest.TestCase):
    def test_modified_archived_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "reproduction").mkdir()
            data = root / "reproduction/scores.csv"
            data.write_bytes(b"original")
            (root / "reproduction/inputs_manifest.json").write_text(json.dumps({"sha256": {
                "reproduction/scores.csv": hashlib.sha256(b"original").hexdigest()}}))
            verify_inputs(root)
            data.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_inputs(root)

    def test_snapshot_rankings_exclude_diagonal_and_use_absolute_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subject = root / "subj01"
            subject.mkdir()
            np.savez(subject / "phasegrid_data_A__B.npz",
                scores=np.array([[[[100.0, -3.0], [2.0, 100.0]]]]),
                ch_names=np.array(["F3", "F4"]), phase="A__B", mode="gc", m=4,
                lags_ms=np.array([50]), group="subj01")
            table = root / "rankings.csv.gz"
            manifest = export_rankings(root, table, topk=2)
            self.assertEqual(manifest["rows"], 2)
            presence, _, _, subjects = collect_edge_presence(dir_root=str(table), topk=1, mode_filter="gc")
            self.assertEqual(subjects, {"subj01"})
            self.assertEqual(list(presence[("A__B", "pc1", 50)]), [("F3", "F4")])
            with self.assertRaises(ValueError):
                collect_edge_presence(dir_root=str(table), topk=3, mode_filter="gc")

    def test_raw_classifier_runs_both_timing_conditions_with_three_epochs(self):
        commands = classifier_commands(Path("data"), Path("work"))
        self.assertEqual(len(commands), 2)
        for (_, args), shift in zip(commands, ["0", "500"]):
            self.assertEqual(args[args.index("--label-shift-ms") + 1], shift)
            self.assertEqual(args[args.index("--epochs") + 1], "3")
            self.assertIn("--fusion-weight-values", args)
            self.assertIn("classifier_candidates.csv", args[args.index("--edge-table") + 1])


if __name__ == "__main__":
    unittest.main()
