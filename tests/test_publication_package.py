import csv
import json
import unittest
import zipfile
from pathlib import Path

from scripts.build_arxiv_submission import FIGURES, TABLES
from scripts.build_article_package import primary_sensitivity_rows
from scripts.reproduce_article import verify_inputs

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "out/article_package"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PublicationPackageTests(unittest.TestCase):
    def test_packaged_methodology_matches_source(self):
        self.assertEqual(
            (ROOT / "docs/METHODOLOGY.md").read_text(encoding="utf-8"),
            (PACKAGE / "documentation/METHODOLOGY.md").read_text(encoding="utf-8"),
        )

    def test_shipped_primary_table_matches_meta_analysis_and_sensitivity(self):
        expected = primary_sensitivity_rows(
            read_csv(PACKAGE / "raw_meta_exports/meta_edges.csv"),
            read_csv(PACKAGE / "sensitivity/edge_stability.csv"),
        )
        actual = read_csv(PACKAGE / "tables/primary_edge_sensitivity.csv")
        self.assertEqual(actual, expected)
        values = json.loads((PACKAGE / "reproduction/article_values.json").read_text())
        self.assertEqual(len(actual), values["primary_bandpass_count"])
        verify_inputs(PACKAGE)

    def test_submission_contains_current_source_and_assets(self):
        expected = {
            "eeg_mfg_article.tex": ROOT / "docs/eeg_mfg_article.tex",
            "references.bib": ROOT / "docs/references.bib",
            **{f"figures/{name}": PACKAGE / "figures" / name for name in FIGURES},
            **{f"tables/{name}": PACKAGE / "tables" / name for name in TABLES},
        }
        with zipfile.ZipFile(PACKAGE / "arxiv/arxiv-source.zip") as archive:
            self.assertEqual(set(archive.namelist()), set(expected) | {"eeg_mfg_article.bbl"})
            for name, path in expected.items():
                archived, current = archive.read(name), path.read_bytes()
                if path.suffix in {".tex", ".bib"}:
                    archived = archived.replace(b"\r\n", b"\n")
                    current = current.replace(b"\r\n", b"\n")
                self.assertEqual(archived, current, name)
        self.assertEqual((ROOT / "docs/eeg_mfg_article.pdf").read_bytes(),
                         (PACKAGE / "documentation/eeg_mfg_article.pdf").read_bytes())


if __name__ == "__main__":
    unittest.main()
