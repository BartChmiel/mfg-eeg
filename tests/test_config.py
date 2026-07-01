from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from src_mfg.config import Config, load_config_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigYamlTests(unittest.TestCase):
    def test_load_config_yaml_reads_defaults(self) -> None:
        cfg = load_config_yaml()
        self.assertEqual(cfg.fs, 500)
        self.assertEqual(cfg.m, 4)
        self.assertEqual(cfg.pca_max_r, 4)
        self.assertAlmostEqual(cfg.ema_half_life_s, 0.1)

    def test_load_config_yaml_override_wins(self) -> None:
        cfg = load_config_yaml(m=7)
        self.assertEqual(cfg.m, 7)

    def test_custom_yaml_path(self) -> None:
        root = REPO_ROOT / "out" / "test_config_yaml"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            path = root / "custom.yaml"
            path.write_text("fs: 256\nm: 2\n", encoding="utf-8")
            cfg = load_config_yaml(path)
            self.assertEqual(cfg.fs, 256)
            self.assertEqual(cfg.m, 2)
            self.assertIsInstance(cfg, Config)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
