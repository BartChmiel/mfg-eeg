from __future__ import annotations

import unittest

import numpy as np

from src_mfg.preprocessing import (
    PREPROCESSING_CHOICES,
    apply_preprocessing,
    bandpass_0_5_48,
    regress_ocular_proxies,
    resolve_channel_names,
)


class PreprocessingTests(unittest.TestCase):
    def test_bandpass_preserves_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 4))
        out = bandpass_0_5_48(X, fs=500)
        self.assertEqual(out.shape, X.shape)

    def test_ocular_regression_reduces_fp_correlation(self) -> None:
        rng = np.random.default_rng(1)
        eye = rng.normal(size=(500, 1))
        channels = ["Fp1", "Fp2", "C3", "C4"]
        X = np.concatenate([eye, eye * 0.9 + rng.normal(scale=0.05, size=(500, 1)), rng.normal(size=(500, 2))], axis=1)
        cleaned = regress_ocular_proxies(X, all_channels=channels, proxy_source=X)
        before = np.corrcoef(X[:, 2], X[:, 0])[0, 1]
        after = np.corrcoef(cleaned[:, 2], X[:, 0])[0, 1]
        self.assertLess(abs(after), abs(before))

    def test_apply_preprocessing_car_is_identity(self) -> None:
        X = np.ones((20, 3))
        out = apply_preprocessing(X, preprocess="car_only", fs=500, all_channels=["A", "B", "C"])
        np.testing.assert_array_equal(out, X)

    def test_resolve_motor_premotor_channels(self) -> None:
        all_channels = [
            "Fp1",
            "Fp2",
            "F3",
            "Fz",
            "F4",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "C3",
            "Cz",
            "C4",
            "O1",
        ]
        selected = resolve_channel_names("motor_premotor", all_channels)
        self.assertIn("C3", selected)
        self.assertIn("F3", selected)
        self.assertNotIn("Fp1", selected)

    def test_preprocessing_choices_are_stable(self) -> None:
        self.assertIn("bandpass_0_5_48", PREPROCESSING_CHOICES)
        self.assertIn("ocular_proxy_regression", PREPROCESSING_CHOICES)


if __name__ == "__main__":
    unittest.main()
