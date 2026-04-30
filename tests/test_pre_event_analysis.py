import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mfg_kaggle_phase_matrix_experimental import (  # noqa: E402
    _align_basis_for_lag,
    _compute_dynamic_coeffs_ema,
)


class PreEventAnalysisTests(unittest.TestCase):
    def test_align_basis_for_zero_lag_keeps_full_length(self) -> None:
        Fy = np.arange(24, dtype=float).reshape(6, 2, 2)
        Fz = Fy + 100.0

        aligned = _align_basis_for_lag(Fy, Fz, 0)
        self.assertIsNotNone(aligned)

        Fy_aligned, Fz_aligned = aligned
        np.testing.assert_array_equal(Fy_aligned, Fy)
        np.testing.assert_array_equal(Fz_aligned, Fz)

    def test_align_basis_for_positive_lag_matches_batch_convention(self) -> None:
        Fy = np.arange(32, dtype=float).reshape(8, 2, 2)
        Fz = Fy + 1000.0

        aligned = _align_basis_for_lag(Fy, Fz, 2)
        self.assertIsNotNone(aligned)

        Fy_aligned, Fz_aligned = aligned
        np.testing.assert_array_equal(Fy_aligned, Fy[:6])
        np.testing.assert_array_equal(Fz_aligned, Fz[2:])

    def test_dynamic_coeffs_ema_constant_signal_centers_to_zero(self) -> None:
        Fy = np.ones((10, 1, 1), dtype=float)
        Fz = np.ones((10, 1, 1), dtype=float)

        coeffs = _compute_dynamic_coeffs_ema(
            Fy,
            Fz,
            alpha=0.5,
            subtract_marginals=True,
        )

        np.testing.assert_allclose(coeffs, 0.0, atol=1e-12)

    def test_dynamic_coeffs_ema_without_centering_keeps_constant_energy(self) -> None:
        Fy = np.ones((10, 1, 1), dtype=float)
        Fz = np.ones((10, 1, 1), dtype=float)

        coeffs = _compute_dynamic_coeffs_ema(
            Fy,
            Fz,
            alpha=0.5,
            subtract_marginals=False,
        )

        np.testing.assert_allclose(coeffs, 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
