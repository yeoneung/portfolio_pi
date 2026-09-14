"""Regression checks for the portable fixed-path empirical calculations."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import reproduce_empirical as report
from empirical_statistics import _crra_cer


class EmpiricalReproductionTests(unittest.TestCase):
    def test_constant_total_return_crra(self):
        self.assertAlmostEqual(_crra_cer(np.full(24, 0.01), gamma=5), 1.01 ** 12 - 1)

    def test_rf_checksum_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory(prefix="rf-check-") as temp:
            path = Path(temp) / "wrong.zip"
            path.write_bytes(b"invalid source")
            with self.assertRaisesRegex(ValueError, "checksum"):
                report.load_rf(path)

    def test_supplied_strategy_dates_align(self):
        for universe in report.UNIVERSES:
            pair, _ = report.load_pair(ROOT / "code" / "empirical_src_bundle", universe)
            self.assertEqual(len(pair), 239)
            self.assertEqual(pair.index.min(), pd.Timestamp("2000-02-29"))
            self.assertEqual(pair.index.max(), pd.Timestamp("2019-12-31"))
            self.assertTrue(np.isfinite(pair.to_numpy()).all())

    def test_complete_empirical_output_reproduction(self):
        with tempfile.TemporaryDirectory(prefix="empirical-regression-") as temp:
            output = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                report.main(["--out-dir", str(output)])
            names = ("primary_metrics_total", "paired_block_bootstrap_total", "hac_inference_total",
                     "cost_sensitivity_total", "subperiod_metrics_total", "break_even_costs")
            for name in names:
                with self.subTest(table=name):
                    actual = pd.read_csv(output / f"{name}.csv")
                    expected = pd.read_csv(ROOT / "results" / "empirical_tables" / f"{name}.csv")
                    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
