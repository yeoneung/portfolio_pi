"""Exact-arithmetic and end-to-end regression checks for the PDE certifier."""
import importlib.util
import json
import contextlib
import io
import tempfile
from fractions import Fraction as F
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "direct_residual_certificate.py"
spec = importlib.util.spec_from_file_location("direct_certificate", SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def evaluate(poly, s, x):
    return sum((v * s ** i * x ** j for (i, j), v in poly.items()), F(0))


class DirectCertificateTests(unittest.TestCase):
    def test_exact_trace(self):
        c = np.asarray([[F(3, 11), F(-5, 17)], [F(1, 19), F(7, 23)]], dtype=object)
        u = d.candidate_polynomial(c)
        for k in range(11):
            v = F(k, 10)
            self.assertEqual(evaluate(u, 0, v), 1)
            self.assertEqual(evaluate(u, v, 0), 1)
            self.assertEqual(evaluate(u, v, 1), 1 - F(3, 100) * v)

    def test_bernstein_and_subdivision_contain_exact_values(self):
        rng = np.random.default_rng(321)
        for _ in range(8):
            p = {(i, j): F(int(rng.integers(-50, 51)), 17)
                 for i in range(5) for j in range(4)}
            root = d.bernstein_enclosure(p)
            nodes = [(root, (F(0), F(1)), (F(0), F(1)))]
            for depth in range(4):
                nxt = []
                axis = depth % 2
                for enclosure, ss, xx in nodes:
                    for s in (ss[0], sum(ss) / 2, ss[1]):
                        for x in (xx[0], sum(xx) / 2, xx[1]):
                            value = evaluate(p, s, x)
                            self.assertLessEqual(F(float(enclosure[0].min())), value)
                            self.assertGreaterEqual(F(float(enclosure[1].max())), value)
                    for k, child in enumerate(d.split_bernstein(enclosure, axis)):
                        bounds = [ss, xx]
                        old = bounds[axis]
                        bounds[axis] = (old[0], sum(old) / 2) if k == 0 else (sum(old) / 2, old[1])
                        nxt.append((child, *bounds))
                nodes = nxt

    def test_residual_branches_match_exact_pde(self):
        c = np.asarray([[F(-1, 20), F(1, 40)]], dtype=object)
        u = d.candidate_polynomial(c)
        ps = d.residual_polynomials(u)
        covered = set()
        for i in range(7):
            for j in range(31):
                s, x = F(i, 6), F(j, 30)
                value = evaluate(u, s, x)
                us = evaluate(d.derivative(u, 0), s, x)
                ux = evaluate(d.derivative(u, 1), s, x)
                uxx = evaluate(d.derivative(d.derivative(u, 1), 1), s, x)
                a = min(F(3, 10), max(F(0), (x - F(3, 10) + ux / (20 * value)) / 2))
                expected = -4 * us + uxx / 32 - (2 * x - 1 + a / 10) * ux / 2
                expected += (-a * (x - F(3, 10)) + a * a) * value
                branch = "lower" if a == 0 else "upper" if a == F(3, 10) else "interior"
                covered.add(branch)
                actual = (evaluate(ps["middle_numerator"], s, x) / (4 * value)
                          if branch == "interior" else evaluate(ps[branch], s, x))
                self.assertEqual(expected, actual)
        self.assertEqual(covered, {"lower", "interior", "upper"})

    def test_switching_cells_include_all_possible_branches(self):
        u = {(0, 0): F(1)}
        cell = {k: d.bernstein_enclosure(v) for k, v in d.residual_polynomials(u).items()}
        lower_u, eta, branches = d.cell_bounds(cell)
        self.assertGreater(lower_u, 0)
        self.assertEqual(set(branches), {"lower", "interior", "upper"})
        for j in range(101):
            x = F(j, 100)
            a = min(F(3, 10), max(F(0), (x - F(3, 10)) / 2))
            actual = abs(-a * (x - F(3, 10)) + a * a)
            self.assertLessEqual(actual, F(eta))

    def test_nonpositive_and_inaccurate_candidates_fail_closed(self):
        bad = d.certify(d.residual_polynomials({(0, 0): F(-1)}), 0.001, max_cells=3)
        self.assertFalse(bad["certified"])
        json.dumps(bad, allow_nan=False)
        inaccurate = d.certify(d.residual_polynomials({(0, 0): F(1)}), 1e-10, max_cells=3)
        self.assertFalse(inaccurate["certified"])

    def test_explicit_barrier_and_upward_output(self):
        from decimal import Decimal
        from mpmath import mp
        mp.dps = 70
        eta = 0.00019932234214238575
        bound = d.explicit_bounds(eta)
        phi = mp.expm1(mp.mpf(9) / 200) / (mp.mpf(9) / 50)
        self.assertGreaterEqual(mp.mpf(bound["barrier_factor_upper"]), phi)
        self.assertGreaterEqual(mp.mpf(bound["value_factor_sup_error_upper"]), phi * mp.mpf(eta))
        for value in [eta, 0.00010193756489785643, 0.99999999, 2.71828]:
            rendered = d.upper_sci(value)
            mantissa, exponent = rendered.split(r"\times10^{")
            decimal = Decimal(mantissa).scaleb(int(exponent[:-1]))
            self.assertGreaterEqual(decimal, Decimal.from_float(value))

    def test_failed_rerun_overwrites_stale_success(self):
        candidate_text = (d.OUT / "candidate.json").read_text()
        with tempfile.TemporaryDirectory(prefix="direct-certificate-test-") as temp:
            out = Path(temp)
            table = out / "table.tex"
            (out / "candidate.json").write_text(candidate_text)
            table.write_text("old successful certificate")
            with patch.object(d, "OUT", out), patch.object(d, "TABLE", table), contextlib.redirect_stdout(io.StringIO()):
                code = d.main(["--verify-only", "--target-residual", "1e-12", "--max-cells", "1"])
            self.assertEqual(code, 2)
            self.assertIn("No certificate returned", table.read_text())
            result = json.loads((out / "certificate_results.json").read_text())
            self.assertFalse(result["certificate"]["certified"])
            self.assertNotIn("bounds", result)

    def test_supplied_candidate_verifies_without_training_or_reference(self):
        candidate = json.loads((d.OUT / "candidate.json").read_text())
        denominator = int(candidate["coefficient_denominator"])
        exact = np.asarray([[F(int(k), denominator) for k in row]
                            for row in candidate["coefficient_numerators"]], dtype=object)
        result = d.certify(d.residual_polynomials(d.candidate_polynomial(exact)), 0.0002)
        self.assertTrue(result["certified"])
        self.assertEqual(result["covered_area_exact"], "1")
        self.assertGreater(result["switching_cells"], 0)
        self.assertGreater(result["u_lower"], 0.96)
        self.assertLessEqual(result["bellman_residual_sup_upper"], 0.0002)


if __name__ == "__main__":
    unittest.main()
