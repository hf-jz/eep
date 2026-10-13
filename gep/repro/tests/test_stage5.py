import sys
import unittest
import mpmath as mp
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import certified_multibranch_gep as g


class CertifiedMultiBranchGEPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = g.main(Path("/tmp") / "stage5_certificate_test.json")

    def test_exact_original_system_has_two_stable_branches_and_one_saddle(self):
        roots = self.result["multistability"]["roots"]
        stable = [r for r in roots if r["stable_by_2x2_hurwitz"]]
        unstable = [r for r in roots if not r["stable_by_2x2_hurwitz"]]
        self.assertEqual(len(stable), 2)
        self.assertEqual(len(unstable), 1)
        self.assertEqual(stable[0]["trace_exact"], "-3/1")
        self.assertEqual(stable[0]["determinant_exact"], "51/25")
        self.assertNotEqual(stable[0]["jacobian_nonsymmetry_gap_exact"], "0/1")

    def test_parameter_tubes_prove_unique_branch_and_stability(self):
        for branch in (-1, 1):
            row = self.result["branch_tubes"][str(branch)]
            self.assertTrue(row["tube_certificate"]["passed"])
            self.assertTrue(row["tube_certificate"]["strict_krawczyk_inclusion"])
            self.assertLess(row["tube_certificate"]["preconditioned_contraction_upper"], 1.0)
            self.assertTrue(row["stability_certificate"]["passed"])

    def test_controller_finds_common_beta_with_certified_gradient_error(self):
        self.assertEqual(self.result["status"], "CERTIFIED_MULTIBRANCH_GEP_PASSED")
        self.assertEqual(self.result["beta_selected_exact"], "1/200")
        self.assertLessEqual(self.result["worst_branch_error_upper"], float(g.TOLERANCE))
        self.assertTrue(self.result["all_branch_gradient_certificates_pass"])
        self.assertEqual(len(self.result["selected_branch_results"]), 2)

    def test_endpoint_boxes_certify_finite_difference_and_enclose_candidate(self):
        for branch in self.result["selected_branch_results"]:
            self.assertTrue(branch["endpoint_boxes_contained_in_branch_tube"])
            self.assertLessEqual(branch["certified_gradient_error_upper"], float(g.TOLERANCE))
            lo, hi = branch["finite_difference_interval_float"]
            self.assertLessEqual(lo, branch["empirical_fd_value"] + 1e-10)
            self.assertGreaterEqual(hi, branch["empirical_fd_value"] - 1e-10)
            for endpoint_key in ("plus_endpoint", "minus_endpoint"):
                cert = branch[endpoint_key]["krawczyk"]
                self.assertTrue(cert["passed"])
                self.assertTrue(cert["strict_krawczyk_inclusion"])

    def test_failure_path_rejects_unvalidated_multibranch_box(self):
        self.assertTrue(self.result["failure_path"]["passed_as_rejection"])
        self.assertFalse(self.result["failure_path"]["certificate"]["passed"])
        self.assertEqual(len(self.result["failure_path"]["known_distinct_roots_in_box"]), 3)
        self.assertFalse(self.result["failure_path"]["certificate"]["strict_krawczyk_inclusion"])
        self.assertGreaterEqual(self.result["failure_path"]["certificate"]["preconditioned_contraction_upper"], 1.0)

    def test_independent_high_precision_roots_lie_inside_certified_boxes(self):
        mp.mp.dps = 70
        for branch_row in self.result["selected_branch_results"]:
            branch = int(branch_row["branch"])
            for key, sign in (("plus_endpoint", 1), ("minus_endpoint", -1)):
                ep = branch_row[key]
                beta = mp.mpf(sign) * mp.mpf(self.result["beta_selected_exact"].split("/")[0]) / mp.mpf(self.result["beta_selected_exact"].split("/")[1])
                s = mp.mpf(branch)
                def f1(x, y): return x-x**3-mp.mpf(1)/5*y-beta*(x-mp.mpf(3)/10)
                def f2(x, y): return -y+mp.mpf(1)/10*(x-x**3)+mp.mpf(2)/5*(x-s)-beta*(y+mp.mpf(1)/5)
                guess = ep["candidate_float"]
                xh, yh = mp.findroot((f1, f2), (mp.mpf(str(guess[0])), mp.mpf(str(guess[1]))), tol=mp.mpf('1e-60'))
                for coord, value in enumerate((xh, yh)):
                    lo_s, hi_s = ep["box"][coord]
                    nlo, dlo = map(int, lo_s.split('/'))
                    nhi, dhi = map(int, hi_s.split('/'))
                    lo = mp.mpf(nlo)/dlo
                    hi = mp.mpf(nhi)/dhi
                    self.assertLessEqual(lo, value)
                    self.assertGreaterEqual(hi, value)

    def test_out_of_domain_requests_fail_closed(self):
        with self.assertRaises(ValueError):
            g.branch_tube(1, F(1, 10))
        with self.assertRaises(ValueError):
            g.certified_endpoint(1, F(1, 10), +1)
        with self.assertRaises(ValueError):
            g.certify_beta(F(1, 10), self.result["branch_tubes"])

    def test_exact_implicit_gradient_values(self):
        self.assertEqual(g.exact_target_negative_gradient(1), F(-11, 34))
        self.assertEqual(g.exact_target_negative_gradient(-1), F(67, 102))


if __name__ == "__main__":
    unittest.main(verbosity=2)
