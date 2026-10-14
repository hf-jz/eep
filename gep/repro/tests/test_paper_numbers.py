"""Assert that every number quoted in paper/cep.tex matches results/.

This is the check that fails when the manuscript and the artefacts drift apart.
Values are the ones printed in the paper; tolerances are only as loose as the
rounding used there (for example 1.55 for 1.55465...).

Run with:  python3 -m unittest discover -s tests -v
"""
import json
import unittest
from fractions import Fraction as F
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return json.loads((RES / name).read_text())


class PaperNumbers(unittest.TestCase):
    def setUp(self):
        self.scan = load("beta_scan.json")
        self.cert = load("stage5_certificate.json")
        self.run4 = load("certified_run.json")
        self.dyadic_res = load("dyadic_residual_certificate.json")
        self.dyadic_fp = load("dyadic_finite_precision_certificate.json")
        self.baselines = load("baselines.json")
        self.sweep = load("nonnormal_sweep_summary.json")
        self.counter = load("nonnormal_counterexample.json")
        self.integr = load("integrator_comparison.json")
        self.stage1 = load("validation_summary.json")
        self.tradeoff = load("beta_noise_tradeoff_summary.json")

    # Section 6.1, Table 1 and Table 2
    def test_table1_nudging_scan(self):
        rows = {r["beta"]: r for r in self.scan["rows"]}
        for beta, bound in [(0.02, 1.201949e-4), (0.015, 6.759348e-5), (0.0125, 4.693783e-5),
                            (0.01, 3.004223e-5), (0.0075, 1.690521e-5), (0.005, 7.527105e-6),
                            (0.004, 4.829474e-6), (0.0025, 1.916717e-6), (0.00125, 5.491755e-7),
                            (0.001, 4.002721e-7)]:
            self.assertAlmostEqual(rows[beta]["worst_branch_error_upper"], bound, delta=bound * 5e-7)
        self.assertAlmostEqual(self.scan["largest_accepted_beta"], 0.005, places=12)
        # the paper's quadratic ratios for successive halvings
        r = [rows[0.02]["worst_branch_error_upper"] / rows[0.01]["worst_branch_error_upper"],
             rows[0.01]["worst_branch_error_upper"] / rows[0.005]["worst_branch_error_upper"],
             rows[0.005]["worst_branch_error_upper"] / rows[0.0025]["worst_branch_error_upper"]]
        for ratio, claimed in zip(r, [4.0009, 3.9912, 3.9271]):
            self.assertAlmostEqual(ratio, claimed, delta=2e-3)
        # branch-level error bounds quoted per row
        # json object keys are strings
        self.assertAlmostEqual(rows[0.005]["branch_error_upper"]["-1"], 7.527105e-6, delta=1e-11)
        self.assertAlmostEqual(rows[0.005]["branch_error_upper"]["1"], 7.882584e-7, delta=1e-12)

    def test_table2_branch_certificates(self):
        rows = {r["branch"]: r for r in self.cert["selected_branch_results"]}
        self.assertEqual(rows[-1]["true_negative_gradient_exact"], "67/102")
        self.assertEqual(rows[1]["true_negative_gradient_exact"], "-11/34")
        self.assertAlmostEqual(rows[-1]["true_negative_gradient_float"], 0.6568627451, places=10)
        self.assertAlmostEqual(rows[1]["true_negative_gradient_float"], -0.3235294118, places=10)
        self.assertEqual(round(rows[-1]["finite_difference_interval_float"][0], 10), 0.6568702322)
        self.assertEqual(round(rows[-1]["finite_difference_interval_float"][1], 10), 0.6568702722)
        self.assertEqual(round(rows[1]["finite_difference_interval_float"][0], 10), -0.3235286635)
        self.assertEqual(round(rows[1]["finite_difference_interval_float"][1], 10), -0.3235286235)
        self.assertAlmostEqual(rows[-1]["certified_gradient_error_upper"], 7.527105e-6, delta=1e-11)
        self.assertAlmostEqual(rows[1]["certified_gradient_error_upper"], 7.882584e-7, delta=1e-12)
        self.assertAlmostEqual(rows[-1]["empirical_abs_error"], 7.507105e-6, delta=1e-11)
        self.assertAlmostEqual(rows[1]["empirical_abs_error"], 7.682584e-7, delta=1e-12)
        # the exact gradient lies outside the certified interval, as the paper states
        for branch in (-1, 1):
            lo, hi = rows[branch]["finite_difference_interval_float"]
            t = rows[branch]["true_negative_gradient_float"]
            self.assertTrue(lo > t or hi < t)
        # enclosure half-width equals the radius form of the budget: r = 1e-10, 2r/(2 beta)
        half = (rows[1]["finite_difference_interval_float"][1] - rows[1]["finite_difference_interval_float"][0]) / 2
        self.assertAlmostEqual(half, 2e-10 / (2 * 0.005), places=14)
        for branch in (-1, 1):
            self.assertTrue(rows[branch]["endpoint_boxes_contained_in_branch_tube"])

    def test_tube_certificate_and_rejection(self):
        for b in ("-1", "1"):
            tube = self.cert["branch_tubes"][b]
            self.assertEqual(tube["tube_certificate"]["preconditioned_contraction_upper_exact"], "2249/13600")
            self.assertEqual(tube["tube_certificate"]["preconditioned_contraction_upper"], float(F(2249, 13600)))
            self.assertEqual(round(float(F(2249, 13600)), 6), 0.165368)  # as printed in the paper
            tr = [F(x) for x in tube["stability_certificate"]["trace_interval"]]
            dt = [F(x) for x in tube["stability_certificate"]["determinant_interval"]]
            self.assertEqual([tr[0], tr[1]], [F(-1339, 400), F(-1067, 400)])
            self.assertEqual([dt[0], dt[1]], [F(4219, 2500), F(24199, 10000)])
            self.assertLess(float(tr[1]), 0.0)
            self.assertGreater(float(dt[0]), 0.0)
        self.assertEqual(self.cert["failure_path"]["certificate"]["preconditioned_contraction_upper_exact"],
                         "183/50")
        self.assertFalse(self.cert["failure_path"]["certificate"]["passed"])

    # Section 6.2, Table 3
    def test_certified_budget(self):
        self.assertEqual(self.run4["beta_selected"], "0.005")
        self.assertEqual(self.run4["T_selected"], 6.0)
        self.assertAlmostEqual(self.run4["truncation_gradient_error_upper"], 3.656e-4, delta=1e-8)
        self.assertAlmostEqual(self.run4["equilibrium_solve_gradient_error_upper"], 7.351e-5, delta=1e-8)
        self.assertAlmostEqual(self.run4["total_gradient_error_upper"], 4.391e-4, delta=1e-7)
        self.assertAlmostEqual(self.run4["empirical_gradient_error"], 1.076e-5, delta=1e-7)
        # the paper quotes it to four significant digits
        self.assertEqual(float(f"{self.run4['empirical_gradient_error']:.4g}"), 1.076e-5)
        self.assertLess(self.run4["empirical_gradient_error"], self.run4["total_gradient_error_upper"])
        # E_z in the paper is the radius form of the enclosure term
        bp = float(F(self.run4["analytic_constants"]["B_frobenius_upper"]))
        r_plus = self.run4["plus_equilibrium_distance_upper"]
        r_minus = self.run4["minus_equilibrium_distance_upper"]
        self.assertAlmostEqual(bp * (r_plus + r_minus) / (2 * 0.005),
                               self.run4["equilibrium_solve_gradient_error_upper"], delta=1e-9)
        self.assertAlmostEqual(self.run4["total_gradient_error_upper"],
                               self.run4["truncation_gradient_error_upper"]
                               + self.run4["equilibrium_solve_gradient_error_upper"], delta=1e-9)
        # the shorter horizon in the table was indeed rejected
        hist = [h for h in self.run4["attempt_history"] if h.get("T") == 4.0]
        self.assertTrue(hist and hist[0]["total_certificate"] > self.run4["tolerance"])
        d = self.run4["selected_beta_derivative_bounds"]
        self.assertAlmostEqual(d["x1_upper"], 0.7714, delta=1e-4)
        self.assertAlmostEqual(d["x2_upper"], 3.6779, delta=1e-3)
        self.assertAlmostEqual(d["x3_upper"], 27.0817, delta=1e-3)

    def test_dyadic_certificates(self):
        self.assertEqual(self.dyadic_res["inverse_transpose_inf_norm"], "55/51")
        for row, exact in zip(self.dyadic_res["rows"],
                              ["11/10200000000", "11/1020000000", "11/102000000", "11/10200000"]):
            self.assertEqual(row["certified_gradient_error_bound_exact"], exact)

    def test_dyadic_finite_precision_and_baselines(self):
        fp = self.dyadic_fp
        self.assertTrue(fp["all_certificates_pass"])
        self.assertEqual(fp["inverse_transpose_inf_norm_exact"], "55/51")
        for row, err, bound in zip(fp["rows"],
                                   [5.7144e-17, 4.5919e-17, 3.5077e-17, 4.3846e-17],
                                   [1.1973e-16, 1.0289e-16, 7.0154e-17, 9.5001e-17]):
            self.assertAlmostEqual(row["actual_gradient_error_decimal"], err, delta=err * 1e-3)
            self.assertAlmostEqual(row["certified_error_upper_decimal"], bound, delta=bound * 1e-3)
            self.assertLess(row["actual_gradient_error_decimal"], row["certified_error_upper_decimal"])
        b = self.baselines
        self.assertAlmostEqual(b["standard_vf_abs_bias"], 3.9216e-2, delta=1e-6)
        self.assertAlmostEqual(b["implicit_gradient_exact"], -0.3235294118, places=10)
        want = {0.08: 1.9737e-4, 0.04: 4.9211e-5, 0.02: 1.2295e-5, 0.01: 3.0732e-6, 0.005: 7.6826e-7}
        for row in b["asymep_finite_beta"]:
            self.assertAlmostEqual(row["asymep_abs_error"], want[row["beta"]],
                                   delta=want[row["beta"]] * 1e-3)
        for row in b["dyadic_ideal_finite_beta"]:
            self.assertEqual(row["abs_error"], 0.0)


class PaperNumbersFromOtherFiles(unittest.TestCase):
    def test_non_normal_table(self):
        rows = {int(r["offdiagonal_k"]): r for r in load("nonnormal_sweep_summary.json")["rows"]}
        for k, kappa, gain, amp in [(0, 10.0, 1.000, 1.000), (1, 20.05, 1.125, 1.000),
                                    (2, 50.08, 1.738, 1.518), (5, 260.10, 3.949, 3.857),
                                    (10, 1010.10, 7.782, 7.735), (20, 4010.10, 15.505, 15.481)]:
            self.assertAlmostEqual(rows[k]["condition_number_2"], kappa, delta=0.01)
            self.assertAlmostEqual(rows[k]["max_transient_gain_0_10"], gain, delta=1e-3)
            self.assertAlmostEqual(rows[k]["gradient_error_amplification_ratio"], amp, delta=1e-3)
        c = load("nonnormal_counterexample.json")
        self.assertAlmostEqual(c["condition_number_2"], 101.99, delta=0.01)
        self.assertAlmostEqual(c["max_transient_gain_on_0_to_5"], 3.716, delta=1e-3)
        self.assertAlmostEqual(c["response_relative_error"], 10.0, places=10)

    def test_integrator_table(self):
        c = load("integrator_comparison.json")
        self.assertEqual(c["adaptive_rk4_rhs_evaluations_estimate"], 168)
        self.assertEqual(c["fixed_rk4_rhs_evaluations"], 1136)
        self.assertEqual(c["dop853_reference_rhs_evaluations"], 8500)
        self.assertAlmostEqual(c["adaptive_gradient_estimate_error_vs_dop853_estimate"], 7.88e-7, delta=1e-9)
        self.assertAlmostEqual(c["fixed_gradient_estimate_error_vs_dop853_estimate"], 2.57e-11, delta=1e-13)

    def test_stage1_and_tradeoff(self):
        s = load("validation_summary.json")
        self.assertAlmostEqual(s["vf_relative_error"], 1.55, delta=5e-3)
        self.assertAlmostEqual(s["empirical_log_log_order_centered_asymep"], 2.002, delta=5e-4)
        self.assertAlmostEqual(s["empirical_log_log_order_one_sided_asymep"], 0.946, delta=5e-4)
        self.assertAlmostEqual(s["implicit_relative_error"], 1.8e-16, delta=1e-17)
        self.assertAlmostEqual(s["central_finite_difference_relative_error"], 6.6e-11, delta=1e-12)
        self.assertAlmostEqual(load("beta_noise_tradeoff_summary.json")["beta_opt_formula_unconstrained"],
                               0.1068, delta=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
