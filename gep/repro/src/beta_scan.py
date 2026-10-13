#!/usr/bin/env python3
"""Dense nudging scan with the exact-rational multibranch certificate.

For every candidate perturbation beta inside the certified parameter tube the
script certifies both stable branches (+/- beta endpoints, containment in the
branch tube, interval enclosure of the centred finite difference) and records
the worst-branch gradient error upper bound. This is the data behind the
nudging-selection figure and table.

Run from the repro directory (or through run_all.py).
"""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import certified_multibranch_gep as g

BETAS = [F(1, 50), F(3, 200), F(1, 80), F(1, 100), F(3, 400),
         F(1, 200), F(1, 250), F(1, 400), F(1, 800), F(1, 1000)]


def main() -> None:
    tubes = {str(s): g.branch_tube(s) for s in g.BRANCHES}
    rows = []
    for beta in BETAS:
        cert = g.certify_beta(beta, tubes)
        rows.append({
            "beta_exact": cert["beta_exact"],
            "beta": cert["beta"],
            "worst_branch_error_upper_exact": cert["worst_branch_error_upper_exact"],
            "worst_branch_error_upper": cert["worst_branch_error_upper"],
            "tolerance": float(g.TOLERANCE),
            "accepted": bool(cert["passed"]),
            "branch_error_upper": {r["branch"]: r["certified_gradient_error_upper"] for r in cert["branch_results"]},
            "branch_error_upper_exact": {r["branch"]: r["certified_gradient_error_upper_exact"] for r in cert["branch_results"]},
            "branch_empirical_error": {r["branch"]: r["empirical_abs_error"] for r in cert["branch_results"]},
        })
        print(f"beta={cert['beta']:.5f} worst={cert['worst_branch_error_upper']:.6e} "
              f"accepted={cert['passed']}")

    out = Path(__file__).resolve().parents[1] / "results" / "beta_scan.json"
    out.write_text(json.dumps({
        "model": "F=(x-x^3+y/5+theta, -y+(x-x^3)/10), C=((x-3/10)^2+(y+1/5)^2)/2, theta=0",
        "certified_beta_domain": [-0.02, 0.02],
        "tolerance_exact": g.frac_text(g.TOLERANCE),
        "branch_tube_contraction_upper": {
            str(s): tubes[str(s)]["tube_certificate"]["preconditioned_contraction_upper_exact"]
            for s in g.BRANCHES},
        "rows": rows,
        "largest_accepted_beta": max([r["beta"] for r in rows if r["accepted"]], default=None),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
