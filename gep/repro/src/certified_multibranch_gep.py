"""Stage 5: branch-aware certified GEP for a non-conservative multistable system.

The floating-point root solver supplies candidates only. Every accepted branch tube
and every +/- beta endpoint is certified by exact-rational interval Krawczyk checks.
The final finite-nudge gradient error is bounded directly by an interval containing
the finite-difference estimator, compared with the exact implicit gradient.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import json
import hashlib
import math
import numpy as np
from scipy.optimize import root

# Exact decimal-rational model constants.
A = F(1, 5)        # F1 y-coupling
B = F(1, 10)       # F2 nonlinear x-coupling
TX = F(3, 10)      # objective target x
TY = F(-1, 5)      # objective target y
BETA_TUBE = F(1, 50)   # certified continuation domain: [-0.02, +0.02]
BOX_RX = F(1, 20)      # local branch tube x-radius: 0.05
BOX_RY = F(1, 25)      # local branch tube y-radius: 0.04
ENDPOINT_RADIUS = F(1, 10**10)
TOLERANCE = F(1, 100000)  # 1e-5
BETA_CANDIDATES = [F(1, 50), F(1, 100), F(1, 200), F(1, 400), F(1, 800)]
BRANCHES = [-1, 1]

@dataclass(frozen=True)
class Iv:
    lo: F
    hi: F
    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("invalid interval")
    def __add__(self, other):
        o = as_iv(other)
        return Iv(self.lo + o.lo, self.hi + o.hi)
    __radd__ = __add__
    def __neg__(self):
        return Iv(-self.hi, -self.lo)
    def __sub__(self, other):
        return self + (-as_iv(other))
    def __rsub__(self, other):
        return as_iv(other) - self
    def __mul__(self, other):
        o = as_iv(other)
        products = (self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi)
        return Iv(min(products), max(products))
    __rmul__ = __mul__
    def scale(self, c):
        return self * F(c)
    @property
    def mid(self):
        return (self.lo + self.hi) / 2
    @property
    def radius(self):
        return (self.hi - self.lo) / 2
    def abs_upper(self):
        return max(abs(self.lo), abs(self.hi))
    def strict_inside(self, other: "Iv") -> bool:
        return self.lo > other.lo and self.hi < other.hi

def as_iv(v) -> Iv:
    if isinstance(v, Iv): return v
    z = F(v)
    return Iv(z, z)

def vec_add(a, b): return [as_iv(x) + y for x, y in zip(a, b)]
def matmul_iv(M, N):
    return [[sum((as_iv(M[i][k]) * as_iv(N[k][j]) for k in range(len(N))), as_iv(0))
             for j in range(len(N[0]))] for i in range(len(M))]
def matvec_iv(M, v):
    return [sum((as_iv(M[i][j]) * as_iv(v[j]) for j in range(len(v))), as_iv(0))
                for i in range(len(M))]
def frac_matmul(M, N):
    return [[sum((F(M[i][k]) * F(N[k][j]) for k in range(len(N))), F(0))
             for j in range(len(N[0]))] for i in range(len(M))]
def inverse_2x2(M):
    det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    if det == 0: raise ZeroDivisionError("singular matrix")
    return [[M[1][1]/det, -M[0][1]/det],
            [-M[1][0]/det, M[0][0]/det]]
def frac_text(x: F) -> str:
    x = F(x)
    return f"{x.numerator}/{x.denominator}"
def num(x) -> float:
    return float(x)
def box_text(X):
    return [[frac_text(i.lo), frac_text(i.hi)] for i in X]

# Original free dynamics F(x,y; theta):
# F1 = x - x^3 + A*y + theta; F2 = -y + B*(x-x^3).
# At theta=0 it has equilibria (-1,0), (0,0), (1,0).
def original_jacobian(branch: int):
    s = F(branch)
    d = F(1) - 3*s*s
    return [[d, A], [B*d, F(-1)]]

def learning_jacobian_at_branch(branch: int):
    J = original_jacobian(branch)
    return [[J[0][0], J[1][0]], [J[0][1], J[1][1]]]  # J^T after AsymEP correction

def preconditioner(branch: int):
    return inverse_2x2(learning_jacobian_at_branch(branch))

def learning_field_interval(X, beta: Iv, branch: int):
    x, y = X
    # g_s = F - 2 A_anti (z-z_s) - beta grad C.
    # A_anti = [[0,1/5],[-1/5,0]] at both stable branches.
    return [x - x*x*x - F(1,5)*y - beta*(x-TX),
            -y + F(1,10)*(x-x*x*x) + F(2,5)*(x-F(branch)) - beta*(y-TY)]

def learning_field_float(z, beta: float, branch: int):
    x, y = z
    return np.array([
        x - x**3 - 0.2*y - beta*(x-0.3),
        -y + 0.1*(x-x**3) + 0.4*(x-branch) - beta*(y+0.2),
    ], dtype=float)

def learning_jacobian_interval(X, beta: Iv):
    x, _ = X
    dx = 1 - 3*x*x
    return [[dx-beta, as_iv(-F(1,5))],
            [F(1,10)*dx + F(2,5), as_iv(-1)-beta]]

def interval_identity(n):
    return [[as_iv(1 if i == j else 0) for j in range(n)] for i in range(n)]

def krawczyk_certificate(X, beta: Iv, branch: int):
    """Uniform parametric Krawczyk certificate over every beta in beta interval.

    If K(X,beta) lies strictly inside X and sup ||I-C*J(X,beta)||_inf < 1,
    then for each beta in the interval the preconditioned fixed-point map is a
    contraction from X into itself. Hence there is exactly one root in X.
    All arithmetic in this function is exact Fraction interval arithmetic.
    """
    m = [v.mid for v in X]
    C = preconditioner(branch)  # exact, nonsingular rational matrix
    f_m = learning_field_interval([as_iv(v) for v in m], beta, branch)
    J_X = learning_jacobian_interval(X, beta)
    C_iv = [[as_iv(v) for v in row] for row in C]
    CJ = matmul_iv(C_iv, J_X)
    IminusCJ = [[interval_identity(2)[i][j] - CJ[i][j] for j in range(2)] for i in range(2)]
    Cf = matvec_iv(C_iv, f_m)
    X_minus_m = [X[i] - m[i] for i in range(2)]
    Mdelta = matvec_iv(IminusCJ, X_minus_m)
    K = [as_iv(m[i]) - Cf[i] + Mdelta[i] for i in range(2)]
    contraction = max(sum(IminusCJ[i][j].abs_upper() for j in range(2)) for i in range(2))
    inside = all(K[i].strict_inside(X[i]) for i in range(2))
    C_det = C[0][0]*C[1][1] - C[0][1]*C[1][0]
    passed = bool(inside and contraction < 1 and C_det != 0)
    return {
        "passed": passed,
        "strict_krawczyk_inclusion": inside,
        "preconditioned_contraction_upper_exact": frac_text(contraction),
        "preconditioned_contraction_upper": float(contraction),
        "preconditioner_det_exact": frac_text(C_det),
        "krawczyk_image": box_text(K),
        "domain_box": box_text(X),
        "beta_domain": [frac_text(beta.lo), frac_text(beta.hi)],
    }

def branch_tube(branch: int, beta_max: F = BETA_TUBE):
    beta_max = abs(F(beta_max))
    if beta_max > BETA_TUBE:
        raise ValueError("requested tube exceeds the configured certified beta domain")
    s = F(branch)
    X = [Iv(s-BOX_RX, s+BOX_RX), Iv(-BOX_RY, BOX_RY)]
    beta_iv = Iv(-beta_max, beta_max)
    cert = krawczyk_certificate(X, beta_iv, branch)
    if not cert["passed"]:
        raise RuntimeError(f"branch tube failed for branch {branch}: {cert}")
    stab = stability_certificate(X, beta_iv)
    if not stab["passed"]:
        raise RuntimeError(f"stability check failed for branch {branch}: {stab}")
    return {"branch": branch, "tube_certificate": cert, "stability_certificate": stab}

def stability_certificate(X, beta: Iv):
    """Prove learning-field Jacobian is Hurwitz throughout a branch tube (2x2)."""
    J = learning_jacobian_interval(X, beta)
    trace = J[0][0] + J[1][1]
    det = J[0][0]*J[1][1] - J[0][1]*J[1][0]
    passed = trace.hi < 0 and det.lo > 0
    return {"passed": passed, "trace_interval": [frac_text(trace.lo), frac_text(trace.hi)],
            "determinant_interval": [frac_text(det.lo), frac_text(det.hi)],
            "criterion": "real 2x2 Jacobian Hurwitz iff trace < 0 and determinant > 0"}

def certified_endpoint(branch: int, beta: F, sign: int, radius: F = ENDPOINT_RADIUS):
    """Use a floating solver for a seed, then certify its exact rational box."""
    beta = abs(F(beta))
    if beta > BETA_TUBE:
        raise ValueError("endpoint beta lies outside the certified branch tube")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    s = int(branch)
    betaf = float(beta) * int(sign)
    sol = root(lambda z: learning_field_float(z, betaf, s), np.array([float(s), 0.0]),
               jac=lambda z: np.array([[1-3*z[0]**2-betaf, -0.2],
                                       [0.1*(1-3*z[0]**2)+0.4, -1-betaf]], dtype=float),
               tol=1e-12)
    if not sol.success and np.linalg.norm(learning_field_float(sol.x, betaf, s)) > 1e-10:
        raise RuntimeError(f"candidate root solver failed: {sol.message}")
    center = [F.from_float(float(v)) for v in sol.x]
    X = [Iv(center[i]-radius, center[i]+radius) for i in range(2)]
    cert = krawczyk_certificate(X, as_iv(F(sign)*beta), branch)
    # This local check is point-parameter Krawczyk; the tube certificate separately
    # establishes the branch identity for the full interval connecting -beta..+beta.
    if not cert["passed"]:
        raise RuntimeError(f"endpoint Krawczyk certificate failed: branch={branch}, beta={betaf}, {cert}")
    return {"branch": branch, "beta": frac_text(F(sign)*beta),
            "candidate_float": [float(v) for v in sol.x],
            "box": box_text(X), "box_radius_exact": frac_text(radius),
            "residual_at_candidate_float": float(np.linalg.norm(learning_field_float(sol.x, betaf, s))),
            "krawczyk": cert, "intervals": X}

def exact_target_negative_gradient(branch: int):
    """Exact B^T J^{-T} q for theta entering F1 with unit gain."""
    J = original_jacobian(branch)
    JT = [[J[0][0], J[1][0]], [J[0][1], J[1][1]]]
    q = [F(branch)-TX, -TY]
    v = [sum((preconditioner_for_matrix(JT)[i][j] * q[j] for j in range(2)), F(0))
         for i in range(2)]
    return v[0]

def preconditioner_for_matrix(M): return inverse_2x2(M)

def gradient_interval_from_endpoints(plus, minus, beta: F):
    difference = plus[0] - minus[0]
    return difference.scale(F(1)/(2*beta))

def scalar_interval_error_upper(I: Iv, target: F):
    return max(abs(I.lo-target), abs(I.hi-target))

def approximate_fd(branch, beta):
    s = int(branch)
    p = root(lambda z: learning_field_float(z, float(beta), s), [float(s), 0.0],
             jac=lambda z: np.array([[1-3*z[0]**2-float(beta), -0.2],
                                     [0.1*(1-3*z[0]**2)+0.4, -1-float(beta)]]), tol=1e-12).x
    m = root(lambda z: learning_field_float(z, -float(beta), s), [float(s), 0.0],
             jac=lambda z: np.array([[1-3*z[0]**2+float(beta), -0.2],
                                     [0.1*(1-3*z[0]**2)+0.4, -1+float(beta)]]), tol=1e-12).x
    return float((p[0]-m[0])/(2*float(beta)))

def _serialized_box_to_intervals(box_text_value):
    return [Iv(F(lo), F(hi)) for lo, hi in box_text_value]

def certify_beta(beta: F, branch_tubes: dict):
    beta = abs(F(beta))
    if beta > BETA_TUBE:
        raise ValueError("beta lies outside certified tube domain")
    branch_rows = []
    max_error = F(0)
    for branch in BRANCHES:
        p = certified_endpoint(branch, beta, +1)
        m = certified_endpoint(branch, beta, -1)
        tube_box = _serialized_box_to_intervals(branch_tubes[str(branch)]["tube_certificate"]["domain_box"])
        endpoint_containment = all(
            all(tube_box[i].lo <= endpoint["intervals"][i].lo and
                endpoint["intervals"][i].hi <= tube_box[i].hi for i in range(2))
            for endpoint in (p, m))
        if not endpoint_containment:
            raise RuntimeError("endpoint certificate is not contained in its branch tube")
        dint = gradient_interval_from_endpoints(p["intervals"], m["intervals"], beta)
        truth = exact_target_negative_gradient(branch)
        err = scalar_interval_error_upper(dint, truth)
        max_error = max(max_error, err)
        branch_rows.append({
            "branch": branch,
            "branch_tube_box": box_text(tube_box),
            "endpoint_boxes_contained_in_branch_tube": endpoint_containment,
            "plus_endpoint": {k:v for k,v in p.items() if k != "intervals"},
            "minus_endpoint": {k:v for k,v in m.items() if k != "intervals"},
            "true_negative_gradient_exact": frac_text(truth),
            "true_negative_gradient_float": float(truth),
            "finite_difference_interval_exact": [frac_text(dint.lo), frac_text(dint.hi)],
            "finite_difference_interval_float": [float(dint.lo), float(dint.hi)],
            "certified_gradient_error_upper_exact": frac_text(err),
            "certified_gradient_error_upper": float(err),
            "empirical_fd_value": approximate_fd(branch, beta),
            "empirical_abs_error": abs(approximate_fd(branch, beta)-float(truth)),
        })
    return {"beta_exact": frac_text(beta), "beta": float(beta), "passed": max_error <= TOLERANCE,
            "worst_branch_error_upper_exact": frac_text(max_error),
            "worst_branch_error_upper": float(max_error), "tolerance_exact": frac_text(TOLERANCE),
            "branch_results": branch_rows}

def original_multistability_certificate():
    roots = []
    for s in [-1, 0, 1]:
        J = original_jacobian(s)
        trace = J[0][0]+J[1][1]
        det = J[0][0]*J[1][1]-J[0][1]*J[1][0]
        anti_gap = J[0][1]-J[1][0]
        stable = trace < 0 and det > 0
        roots.append({"root_exact": [str(s), "0"], "trace_exact": frac_text(trace),
                      "determinant_exact": frac_text(det), "stable_by_2x2_hurwitz": stable,
                      "jacobian_nonsymmetry_gap_exact": frac_text(anti_gap)})
    assert roots[0]["stable_by_2x2_hurwitz"] and roots[2]["stable_by_2x2_hurwitz"]
    assert not roots[1]["stable_by_2x2_hurwitz"]
    return {"roots": roots, "claim": "at least two distinct stable equilibria; global uniqueness cannot hold"}

def krawczyk_original_free_system(X):
    """Apply Krawczyk to the original theta=0 free field, not the branch-corrected field."""
    m = [v.mid for v in X]
    x, y = [as_iv(v) for v in m]
    f_m = [x-x*x*x+F(1,5)*y, -y+F(1,10)*(x-x*x*x)]
    xb, yb = X
    dx = 1-3*xb*xb
    J_X = [[dx, as_iv(F(1,5))], [F(1,10)*dx, as_iv(-1)]]
    C = inverse_2x2(original_jacobian(1))
    C_iv = [[as_iv(v) for v in row] for row in C]
    CJ = matmul_iv(C_iv, J_X)
    IminusCJ = [[interval_identity(2)[i][j]-CJ[i][j] for j in range(2)] for i in range(2)]
    Cf = matvec_iv(C_iv, f_m)
    Xm = [X[i]-m[i] for i in range(2)]
    Mdelta = matvec_iv(IminusCJ, Xm)
    K = [as_iv(m[i])-Cf[i]+Mdelta[i] for i in range(2)]
    contraction = max(sum(IminusCJ[i][j].abs_upper() for j in range(2)) for i in range(2))
    inside = all(K[i].strict_inside(X[i]) for i in range(2))
    cdet = C[0][0]*C[1][1]-C[0][1]*C[1][0]
    passed = bool(inside and contraction < 1 and cdet != 0)
    return {"passed":passed, "strict_krawczyk_inclusion":inside,
            "preconditioned_contraction_upper_exact":frac_text(contraction),
            "preconditioned_contraction_upper":float(contraction),
            "preconditioner_det_exact":frac_text(cdet), "krawczyk_image":box_text(K),
            "domain_box":box_text(X), "field":"original free field at theta=0"}

def deliberate_rejection_test():
    # This box contains all three exact roots of the original free system.
    # A unique-root certificate for this entire box must fail.
    X = [Iv(F(-6,5), F(6,5)), Iv(F(-1,10), F(1,10))]
    cert = krawczyk_original_free_system(X)
    return {"passed_as_rejection": not cert["passed"],
            "known_distinct_roots_in_box": [["-1", "0"], ["0", "0"], ["1", "0"]],
            "certificate": cert}

def main(output_path=None):
    multi = original_multistability_certificate()
    tubes = {str(s): branch_tube(s) for s in BRANCHES}
    reject = deliberate_rejection_test()
    if not reject["passed_as_rejection"]:
        raise RuntimeError("failure-path test failed: a deliberately broad multibranch box was accepted")
    attempts=[]
    selected=None
    # The tube proved over [-0.02,+0.02] is a hard safety envelope. Choose the
    # largest candidate beta for which every stable branch has a certified gradient.
    for beta in BETA_CANDIDATES:
        if beta > BETA_TUBE:
            attempts.append({"beta":float(beta),"status":"REJECTED_OUTSIDE_CERTIFIED_TUBE"})
            continue
        row=certify_beta(beta,tubes)
        row["status"] = "CERTIFIED" if row["passed"] else "ERROR_BOUND_EXCEEDS_TOLERANCE"
        attempts.append(row)
        if row["passed"]:
            selected=row
            break
    if selected is None:
        result={"status":"NO_CERTIFIED_BETA_FOUND","tolerance":float(TOLERANCE),"attempts":attempts,
                "multistability":multi,"branch_tubes":tubes,"failure_path":reject}
        status_ok=False
    else:
        result={"status":"CERTIFIED_MULTIBRANCH_GEP_PASSED","scope":"exact rational-coefficient 2D polynomial model; two stable local branches; |beta|<=1/50",
                "tolerance":float(TOLERANCE),"tolerance_exact":frac_text(TOLERANCE),
                "beta_selected":selected["beta"],"beta_selected_exact":selected["beta_exact"],
                "worst_branch_error_upper":selected["worst_branch_error_upper"],
                "worst_branch_error_upper_exact":selected["worst_branch_error_upper_exact"],
                "all_branch_gradient_certificates_pass":selected["passed"],
                "selected_branch_results":selected["branch_results"],
                "attempts":attempts,"multistability":multi,"branch_tubes":tubes,"failure_path":reject,
                "proof_obligations_checked":[
                    "exact rational Krawczyk inclusion for parameter tube on each branch",
                    "preconditioned contraction factor below one on each tube",
                    "Hurwitz trace/determinant interval signs throughout each tube",
                    "exact rational Krawczyk inclusion for every +/-beta endpoint box",
                    "interval enclosure of the finite-difference gradient",
                    "exact comparison with implicit-function-theorem gradient",
                    "explicit rejection of a broad box that cannot certify one branch"],
                "caveat":"The theorem-backed numerical certificate applies to this explicitly defined polynomial benchmark. Floating root finding only supplies candidates; accepted claims are based on exact rational interval checks. This is not a proof-assistant-checked formalization and does not imply a universal guarantee for arbitrary GEP systems."}
        status_ok=True
    out=Path(output_path) if output_path else Path(__file__).resolve().parents[1]/"results"/"stage5_certificate.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:result.get(k) for k in ["status","beta_selected","worst_branch_error_upper","tolerance","all_branch_gradient_certificates_pass"]},indent=2))
    if not status_ok: raise SystemExit("NO CERTIFIED BETA")
    print("STAGE 5 MULTIBRANCH CERTIFICATE PASSED")
    return result

if __name__ == "__main__": main()
