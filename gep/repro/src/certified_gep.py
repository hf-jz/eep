"""Certified Adaptive GEP prototype.

The ODE integrator is used only to propose endpoint states.  The certificate
is computed independently from exact rational enclosures of the residual of
an explicitly defined rational-coefficient vector field, using rational Taylor
bounds for sin/tanh and an exact integer-square-root upper enclosure for the
Euclidean norm.  Thus no ODE step-size estimate is mislabeled as a proof.

Scope: the finite-dimensional three-state benchmark described in the report,
for |beta| <= 0.32, and the stated exact decimal-rational model coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import factorial, isqrt, log
from pathlib import Path
import json
import hashlib

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------- Exact problem definition ----------------------
A0 = [
    [F('-2.0'), F('1.2'), F('-0.4')],
    [F('-0.3'), F('-1.8'), F('0.5')],
    [F('0.2'), F('-0.7'), F('-1.5')],
]
K = [
    [F('0.0'), F('0.5'), F('0.1')],
    [F('-0.2'), F('0.0'), F('0.3')],
    [F('0.1'), F('-0.4'), F('0.0')],
]
GAMMA = F('0.25')
QDIAG = [F('1.0'), F('1.4'), F('0.8')]
COSSCALE = F('0.1')
TARGET = [F('0.5'), F('-0.1'), F('0.3')]
X0 = [F('0.27'), F('-0.385'), F('0.185')]
L = [
    [F('1.0'), F('0.2')],
    [F('-0.4'), F('0.7')],
    [F('0.3'), F('-0.5')],
]
# Original free-equilibrium Jacobian A = A0 + gamma K.
A = [[A0[i][j] + GAMMA * K[i][j] for j in range(3)] for i in range(3)]
ANTI = [[(A[i][j] - A[j][i]) / 2 for j in range(3)] for i in range(3)]
# Parameter Jacobian B = -A L. The Frobenius norm is certified below by 3.24.
B = [[-sum(A[i][k] * L[k][j] for k in range(3)) for j in range(2)] for i in range(3)]

ALPHA = F('0.64')        # global one-sided contraction rate, analytically proved
B_FROBENIUS_UPPER = F('3.24')
Q0_NORM_UPPER = F('0.488')
LQ_UPPER = F('1.5')
M2F_UPPER = F('0.066')
M3F_UPPER = F('0.080')
M2Q_UPPER = F('0.1')
M3Q_UPPER = F('0.1')
BETA_DOMAIN = F('0.32')

# ---------------------- Small exact interval type --------------------
@dataclass(frozen=True)
class RI:
    lo: F
    hi: F

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("invalid rational interval")

    def __add__(self, other):
        other = as_interval(other)
        return RI(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return RI(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-as_interval(other))

    def __rsub__(self, other):
        return as_interval(other) - self

    def __mul__(self, other):
        other = as_interval(other)
        xs = [self.lo * other.lo, self.lo * other.hi,
              self.hi * other.lo, self.hi * other.hi]
        return RI(min(xs), max(xs))

    __rmul__ = __mul__

    def scale(self, scalar: F):
        scalar = F(scalar)
        if scalar >= 0:
            return RI(self.lo * scalar, self.hi * scalar)
        return RI(self.hi * scalar, self.lo * scalar)


def as_interval(x) -> RI:
    if isinstance(x, RI):
        return x
    x = F(x)
    return RI(x, x)


def sin_enclosure(x: F, n: int = 12) -> RI:
    """Rigorous rational enclosure of sin(x) for |x| <= 1.

    Alternating Taylor series: S_n=sum_{k=0}^n (-1)^k x^(2k+1)/(2k+1)!.
    The absolute remainder is bounded by |x|^(2n+3)/(2n+3)!.
    We use a symmetric enclosure, which is slightly wider but simple to audit.
    """
    x = F(x)
    if abs(x) > 1:
        raise ValueError("sin certificate currently requires |x| <= 1")
    y = abs(x)
    s = sum((F(-1) ** k) * y ** (2 * k + 1) / factorial(2 * k + 1)
            for k in range(n + 1))
    rem = y ** (2 * n + 3) / factorial(2 * n + 3)
    pos = RI(s - rem, s + rem)
    return pos if x >= 0 else -pos


def exp_enclosure(x: F, n: int = 20) -> RI:
    """Rigorous rational enclosure of exp(x) for |x| <= 1.

    Taylor polynomial through degree n; Lagrange remainder <= e*|x|^(n+1)/(n+1)!.
    Since e < 3, use 3*|x|^(n+1)/(n+1)!, entirely rational.
    """
    x = F(x)
    if abs(x) > 1:
        raise ValueError("exp certificate currently requires |x| <= 1")
    s = sum(x ** k / factorial(k) for k in range(n + 1))
    rem = F(3) * abs(x) ** (n + 1) / factorial(n + 1)
    out = RI(s - rem, s + rem)
    if out.lo <= 0:
        raise ArithmeticError("exp enclosure unexpectedly crosses zero")
    return out


def tanh_enclosure(x: F) -> RI:
    """Enclose tanh(x) by tanh(x)=(1-exp(-2x))/(1+exp(-2x))."""
    x = F(x)
    ex = exp_enclosure(-2 * x)
    # (1-r)/(1+r) is decreasing for r > 0.
    lo = (F(1) - ex.hi) / (F(1) + ex.hi)
    hi = (F(1) - ex.lo) / (F(1) + ex.lo)
    return RI(lo, hi)


def sqrt_upper_fraction(q: F, decimal_digits: int = 30) -> F:
    """Return a rational upper bound to sqrt(q), via exact integer arithmetic."""
    q = F(q)
    if q < 0:
        raise ValueError("sqrt argument must be nonnegative")
    if q == 0:
        return F(0)
    scale = 10 ** decimal_digits
    num = q.numerator * scale * scale
    den = q.denominator
    ceil_scaled_square = (num + den - 1) // den
    k = isqrt(ceil_scaled_square)
    if k * k < ceil_scaled_square:
        k += 1
    return F(k, scale)


def interval_abs_upper(iv: RI) -> F:
    return max(abs(iv.lo), abs(iv.hi))


def interval_vector_norm_upper(v: list[RI]) -> F:
    s = sum((interval_abs_upper(x) ** 2 for x in v), F(0))
    return sqrt_upper_fraction(s)

# ---------------------- Residual enclosure ---------------------------
def q_interval(x: list[F]) -> list[RI]:
    out = []
    for i in range(3):
        lin = QDIAG[i] * (x[i] - TARGET[i])
        out.append(RI(lin, lin) + sin_enclosure(x[i]).scale(COSSCALE))
    return out


def exact_model_residual_enclosure(x_float: np.ndarray, beta_text: str) -> list[RI]:
    """Enclose each exact model residual at the exact binary64 endpoint x_float.

    Input floating-point states are interpreted as exact binary rationals using
    Fraction.from_float. All coefficients and beta are exact decimal rationals.
    Trig functions are enclosed using explicit rational Taylor remainder bounds.
    """
    x = [F.from_float(float(v)) for v in x_float]
    beta = F(beta_text)
    d = [x[i] - X0[i] for i in range(3)]
    kd = [sum(K[i][j] * d[j] for j in range(3)) for i in range(3)]
    tanhs = [tanh_enclosure(v) for v in kd]
    qv = q_interval(x)
    residual = []
    for i in range(3):
        base = sum((A0[i][j] - 2 * ANTI[i][j]) * d[j] for j in range(3))
        value = RI(base, base) + tanhs[i].scale(GAMMA) - qv[i].scale(beta)
        residual.append(value)
    return residual


def certified_residual_norm(x_float: np.ndarray, beta_text: str) -> F:
    """Exact rational upper bound on ||g_beta(x)||_2 for the specified model."""
    return interval_vector_norm_upper(exact_model_residual_enclosure(x_float, beta_text))

# ---------------------- Analytic contraction proof constants ----------
def exact_gershgorin_row_bounds() -> list[F]:
    """Exact rational Gershgorin upper bounds for sym(D_x g_beta), |beta|<=0.32."""
    row_bounds = []
    for i in range(3):
        # Worst diagonal occurs at |beta|=0.32 and |cos(x_i)|=1.
        diag_upper = A0[i][i] + BETA_DOMAIN * (QDIAG[i] + COSSCALE)
        off_sum = F(0)
        for j in range(3):
            if i == j:
                continue
            base = (A0[i][j] + A0[j][i]) / 2
            corner_values = [
                base + GAMMA / 2 * (F(di) * K[i][j] + F(dj) * K[j][i])
                for di in (0, 1) for dj in (0, 1)
            ]
            off_sum += max(abs(v) for v in corner_values)
        row_bounds.append(diag_upper + off_sum)
    return row_bounds


def check_analytic_constants() -> dict:
    """Audit exact rational constants underlying every certificate bound."""
    b_sq = sum((B[i][j] ** 2 for i in range(3) for j in range(2)), F(0))
    if not b_sq < B_FROBENIUS_UPPER ** 2:
        raise AssertionError("claimed B norm upper bound failed")
    q0_bound = interval_vector_norm_upper(q_interval(X0))
    if not q0_bound < Q0_NORM_UPPER:
        raise AssertionError("claimed ||q(x0)|| upper bound failed")

    # Upper bounds on row norms of K are rational and strictly exceed the exact values.
    row_norm_bounds = [F('0.511'), F('0.361'), F('0.413')]
    for i, row_bound in enumerate(row_norm_bounds):
        row_sq = sum((K[i][j] ** 2 for j in range(3)), F(0))
        if not row_sq < row_bound ** 2:
            raise AssertionError(f"K row-norm upper bound failed at row {i}")
    m2_sq = (GAMMA * F('0.77')) ** 2 * sum((r ** 4 for r in row_norm_bounds), F(0))
    m3_sq = (GAMMA * F('2.0')) ** 2 * sum((r ** 6 for r in row_norm_bounds), F(0))
    if not m2_sq < M2F_UPPER ** 2:
        raise AssertionError("claimed second derivative bound for force failed")
    if not m3_sq < M3F_UPPER ** 2:
        raise AssertionError("claimed third derivative bound for force failed")

    row_bounds = exact_gershgorin_row_bounds()
    if max(row_bounds) > -ALPHA:
        raise AssertionError("global contraction constant failed the exact row-sum check")

    # A conservative global Lipschitz bound L<10 follows from Frobenius bounds:
    # ||A0||F<3.5, ||K||F<0.75, ||ANTI||F<1.6, and ||Dq||<=1.5.
    a0_sq = sum((x * x for row in A0 for x in row), F(0))
    k_sq = sum((x * x for row in K for x in row), F(0))
    anti_sq = sum((x * x for row in ANTI for x in row), F(0))
    if not (a0_sq < F('3.5') ** 2 and k_sq < F('0.75') ** 2 and anti_sq < F('1.6') ** 2):
        raise AssertionError("global Lipschitz norm bounds failed")
    lipschitz_upper = F('3.5') + GAMMA * F('0.75') + 2 * F('1.6') + BETA_DOMAIN * LQ_UPPER
    if not lipschitz_upper < F(10):
        raise AssertionError("claimed global Lipschitz bound L<10 failed")

    return {
        "B_frobenius_squared_exact": f"{b_sq.numerator}/{b_sq.denominator}",
        "B_frobenius_numeric": float(sqrt_upper_fraction(b_sq)),
        "B_frobenius_upper": float(B_FROBENIUS_UPPER),
        "q0_norm_interval_upper": float(q0_bound),
        "q0_norm_interval_upper_exact": frac_exact(q0_bound),
        "q0_norm_claimed_upper": float(Q0_NORM_UPPER),
        "alpha_contraction": float(ALPHA),
        "gershgorin_row_upper_bounds": [float(v) for v in row_bounds],
        "gershgorin_row_upper_bounds_exact": [frac_exact(v) for v in row_bounds],
        "alpha_proved_by_gershgorin_row_upper_max": float(max(row_bounds)),
        "alpha_proved_by_gershgorin_row_upper_max_exact": frac_exact(max(row_bounds)),
        "global_lipschitz_upper": float(lipschitz_upper),
        "force_second_derivative_upper": float(M2F_UPPER),
        "force_third_derivative_upper": float(M3F_UPPER),
    }


def branch_derivative_bounds(beta: F) -> dict:
    """Certified conservative bound on sup ||d^3 x_beta / d beta^3||.

    Uses implicit differentiation of g(x_beta,beta)=0 and ||D_x g^{-1}||<=1/alpha.
    Applies over |beta'| <= |beta|, not just at the endpoint.
    """
    b = abs(F(beta))
    if b > BETA_DOMAIN:
        raise ValueError("beta outside certified domain")
    radius = b * Q0_NORM_UPPER / ALPHA
    qmax = Q0_NORM_UPPER + LQ_UPPER * radius
    x1 = qmax / ALPHA
    m2 = M2F_UPPER + b * M2Q_UPPER
    m3 = M3F_UPPER + b * M3Q_UPPER
    x2 = (m2 * x1 * x1 + 2 * LQ_UPPER * x1) / ALPHA
    x3 = (3 * m2 * x1 * x2 + m3 * x1 ** 3
          + 3 * LQ_UPPER * x2 + 3 * M2Q_UPPER * x1 * x1) / ALPHA
    return {
        "root_radius_upper": radius,
        "q_norm_upper": qmax,
        "x1_upper": x1,
        "x2_upper": x2,
        "x3_upper": x3,
        "finite_nudge_gradient_error_upper": B_FROBENIUS_UPPER * b * b * x3 / 6,
    }

# ---------------------- Candidate-state generation -------------------
def x0_float() -> np.ndarray:
    return np.array([float(v) for v in X0], dtype=float)

A0_NP = np.array([[float(v) for v in row] for row in A0])
K_NP = np.array([[float(v) for v in row] for row in K])
GAMMA_NP = float(GAMMA)
Q_NP = np.diag([float(v) for v in QDIAG])
COSSCALE_NP = float(COSSCALE)
TARGET_NP = np.array([float(v) for v in TARGET])
ANTI_NP = np.array([[float(v) for v in row] for row in ANTI])
B_NP = np.array([[float(v) for v in row] for row in B])


def field_float(x: np.ndarray, beta: float) -> np.ndarray:
    d = x - x0_float()
    q = Q_NP @ (x - TARGET_NP) + COSSCALE_NP * np.sin(x)
    return A0_NP @ d + GAMMA_NP * np.tanh(K_NP @ d) - 2 * ANTI_NP @ d - beta * q


def integrate_candidate(beta_text: str, T: float, max_step: float) -> dict:
    beta = float(F(beta_text))
    x0 = x0_float()
    sol = solve_ivp(lambda t, x: field_float(x, beta), (0.0, T), x0,
                    method="DOP853", rtol=2e-12, atol=2e-14,
                    max_step=max_step)
    if not sol.success:
        raise RuntimeError(f"DOP853 failed: {sol.message}")
    x_end = sol.y[:, -1]
    residual = certified_residual_norm(x_end, beta_text)
    return {
        "state": x_end,
        "residual_upper": residual,
        "residual_upper_float": float(residual),
        "nfev": int(sol.nfev),
        "accepted_steps": int(len(sol.t) - 1),
        "max_observed_step": float(np.max(np.diff(sol.t))) if len(sol.t) > 1 else 0.0,
    }


def total_gradient_error_certificate(beta_text: str, rho_plus: F, rho_minus: F) -> dict:
    beta = F(beta_text)
    db = branch_derivative_bounds(beta)
    trunc = db["finite_nudge_gradient_error_upper"]
    root = B_FROBENIUS_UPPER * (rho_plus + rho_minus) / (2 * ALPHA * beta)
    total = trunc + root
    return {"truncation_upper": trunc, "equilibrium_residual_upper": root,
            "total_upper": total, "total_upper_float": float(total)}


def frac_dec(x: F, digits: int = 15) -> str:
    return f"{float(x):.{digits}g}"


def frac_exact(x: F) -> str:
    x = F(x)
    return f"{x.numerator}/{x.denominator}"


def adaptive_certified_run(tol: F = F('0.0005')) -> dict:
    """Select beta, T, and an integration max-step using certificate checks.

    We select the largest beta whose conservative finite-nudge bound can fit
    within the total error budget. For each beta we seek the shortest T and the
    coarsest max_step whose independently checked residuals make the full
    certificate pass. The ODE solver's internal error estimate is not used as
    a proof; acceptance is based on exact residual enclosures.
    """
    summary = check_analytic_constants()
    beta_candidates = ["0.02", "0.01", "0.005", "0.0025", "0.00125"]
    T_grid = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0]
    h_grid = [1.0, 0.5, 0.25, 0.125]
    history = []
    root_budget = tol * F('0.25')
    for beta_text in beta_candidates:
        beta = F(beta_text)
        trunc = branch_derivative_bounds(beta)["finite_nudge_gradient_error_upper"]
        if trunc >= tol:
            history.append({"beta": beta_text, "status": "skip_truncation_bound_exceeds_tolerance",
                            "truncation_bound": float(trunc)})
            continue
        for T in T_grid:
            passing_at_T = None
            for hmax in h_grid:
                plus = integrate_candidate(beta_text, T, hmax)
                minus = integrate_candidate(str(-beta), T, hmax)
                cert = total_gradient_error_certificate(beta_text,
                                                        plus["residual_upper"],
                                                        minus["residual_upper"])
                row = {
                    "beta": beta_text, "T": T, "max_step": hmax,
                    "rho_plus": plus["residual_upper_float"],
                    "rho_minus": minus["residual_upper_float"],
                    "truncation_bound": float(cert["truncation_upper"]),
                    "root_residual_bound": float(cert["equilibrium_residual_upper"]),
                    "total_certificate": cert["total_upper_float"],
                    "nfev_total": plus["nfev"] + minus["nfev"],
                    "steps_total": plus["accepted_steps"] + minus["accepted_steps"],
                }
                history.append(row)
                # Do not require the split to be exact, but keep a root-error budget
                # as a safety check to avoid certificates dominated by residuals.
                if cert["total_upper"] <= tol and cert["equilibrium_residual_upper"] <= root_budget:
                    passing_at_T = (plus, minus, cert, hmax)
                    break  # coarsest h that passes at this T
            if passing_at_T is not None:
                plus, minus, cert, chosen_h = passing_at_T
                xplus, xminus = plus["state"], minus["state"]
                finite_diff = B_NP.T @ ((xplus - xminus) / (2.0 * float(beta)))
                q0_float = np.array([float(v.lo + (v.hi - v.lo) / 2) for v in q_interval(X0)])
                true_descent = -np.array([[float(v) for v in row] for row in L]).T @ q0_float
                empirical_error = float(np.linalg.norm(finite_diff - true_descent))
                return {
                    "status": "CERTIFICATE_PASSED",
                    "scope": "exact decimal-rational benchmark; all beta in [-0.32,0.32]",
                    "tolerance": float(tol),
                    "beta_selected": beta_text,
                    "T_selected": T,
                    "max_step_selected": chosen_h,
                    "plus_state_float": xplus.tolist(),
                    "minus_state_float": xminus.tolist(),
                    "plus_residual_norm_upper": plus["residual_upper_float"],
                    "plus_residual_norm_upper_exact": frac_exact(plus["residual_upper"]),
                    "minus_residual_norm_upper": minus["residual_upper_float"],
                    "minus_residual_norm_upper_exact": frac_exact(minus["residual_upper"]),
                    "plus_equilibrium_distance_upper": float(plus["residual_upper"] / ALPHA),
                    "plus_equilibrium_distance_upper_exact": frac_exact(plus["residual_upper"] / ALPHA),
                    "minus_equilibrium_distance_upper": float(minus["residual_upper"] / ALPHA),
                    "minus_equilibrium_distance_upper_exact": frac_exact(minus["residual_upper"] / ALPHA),
                    "truncation_gradient_error_upper": float(cert["truncation_upper"]),
                    "truncation_gradient_error_upper_exact": frac_exact(cert["truncation_upper"]),
                    "equilibrium_solve_gradient_error_upper": float(cert["equilibrium_residual_upper"]),
                    "equilibrium_solve_gradient_error_upper_exact": frac_exact(cert["equilibrium_residual_upper"]),
                    "total_gradient_error_upper": float(cert["total_upper"]),
                    "total_gradient_error_upper_exact": frac_exact(cert["total_upper"]),
                    "tolerance_exact": frac_exact(tol),
                    "empirical_gradient_error": empirical_error,
                    "empirical_error_below_certificate": empirical_error <= float(cert["total_upper"]) + 1e-10,
                    "nfev_total": plus["nfev"] + minus["nfev"],
                    "accepted_steps_total": plus["accepted_steps"] + minus["accepted_steps"],
                    "analytic_constants": summary,
                    "selected_beta_derivative_bounds": {
                        k: float(v) for k, v in branch_derivative_bounds(beta).items()
                    },
                    "selected_beta_derivative_bounds_exact": {
                        k: frac_exact(v) for k, v in branch_derivative_bounds(beta).items()
                    },
                    "attempt_history": history,
                    "certificate_note": (
                        "The certificate is a theorem-backed bound for the defined model, conditional only on the explicitly proved derivative/contractivity bounds and exact rational enclosure implementation. "
                        "It certifies distance to the equilibrium of the exact decimal-rational vector field at the returned binary64 endpoints; it does not certify the entire computed ODE trajectory or any noisy/uncertain physical model."
                    ),
                }
    return {"status": "NO_CERTIFIED_CONFIGURATION_FOUND", "tolerance": float(tol),
            "attempt_history": history, "analytic_constants": summary}


def main(output_path: str | None = None) -> dict:
    result = adaptive_certified_run()
    out = Path(output_path) if output_path else Path(__file__).resolve().parents[1] / "results" / "certified_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ["status", "beta_selected", "T_selected", "max_step_selected",
                                                "truncation_gradient_error_upper", "equilibrium_solve_gradient_error_upper",
                                                "total_gradient_error_upper", "empirical_gradient_error",
                                                "empirical_error_below_certificate", "nfev_total"]}, indent=2))
    if result["status"] != "CERTIFICATE_PASSED":
        raise SystemExit("certified controller did not find a passing configuration")
    if not result["empirical_error_below_certificate"]:
        raise SystemExit("numerical empirical error exceeded the proposed certificate; inspect model/implementation")
    print("STAGE 4 CERTIFICATE PASSED")
    return result

if __name__ == "__main__":
    main()
