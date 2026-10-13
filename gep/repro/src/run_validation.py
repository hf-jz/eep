"""Reproducible validation suite for deterministic generalized equilibrium propagation.

Requires: numpy, scipy, torch, matplotlib.
Run from project root with: python src/run_validation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
np.set_printoptions(precision=7, suppress=True)
torch.set_default_dtype(torch.float64)

# Reproducible nonlinear equilibrium system with a closed-form equilibrium x*(theta).
A0 = np.array([
    [-2.0, 1.2, -0.4],
    [-0.3, -1.8, 0.5],
    [0.2, -0.7, -1.5],
], dtype=float)
K = np.array([
    [0.0, 0.5, 0.1],
    [-0.2, 0.0, 0.3],
    [0.1, -0.4, 0.0],
], dtype=float)
GAMMA = 0.25
L = np.array([[1.0, 0.2], [-0.4, 0.7], [0.3, -0.5]], dtype=float)
OFFSET = np.array([0.1, -0.2, 0.05], dtype=float)
THETA = np.array([0.2, -0.15], dtype=float)
TARGET = np.array([0.5, -0.1, 0.3], dtype=float)
QMAT = np.diag([1.0, 1.4, 0.8])
COSSCALE = 0.1


def x_star(theta: np.ndarray) -> np.ndarray:
    return L @ theta + OFFSET


def force(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Smooth non-conservative force with known local equilibrium x_star(theta)."""
    d = x - x_star(theta)
    return A0 @ d + GAMMA * np.tanh(K @ d)


def jacobian_at_state(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    d = x - x_star(theta)
    pre = K @ d
    return A0 + GAMMA * np.diag(1.0 - np.tanh(pre) ** 2) @ K


def jacobian_at_equilibrium() -> np.ndarray:
    return A0 + GAMMA * K


def parameter_jacobian_at_state(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # By the chain rule, D_theta F at fixed x equals -D_x F D_theta x_star.
    return -jacobian_at_state(x, theta) @ L


def parameter_jacobian_at_equilibrium() -> np.ndarray:
    return parameter_jacobian_at_state(x_star(THETA), THETA)


def cost(x: np.ndarray) -> float:
    d = x - TARGET
    return float(0.5 * d @ QMAT @ d + COSSCALE * np.sum(1.0 - np.cos(x)))


def cost_grad_x(x: np.ndarray) -> np.ndarray:
    return QMAT @ (x - TARGET) + COSSCALE * np.sin(x)


def augmented_force(x: np.ndarray, beta: float, theta: np.ndarray, x0: np.ndarray, j: np.ndarray) -> np.ndarray:
    antisym = 0.5 * (j - j.T)
    return force(x, theta) - beta * cost_grad_x(x) - 2.0 * antisym @ (x - x0)


def find_asym_root(beta: float, theta: np.ndarray, x0: np.ndarray, j: np.ndarray) -> np.ndarray:
    sol = root(lambda x: augmented_force(x, beta, theta, x0, j), x0, method="hybr", tol=1e-12)
    residual = np.linalg.norm(augmented_force(sol.x, beta, theta, x0, j))
    if not sol.success and residual > 1e-9:
        raise RuntimeError(f"AsymEP root failed: {sol.message}; residual={residual:g}")
    if residual > 1e-8:
        raise RuntimeError(f"AsymEP root residual too large: {residual:g}")
    return sol.x


def find_dyadic_stationary(beta: float, theta: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the extended-space stationarity equations for mean m and difference d.

    At stationarity: F(m,theta)=0 and J_F(m,theta)^T d = beta * grad C(m).
    """
    j0 = jacobian_at_state(x0, theta)
    d0 = beta * np.linalg.solve(j0.T, cost_grad_x(x0))
    guess = np.concatenate([x0, d0])

    def residual(y: np.ndarray) -> np.ndarray:
        m, d = y[:3], y[3:]
        jm = jacobian_at_state(m, theta)
        return np.concatenate([force(m, theta), jm.T @ d - beta * cost_grad_x(m)])

    sol = root(residual, guess, method="hybr", tol=1e-12)
    res = float(np.linalg.norm(residual(sol.x)))
    if not sol.success and res > 1e-9:
        raise RuntimeError(f"Dyadic root failed: {sol.message}; residual={res:g}")
    if res > 1e-8:
        raise RuntimeError(f"Dyadic root residual too large: {res:g}")
    return sol.x[:3], sol.x[3:], res


def autodiff_descent(theta_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = torch.tensor(theta_np, requires_grad=True)
    L_t = torch.tensor(L)
    offset_t = torch.tensor(OFFSET)
    target_t = torch.tensor(TARGET)
    Q_t = torch.tensor(QMAT)
    x = L_t @ theta + offset_t
    d = x - target_t
    obj = 0.5 * d @ Q_t @ d + COSSCALE * torch.sum(1.0 - torch.cos(x))
    obj.backward()
    return x.detach().numpy(), theta.grad.detach().numpy(), (-theta.grad).detach().numpy()


def relerr(a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> float:
    return float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), eps))


def euler_asym(beta: float, theta: np.ndarray, x0: np.ndarray, j: np.ndarray,
               dt: float, steps: int) -> np.ndarray:
    x = x0.copy()
    for _ in range(steps):
        x += dt * augmented_force(x, beta, theta, x0, j)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("Euler integration diverged")
    return x


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with (RES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    J = jacobian_at_equilibrium()
    B = parameter_jacobian_at_equilibrium()
    x0 = x_star(THETA)
    q = cost_grad_x(x0)
    eigs = np.linalg.eigvals(J)
    if np.max(np.real(eigs)) >= 0:
        raise RuntimeError(f"Chosen equilibrium must be locally stable; eigenvalues={eigs}")

    x_ad, grad_ad, descent_ad = autodiff_descent(THETA)
    fd_eps = 1e-6
    fd_grad = np.empty_like(THETA)
    for k in range(len(THETA)):
        e = np.zeros_like(THETA); e[k] = 1.0
        fd_grad[k] = (cost(x_star(THETA + fd_eps * e)) - cost(x_star(THETA - fd_eps * e))) / (2.0 * fd_eps)
    descent_implicit = B.T @ np.linalg.solve(J.T, q)
    vf_descent = B.T @ np.linalg.solve(J, q)
    dyadic_descent = B.T @ np.linalg.solve(J.T, q)

    # Validate the equilibrium itself and compare analytic implicit gradient to torch autograd.
    analytical_rows = [
        {"method": "implicit_true_gradient", "relative_error_vs_autograd_descent": relerr(descent_implicit, descent_ad),
         "cosine_vs_autograd_descent": cos_sim(descent_implicit, descent_ad)},
        {"method": "Dyadic_EP_stationary", "relative_error_vs_autograd_descent": relerr(dyadic_descent, descent_ad),
         "cosine_vs_autograd_descent": cos_sim(dyadic_descent, descent_ad)},
        {"method": "central_finite_difference_check", "relative_error_vs_autograd_gradient": relerr(fd_grad, grad_ad),
         "cosine_vs_autograd_gradient": cos_sim(fd_grad, grad_ad)},
        {"method": "Vector_Field_EP", "relative_error_vs_autograd_descent": relerr(vf_descent, descent_ad),
         "cosine_vs_autograd_descent": cos_sim(vf_descent, descent_ad)},
    ]
    write_csv("gradient_comparison.csv", analytical_rows)

    beta_values = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    finite_rows = []
    one_sided_errors = []
    for beta in beta_values:
        xp = find_asym_root(beta, THETA, x0, J)
        xm = find_asym_root(-beta, THETA, x0, J)
        v_center = (xp - xm) / (2.0 * beta)
        v_one = (xp - x0) / beta
        d_center = B.T @ v_center
        d_one = B.T @ v_one
        # Solve the extended-space stationary equations rather than inserting the formula directly.
        m_dyad, d_dyad_state, dyad_residual = find_dyadic_stationary(beta, THETA, x0)
        B_dyad = parameter_jacobian_at_state(m_dyad, THETA)
        d_dyadic = B_dyad.T @ (d_dyad_state / beta)
        finite_rows.append({
            "beta": beta,
            "asymep_centered_relative_error": relerr(d_center, descent_ad),
            "asymep_one_sided_relative_error": relerr(d_one, descent_ad),
            "dyadic_stationary_relative_error": relerr(d_dyadic, descent_ad),
            "dyadic_mean_state_error": float(np.linalg.norm(m_dyad - x0)),
            "dyadic_stationarity_residual": dyad_residual,
            "plus_residual": float(np.linalg.norm(augmented_force(xp, beta, THETA, x0, J))),
            "minus_residual": float(np.linalg.norm(augmented_force(xm, -beta, THETA, x0, J))),
        })
        one_sided_errors.append((beta, relerr(d_one, descent_ad)))
    write_csv("finite_nudge_errors.csv", finite_rows)

    # Finite-time Euler integration: exact free equilibrium is initial condition for both phases.
    time_rows = []
    dt = 0.01
    beta = 0.1
    xp_star = find_asym_root(beta, THETA, x0, J)
    xm_star = find_asym_root(-beta, THETA, x0, J)
    v_star = (xp_star - xm_star) / (2.0 * beta)
    d_star_asym = B.T @ v_star
    for T in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        steps = int(round(T / dt))
        xp_t = euler_asym(beta, THETA, x0, J, dt, steps)
        xm_t = euler_asym(-beta, THETA, x0, J, dt, steps)
        d_t = B.T @ ((xp_t - xm_t) / (2.0 * beta))
        time_rows.append({
            "time": T,
            "dt": dt,
            "asymep_finite_time_relative_error_vs_autograd": relerr(d_t, descent_ad),
            "state_plus_error_to_equilibrium": float(np.linalg.norm(xp_t - xp_star)),
            "state_minus_error_to_equilibrium": float(np.linalg.norm(xm_t - xm_star)),
            "relative_error_to_asymep_finite_beta_limit": relerr(d_t, d_star_asym),
        })
    write_csv("finite_time_errors.csv", time_rows)

    # Integration-step sweep at fixed time; Euler scheme should converge with first order in dt.
    dt_rows = []
    T_final = 6.0
    beta = 0.1
    # High-accuracy continuous-time trajectory is the reference for isolating integrator error.
    def integrate_reference(b: float) -> np.ndarray:
        sol = solve_ivp(
            lambda t, x: augmented_force(x, b, THETA, x0, J),
            (0.0, T_final), x0, method="DOP853", rtol=1e-12, atol=1e-14,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]
    xp_ivp = integrate_reference(beta)
    xm_ivp = integrate_reference(-beta)
    d_ivp = B.T @ ((xp_ivp - xm_ivp) / (2.0 * beta))
    for h in [0.08, 0.04, 0.02, 0.01, 0.005]:
        steps = int(round(T_final / h))
        xp_h = euler_asym(beta, THETA, x0, J, h, steps)
        xm_h = euler_asym(-beta, THETA, x0, J, h, steps)
        d_h = B.T @ ((xp_h - xm_h) / (2.0 * beta))
        dt_rows.append({
            "dt": h,
            "time": T_final,
            "asymep_relative_error_vs_autograd": relerr(d_h, descent_ad),
            "relative_error_vs_root_based_asymep": relerr(d_h, d_star_asym),
            "relative_integrator_error_vs_high_accuracy_ivp": relerr(d_h, d_ivp),
        })
    write_csv("integration_step_errors.csv", dt_rows)

    # A stable but non-normal counterexample showing large transient amplification.
    a = 10.0
    N = np.array([[-1.0, a], [0.0, -1.0]])
    qn = np.array([0.0, 1.0])
    vf_n = np.linalg.solve(N, qn)
    true_n = np.linalg.solve(N.T, qn)
    ts = np.linspace(0.0, 5.0, 501)
    gains = []
    for t in ts:
        expNt = np.exp(-t) * np.array([[1.0, a * t], [0.0, 1.0]])
        gains.append(np.linalg.norm(expNt, 2))
    imax = int(np.argmax(gains))
    nonnormal = {
        "J": N.tolist(),
        "eigenvalues": [{"real": float(np.real(v)), "imag": float(np.imag(v))} for v in np.linalg.eigvals(N)],
        "condition_number_2": float(np.linalg.cond(N, 2)),
        "loss_gradient_q": qn.tolist(),
        "vector_field_response_J_inverse_q": vf_n.tolist(),
        "true_adjoint_response_J_transpose_inverse_q": true_n.tolist(),
        "response_relative_error": relerr(vf_n, true_n),
        "max_transient_gain_on_0_to_5": float(gains[imax]),
        "time_of_max_transient_gain": float(ts[imax]),
    }
    (RES / "nonnormal_counterexample.json").write_text(json.dumps(nonnormal, indent=2), encoding="utf-8")

    # Plot: error vs beta (log-log).
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog([r["beta"] for r in finite_rows], [r["asymep_centered_relative_error"] for r in finite_rows], "o-", label="AsymEP centered difference")
    ax.loglog([r["beta"] for r in finite_rows], [r["asymep_one_sided_relative_error"] for r in finite_rows], "s--", label="AsymEP one-sided difference")
    ax.loglog([r["beta"] for r in finite_rows], [r["dyadic_stationary_relative_error"] for r in finite_rows], "^-", label="Dyadic stationary update")
    ax.set_xlabel("nudging strength |beta|")
    ax.set_ylabel("relative error vs. autodiff descent")
    ax.set_title("Finite-nudge gradient error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "finite_nudge_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy([r["time"] for r in time_rows], [r["asymep_finite_time_relative_error_vs_autograd"] for r in time_rows], "o-", label="Finite-time AsymEP")
    ax.set_xlabel("relaxation time T")
    ax.set_ylabel("relative error vs. autodiff descent")
    ax.set_title("Finite-time relaxation error (Euler dt=0.01, beta=0.1)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "finite_time_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ts, gains)
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$||e^{Jt}||_2$")
    ax.set_title("Transient amplification in a stable non-normal system")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "nonnormal_transient.png", dpi=180)
    plt.close(fig)

    summary = {
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "J_eigenvalues": [{"real": float(np.real(v)), "imag": float(np.imag(v))} for v in eigs],
        "x_equilibrium": x0.tolist(),
        "loss_at_equilibrium": cost(x0),
        "state_gradient_q": q.tolist(),
        "autograd_loss_gradient": grad_ad.tolist(),
        "central_finite_difference_gradient": fd_grad.tolist(),
        "central_finite_difference_relative_error": relerr(fd_grad, grad_ad),
        "autograd_descent_direction": descent_ad.tolist(),
        "implicit_true_descent_direction": descent_implicit.tolist(),
        "dyadic_stationary_descent_direction": dyadic_descent.tolist(),
        "vector_field_descent_direction": vf_descent.tolist(),
        "implicit_relative_error": relerr(descent_implicit, descent_ad),
        "dyadic_relative_error": relerr(dyadic_descent, descent_ad),
        "vf_relative_error": relerr(vf_descent, descent_ad),
        "asymep_centered_error_orders": [
            {"beta": r["beta"], "relative_error": r["asymep_centered_relative_error"]} for r in finite_rows
        ],
        "empirical_log_log_order_centered_asymep": float(np.polyfit(
            np.log([r["beta"] for r in finite_rows]),
            np.log([r["asymep_centered_relative_error"] for r in finite_rows]), 1)[0]),
        "empirical_log_log_order_one_sided_asymep": float(np.polyfit(
            np.log([r["beta"] for r in finite_rows]),
            np.log([r["asymep_one_sided_relative_error"] for r in finite_rows]), 1)[0]),
        "dyadic_finite_beta_errors": [
            {"beta": r["beta"], "relative_error": r["dyadic_stationary_relative_error"]} for r in finite_rows
        ],
        "nonnormal": nonnormal,
        "assertions": {
            "implicit_matches_autograd": relerr(descent_implicit, descent_ad) < 1e-9,
            "central_finite_difference_matches_autograd": relerr(fd_grad, grad_ad) < 1e-7,
            "dyadic_matches_autograd": relerr(dyadic_descent, descent_ad) < 1e-9,
            "vf_not_exact_for_nonsymmetric_example": relerr(vf_descent, descent_ad) > 1e-3,
            "asymep_centered_error_decreases_as_beta_shrinks": finite_rows[-1]["asymep_centered_relative_error"] < finite_rows[0]["asymep_centered_relative_error"],
            "dyadic_finite_beta_exact_up_to_numerical_root_tolerance": max(r["dyadic_stationary_relative_error"] for r in finite_rows) < 1e-9,
            "stable_nonnormal_transient_gain_exceeds_one": max(gains) > 1.0,
        },
    }
    if not all(summary["assertions"].values()):
        raise AssertionError(f"Validation assertions failed: {summary['assertions']}")
    (RES / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj)), encoding="utf-8")

    print("=== Exact gradient comparison ===")
    for row in analytical_rows:
        print(row)
    print("\n=== Finite nudge ===")
    for row in finite_rows:
        print(row)
    print("\n=== Finite time ===")
    for row in time_rows:
        print(row)
    print("\n=== Non-normal counterexample ===")
    print(json.dumps(nonnormal, indent=2))
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
