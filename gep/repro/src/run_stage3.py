"""Stage 3: adaptive beta/T/h selection and tighter nonnormal EP error bounds.

This is a reproducible finite-dimensional benchmark, not a formal proof assistant.
The controller distinguishes theorem-backed inequalities from a-posteriori numerical
indicators. Run from project root: python src/run_stage3.py
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
SEED = 20261013
np.set_printoptions(precision=10, suppress=True)

# Nonlinear equilibrium benchmark reused from Stage 2.
A0 = np.array([[-2.0, 1.2, -0.4], [-0.3, -1.8, 0.5], [0.2, -0.7, -1.5]], dtype=float)
K = np.array([[0.0, 0.5, 0.1], [-0.2, 0.0, 0.3], [0.1, -0.4, 0.0]], dtype=float)
GAMMA = 0.25
L = np.array([[1.0, 0.2], [-0.4, 0.7], [0.3, -0.5]], dtype=float)
OFFSET = np.array([0.1, -0.2, 0.05], dtype=float)
THETA = np.array([0.2, -0.15], dtype=float)
TARGET = np.array([0.5, -0.1, 0.3], dtype=float)
QMAT = np.diag([1.0, 1.4, 0.8])
COSSCALE = 0.1
BETA_LEVELS = np.array([0.32, 0.16, 0.08, 0.04, 0.02], dtype=float)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (RES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def x_star(theta: np.ndarray) -> np.ndarray:
    return L @ theta + OFFSET


def force(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    d = x - x_star(theta)
    return A0 @ d + GAMMA * np.tanh(K @ d)


def free_jacobian(x0: np.ndarray) -> np.ndarray:
    # x0 is the exact free equilibrium, hence d=0 and tanh'(0)=1.
    return A0 + GAMMA * K


def cost_grad(x: np.ndarray) -> np.ndarray:
    d = x - TARGET
    return QMAT @ d + COSSCALE * np.sin(x)


def cost_hessian(x: np.ndarray) -> np.ndarray:
    return QMAT + COSSCALE * np.diag(np.cos(x))


def param_jacobian(A: np.ndarray) -> np.ndarray:
    return -A @ L


def true_descent(theta: np.ndarray) -> np.ndarray:
    return -L.T @ cost_grad(x_star(theta))


def anti_part(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A - A.T)


def augmented_force(x: np.ndarray, beta: float, theta: np.ndarray,
                    x0: np.ndarray, A: np.ndarray) -> np.ndarray:
    anti = anti_part(A)
    return force(x, theta) - beta * cost_grad(x) - 2.0 * anti @ (x - x0)


def augmented_jacobian(x: np.ndarray, beta: float, theta: np.ndarray,
                       x0: np.ndarray, A: np.ndarray) -> np.ndarray:
    d = x - x_star(theta)
    dvec = 1.0 - np.tanh(K @ d) ** 2
    return (A0 + GAMMA * np.diag(dvec) @ K
            - beta * cost_hessian(x) - 2.0 * anti_part(A))


def global_log_norm_upper(beta: float) -> float:
    """Global upper bound on mu_2(D_x f_beta) for this benchmark.

    In the augmented vector field, tanh' entries lie in [0,1] and cos(x_i) in
    [-1,1]. The symmetric part's largest eigenvalue is a convex function of
    these box variables, so its maximum over the box is attained at a vertex.
    Enumerating all vertices gives a conservative global upper bound.
    """
    vals: list[float] = []
    anti = anti_part(A0 + GAMMA * K)
    for d_bits in itertools.product([0.0, 1.0], repeat=3):
        D = np.diag(d_bits)
        for c_bits in itertools.product([-1.0, 1.0], repeat=3):
            C = np.diag(c_bits)
            J = A0 + GAMMA * D @ K - beta * (QMAT + COSSCALE * C) - 2.0 * anti
            S = 0.5 * (J + J.T)
            vals.append(float(np.linalg.eigvalsh(S)[-1]))
    return max(vals)


def rk4_step(fun: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float) -> np.ndarray:
    k1 = fun(y)
    k2 = fun(y + 0.5 * h * k1)
    k3 = fun(y + 0.5 * h * k2)
    k4 = fun(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def adaptive_rk4(beta: float, T: float, state_goal: float, *, h_initial: float = 1.1,
                 h_max: float = 1.1, h_min: float = 1e-7) -> dict:
    """RK4 step-doubling with contraction-weighted a-posteriori error indicator.

    The difference between one full RK4 step and two half steps divided by 15
    estimates the local error of the fine RK4 solution. A global log-norm bound
    weights accumulated local indicators. Because the embedded difference is an
    estimator (not a validated interval enclosure), the resulting error value is
    an a-posteriori numerical indicator, not an unconditional rigorous bound.
    """
    theta = THETA.copy()
    x0 = x_star(theta)
    A = free_jacobian(x0)
    mu = global_log_norm_upper(beta)
    fun = lambda y: augmented_force(y, beta, theta, x0, A)
    y = x0.copy()
    t = 0.0
    h = min(h_initial, h_max, T)
    weighted_error = 0.0
    accepted = rejected = 0
    max_est_local = 0.0
    min_accepted_h = float("inf")
    max_accepted_h = 0.0
    last_accepted_h = 0.0
    while t < T - 1e-14:
        h = min(h, h_max, T - t)
        full = rk4_step(fun, y, h)
        half = rk4_step(fun, y, h / 2.0)
        fine = rk4_step(fun, half, h / 2.0)
        local_est = float(np.linalg.norm(fine - full) / 15.0)
        t_new = t + h
        # Allocate final-time state-error budget proportional to h/T, undoing
        # the log-norm propagation weight exp(mu*(T-t_new)).
        exponent = float(np.clip(-mu * (T - t_new), -50.0, 50.0))
        local_allowed = state_goal * (h / T) * math.exp(exponent)
        # If the field is contractive, existing endpoint error contracts by e^(mu h).
        # This recurrence estimates the propagated global error at t_new.
        predicted_weighted = math.exp(float(np.clip(mu * h, -50.0, 50.0))) * weighted_error + local_est
        if local_est <= local_allowed and np.all(np.isfinite(fine)):
            weighted_error = predicted_weighted
            y = fine
            t = t_new
            accepted += 1
            max_est_local = max(max_est_local, local_est)
            min_accepted_h = min(min_accepted_h, h)
            max_accepted_h = max(max_accepted_h, h)
            last_accepted_h = h
            if local_est == 0.0:
                fac = 2.0
            else:
                fac = 0.9 * (local_allowed / local_est) ** 0.2
            h *= float(np.clip(fac, 0.25, 2.0))
        else:
            rejected += 1
            if h <= h_min:
                raise RuntimeError(
                    f"RK4 controller hit h_min; beta={beta}, T={T}, "
                    f"local_est={local_est:g}, allowed={local_allowed:g}"
                )
            ratio = max(local_allowed / max(local_est, 1e-300), 1e-12)
            h *= float(np.clip(0.85 * ratio ** 0.2, 0.1, 0.7))
    residual = float(np.linalg.norm(fun(y)))
    return {
        "state": y,
        "state_error_indicator": float(weighted_error),
        "field_residual": residual,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "max_local_error_estimate": max_est_local,
        "log_norm_upper": mu,
        "min_accepted_h": min_accepted_h if np.isfinite(min_accepted_h) else 0.0,
        "max_accepted_h": max_accepted_h,
        "last_accepted_h": last_accepted_h,
        "next_proposed_h": h,
    }


def fixed_rk4_integrate(beta: float, T: float, h: float) -> tuple[np.ndarray, int]:
    theta = THETA.copy()
    x0 = x_star(theta)
    A = free_jacobian(x0)
    fun = lambda y: augmented_force(y, beta, theta, x0, A)
    y = x0.copy()
    t = 0.0
    steps = 0
    while t < T - 1e-14:
        hh = min(h, T - t)
        y = rk4_step(fun, y, hh)
        t += hh
        steps += 1
    return y, steps


def dop853_reference(beta: float, T: float) -> tuple[np.ndarray, int]:
    theta = THETA.copy()
    x0 = x_star(theta)
    A = free_jacobian(x0)
    fun = lambda t, y: augmented_force(y, beta, theta, x0, A)
    sol = solve_ivp(fun, (0.0, T), x0, method="DOP853", rtol=2e-13, atol=2e-15,
                    max_step=0.02)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1], int(sol.nfev), int(len(sol.t) - 1)


def run_phase_pair(beta: float, T: float, grad_num_budget: float,
                   B: np.ndarray) -> dict:
    x0 = x_star(THETA)
    A = free_jacobian(x0)
    BT_norm = float(np.linalg.norm(B.T, 2))
    state_goal = grad_num_budget * beta / max(BT_norm, 1e-15)
    plus = adaptive_rk4(beta, T, state_goal)
    minus = adaptive_rk4(-beta, T, state_goal)
    direction = B.T @ ((plus["state"] - minus["state"]) / (2.0 * beta))
    numerical_indicator = BT_norm * (plus["state_error_indicator"] + minus["state_error_indicator"]) / (2.0 * beta)
    return {
        "direction": direction,
        "positive": plus,
        "negative": minus,
        "numerical_gradient_indicator": float(numerical_indicator),
        "steps_total": int(plus["accepted_steps"] + minus["accepted_steps"]),
        "rejections_total": int(plus["rejected_steps"] + minus["rejected_steps"]),
        "state_goal_per_phase": state_goal,
    }


def directional_linear_time_tail(A: np.ndarray, B: np.ndarray, q: np.ndarray, T: float) -> float:
    """Exact finite-time gradient tail for the first-order AsymEP response."""
    rhs = expm(A.T * T) @ q
    return float(np.linalg.norm(B.T @ np.linalg.solve(A.T, rhs)))


def choose_directional_start_time(A: np.ndarray, B: np.ndarray, q: np.ndarray,
                                  tolerance: float, *, max_T: float = 16.0,
                                  dt: float = 0.01) -> tuple[float, dict]:
    times = np.arange(0.0, max_T + dt / 2.0, dt)
    tails = np.array([directional_linear_time_tail(A, B, q, float(t)) for t in times])
    # Suffix maxima make this robust to non-monotone transient growth on the scan grid.
    suffix = np.maximum.accumulate(tails[::-1])[::-1]
    candidates = np.flatnonzero(suffix <= tolerance)
    if not len(candidates):
        raise RuntimeError("No directional time satisfying requested tolerance found")
    i = int(candidates[0])
    return float(times[i]), {
        "time_grid_step": dt,
        "chosen_T": float(times[i]),
        "directional_tail_at_T": float(tails[i]),
        "sampled_suffix_max_tail": float(suffix[i]),
        "global_sampled_max_tail_after_T": float(np.max(tails[i:])),
        "note": "Grid-checked first-order directional indicator; not a continuous-time validated bound for the nonlinear finite-beta system.",
    }


def estimate_nudge_order(directions: list[np.ndarray]) -> list[dict]:
    deltas = [float(np.linalg.norm(directions[i] - directions[i + 1]))
              for i in range(len(directions) - 1)]
    rows: list[dict] = []
    for i in range(len(deltas) - 1):
        if deltas[i + 1] > 1e-16 and deltas[i] > 0:
            p = float(np.log2(deltas[i] / deltas[i + 1]))
        else:
            p = None
        rows.append({
            "coarse_beta": float(BETA_LEVELS[i]),
            "middle_beta": float(BETA_LEVELS[i + 1]),
            "fine_beta": float(BETA_LEVELS[i + 2]),
            "difference_coarse_middle": deltas[i],
            "difference_middle_fine": deltas[i + 1],
            "observed_order": p,
        })
    return rows


def choose_beta(directions: list[np.ndarray], numerical_indicators: list[float],
                time_indicators: list[float], *, BT_norm: float,
                state_uncertainty_per_phase: float) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    deltas = [float(np.linalg.norm(directions[i] - directions[i + 1]))
              for i in range(len(directions) - 1)]
    orders = {}
    for i in range(len(deltas) - 1):
        if deltas[i + 1] > 1e-16 and deltas[i] > 0:
            orders[i + 1] = float(np.log2(deltas[i] / deltas[i + 1]))
    for j in range(1, len(BETA_LEVELS)):
        delta_coarse_fine = deltas[j - 1]
        p = orders.get(j)
        if p is not None and 1.4 <= p <= 2.6:
            nudge = delta_coarse_fine / (2.0 ** p - 1.0)
            order_status = "estimated from adjacent scale differences"
        elif p is not None:
            # Conservative indicator if second-order asymptotics do not look established.
            nudge = delta_coarse_fine
            order_status = "order outside [1.4,2.6]; use full adjacent difference conservatively"
        else:
            nudge = delta_coarse_fine / 3.0
            order_status = "assume second order; no finer order check available"
        beta = float(BETA_LEVELS[j])
        # If each phase has an unknown state/measurement error bounded by eps,
        # ||B^T (xi+ - xi-)/(2 beta)|| <= ||B^T|| eps / beta.
        uncertainty = BT_norm * state_uncertainty_per_phase / beta
        score = float(nudge + numerical_indicators[j] + time_indicators[j] + uncertainty)
        rows.append({
            "beta_candidate": beta,
            "adjacent_difference": delta_coarse_fine,
            "observed_order": p,
            "nudge_error_indicator": float(nudge),
            "time_error_indicator": float(time_indicators[j]),
            "numerical_error_indicator": float(numerical_indicators[j]),
            "state_uncertainty_indicator": float(uncertainty),
            "combined_error_indicator": score,
            "order_status": order_status,
        })
    best = min(rows, key=lambda r: r["combined_error_indicator"])
    return best, rows


def study_adaptive_controller() -> dict:
    theta = THETA.copy()
    x0 = x_star(theta)
    A = free_jacobian(x0)
    B = param_jacobian(A)
    q = cost_grad(x0)
    exact = true_descent(theta)
    norm_BT = float(np.linalg.norm(B.T, 2))
    total_tolerance = 1.2e-3
    time_budget = 2.5e-4
    numerical_budget = 1.5e-4
    time_check_tolerance = 4.0e-4
    # Configured per-phase bound for residual sensor/model/state uncertainty.
    # It is deliberately separate from the adaptive ODE solver's numerical indicator.
    state_uncertainty_per_phase = 5.0e-6
    # Use the tighter task-direction first-order tail to seed T; the actual
    # finite-beta trajectory is then checked a posteriori at T and 1.25*T.
    T0, time_seed = choose_directional_start_time(A, B, q, time_budget, max_T=16.0, dt=0.01)
    pair_cache: dict[tuple[float, float], dict] = {}

    def get_pair(beta: float, T: float) -> dict:
        key = (round(float(beta), 10), round(float(T), 10))
        if key not in pair_cache:
            pair_cache[key] = run_phase_pair(beta, T, numerical_budget, B)
        return pair_cache[key]

    # Increase T geometrically if the observed finite-beta output has not settled.
    base_T = T0
    time_history: list[dict] = []
    final_T = None
    final_Tprev = None
    final_data = None
    for attempt in range(6):
        next_T = 1.25 * base_T
        mid_pairs = [get_pair(float(b), base_T) for b in BETA_LEVELS]
        end_pairs = [get_pair(float(b), next_T) for b in BETA_LEVELS]
        time_diffs = [float(np.linalg.norm(end_pairs[i]["direction"] - mid_pairs[i]["direction"]))
                      for i in range(len(BETA_LEVELS))]
        max_time_diff = max(time_diffs)
        analytic_tail = directional_linear_time_tail(A, B, q, next_T)
        ok = max_time_diff <= time_check_tolerance and analytic_tail <= time_budget
        time_history.append({
            "attempt": attempt + 1,
            "base_T": base_T,
            "candidate_output_T": next_T,
            "max_observed_gradient_change": max_time_diff,
            "linear_directional_tail_at_output_T": analytic_tail,
            "time_check_tolerance": time_check_tolerance,
            "passed": bool(ok),
        })
        if ok:
            final_Tprev, final_T, final_data = base_T, next_T, end_pairs
            break
        base_T = next_T
    if final_data is None or final_T is None or final_Tprev is None:
        raise RuntimeError("Adaptive T controller did not meet the time criteria within six attempts")

    directions = [p["direction"] for p in final_data]
    num_ind = [p["numerical_gradient_indicator"] for p in final_data]
    time_ind = [float(np.linalg.norm(final_data[i]["direction"] - get_pair(float(BETA_LEVELS[i]), final_Tprev)["direction"]))
                for i in range(len(BETA_LEVELS))]
    best, candidates = choose_beta(
        directions, num_ind, time_ind, BT_norm=norm_BT,
        state_uncertainty_per_phase=state_uncertainty_per_phase,
    )
    best_idx = int(np.where(np.isclose(BETA_LEVELS, best["beta_candidate"]))[0][0])
    selected_beta = float(BETA_LEVELS[best_idx])
    selected_pair = final_data[best_idx]
    estimated_direction = selected_pair["direction"]
    actual_error = float(np.linalg.norm(estimated_direction - exact))
    actual_relative_error = actual_error / max(float(np.linalg.norm(exact)), 1e-15)
    selected_error_indicator = float(best["combined_error_indicator"])
    selected_time_tail = directional_linear_time_tail(A, B, q, final_T)
    selected_num_indicator = num_ind[best_idx]
    estimated_h_max = max(selected_pair["positive"]["max_accepted_h"], selected_pair["negative"]["max_accepted_h"])
    estimated_h_min = min(selected_pair["positive"]["min_accepted_h"], selected_pair["negative"]["min_accepted_h"])
    # Cross-check adaptive RK4 against a high-accuracy DOP853 reference and fixed-step RK4.
    ref_plus, ref_plus_nfev, ref_plus_intervals = dop853_reference(selected_beta, final_T)
    ref_minus, ref_minus_nfev, ref_minus_intervals = dop853_reference(-selected_beta, final_T)
    fixed_h = 0.05
    fixed_plus, fixed_plus_steps = fixed_rk4_integrate(selected_beta, final_T, fixed_h)
    fixed_minus, fixed_minus_steps = fixed_rk4_integrate(-selected_beta, final_T, fixed_h)
    adaptive_plus = selected_pair["positive"]["state"]
    adaptive_minus = selected_pair["negative"]["state"]
    ref_direction = B.T @ ((ref_plus - ref_minus) / (2.0 * selected_beta))
    fixed_direction = B.T @ ((fixed_plus - fixed_minus) / (2.0 * selected_beta))
    adaptive_direction = selected_pair["direction"]
    integrator_comparison = {
        "selected_beta": selected_beta,
        "selected_T": final_T,
        "adaptive_rk4_steps_both_phases": selected_pair["steps_total"],
        "adaptive_rk4_rejections_both_phases": selected_pair["rejections_total"],
        "adaptive_rk4_rhs_evaluations_estimate": 12 * (selected_pair["steps_total"] + selected_pair["rejections_total"]),
        "adaptive_min_accepted_h": estimated_h_min,
        "adaptive_max_accepted_h": estimated_h_max,
        "fixed_rk4_h": fixed_h,
        "fixed_rk4_steps_both_phases": fixed_plus_steps + fixed_minus_steps,
        "fixed_rk4_rhs_evaluations": 4 * (fixed_plus_steps + fixed_minus_steps),
        "dop853_reference_rhs_evaluations": ref_plus_nfev + ref_minus_nfev,
        "dop853_reference_accepted_intervals": ref_plus_intervals + ref_minus_intervals,
        "adaptive_positive_state_error_vs_dop853": float(np.linalg.norm(adaptive_plus - ref_plus)),
        "adaptive_negative_state_error_vs_dop853": float(np.linalg.norm(adaptive_minus - ref_minus)),
        "fixed_positive_state_error_vs_dop853": float(np.linalg.norm(fixed_plus - ref_plus)),
        "fixed_negative_state_error_vs_dop853": float(np.linalg.norm(fixed_minus - ref_minus)),
        "adaptive_gradient_estimate_error_vs_dop853_estimate": float(np.linalg.norm(adaptive_direction - ref_direction)),
        "fixed_gradient_estimate_error_vs_dop853_estimate": float(np.linalg.norm(fixed_direction - ref_direction)),
        "adaptive_gradient_estimate_error_indicator": float(selected_pair["numerical_gradient_indicator"]),
        "warning": "DOP853 is a high-accuracy numerical reference, not an exact solution; its reported step count is solver internal output interval count, not a directly comparable RHS-call count.",
    }
    (RES / "integrator_comparison.json").write_text(json.dumps(integrator_comparison, indent=2), encoding="utf-8")
    # Monte-Carlo stress test of the configured bounded per-phase state uncertainty.
    rng_noise = np.random.default_rng(SEED + 303)
    monte_carlo_rows = []
    noise_trials = 500
    for i, b in enumerate(BETA_LEVELS):
        pair = final_data[i]
        clean = pair["direction"]
        noisy_errors = []
        noise_bound = norm_BT * state_uncertainty_per_phase / float(b)
        for _ in range(noise_trials):
            eps_plus = rng_noise.normal(size=x0.size)
            eps_minus = rng_noise.normal(size=x0.size)
            eps_plus *= state_uncertainty_per_phase / np.linalg.norm(eps_plus)
            eps_minus *= state_uncertainty_per_phase / np.linalg.norm(eps_minus)
            noisy_direction = B.T @ ((pair["positive"]["state"] + eps_plus
                                       - pair["negative"]["state"] - eps_minus) / (2.0 * float(b)))
            noisy_errors.append(float(np.linalg.norm(noisy_direction - exact)))
            if np.linalg.norm(noisy_direction - clean) > noise_bound + 1e-10:
                raise AssertionError("Bounded state-noise inequality was violated")
        monte_carlo_rows.append({
            "beta": float(b),
            "noise_trials": noise_trials,
            "state_uncertainty_per_phase": state_uncertainty_per_phase,
            "worst_case_noise_bound": noise_bound,
            "clean_error_vs_exact": float(np.linalg.norm(clean - exact)),
            "noisy_rms_error_vs_exact": float(np.sqrt(np.mean(np.square(noisy_errors)))),
            "noisy_max_error_vs_exact": float(np.max(noisy_errors)),
            "triangle_bound_clean_plus_noise": float(np.linalg.norm(clean - exact) + noise_bound),
            "all_noise_trials_within_triangle_bound": bool(max(noisy_errors) <= np.linalg.norm(clean - exact) + noise_bound + 1e-10),
        })
    write_csv("adaptive_noise_stress.csv", monte_carlo_rows)
    beta_rows = []
    for i, b in enumerate(BETA_LEVELS):
        pair = final_data[i]
        noise_row = monte_carlo_rows[i]
        beta_rows.append({
            "beta": float(b),
            "gradient_estimate_norm": float(np.linalg.norm(pair["direction"])),
            "error_vs_analytic_gradient": float(np.linalg.norm(pair["direction"] - exact)),
            "relative_error_vs_analytic_gradient": float(np.linalg.norm(pair["direction"] - exact) / max(np.linalg.norm(exact), 1e-15)),
            "integration_error_indicator": float(pair["numerical_gradient_indicator"]),
            "state_uncertainty_per_phase": state_uncertainty_per_phase,
            "worst_case_state_uncertainty_gradient_bound": float(norm_BT * state_uncertainty_per_phase / float(b)),
            "noisy_rms_error_vs_analytic_gradient": noise_row["noisy_rms_error_vs_exact"],
            "steps_total": pair["steps_total"],
            "rejections_total": pair["rejections_total"],
            "positive_residual": pair["positive"]["field_residual"],
            "negative_residual": pair["negative"]["field_residual"],
            "positive_state_error_indicator": pair["positive"]["state_error_indicator"],
            "negative_state_error_indicator": pair["negative"]["state_error_indicator"],
            "minimum_accepted_step": min(pair["positive"]["min_accepted_h"], pair["negative"]["min_accepted_h"]),
            "maximum_accepted_step": max(pair["positive"]["max_accepted_h"], pair["negative"]["max_accepted_h"]),
            "log_norm_upper_positive": pair["positive"]["log_norm_upper"],
            "log_norm_upper_negative": pair["negative"]["log_norm_upper"],
        })
    write_csv("adaptive_beta_candidates.csv", beta_rows)
    write_csv("adaptive_beta_scores.csv", candidates)
    write_csv("adaptive_T_history.csv", time_history)
    write_csv("adaptive_nudge_order.csv", estimate_nudge_order(directions))

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    attempts = np.arange(1, len(time_history) + 1)
    ax.semilogy(attempts, [max(r["max_observed_gradient_change"], 1e-16) for r in time_history], "o-", label="observed change between T and 1.25T")
    ax.semilogy(attempts, [max(r["linear_directional_tail_at_output_T"], 1e-16) for r in time_history], "s--", label="linearized directional tail")
    ax.axhline(time_check_tolerance, linestyle=":", label="observed-change tolerance")
    ax.set_xlabel("time-controller attempt")
    ax.set_ylabel("gradient change / first-order tail")
    ax.set_title("Adaptive relaxation-time selection")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "adaptive_T_selection.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(BETA_LEVELS, [r["minimum_accepted_step"] for r in beta_rows], "o-", label="minimum accepted h")
    ax.plot(BETA_LEVELS, [r["maximum_accepted_step"] for r in beta_rows], "s--", label="maximum accepted h")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("nudging strength beta (large to small)")
    ax.set_ylabel("accepted RK4 step size h")
    ax.set_title("Adaptive integration step sizes by nudge level")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "adaptive_h_by_beta.png", dpi=180)
    plt.close(fig)

    # Convergence/order plot and selected estimate vs beta.
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.loglog(BETA_LEVELS, [max(np.linalg.norm(d - exact), 1e-14) for d in directions], "o-", label="clean error vs exact gradient (post-run audit)")
    ax.loglog(BETA_LEVELS, [max(r["noisy_rms_error_vs_exact"], 1e-14) for r in monte_carlo_rows], "d-.", label="RMS error with bounded state noise")
    deltas = [np.linalg.norm(directions[i] - directions[i + 1]) for i in range(len(directions) - 1)]
    ax.loglog(BETA_LEVELS[1:], [max(v, 1e-14) for v in deltas], "s--", label="adjacent-scale difference")
    score_by_beta = {float(r["beta_candidate"]): float(r["combined_error_indicator"]) for r in candidates}
    ax.loglog(BETA_LEVELS[1:], [score_by_beta[float(b)] for b in BETA_LEVELS[1:]], "^:", label="adaptive error-budget score")
    ax.axvline(selected_beta, linestyle=":", label=f"selected beta={selected_beta:g}")
    ax.set_xlabel("nudging strength beta")
    ax.set_ylabel("gradient error / scale difference")
    ax.set_title("Adaptive beta selection: truncation vs numerical uncertainty")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "adaptive_beta_selection.png", dpi=180)
    plt.close(fig)

    summary = {
        "seed": SEED,
        "true_gradient_descent_direction": exact.tolist(),
        "true_gradient_norm": float(np.linalg.norm(exact)),
        "A_free": A.tolist(),
        "B_parameter_jacobian": B.tolist(),
        "q_loss_state_gradient": q.tolist(),
        "global_log_norm_upper_beta_zero": global_log_norm_upper(0.0),
        "time_tolerance_for_directional_seed": time_budget,
        "time_seed": time_seed,
        "time_controller_tolerance": time_check_tolerance,
        "time_controller_history": time_history,
        "selected_T_previous_check": final_Tprev,
        "selected_T_output": final_T,
        "selected_directional_linear_tail": selected_time_tail,
        "numerical_gradient_budget": numerical_budget,
        "state_uncertainty_per_phase": state_uncertainty_per_phase,
        "selected_beta": selected_beta,
        "selected_beta_score": best,
        "selected_h_max_accepted": estimated_h_max,
        "selected_h_min_accepted": estimated_h_min,
        "selected_state_uncertainty_gradient_bound": norm_BT * state_uncertainty_per_phase / selected_beta,
        "selected_estimated_direction": estimated_direction.tolist(),
        "selected_absolute_error_vs_exact_gradient": actual_error,
        "selected_relative_error_vs_exact_gradient": actual_relative_error,
        "selected_numerical_error_indicator": selected_num_indicator,
        "selected_time_change_indicator": time_ind[best_idx],
        "selected_total_indicator": selected_error_indicator,
        "all_candidate_rows": beta_rows,
        "beta_score_rows": candidates,
        "noise_stress_rows": monte_carlo_rows,
        "integrator_comparison": integrator_comparison,
        "interpretation": [
            "T seeding uses exact matrix-exponential first-order directional tail evaluated on a time grid.",
            "T is then extended until both the directional tail and observed finite-beta gradient changes pass thresholds.",
            "Beta uses central-difference scale comparisons plus a configured bounded per-phase state-uncertainty budget; the order-based nudge estimate is an a-posteriori indicator.",
            "A 500-trial bounded-noise stress test checks the state-noise triangle bound on the selected set of beta values.",
            "RK4 step-doubling estimates local error; contraction-weighted totals rely on the embedded estimator and are not validated interval bounds.",
            "The reference true gradient is used only for post-run evaluation, not by the adaptive controller."
        ],
    }
    (RES / "adaptive_controller_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def study_tighter_nonnormal_bounds() -> dict:
    """Compare directional and operator bounds for stable nonnormal systems."""
    rows: list[dict] = []
    curves: dict[str, list] = {}
    times = np.linspace(0.0, 10.0, 501)
    for k in [1.0, 2.0, 5.0, 10.0, 20.0]:
        A = np.array([[-1.0, k], [0.0, -0.1]])
        B = np.eye(2)
        AinvT = np.linalg.inv(A.T)
        _, _, Vh = np.linalg.svd(AinvT)
        q = Vh[-1, :].copy()
        q /= np.linalg.norm(q)
        q_basis = q[:, None]
        lyap_P = solve_continuous_lyapunov(A, -np.eye(2))
        lyap_P = 0.5 * (lyap_P + lyap_P.T)
        peigs = np.linalg.eigvalsh(lyap_P)
        lam = 1.0 / (2.0 * float(peigs.max()))
        kappa_p_sqrt = math.sqrt(float(peigs.max() / peigs.min()))
        curve = []
        for t in times:
            E = expm(A.T * float(t))
            # The actual q-specific linearized gradient tail.
            direction = float(np.linalg.norm(B.T @ AinvT @ E @ q))
            # Exact induced norm over the full q-space; tighter than product of norms.
            full_operator = float(np.linalg.norm(B.T @ AinvT @ E, 2))
            # Exact worst case over the one-dimensional task subspace span(q).
            subspace_operator = float(np.linalg.norm(B.T @ AinvT @ E @ q_basis, 2))
            product_bound = float(np.linalg.norm(B.T @ AinvT, 2) * np.linalg.norm(E, 2) * np.linalg.norm(q))
            lyap_semigroup = kappa_p_sqrt * math.exp(-lam * float(t))
            lyap_bound = float(np.linalg.norm(B.T @ AinvT, 2) * lyap_semigroup * np.linalg.norm(q))
            curve.append({
                "time": float(t),
                "directional_error": direction,
                "full_operator_bound": full_operator,
                "task_subspace_bound": subspace_operator,
                "product_operator_bound": product_bound,
                "lyapunov_bound": lyap_bound,
                "semigroup_gain": float(np.linalg.norm(E, 2)),
            })
        # Check pointwise containment relationships, with floating-point margin.
        tol = 1e-9
        valid_direction = all(c["directional_error"] <= c["full_operator_bound"] + tol for c in curve)
        valid_full_vs_product = all(c["full_operator_bound"] <= c["product_operator_bound"] + tol for c in curve)
        valid_task_subspace = all(c["task_subspace_bound"] <= c["full_operator_bound"] + tol for c in curve)
        valid_lyap = all(c["directional_error"] <= c["lyapunov_bound"] + tol for c in curve)
        idx = int(np.argmax([c["product_operator_bound"] / max(c["directional_error"], 1e-15) for c in curve]))
        row = {
            "k": k,
            "eigenvalues": [-1.0, -0.1],
            "condition_number_A": float(np.linalg.cond(A, 2)),
            "q_direction": q.tolist(),
            "log_norm_mu2": float(np.linalg.eigvalsh(0.5 * (A + A.T))[-1]),
            "max_semigroup_gain_0_10": max(c["semigroup_gain"] for c in curve),
            "max_product_bound_over_directional_error": max(c["product_operator_bound"] / max(c["directional_error"], 1e-15) for c in curve),
            "max_full_operator_over_directional_error": max(c["full_operator_bound"] / max(c["directional_error"], 1e-15) for c in curve),
            "max_lyapunov_bound_over_directional_error": max(c["lyapunov_bound"] / max(c["directional_error"], 1e-15) for c in curve),
            "time_at_worst_product_gap": curve[idx]["time"],
            "directional_below_full_operator_all_times": valid_direction,
            "full_operator_below_product_bound_all_times": valid_full_vs_product,
            "subspace_bound_below_full_operator_all_times": valid_task_subspace,
            "directional_below_lyapunov_bound_all_times": valid_lyap,
        }
        rows.append(row)
        curves[str(int(k))] = curve
    with (RES / "nonnormal_bound_tightness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RES / "nonnormal_bound_curves.json").write_text(json.dumps(curves), encoding="utf-8")

    c = curves["10"]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for key, label, style in [
        ("directional_error", "task-direction exact linearized tail", "-"),
        ("full_operator_bound", "exact full-space operator norm", "--"),
        ("product_operator_bound", "product-of-norms bound", ":"),
        ("lyapunov_bound", "Lyapunov envelope", "-."),
    ]:
        ax.semilogy([r["time"] for r in c], [max(r[key], 1e-16) for r in c], style, label=label)
    ax.set_xlabel("time T")
    ax.set_ylabel("finite-time gradient error / bound")
    ax.set_title("Tightness of nonnormal finite-time gradient bounds (k=10)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "nonnormal_bound_tightness.png", dpi=180)
    plt.close(fig)

    result = {
        "all_bound_relations_pass": bool(all(r["directional_below_full_operator_all_times"]
                                               and r["full_operator_below_product_bound_all_times"]
                                               and r["subspace_bound_below_full_operator_all_times"]
                                               and r["directional_below_lyapunov_bound_all_times"] for r in rows)),
        "rows": rows,
        "interpretation": [
            "The q-direction quantity is the exact first-order finite-time gradient tail for the chosen task direction.",
            "The full-space operator norm is the exact worst-case linearized tail over unit q; it is no larger than the product-of-norms bound.",
            "The one-dimensional subspace span(q) bound equals the directional quantity; more generally, a restricted gradient subspace U uses ||B^T A^{-T} exp(A^T T) U||.",
            "The Lyapunov envelope is valid under the stated Hurwitz and Lyapunov assumptions, but can be conservative.",
        ],
    }
    (RES / "nonnormal_bound_tightness_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    adaptive = study_adaptive_controller()
    bounds = study_tighter_nonnormal_bounds()
    assertions = {
        "adaptive_beta_in_candidate_set": adaptive["selected_beta"] in list(BETA_LEVELS[1:]),
        "adaptive_T_passed_observed_checks": bool(adaptive["time_controller_history"][-1]["passed"]),
        "adaptive_direction_finite": bool(np.all(np.isfinite(adaptive["selected_estimated_direction"]))),
        "adaptive_direction_error_finite": bool(np.isfinite(adaptive["selected_absolute_error_vs_exact_gradient"])),
        "selected_numerical_indicator_within_budget_factor": adaptive["selected_numerical_error_indicator"] <= 2.0 * adaptive["numerical_gradient_budget"],
        "adaptive_integrator_agrees_with_dop853": adaptive["integrator_comparison"]["adaptive_gradient_estimate_error_vs_dop853_estimate"] <= 10.0 * adaptive["integrator_comparison"]["adaptive_gradient_estimate_error_indicator"] + 1e-7,
        "noise_triangle_bounds_pass": all(r["all_noise_trials_within_triangle_bound"] for r in adaptive["noise_stress_rows"]),
        "directional_and_operator_bounds_pass": bounds["all_bound_relations_pass"],
        "non_normal_conditioning_visible": max(r["condition_number_A"] for r in bounds["rows"]) > 100.0,
    }
    summary = {
        "stage": 3,
        "seed": SEED,
        "assertions": assertions,
        "all_assertions_passed": bool(all(assertions.values())),
        "adaptive_controller": adaptive,
        "integrator_comparison": adaptive["integrator_comparison"],
        "nonnormal_bounds": bounds,
    }
    (RES / "stage3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "assertions": assertions,
        "selected_beta": adaptive["selected_beta"],
        "selected_T": adaptive["selected_T_output"],
        "selected_abs_error": adaptive["selected_absolute_error_vs_exact_gradient"],
        "selected_rel_error": adaptive["selected_relative_error_vs_exact_gradient"],
        "selected_error_indicator": adaptive["selected_total_indicator"],
        "max_product_bound_gap_k10": next(r["max_product_bound_over_directional_error"] for r in bounds["rows"] if r["k"] == 10.0),
    }, indent=2))
    if not all(assertions.values()):
        raise AssertionError(f"Stage 3 assertions failed: {assertions}")
    print("STAGE 3 ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
