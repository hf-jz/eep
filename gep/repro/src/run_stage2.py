"""Stage 2 quantitative audit for generalized equilibrium propagation.

Studies:
1. Necessary/sufficient VF exactness over a restricted loss-gradient subspace.
2. Joint nudge/noise trade-off and empirical optimal beta.
3. Decomposition of finite-nudge, finite-time, integration, and state-noise errors.
4. Nonnormal transient amplification and a Lyapunov-based decay envelope.

Run from project root:
    python src/run_stage2.py
Requires numpy, scipy, matplotlib. The script is deterministic under its fixed seed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 20261012
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=8, suppress=True)

# Reuse the exact baseline model from Stage 1, but keep this script standalone.
A0 = np.array([[-2.0, 1.2, -0.4], [-0.3, -1.8, 0.5], [0.2, -0.7, -1.5]], dtype=float)
K = np.array([[0.0, 0.5, 0.1], [-0.2, 0.0, 0.3], [0.1, -0.4, 0.0]], dtype=float)
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
    d = x - x_star(theta)
    return A0 @ d + GAMMA * np.tanh(K @ d)


def jacobian(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    d = x - x_star(theta)
    return A0 + GAMMA * np.diag(1.0 - np.tanh(K @ d) ** 2) @ K


def cost_grad(x: np.ndarray) -> np.ndarray:
    d = x - TARGET
    return QMAT @ d + COSSCALE * np.sin(x)


def param_jacobian(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return -jacobian(x, theta) @ L


def descent_true(theta: np.ndarray) -> np.ndarray:
    """Analytic gradient direction since x*(theta)=L theta + offset."""
    x = x_star(theta)
    q = cost_grad(x)
    return -L.T @ q


def augmented_force(x: np.ndarray, beta: float, theta: np.ndarray,
                    x0: np.ndarray, A_free: np.ndarray) -> np.ndarray:
    anti = 0.5 * (A_free - A_free.T)
    return force(x, theta) - beta * cost_grad(x) - 2.0 * anti @ (x - x0)


def root_asym(beta: float, theta: np.ndarray, x0: np.ndarray,
              A_free: np.ndarray) -> tuple[np.ndarray, float]:
    fun = lambda x: augmented_force(x, beta, theta, x0, A_free)
    sol = root(fun, x0, method="hybr", tol=1e-12)
    residual = float(np.linalg.norm(fun(sol.x)))
    if (not sol.success and residual > 1e-9) or residual > 1e-8:
        raise RuntimeError(f"AsymEP root failed: {sol.message}; residual={residual:g}")
    return sol.x, residual


def integrate(beta: float, theta: np.ndarray, x0: np.ndarray, A_free: np.ndarray,
              T: float, *, method: str, h: float | None = None) -> tuple[np.ndarray, float]:
    f = lambda t, x: augmented_force(x, beta, theta, x0, A_free)
    if T == 0:
        return x0.copy(), 0.0
    if method == "DOP853":
        sol = solve_ivp(f, (0.0, T), x0, method="DOP853", rtol=2e-12, atol=2e-14,
                        max_step=min(0.05, T))
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1], float(np.linalg.norm(f(T, sol.y[:, -1])))
    if method == "Euler":
        if h is None or h <= 0:
            raise ValueError("Euler needs positive h")
        n_steps = int(round(T / h))
        if abs(n_steps * h - T) > 1e-10:
            raise ValueError("T must be an integer multiple of h in this study")
        x = x0.copy()
        for _ in range(n_steps):
            x = x + h * f(0.0, x)
            if not np.all(np.isfinite(x)):
                raise FloatingPointError("Euler diverged")
        return x, float(np.linalg.norm(f(T, x)))
    raise ValueError(method)


def root_centered_direction(beta: float, theta: np.ndarray, x0: np.ndarray,
                             A_free: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xp, rp = root_asym(beta, theta, x0, A_free)
    xm, rm = root_asym(-beta, theta, x0, A_free)
    d = B.T @ ((xp - xm) / (2.0 * beta))
    return d, xp, xm


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (RES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def study_task_restricted_exactness() -> dict:
    """Show nonsymmetric A can still be exact on a task-restricted q-subspace."""
    k = 6.0
    A = np.array([[-1.0, k], [0.0, -1.0]])
    B = np.array([[1.0], [0.0]])  # parameter only changes the first force component
    D = np.linalg.inv(A) - np.linalg.inv(A).T
    q_good = np.array([1.0, 0.0])
    q_bad = np.array([0.0, 1.0])
    S_good = q_good[:, None]  # the allowed loss-gradient subspace is span(e1)
    M_good = B.T @ D @ S_good
    M_all = B.T @ D @ np.eye(2)
    # Required criterion: B^T D S = 0 for all allowed q directions.
    result = {
        "A": A.tolist(), "B": B.tolist(), "D_Ainv_minus_AinvT": D.tolist(),
        "A_is_symmetric": bool(np.allclose(A, A.T)),
        "allowed_loss_gradient_subspace_basis": S_good.tolist(),
        "restricted_operator_norm": float(np.linalg.norm(M_good, 2)),
        "all_loss_gradients_operator_norm": float(np.linalg.norm(M_all, 2)),
        "q_good": q_good.tolist(),
        "q_good_vf_error": float(np.linalg.norm(B.T @ D @ q_good)),
        "q_bad": q_bad.tolist(),
        "q_bad_vf_error": float(np.linalg.norm(B.T @ D @ q_bad)),
        "restricted_exactness_passes": bool(np.linalg.norm(M_good, 2) < 1e-12),
        "global_exactness_fails": bool(np.linalg.norm(M_all, 2) > 1e-3),
    }
    (RES / "task_restricted_exactness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def study_combined_error_and_optimal_beta() -> dict:
    theta = THETA.copy()
    x0 = x_star(theta)
    A = jacobian(x0, theta)
    B = param_jacobian(x0, theta)
    q = cost_grad(x0)
    true_d = B.T @ np.linalg.solve(A.T, q)
    # Cross-check with direct differentiation of the known equilibrium x*=L theta+b.
    true_d_direct = descent_true(theta)
    if relative_error(true_d, true_d_direct) > 1e-10:
        raise AssertionError("Implicit gradient does not match direct equilibrium derivative")

    # beta sweep for truncation/noise balance
    betas = np.array([0.004, 0.006, 0.009, 0.013, 0.019, 0.028, 0.04, 0.06,
                      0.085, 0.12, 0.17, 0.24, 0.34, 0.45])
    eps_state = 1.0e-4  # Euclidean state error budget per phase
    Bnorm = float(np.linalg.norm(B.T, 2))
    rng_local = np.random.default_rng(SEED + 101)
    beta_rows: list[dict] = []
    root_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for beta in betas:
        d_root, xp, xm = root_centered_direction(float(beta), theta, x0, A, B)
        root_cache[float(beta)] = (d_root, xp, xm)
        trunc_abs = float(np.linalg.norm(d_root - true_d))
        noises = []
        for _ in range(400):
            ep = rng_local.normal(size=x0.size); ep = eps_state * ep / np.linalg.norm(ep)
            em = rng_local.normal(size=x0.size); em = eps_state * em / np.linalg.norm(em)
            d_noisy = B.T @ ((xp + ep - xm - em) / (2.0 * beta))
            noises.append(float(np.linalg.norm(d_noisy - true_d)))
        rms_noisy = float(np.sqrt(np.mean(np.square(noises))))
        # A conservative bound for arbitrary state errors of norm <= eps_state in each phase.
        noise_bound = Bnorm * eps_state / float(beta)
        beta_rows.append({
            "beta": float(beta),
            "root_centered_abs_error": trunc_abs,
            "root_centered_rel_error": relative_error(d_root, true_d),
            "mc_rms_error_with_state_noise": rms_noisy,
            "worst_case_state_noise_bound": noise_bound,
            "empirical_total_upper_proxy": trunc_abs + noise_bound,
        })

    # Estimate a in ||error_trunc|| ~= a beta^2 from a small-beta range, least squares through origin.
    small = [row for row in beta_rows if row["beta"] <= 0.085]
    b2 = np.array([row["beta"] ** 2 for row in small])
    e2 = np.array([row["root_centered_abs_error"] for row in small])
    a_est = float(np.dot(b2, e2) / np.dot(b2, b2))
    d_est = Bnorm * eps_state
    beta_opt_unconstrained = float((d_est / (2.0 * a_est)) ** (1.0 / 3.0))
    beta_opt = float(np.clip(beta_opt_unconstrained, betas.min(), betas.max()))
    def proxy(beta: float) -> float:
        return a_est * beta**2 + d_est / beta
    beta_empirical = min(beta_rows, key=lambda r: r["mc_rms_error_with_state_noise"])["beta"]
    beta_summary = {
        "state_error_budget_per_phase": eps_state,
        "spectral_norm_B_transpose": Bnorm,
        "estimated_truncation_coefficient_a": a_est,
        "worst_case_noise_coefficient_d": d_est,
        "beta_opt_formula_unconstrained": beta_opt_unconstrained,
        "beta_opt_formula_clipped_to_sweep": beta_opt,
        "proxy_error_at_beta_opt": proxy(beta_opt),
        "empirical_rms_best_beta_on_grid": beta_empirical,
        "warning": "beta_opt is optimal for the stated upper-bound proxy, not universally optimal for actual training or all systems.",
    }
    write_csv("beta_noise_tradeoff.csv", beta_rows)

    # Plot truncation, Monte-Carlo error, and theoretical proxy.
    fine = np.geomspace(float(betas.min()), float(betas.max()), 240)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.loglog([r["beta"] for r in beta_rows], [r["root_centered_abs_error"] for r in beta_rows], "o-", label="finite-nudge truncation")
    ax.loglog([r["beta"] for r in beta_rows], [r["mc_rms_error_with_state_noise"] for r in beta_rows], "s-", label="RMS error with state noise")
    ax.loglog(fine, [proxy(float(b)) for b in fine], "--", label="upper-bound proxy a beta^2 + d/beta")
    ax.axvline(beta_opt, linestyle=":", label=f"bound-optimal beta={beta_opt:.3g}")
    ax.set_xlabel("nudging strength |beta|")
    ax.set_ylabel("absolute gradient error")
    ax.set_title("Trade-off: finite nudge vs. additive state error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "beta_noise_tradeoff.png", dpi=180)
    plt.close(fig)

    # Decompose finite-nudge root error + finite-time relaxation + integrator error + state errors.
    decomp_rows: list[dict] = []
    betas_decomp = [0.05, 0.1, 0.2]
    times = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    h = 0.01
    obs_eps = 2.0e-6
    rng_noise = np.random.default_rng(SEED + 202)
    for beta in betas_decomp:
        d_root, xrp, xrm = root_cache.get(float(beta), (None, None, None))
        if d_root is None:
            d_root, xrp, xrm = root_centered_direction(beta, theta, x0, A, B)
        root_trunc_vec = d_root - true_d
        for T in times:
            xcp, resid_cp = integrate(beta, theta, x0, A, T, method="DOP853")
            xcm, resid_cm = integrate(-beta, theta, x0, A, T, method="DOP853")
            xhp, resid_hp = integrate(beta, theta, x0, A, T, method="Euler", h=h)
            xhm, resid_hm = integrate(-beta, theta, x0, A, T, method="Euler", h=h)
            time_vec = B.T @ (((xcp - xrp) - (xcm - xrm)) / (2.0 * beta))
            integ_vec = B.T @ (((xhp - xcp) - (xhm - xcm)) / (2.0 * beta))
            # Perturb the computed phase states by bounded, independent state errors.
            ep = rng_noise.normal(size=x0.size); ep = obs_eps * ep / np.linalg.norm(ep)
            em = rng_noise.normal(size=x0.size); em = obs_eps * em / np.linalg.norm(em)
            d_hat = B.T @ ((xhp + ep - xhm - em) / (2.0 * beta))
            noise_vec = B.T @ ((ep - em) / (2.0 * beta))
            total_vec = d_hat - true_d
            component_norm_sum = (np.linalg.norm(root_trunc_vec) + np.linalg.norm(time_vec)
                                  + np.linalg.norm(integ_vec) + np.linalg.norm(noise_vec))
            actual_component_sum = root_trunc_vec + time_vec + integ_vec + noise_vec
            if np.linalg.norm(total_vec - actual_component_sum) > 1e-8:
                raise AssertionError("Error decomposition identity failed")
            if np.linalg.norm(total_vec) > component_norm_sum + 1e-10:
                raise AssertionError("Triangle-inequality error bound failed")
            decomp_rows.append({
                "beta": beta, "time_T": T, "euler_h": h, "state_noise_norm_each_phase": obs_eps,
                "root_truncation_component_norm": float(np.linalg.norm(root_trunc_vec)),
                "finite_time_component_norm": float(np.linalg.norm(time_vec)),
                "integrator_component_norm": float(np.linalg.norm(integ_vec)),
                "state_noise_component_norm": float(np.linalg.norm(noise_vec)),
                "actual_total_gradient_error_norm": float(np.linalg.norm(total_vec)),
                "sum_component_norms_upper_bound": float(component_norm_sum),
                "finite_time_plus_integrator_state_residuals": float(np.linalg.norm(xhp - xrp) + np.linalg.norm(xhm - xrm)),
                "ode_reference_final_vector_field_residuals": resid_cp + resid_cm,
                "euler_final_vector_field_residuals": resid_hp + resid_hm,
            })
    write_csv("unified_error_decomposition.csv", decomp_rows)

    # Visualize error components for beta=.1 at different T.
    rows_mid = [r for r in decomp_rows if np.isclose(r["beta"], 0.1)]
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for key, label in [
        ("root_truncation_component_norm", "finite-nudge component"),
        ("finite_time_component_norm", "finite-time relaxation"),
        ("integrator_component_norm", "Euler integration"),
        ("state_noise_component_norm", "state error/noise"),
        ("actual_total_gradient_error_norm", "actual total error"),
        ("sum_component_norms_upper_bound", "sum-of-components bound"),
    ]:
        ax.semilogy([r["time_T"] for r in rows_mid], [max(r[key], 1e-16) for r in rows_mid], marker="o", label=label)
    ax.set_xlabel("relaxation time T")
    ax.set_ylabel("absolute gradient error / bound")
    ax.set_title("Unified EP error decomposition (beta=0.1, Euler h=0.01)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "unified_error_decomposition.png", dpi=180)
    plt.close(fig)

    beta_summary["combined_error_decomposition_rows"] = len(decomp_rows)
    beta_summary["triangle_inequality_bound_verified"] = True
    (RES / "beta_noise_tradeoff_summary.json").write_text(json.dumps(beta_summary, indent=2), encoding="utf-8")
    return {"beta_tradeoff": beta_summary, "decomposition_rows": len(decomp_rows)}


def study_nonnormal_transient() -> dict:
    """Quantify transient semigroup gain and its effect on finite-time adjoint response.

    A_k=[[-1,k],[0,-0.1]] is Hurwitz for all k, but becomes increasingly non-normal.
    For each k, q is chosen as the right singular vector of A^{-T} with the smallest
    singular value. Thus the initial adjoint-response tail is small and can be
    transiently amplified; this is a deliberately adverse, fully disclosed direction.
    """
    rows: list[dict] = []
    all_curves: dict[str, list] = {}
    times = np.linspace(0.0, 10.0, 1001)
    B = np.eye(2)
    for k in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
        A = np.array([[-1.0, k], [0.0, -0.1]])
        AinvT = np.linalg.inv(A.T)
        _, _, Vh = np.linalg.svd(AinvT)
        q = Vh[-1, :].copy()
        q /= np.linalg.norm(q)
        gains = []
        grad_errors = []
        direct_bounds = []
        for t in times:
            E = expm(A.T * float(t))
            gains.append(float(np.linalg.norm(E, 2)))
            # AsymEP first-order sensitivity after finite time T:
            # v_T = A^{-T}(I - exp(A^T T))q; error to v_inf=A^{-T}q is -A^{-T}exp(A^T T)q.
            er = -B.T @ AinvT @ E @ q
            grad_errors.append(float(np.linalg.norm(er)))
            direct_bounds.append(float(np.linalg.norm(B.T @ AinvT, 2) * np.linalg.norm(E, 2) * np.linalg.norm(q)))
        imax = int(np.argmax(gains))
        emax = int(np.argmax(grad_errors))
        mu2 = float(np.linalg.eigvalsh(0.5 * (A + A.T)).max())
        # Lyapunov function for M=A.T: M.T P + P M = A P + P A.T = -I.
        P = solve_continuous_lyapunov(A, -np.eye(2))
        P = 0.5 * (P + P.T)
        p_eigs = np.linalg.eigvalsh(P)
        if np.min(p_eigs) <= 0:
            raise AssertionError("Lyapunov solution should be positive definite for Hurwitz A")
        lam = float(1.0 / (2.0 * np.max(p_eigs)))  # Q=I: lambda_min(Q)/(2 lambda_max(P))
        lyap_semigroup_bound = float(np.sqrt(np.max(p_eigs) / np.min(p_eigs))) * np.exp(-lam * times)
        lyap_gradient_bound = np.linalg.norm(B.T @ AinvT, 2) * lyap_semigroup_bound * np.linalg.norm(q)
        initial_error = max(grad_errors[0], 1e-15)
        rows.append({
            "offdiagonal_k": k,
            "eigenvalues": "[-1.0, -0.1]",
            "logarithmic_norm_mu2": mu2,
            "condition_number_2": float(np.linalg.cond(A, 2)),
            "q_selected_as_min_right_singular_vector_of_AinvT": q.tolist(),
            "initial_finite_time_gradient_error": grad_errors[0],
            "max_transient_gain_0_10": gains[imax],
            "time_at_max_gain": float(times[imax]),
            "max_finite_time_gradient_error_0_10": grad_errors[emax],
            "time_at_max_gradient_error": float(times[emax]),
            "gradient_error_amplification_ratio": float(max(grad_errors) / initial_error),
            "max_actual_gradient_error_minus_direct_bound": float(np.max(np.array(grad_errors) - np.array(direct_bounds))),
            "lyapunov_lambda": lam,
            "lyapunov_P_condition": float(np.linalg.cond(P, 2)),
            "lyapunov_envelope_valid": bool(np.all(np.array(gains) <= lyap_semigroup_bound + 1e-10)),
            "lyapunov_gradient_bound_valid": bool(np.all(np.array(grad_errors) <= lyap_gradient_bound + 1e-10)),
        })
        all_curves[str(int(k))] = [
            {"time": float(t), "semigroup_gain": float(g), "gradient_error": float(e),
             "direct_gradient_error_bound": float(b), "lyapunov_gradient_error_bound": float(lb)}
            for t, g, e, b, lb in zip(times, gains, grad_errors, direct_bounds, lyap_gradient_bound)
        ]
    write_csv("nonnormal_sweep.csv", rows)
    (RES / "nonnormal_transient_curves.json").write_text(json.dumps(all_curves), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    for k in [1, 2, 5, 10, 20]:
        curve = all_curves[str(k)]
        ax.plot([p["time"] for p in curve], [p["semigroup_gain"] for p in curve], label=f"k={k}")
    ax.set_xlabel("time T")
    ax.set_ylabel(r"$\|e^{A^T T}\|_2$")
    ax.set_title("Transient growth despite stable eigenvalues (-1, -0.1)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "nonnormal_sweep_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    for k in [1, 2, 5, 10, 20]:
        curve = all_curves[str(k)]
        initial = max(curve[0]["gradient_error"], 1e-14)
        ax.plot([p["time"] for p in curve], [max(p["gradient_error"], 1e-14) / initial for p in curve], label=f"k={k}")
    ax.set_xlabel("time T")
    ax.set_ylabel("gradient-response error / initial error")
    ax.set_title("Transient amplification of finite-time gradient response")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "nonnormal_gradient_error.png", dpi=180)
    plt.close(fig)

    # Bounds can be much looser than the measured response, so plot them separately.
    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    curve10 = all_curves["10"]
    ax.semilogy([p["time"] for p in curve10], [max(p["gradient_error"], 1e-14) for p in curve10], label="actual error k=10")
    ax.semilogy([p["time"] for p in curve10], [max(p["direct_gradient_error_bound"], 1e-14) for p in curve10], "--", label="direct operator-norm bound")
    ax.semilogy([p["time"] for p in curve10], [max(p["lyapunov_gradient_error_bound"], 1e-14) for p in curve10], ":", label="Lyapunov-envelope bound")
    ax.set_xlabel("time T")
    ax.set_ylabel("gradient-response error / upper bound")
    ax.set_title("Upper bounds can be valid but conservative (k=10)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "nonnormal_bounds.png", dpi=180)
    plt.close(fig)

    row_k10 = next(r for r in rows if r["offdiagonal_k"] == 10.0)
    row_k20 = next(r for r in rows if r["offdiagonal_k"] == 20.0)
    result = {
        "all_systems_have_eigenvalues_minus_one_and_minus_point_one": True,
        "lyapunov_envelope_checks_all_pass": bool(all(r["lyapunov_envelope_valid"] for r in rows)),
        "lyapunov_gradient_bounds_all_pass": bool(all(r["lyapunov_gradient_bound_valid"] for r in rows)),
        "max_transient_gain_k10": row_k10["max_transient_gain_0_10"],
        "max_transient_gain_k20": row_k20["max_transient_gain_0_10"],
        "gradient_error_amplification_ratio_k10": row_k10["gradient_error_amplification_ratio"],
        "gradient_error_amplification_ratio_k20": row_k20["gradient_error_amplification_ratio"],
        "maximum_actual_minus_direct_bound": max(r["max_actual_gradient_error_minus_direct_bound"] for r in rows),
        "rows": rows,
    }
    (RES / "nonnormal_sweep_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    exactness = study_task_restricted_exactness()
    combined = study_combined_error_and_optimal_beta()
    nonnormal = study_nonnormal_transient()

    assertions = {
        "task_restricted_exactness_can_hold_without_symmetric_A": exactness["A_is_symmetric"] is False and exactness["restricted_exactness_passes"],
        "same_nonsymmetric_A_not_exact_for_all_q": exactness["global_exactness_fails"],
        "formula_beta_opt_positive": combined["beta_tradeoff"]["beta_opt_formula_unconstrained"] > 0,
        "finite_time_error_bound_identity_verified": combined["beta_tradeoff"]["triangle_inequality_bound_verified"],
        "nonnormal_lyapunov_envelope_valid": nonnormal["lyapunov_envelope_checks_all_pass"],
        "nonnormal_lyapunov_gradient_bound_valid": nonnormal["lyapunov_gradient_bounds_all_pass"],
        "nonnormal_gain_exceeds_one": nonnormal["max_transient_gain_k10"] > 1.0,
        "nonnormal_finite_time_gradient_response_amplifies": nonnormal["gradient_error_amplification_ratio_k10"] > 1.0,
        "nonnormal_gradient_direct_bound_valid": nonnormal["maximum_actual_minus_direct_bound"] < 1e-9,
    }
    summary = {
        "seed": SEED,
        "numpy_version": np.__version__,
        "study_1_task_restricted_exactness": exactness,
        "study_2_combined_error_and_beta_opt": combined,
        "study_3_nonnormal_transient": nonnormal,
        "assertions": assertions,
        "all_assertions_passed": bool(all(assertions.values())),
    }
    (RES / "stage2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"assertions": assertions,
                      "beta_opt": combined["beta_tradeoff"]["beta_opt_formula_clipped_to_sweep"],
                      "empirical_best_beta": combined["beta_tradeoff"]["empirical_rms_best_beta_on_grid"],
                      "max_gain_k10": nonnormal["max_transient_gain_k10"],
                      "max_gain_k20": nonnormal["max_transient_gain_k20"],
                      "gradient_error_amplification_k10": nonnormal["gradient_error_amplification_ratio_k10"]}, indent=2))
    if not all(assertions.values()):
        raise AssertionError(f"Stage 2 assertions failed: {assertions}")
    print("STAGE 2 ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
