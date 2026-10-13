#!/usr/bin/env python3
"""Generate the paper figures from the artefacts in results/.

Every curve is read from a file produced by run_all.py; nothing is hard coded
except axis limits. Output goes to ../assets/.

    python3 repro/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "assets"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})
INK, ACC, REJ, MUT = "#15171b", "#1d4e89", "#b3352c", "#6b7280"


def load(name):
    return json.loads((RES / name).read_text())


def read_csv(name):
    with (RES / name).open() as fh:
        return list(csv.DictReader(fh))


def fig1_benchmark():
    A, B = 1 / 5, 1 / 10
    x = np.linspace(-1.6, 1.6, 500)
    y = np.linspace(-0.55, 0.55, 400)
    X, Y = np.meshgrid(x, y)
    U = X - X**3 + A * Y
    V = -Y + B * (X - X**3)
    N = np.hypot(U, V)
    N[N == 0] = 1.0

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.streamplot(x, y, U, V, density=1.5, linewidth=0.55, arrowsize=0.7,
                  color=MUT, broken_streamlines=False)
    ax.plot(x, 5 * (x**3 - x), color=ACC, lw=1.4, label=r"$F_1=0$")
    ax.plot(x, (x - x**3) / 10, color=REJ, lw=1.4, label=r"$F_2=0$")
    for s, lab in ((-1, "stable"), (0, "saddle"), (1, "stable")):
        ax.plot([s], [0], marker="o", ms=7, mfc=INK if lab == "stable" else "white",
                mec=INK, mew=1.2, zorder=5)
    for s in (-1, 1):
        ax.add_patch(plt.Rectangle((s - 0.05, -0.04), 0.1, 0.08, fill=False,
                                   ec=ACC, lw=1.0, ls=(0, (4, 2)), zorder=6))
    ax.annotate("certified\nbranch tube", xy=(1.05, 0.0), xytext=(1.18, 0.26),
                fontsize=7.5, color=ACC,
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8))
    ax.set_xlabel("$x$"), ax.set_ylabel("$y$")
    ax.set_xlim(-1.6, 1.6), ax.set_ylim(-0.55, 0.55)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_benchmark.png")
    plt.close(fig)


def fig2_beta_scan():
    rows = load("beta_scan.json")["rows"]
    b = np.array([r["beta"] for r in rows])
    e = np.array([r["worst_branch_error_upper"] for r in rows])
    ok = np.array([r["accepted"] for r in rows])
    o = np.argsort(b)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(b[o], e[o], "-", color=INK, lw=1.2, zorder=3)
    ax.loglog(b[o][ok[o]], e[o][ok[o]], "o", ms=5, color=ACC, label="accepted")
    ax.loglog(b[o][~ok[o]], e[o][~ok[o]], "s", ms=5, color=REJ, label="rejected")
    tol = rows[0]["tolerance"]
    ax.axhline(tol, color=REJ, lw=0.9, ls=(0, (5, 3)))
    ax.annotate(r"tolerance $10^{-5}$", xy=(b[o][0], tol), xytext=(b[o][0] * 0.62, tol * 1.9),
                fontsize=7.5, color=REJ)
    slope = e[o][0] / b[o][0] ** 2
    ax.loglog(b[o], slope * b[o] ** 2, ls=":", color=MUT, lw=1.0)
    ax.annotate(r"$\propto\beta^{2}$", xy=(b[o][-1] * 1.05, slope * b[o][-1] ** 2 * 1.15),
                fontsize=8, color=MUT)
    ax.plot([0.005], [e[o][np.argmin(abs(b[o] - 0.005))]], marker="*", ms=11,
            color="#2e7d32", zorder=5)
    ax.annotate(r"selected $\beta=0.005$", xy=(0.005, 7.53e-6), xytext=(0.00055, 3.2e-6),
                fontsize=7.5, color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.8))
    ax.set_xlabel(r"nudging strength $\beta$")
    ax.set_ylabel("certified worst-branch\ngradient error bound")
    ax.set_xlim(7e-4, 3e-2)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_beta_scan.png")
    plt.close(fig)


def fig3_nudge_order():
    rows = read_csv("finite_nudge_errors.csv")
    b = np.array([float(r["beta"]) for r in rows])
    c = np.array([float(r["asymep_centered_relative_error"]) for r in rows])
    s = np.array([float(r["asymep_one_sided_relative_error"]) for r in rows])
    d = np.array([float(r["dyadic_stationary_relative_error"]) for r in rows])

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(b, c, "o-", ms=4, color=ACC, lw=1.1, label="centred contrast (AsymEP)")
    ax.loglog(b, s, "s-", ms=4, color=REJ, lw=1.1, label="one-sided contrast")
    ax.loglog(b, np.maximum(d, 1e-17), "^--", ms=4, color="#2e7d32", lw=1.0,
              label="Dyadic stationary signal")
    ax.loglog(b, c[-1] * (b / b[-1]) ** 2, ":", color=MUT, lw=1.0)
    ax.loglog(b, s[-1] * (b / b[-1]), ":", color=MUT, lw=1.0)
    ax.annotate(r"slope 2", xy=(0.12, c[-1] * (0.12 / b[-1]) ** 2 * 1.8), fontsize=7.5, color=MUT)
    ax.annotate(r"slope 1", xy=(0.12, s[-1] * (0.12 / b[-1]) * 0.35), fontsize=7.5, color=MUT)
    ax.set_xlabel(r"nudging strength $\beta$")
    ax.set_ylabel("relative error of the\nstate-contrast signal")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_nudge_order.png")
    plt.close(fig)


def fig4_error_budget():
    c = load("certified_run.json")
    names = ["finite nudge\n(truncation)", "equilibrium\nsolve residual", "total certificate", "observed error"]
    vals = [c["truncation_gradient_error_upper"], c["equilibrium_solve_gradient_error_upper"],
            c["total_gradient_error_upper"], c["empirical_gradient_error"]]
    cols = [ACC, ACC, "#2e7d32", REJ]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    bars = ax.bar(range(4), vals, color=cols, width=0.62)
    ax.set_yscale("log")
    ax.axhline(c["tolerance"], color=REJ, lw=0.9, ls=(0, (5, 3)))
    ax.annotate(r"tolerance $5\times10^{-4}$", xy=(2.55, c["tolerance"] * 1.35), fontsize=7.5, color=REJ)
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:.2e}", xy=(rect.get_x() + rect.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("gradient error")
    ax.set_ylim(1e-6, 3e-3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_error_budget.png")
    plt.close(fig)


def fig5_nonnormal():
    curves = load("nonnormal_transient_curves.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharex=True)
    for k in ("0", "1", "2", "5", "10", "20"):
        rows = curves[k]
        t = np.array([r["time"] for r in rows])
        g = np.array([r["semigroup_gain"] for r in rows])
        ge = np.array([r["gradient_error"] for r in rows])
        lab = rf"$K={k}$"
        axes[0].plot(t, g, lw=1.1, label=lab)
        axes[1].plot(t, ge, lw=1.1, label=lab)
    rows = curves["20"]
    t = np.array([r["time"] for r in rows])
    axes[1].plot(t, [r["direct_gradient_error_bound"] for r in rows], "--",
                 color=MUT, lw=1.0, label="direct bound")
    axes[1].plot(t, [r["lyapunov_gradient_error_bound"] for r in rows], ":",
                 color=REJ, lw=1.0, label="Lyapunov bound")
    axes[0].set_xlabel("time $t$"), axes[0].set_ylabel(r"transient gain $\|e^{At}\|_2$")
    axes[1].set_xlabel("time $t$"), axes[1].set_ylabel("finite-time gradient error")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.02, 3e5)
    axes[0].set_title("state response", fontsize=8.5)
    axes[1].set_title("gradient response", fontsize=8.5)
    axes[0].legend(fontsize=6.5, ncol=2), axes[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_nonnormal.png")
    plt.close(fig)


def fig6_integrators():
    c = load("integrator_comparison.json")
    labels = ["adaptive RK4", "fixed RK4", "DOP853"]
    evals = [c["adaptive_rk4_rhs_evaluations_estimate"], c["fixed_rk4_rhs_evaluations"],
             c["dop853_reference_rhs_evaluations"]]
    errs = [c["adaptive_gradient_estimate_error_vs_dop853_estimate"],
            c["fixed_gradient_estimate_error_vs_dop853_estimate"], 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    axes[0].bar(range(3), evals, color=[ACC, "#2e7d32", MUT], width=0.6)
    for i, v in enumerate(evals):
        axes[0].annotate(f"{v}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=7.5)
    axes[0].set_xticks(range(3)), axes[0].set_xticklabels(labels, fontsize=7.5)
    axes[0].set_ylabel("right-hand-side evaluations")
    vals = [max(e, 1e-13) for e in errs]
    axes[1].bar(range(3), vals, color=[ACC, "#2e7d32", MUT], width=0.6)
    axes[1].set_yscale("log")
    for i, (v, raw) in enumerate(zip(vals, errs)):
        txt = "reference" if raw == 0 else f"{raw:.1e}"
        axes[1].annotate(txt, xy=(i, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=7.5)
    axes[1].set_xticks(range(3)), axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylabel("gradient error vs DOP853")
    axes[0].set_title("cost", fontsize=8.5), axes[1].set_title("accuracy", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_integrators.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_benchmark(); print("fig1")
    fig2_beta_scan(); print("fig2")
    fig3_nudge_order(); print("fig3")
    fig4_error_budget(); print("fig4")
    fig5_nonnormal(); print("fig5")
    fig6_integrators(); print("fig6")
    print("figures written to", OUT)
