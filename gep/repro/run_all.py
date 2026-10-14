#!/usr/bin/env python3
"""Run every certificate and experiment behind the paper, then the test suite.

    python3 repro/run_all.py            # from the repository root
    python3 run_all.py                  # from this directory

Writes JSON/CSV artefacts into results/ and console logs into logs/. Exits
non-zero if any stage fails, so the script doubles as an acceptance check.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import json

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOGS = HERE / "logs"

STAGES = [
    ("stage1_validation", ["src/run_validation.py"],
     "3-state non-normal benchmark: analytic gradient vs finite differences, error decomposition"),
    ("stage2_audit", ["src/run_stage2.py"],
     "task-restricted VF exactness, nudge/noise trade-off, non-normal transient study"),
    ("stage3_adaptive", ["src/run_stage3.py"],
     "adaptive beta/T/h controller, integrator comparison, tighter non-normal bounds"),
    ("stage4_certified_residual", ["src/certified_gep.py"],
     "exact-rational certified residual and total gradient error bound"),
    ("stage5_multibranch_certificate", ["src/certified_multibranch_gep.py"],
     "exact-rational Krawczyk branch/endpoint certificate on the multistable benchmark"),
    ("baselines", ["src/unified_baselines.py"],
     "floating-point baseline comparison: implicit gradient, VF-EP, AsymEP, Dyadic EP"),
    ("dyadic_residual_certificate", ["src/dyadic_residual_certificate.py"],
     "residual-only Dyadic EP bound with exact rational constants"),
    ("dyadic_finite_precision_certificate", ["src/dyadic_finite_precision_certificate.py"],
     "exact re-evaluation of the float64 difference-solve residual"),
    ("beta_scan", ["src/beta_scan.py"],
     "certified worst-branch gradient error over a dense nudging grid"),
]


def main() -> int:
    RES.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    failures = []
    for name, argv, blurb in STAGES:
        print(f"[run] {name}: {blurb}")
        proc = subprocess.run([sys.executable, *argv], cwd=HERE,
                              capture_output=True, text=True)
        log = LOGS / f"{name}.log"
        log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        # Stages that print a JSON document also deposit it under results/.
        try:
            json.loads(proc.stdout)
        except (ValueError, TypeError):
            pass
        else:
            (RES / f"{name}.json").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            failures.append(name)
            print(f"  FAILED (exit {proc.returncode}); see {log}")
            continue
        print(f"  ok -> results/ and {log.relative_to(HERE)}")

    print("[run] unittest suite")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=HERE, capture_output=True, text=True)
    (LOGS / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    tail = [ln for ln in (tests.stdout + tests.stderr).splitlines() if ln.startswith("Ran ") or ln == "OK"]
    print("  " + " | ".join(tail))
    if tests.returncode != 0:
        failures.append("tests")

    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print("\nAll certificates, experiments and tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
