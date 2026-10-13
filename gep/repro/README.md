# Certified equilibrium propagation: reproducibility package

Every number, table and figure in `paper/cep.tex` is produced here.

```bash
python3 repro/run_all.py        # nine stages + the test suite, ~10 s
python3 repro/make_figures.py   # the six paper figures, from results/
```

`run_all.py` writes `results/` (JSON and CSV artefacts) and `logs/` (console output of each
stage), runs `tests/test_stage5.py`, and exits non-zero if any stage or test fails.

## Stages

| File | Purpose |
| --- | --- |
| `src/run_validation.py` | three-state benchmark: analytic gradient against finite differences, error decomposition |
| `src/run_stage2.py` | task-restricted exactness of vector-field EP, nudging trade-off, non-normal transient study |
| `src/run_stage3.py` | adaptive controller over (beta, T, h), integrator comparison, tighter non-normal bounds |
| `src/certified_gep.py` | exact rational residual certificate and total gradient error bound, three-state benchmark |
| `src/certified_multibranch_gep.py` | exact rational Krawczyk branch, endpoint and gradient certificate, multistable benchmark |
| `src/beta_scan.py` | dense nudging scan (Table 1, Figure 2) |
| `src/unified_baselines.py` | implicit gradient, vector-field EP bias, AsymEP finite-nudge errors, Dyadic ideal signal |
| `src/dyadic_residual_certificate.py` | residual-only doubled-state bound with exact rational constants |
| `src/dyadic_finite_precision_certificate.py` | exact rational re-evaluation of the float64 difference solve |

## Evidence levels

Certificates are produced by interval arithmetic over `fractions.Fraction`, from the exact binary
rational conversion of candidate states supplied by floating-point solvers. Solver output is a
candidate, never a conclusion. Console logs and JSON artefacts record both decimal and exact
fractional forms of each accepted bound, so an accepted claim can be re-verified by reading
`results/` without re-running anything.

The `check_*.py`-style exploration scripts used while writing the paper are not shipped; every
table value is reproducible from `results/` directly.

## Requirements

Python 3.9 or newer with `numpy`, `scipy`, `mpmath` and `matplotlib` (only for figures and for the
floating-point stages). The certified stages need `numpy`, `scipy` and the standard library.
