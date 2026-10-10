# Evolutionary Equilibrium Propagation

**Settling to a stable equilibrium as the single primitive of continual learning.**

Paper: [eep.pdf](eep.pdf) &nbsp;|&nbsp; Project page: https://hf-jz.github.io/eep/

A patch net has no training phase. Each unit holds local state and predicts it from its
neighbours, the network repairs disagreement by projected gradient descent on one energy, and the
settled state is the answer. This repository contains the paper that proves what follows from that
single operator, plus the scripts that reproduce every number in it.

## What the paper proves

1. **The repair always reaches a stable equilibrium.** The energy is coercive, because every patch
   predicts its own state and every prediction is a bounded function of it, so sublevel sets are
   compact. The energy is real analytic, so the Łojasiewicz inequality holds everywhere and a
   line-search descent method has bounded iterates converging to a single stationary point, at a
   rate set by the Łojasiewicz exponent.
2. **Inference and learning are one operator.** The learning objective differs from the query
   objective by an anchored quadratic term and by which coordinate blocks are eligible. The state
   block of the gradient is provably unchanged. There is no train/inference boundary to draw.
3. **The contrast is second order.** The learning rule is equilibrium propagation in the sense of
   Scellier and Bengio, and the centred contrast used by implementations has error `O(beta^2)`
   against `O(beta)` for the one-sided one. An exactly solvable case gives both constants in
   closed form as `dL/dw * (1+beta)^-1` and `dL/dw * (1-beta^2)^-1`.
4. **An evolutionary layer with an averaging theorem.** Evolution acts on meta-parameters (the
   wiring mask, the energy's form, the plasticity constants) and never on the weights learned
   within a life. The population flow averages to a replicator-mutator equation whose fitness is
   the post-learning (Baldwinian) fitness. A Price-type decomposition and a free-energy
   dissipation identity complete the layer.
5. **Biological compliance is proved, not asserted.** Six formal criteria of biological evolution
   are satisfied by construction, and inheriting plasticity but not learned weights is shown to be
   necessary by a counterexample that is adaptive, not merely biological.

## Reproduce

```bash
# 1. The settling, learning and evolutionary results (numpy only, Python 3.9+)
python3 src/verify_eep.py

# 2. A cross-check against the reference implementation of the patch law
PYTHONPATH=../cadence/src python3.13 src/cadence_crosscheck.py

# 3. Page self-check (balanced tags, no em/en dashes, assets present)
python3 tools/check_site.py index.html
```

`verify_eep.py` asserts each claim and exits non-zero if one fails. Parts A to F cover analytic
gradients against finite differences, settling monotonicity and convergence, coercivity, the order
of both contrasts, one operator across two coordinate blocks, the evolutionary layer, and the
closed-form case.

The cross-check runs the paper's own implementation of the patch law against
[cadence-net](https://github.com/muellerberndt/cadence) 0.80.0 on the same graph, state and
parameters. The largest absolute difference across the energy, the prediction, the error and all
three gradients is `1.1e-16`, which is float64 rounding.

Run logs are archived in [`evidence/`](evidence).

## Layout

```
index.html      project page (static, single file)
eep.pdf         the paper, 17 pages
paper/eep.tex   LaTeX source, builds with xelatex
src/            the two verification scripts
evidence/       archived run logs
tools/          page self-check
docs/           Chinese background notes: literature, the extension, the maturity verdict
refs/           bibliography with source URLs
```

## Scope

The paper proves statements about the settling dynamics and the optimisation they define. It does
not claim that settling yields intelligence, that a patch net matches a transformer at any task, or
that this saves energy while learning. The open problems and four falsifiable predictions are in
the paper's final section.

## Citation

```bibtex
@article{cheng2026eep,
  title  = {Evolutionary Equilibrium Propagation: Settling to a Stable
            Equilibrium as the Single Primitive of Continual Learning},
  author = {Cheng, Sheng},
  year   = {2026},
  month  = {October},
  note   = {Hefei Jiuzhai Big Data Technology Co., Ltd.},
  url    = {https://hf-jz.github.io/eep/}
}
```

## Attribution

The patch law formalised here is the settling rule of the Cadence library by Bernhard Mueller and
Pragma Research, which is also the implementation the cross-check runs against. This work is an
independent extension and is not affiliated with that project. Equilibrium propagation is due to
Scellier and Bengio.

Text and figures: CC BY 4.0. Code in `src/` and `tools/`: MIT.
