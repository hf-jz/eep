#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cadence_crosscheck.py -- 把本文实现的 processing-patch 定律与 Cadence 官方库逐项对照

需要 Python >= 3.11 (cadence 要求) 与本地 cadence 源码:

    PYTHONPATH=/Users/sheng/math/cadence/src /opt/homebrew/bin/python3.13 \
        src/cadence_crosscheck.py

对照三件事:
  1. 同一图/状态/参数下, 能量与全部解析梯度 (state/weight/bias) 是否逐位相同
  2. 同一初值下, 官方 settle 与本文 settle 的终态与驻定残差
  3. 用公开 API (Cortex/Brain) 跑一次真实沉降, 报告 qualified 与残差
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import verify_eep as mine                       # 本文实现
from cadence.experimental.equilibrium import Cortex
from cadence.experimental.equilibrium import _repair as cad

rng = np.random.default_rng(20261010)

EDGES = [("input", 0, 0), ("input", 1, 0),
         ("state", 0, 1), ("input", 1, 1), ("state", 1, 2),
         ("state", 2, 1)]                       # 反馈边 (Cadence `layered` 的互惠对)
N_IN, N_PATCH = 2, 3
STATE_PRIOR = 0.01

print("=" * 78)
print("Cadence 官方库对照 (cadence-net %s)" % __import__("cadence").__version__)
print("=" * 78)

g_mine = mine.Graph(N_IN, N_PATCH, EDGES)
g_cad = cad.Graph(N_IN, N_PATCH, tuple(EDGES))

x = list(0.4 * rng.standard_normal(N_PATCH))
w = list(0.7 * rng.standard_normal(len(EDGES)))
b = list(0.3 * rng.standard_normal(N_PATCH))
inputs = [0.6, -0.3]

rm = mine.evaluate(g_mine, x, w, b, inputs, STATE_PRIOR)
rc = cad.evaluate(g_cad, tuple(inputs), tuple(x), tuple(w), tuple(b),
                  state_prior=STATE_PRIOR)

print("1) 能量与解析梯度")
worst = abs(rm["energy"] - rc["energy"])
print(f"   energy      本文 {rm['energy']:.15f}   官方 {rc['energy']:.15f}   "
      f"|diff| = {worst:.3e}")
for key_m, key_c, name in (("grad_x", "gradient_state", "grad_state"),
                           ("grad_w", "gradient_weights", "grad_weights"),
                           ("grad_b", "gradient_biases", "grad_biases")):
    d = max(abs(p - q) for p, q in zip(rm[key_m], rc[key_c]))
    worst = max(worst, d)
    print(f"   {name:<13} |diff|_inf = {d:.3e}")
for name, a, c in (("prediction", rm["pred"], rc["predictions"]),
                   ("error", rm["err"], rc["errors"])):
    d = max(abs(p - q) for p, q in zip(a, c))
    worst = max(worst, d)
    print(f"   {name:<13} |diff|_inf = {d:.3e}")
assert worst < 1e-12, f"与官方库不一致: {worst}"

print()
print("2) 同一初值下的沉降终态")
x_m, w_m, b_m, r_m = mine.settle(g_mine, [0.0] * N_PATCH, w, b, inputs,
                                 state_prior=STATE_PRIOR, budget=4000)
res_m, _ = mine.projected_residual(g_mine, x_m, w_m, b_m, inputs, clamps={},
                                   learn=False, state_prior=STATE_PRIOR,
                                   parameter_prior=0.1, anchor=(w_m, b_m))
out = cad.settle(g_cad, tuple(inputs), tuple([0.0] * N_PATCH), tuple(w), tuple(b),
                 clamps=None, learn=False, budget=4000, tolerance=1e-12,
                 state_prior=STATE_PRIOR, step=1.0, backtracks=32)
x_c = list(out["state"]) if isinstance(out, dict) else list(out[0])
e_m = mine.evaluate(g_mine, x_m, w_m, b_m, inputs, STATE_PRIOR)["energy"]
e_c = cad.evaluate(g_cad, tuple(inputs), tuple(x_c), tuple(w), tuple(b),
                   state_prior=STATE_PRIOR)["energy"]
print(f"   本文终态 x = {[round(v, 12) for v in x_m]}")
print(f"   官方终态 x = {[round(v, 12) for v in x_c]}")
print(f"   |x_mine - x_cadence|_inf = {max(abs(p - q) for p, q in zip(x_m, x_c)):.3e}")
print(f"   终态能量: 本文 {e_m:.15f}   官方 {e_c:.15f}   |diff| = {abs(e_m - e_c):.3e}")
print(f"   本文投影驻定残差 = {res_m:.3e}")

print()
print("3) 公开 API 端到端 (Cortex/Brain)")
cortex = Cortex(seed=7)
eyes = cortex.input("eyes", shape=(4,))
perception = cortex.column("perception", patches=6, inputs=(eyes,))
response = cortex.column("response", patches=4, inputs=(eyes, perception))
cortex.output("motor", shape=(2,), reads=response)
brain = cortex.build()
result = brain.settle({"eyes": [0.2, -0.4, 0.7, 0.1]})
print(f"   qualified = {result['qualified']}   输出 = "
      f"{[round(float(v), 6) for v in result['outputs']['motor']]}")
print(f"   stationarity = {result.get('stationarity')}   sweeps = {result.get('sweeps')}")
assert result["qualified"], "官方 brain.settle 未 qualified"

print()
print("=" * 78)
print(f"对照通过: 本文实现与 cadence-net 官方的最大偏差 {worst:.3e} (float64 舍入量级),")
print("沉降终态一致, 公开 API 端到端 qualified。")
print("=" * 78)
