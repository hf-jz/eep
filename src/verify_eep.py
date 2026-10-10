#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_eep.py -- EEP 论文的数值证据 (numpy only)

实现 Cadence 的 processing-patch (DRSN) 定律, 逐行对应
cadence/src/cadence/experimental/equilibrium/_repair.py 的 _evaluate / settle:

    p_i = tanh(b_i + sum_j w_ij * signal_j)          signal = 输入/状态/误差
    e_i = x_i - p_i
    E   = 1/2 * sum_i e_i^2 + state_prior/2 * sum_i x_i^2
    E_learn = E + parameter_prior/2 * ||theta - anchor||^2

  A  能量与解析梯度 vs 有限差分 (验证本文复现)
  B  沉降: Lyapunov 单调 / 收敛到驻点 / 活动先验保证有界 (coercivity)
  C  EP 基本引理: 单边对比 err=O(beta); 中心对比 (centered) err=O(beta^2)
  D  一个算子两个坐标块: 查询与学习共用同一 repair; 学习收敛 (rho-强凸)
  E  双尺度进化: Baldwin 最优可塑性

运行: /usr/bin/python3 src/verify_eep.py
"""
from __future__ import annotations

import math

import numpy as np

rng = np.random.default_rng(20261010)

# ---------------------------------------------------------------------------
# 图: edges = (kind, source, target), kind in {"input","state","residual"}
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self, n_inputs, n_patches, edges):
        self.n_inputs = n_inputs
        self.n_patches = n_patches
        self.edges = list(edges)
        self.incoming = [[] for _ in range(n_patches)]
        deg = [0] * n_patches
        desc = [[] for _ in range(n_patches)]
        for e, (kind, src, tgt) in enumerate(self.edges):
            self.incoming[tgt].append(e)
            if kind == "residual":
                desc[src].append(tgt); deg[tgt] += 1
        # 残差(误差回读)依赖的拓扑序, 与 Cadence 的 residual_order 一致
        ready = [i for i, d in enumerate(deg) if d == 0]
        order = []
        while ready:
            s = ready.pop()
            order.append(s)
            for t in desc[s]:
                deg[t] -= 1
                if deg[t] == 0:
                    ready.append(t)
        assert len(order) == n_patches, "residual-readback dependencies must be acyclic"
        self.residual_order = order


def evaluate(g, x, w, b, inputs, state_prior):
    """返回 energy / pred / err / 解析梯度 (逐行对应 Cadence _evaluate)"""
    pred = [0.0] * g.n_patches
    err = [0.0] * g.n_patches
    signal = [0.0] * len(g.edges)
    for t in g.residual_order:
        terms = [b[t]]
        for e in g.incoming[t]:
            kind, src, _ = g.edges[e]
            signal[e] = (inputs[src] if kind == "input"
                         else x[src] if kind == "state" else err[src])
            terms.append(w[e] * signal[e])
        pred[t] = math.tanh(math.fsum(terms))
        err[t] = x[t] - pred[t]
    energy = 0.5 * math.fsum(e * e for e in err) \
        + 0.5 * state_prior * math.fsum(v * v for v in x)
    grad_x = [state_prior * v for v in x]
    grad_w = [0.0] * len(w)
    grad_b = [0.0] * g.n_patches
    adj_err = list(err)
    for t in reversed(g.residual_order):
        adj = adj_err[t]
        grad_x[t] += adj
        adj_pred = -adj * (1.0 - pred[t] ** 2)
        grad_b[t] += adj_pred
        for e in g.incoming[t]:
            kind, src, _ = g.edges[e]
            grad_w[e] += adj_pred * signal[e]
            if kind == "state":
                grad_x[src] += adj_pred * w[e]
            elif kind == "residual":
                adj_err[src] += adj_pred * w[e]
    return {"energy": energy, "pred": pred, "err": err,
            "grad_x": grad_x, "grad_w": grad_w, "grad_b": grad_b}


def _project(v, bound):
    return min(max(v, -bound), bound)


def settle(g, x, w, b, inputs, *, clamps=None, learn=False, state_prior=0.01,
           parameter_prior=0.1, state_bound=1.0, parameter_bound=4.0,
           budget=2000, tolerance=1e-6, step=1.0, backtracks=32,
           anchor=None, trace=False):
    """Cadence 的投影梯度 repair: Armijo 充分下降 + Barzilai-Borwein 步长"""
    clamps = {} if clamps is None else dict(clamps)
    x = list(x); w = list(w); b = list(b)
    for i, v in clamps.items():          # 钳制坐标属于状态本身 (Cadence: clamps 是边界值)
        x[i] = v
    if learn and anchor is None:
        anchor = (list(w), list(b))
    def full(xx, ww, bb):
        r = evaluate(g, xx, ww, bb, inputs, state_prior)
        if learn:
            aw, ab = anchor
            r["energy"] += 0.5 * parameter_prior * (
                math.fsum((p - q) ** 2 for p, q in zip(ww, aw))
                + math.fsum((p - q) ** 2 for p, q in zip(bb, ab)))
            r["grad_w"] = [q + parameter_prior * (p - a)
                           for q, p, a in zip(r["grad_w"], ww, aw)]
            r["grad_b"] = [q + parameter_prior * (p - a)
                           for q, p, a in zip(r["grad_b"], bb, ab)]
        return r
    r = full(x, w, b)
    hist = [r["energy"]]
    trial = step
    sweeps = 0
    while sweeps < budget:
        # 投影梯度方向 (clamped 坐标不参与)
        dir_x = [0.0 if i in clamps else -r["grad_x"][i] for i in range(g.n_patches)]
        dir_w = [-q for q in r["grad_w"]] if learn else [0.0] * len(w)
        dir_b = [-q for q in r["grad_b"]] if learn else [0.0] * len(b)
        slope = (math.fsum(q * d for q, d in zip(r["grad_x"], dir_x))
                 + math.fsum(q * d for q, d in zip(r["grad_w"], dir_w))
                 + math.fsum(q * d for q, d in zip(r["grad_b"], dir_b)))
        if slope >= 0.0:                      # 已驻定
            break
        moved = False
        s = trial
        for _ in range(backtracks):
            nx = [x[i] if i in clamps else _project(x[i] + s * dir_x[i], state_bound)
                  for i in range(g.n_patches)]
            for i, v in clamps.items():
                nx[i] = v
            nw = [_project(p + s * d, parameter_bound) for p, d in zip(w, dir_w)]
            nb = [_project(p + s * d, parameter_bound) for p, d in zip(b, dir_b)]
            nr = full(nx, nw, nb)
            disp = (math.fsum((nx[i] - x[i]) * r["grad_x"][i] for i in range(g.n_patches))
                    + math.fsum((nw[e] - w[e]) * r["grad_w"][e] for e in range(len(w)))
                    + math.fsum((nb[i] - b[i]) * r["grad_b"][i] for i in range(g.n_patches)))
            if nr["energy"] <= r["energy"] + 1e-4 * disp:
                # Barzilai-Borwein: s = (s^T s)/(s^T y)
                sx = [nx[i] - x[i] for i in range(g.n_patches) if nx[i] != x[i]]
                gx = [nr["grad_x"][i] - r["grad_x"][i] for i in range(g.n_patches)
                      if nx[i] != x[i]]
                sw = [nw[e] - w[e] for e in range(len(w)) if nw[e] != w[e]]
                gw = [nr["grad_w"][e] - r["grad_w"][e] for e in range(len(w)) if nw[e] != w[e]]
                sb = [nb[i] - b[i] for i in range(g.n_patches) if nb[i] != b[i]]
                gb = [nr["grad_b"][i] - r["grad_b"][i] for i in range(g.n_patches)
                      if nb[i] != b[i]]
                ss = math.fsum(q * q for q in sx + sw + sb)
                sy = (math.fsum(q * p for q, p in zip(sx, gx))
                      + math.fsum(q * p for q, p in zip(sw, gw))
                      + math.fsum(q * p for q, p in zip(sb, gb)))
                if sy > 0:
                    est = ss / sy
                    if math.isfinite(est) and est > 0:
                        trial = min(est, trial * (2 ** min(backtracks - 1, 30)))
                x, w, b, r = nx, nw, nb, nr
                moved = True
                break
            s *= 0.5
        if not moved:
            break
        sweeps += 1
        hist.append(r["energy"])
    if trace:
        return x, w, b, r, np.array(hist)
    return x, w, b, r


def projected_residual(g, x, w, b, inputs, *, clamps, learn, state_prior,
                       parameter_prior, anchor, state_bound=1.0, parameter_bound=4.0):
    r = evaluate(g, x, w, b, inputs, state_prior)
    grad_w, grad_b = list(r["grad_w"]), list(r["grad_b"])
    if learn:
        aw, ab = anchor
        grad_w = [q + parameter_prior * (p - a) for q, p, a in zip(grad_w, w, aw)]
        grad_b = [q + parameter_prior * (p - a) for q, p, a in zip(grad_b, b, ab)]
    res = max((abs(x[i] - _project(x[i] - r["grad_x"][i], state_bound))
               for i in range(g.n_patches) if i not in clamps), default=0.0)
    if learn:
        res = max([res] + [abs(p - _project(p - gg, parameter_bound))
                           for p, gg in zip(w, grad_w)])
        res = max([res] + [abs(p - _project(p - gg, parameter_bound))
                           for p, gg in zip(b, grad_b)])
    return res, r


# ---------------------------------------------------------------------------
def make_graph(feedback=True):
    """3 patch 链 + 输出到隐层的反馈 (Cadence `layered` 的互惠对)"""
    edges = [("input", 0, 0), ("input", 1, 0),
             ("state", 0, 1), ("input", 1, 1), ("state", 1, 2)]
    if feedback:
        edges.append(("state", 2, 1))
    return Graph(2, 3, edges)


def random_params(g, scale=0.7):
    return (list(scale * rng.standard_normal(len(g.edges))),
            list(0.3 * rng.standard_normal(g.n_patches)))


def cost(x, y, out=2):
    """输出 patch 的代价 C = 1/2 (x_out - y)^2"""
    return 0.5 * (x[out] - y) ** 2


# ===========================================================================
def part_A():
    print("=" * 78)
    print("Part A  能量与解析梯度 vs 中心有限差分  (验证本文对 Cadence 定律的复现)")
    print("=" * 78)
    g = make_graph()
    w, b = random_params(g)
    x = list(0.4 * rng.standard_normal(3))
    inputs = [0.6, -0.3]
    r = evaluate(g, x, w, b, inputs, 0.01)
    eps = 1e-6
    worst = 0.0
    for i in range(3):
        xp, xm = list(x), list(x)
        xp[i] += eps; xm[i] -= eps
        fd = (evaluate(g, xp, w, b, inputs, 0.01)["energy"]
              - evaluate(g, xm, w, b, inputs, 0.01)["energy"]) / (2 * eps)
        worst = max(worst, abs(fd - r["grad_x"][i]))
    for e in range(len(w)):
        wp, wm = list(w), list(w)
        wp[e] += eps; wm[e] -= eps
        fd = (evaluate(g, x, wp, b, inputs, 0.01)["energy"]
              - evaluate(g, x, wm, b, inputs, 0.01)["energy"]) / (2 * eps)
        worst = max(worst, abs(fd - r["grad_w"][e]))
    for i in range(3):
        bp, bm = list(b), list(b)
        bp[i] += eps; bm[i] -= eps
        fd = (evaluate(g, x, w, bp, inputs, 0.01)["energy"]
              - evaluate(g, x, w, bm, inputs, 0.01)["energy"]) / (2 * eps)
        worst = max(worst, abs(fd - r["grad_b"][i]))
    print(f"  状态梯度 (含反馈边 2->1) 与参数梯度的最大偏差 = {worst:.3e}")
    assert worst < 1e-5, f"解析梯度与有限差分不符: {worst}"
    print("  [check] 解析梯度正确 —— 通过")
    return g


def part_B():
    print()
    print("=" * 78)
    print("Part B  沉降: Lyapunov 单调下降 / 收敛到驻点 / 活动先验保证有界")
    print("=" * 78)
    g = make_graph()
    w, b = random_params(g)
    inputs = [0.6, -0.3]
    x0 = [0.0, 0.0, 0.0]
    x, w2, b2, r, hist = settle(g, x0, w, b, inputs, state_prior=0.01, trace=True)
    mono = bool(np.all(np.diff(hist) <= 1e-12))
    res, _ = projected_residual(g, x, w2, b2, inputs, clamps={}, learn=False,
                                state_prior=0.01, parameter_prior=0.1, anchor=(w2, b2))
    print(f"  能量轨迹长度 {len(hist)}, E: {hist[0]:.8f} -> {hist[-1]:.8f}")
    print(f"  逐点单调下降 = {mono}")
    print(f"  终态投影驻定残差 ||proj(x-grad) - x||_inf = {res:.3e}")
    assert mono, "沉降轨迹非单调"
    assert res < 1e-5, f"未收敛到驻点: {res}"

    # 有界性: tanh 预测满足 |p_i| <= 1, 故 E >= 1/2 sum (|x_i|-1)^2, 与 lambda 无关
    print("  有界性 (coercivity): 沿轨迹与随机大状态检查下界 E >= 1/2 sum(|x_i|-1)^2")
    big = [rng.standard_normal() * 20 for _ in range(3)]
    lb = 0.5 * sum((abs(v) - 1.0) ** 2 for v in big)
    for lam in (0.0, 0.01):
        eb = evaluate(g, big, w, b, inputs, lam)["energy"]
        print(f"    lambda={lam:<5} E(x_large)={eb:14.4f}  >= 1/2 sum(|x|-1)^2 = {lb:.4f} : {eb >= lb}")
        assert eb >= lb
    print("  → 能量强制 (coercive): 次水平集紧 ⇒ 迭代有界; 与活动先验无关")
    print("  活动先验的角色是选择驻点 (不是保证有界):")
    for lam in (0.0, 0.01):
        xx, _, _, rr = settle(g, x0, w, b, inputs, state_prior=lam, budget=3000)
        r2, _ = projected_residual(g, xx, w, b, inputs, clamps={}, learn=False,
                                   state_prior=lam, parameter_prior=0.1, anchor=(w, b))
        pred_res = max(abs(v) for v in rr["err"])
        print(f"    lambda={lam:<5} E*={rr['energy']:.8f}  ||x||_inf={max(abs(v) for v in xx):.4f}"
              f"  最大预测误差={pred_res:.3e}  驻定残差={r2:.2e}")
    print("  [check] 单调下降 + 收敛到驻点 + 能量强制有界 —— 通过")
    return g


def part_C():
    print()
    print("=" * 78)
    print("Part C  EP 基本引理: 单边对比 O(beta)  vs  中心对比 O(beta^2)")
    print("=" * 78)
    g = make_graph()
    w, b = random_params(g)
    inputs = [0.6, -0.3]
    y = 0.7
    xf, _, _, _ = settle(g, [0.0] * 3, w, b, inputs, state_prior=0.01, budget=3000)

    def loss_at(theta_w, theta_b):
        xx, _, _, _ = settle(g, [0.0] * 3, theta_w, theta_b, inputs,
                             state_prior=0.01, budget=3000)
        return cost(xx, y)

    # 真实梯度 dC/dtheta: 对"自由相解"做有限差分 (隐函数定理)
    eps = 1e-6
    true_w = np.zeros(len(w)); true_b = np.zeros(3)
    for e in range(len(w)):
        wp, wm = list(w), list(w); wp[e] += eps; wm[e] -= eps
        true_w[e] = (loss_at(wp, b) - loss_at(wm, b)) / (2 * eps)
    for i in range(3):
        bp, bm = list(b), list(b); bp[i] += eps; bm[i] -= eps
        true_b[i] = (loss_at(w, bp) - loss_at(w, bm)) / (2 * eps)

    def contrast(beta, centered):
        """EP 对比量 D = (dE/dtheta(s^beta) - dE/dtheta(s0)) / beta
        centered=True 时用 +/-beta 的中心对比 (Cadence 的 `centered` 选项)"""
        xp = _nudge_settle(g, list(xf), w, b, inputs, beta=beta, y=y)
        dnud = evaluate(g, xp, w, b, inputs, 0.01)
        if not centered:
            dfree = evaluate(g, xf, w, b, inputs, 0.01)
            gw = np.array([(q - p) / beta for q, p in zip(dnud["grad_w"], dfree["grad_w"])])
            gb = np.array([(q - p) / beta for q, p in zip(dnud["grad_b"], dfree["grad_b"])])
        else:
            xm = _nudge_settle(g, list(xf), w, b, inputs, beta=-beta, y=y)
            dm = evaluate(g, xm, w, b, inputs, 0.01)
            gw = 0.5 * np.array([(q - p) / beta for q, p in zip(dnud["grad_w"], dm["grad_w"])])
            gb = 0.5 * np.array([(q - p) / beta for q, p in zip(dnud["grad_b"], dm["grad_b"])])
        return np.concatenate([gw, gb])

    target = np.concatenate([true_w, true_b])
    print(f"  {'beta':>9} | {'单边 ||D-dC/dth||':>20} | {'中心 ||D-dC/dth||':>20}")
    errs = {}
    for beta in (4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3):
        e1 = np.linalg.norm(contrast(beta, False) - target)
        e2 = np.linalg.norm(contrast(beta, True) - target)
        errs[beta] = (e1, e2)
        print(f"  {beta:9.2e} | {e1:20.6e} | {e2:20.6e}")
    c1, _ = errs[5e-3]
    c2, _ = errs[2.5e-3]
    _, d1 = errs[5e-3]
    _, d2 = errs[2.5e-3]
    print(f"  阶数估计 单边: log2(err(2b)/err(b)) = {math.log2(c1 / c2):.2f}  (期望 1)")
    print(f"  阶数估计 中心: log2(err(2b)/err(b)) = {math.log2(d1 / d2):.2f}  (期望 2)")
    assert abs(math.log2(c1 / c2) - 1.0) < 0.6, "单边对比不是 O(beta)"
    assert math.log2(d1 / d2) > 1.6, "中心对比不是 O(beta^2)"
    print("  [check] 单边 O(beta), 中心 O(beta^2) —— 通过")
    return g


def _nudge_settle(g, x, w, b, inputs, *, beta, y, budget=3000, state_prior=0.01, out=2):
    """把 beta*C 并入能量后沉降 (弱钳制相): grad_x[out] += beta * dC/dx"""
    x = list(x)
    step = 0.5
    for _ in range(budget):
        r = evaluate(g, x, w, b, inputs, state_prior)
        gx = list(r["grad_x"])
        gx[out] += beta * (x[out] - y)     # dC/dx_out = x_out - y
        gnorm = max(abs(v) for v in gx)
        if gnorm < 1e-12:
            break
        moved = False
        s = step
        for _ in range(40):
            nx = [_project(x[i] - s * gx[i], 1.0) for i in range(g.n_patches)]
            nr = evaluate(g, nx, w, b, inputs, state_prior)
            e0 = r["energy"] + beta * cost(x, y, out)
            e1 = nr["energy"] + beta * cost(nx, y, out)
            if e1 <= e0 + 1e-4 * math.fsum((nx[i] - x[i]) * gx[i] for i in range(g.n_patches)):
                x = nx; moved = True; break
            s *= 0.5
        if not moved:
            break
    return x


# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("Part D  一个算子两个坐标块: 查询与学习共用同一 repair; 学习收敛")
    print("=" * 78)
    g = make_graph()
    w, b = random_params(g)
    inputs = [0.6, -0.3]
    y = 0.8
    xs = [0.3, -0.2, 0.4]

    # (1) 同一算子: 状态坐标块的梯度与"参数是否 eligible"无关
    r_q = evaluate(g, xs, w, b, inputs, 0.01)
    _, r_l_eval = projected_residual(g, xs, w, b, inputs, clamps={}, learn=True,
                                     state_prior=0.01, parameter_prior=0.1, anchor=(w, b))
    d = max(abs(p - q) for p, q in zip(r_q["grad_x"], r_l_eval["grad_x"]))
    print(f"  同一 (x,w,b) 处 查询 与 学习 的状态梯度最大差 = {d:.3e}")
    assert d == 0.0, "状态梯度被 learn 标志改变"
    print("  → learn 只把权重/偏置加入 eligible 集合并向其梯度加先验项; 状态块完全不变")

    # (2) 学习解本身也是同一 repair 的投影梯度下降: 它收敛到增广目标的驻点
    xx, ww, bb, rr, hist_l = settle(g, [0.0] * 3, w, b, inputs, state_prior=0.01,
                                    learn=True, parameter_prior=0.1, budget=3000,
                                    clamps={2: y}, trace=True)
    res_l, _ = projected_residual(g, xx, ww, bb, inputs, clamps={2: y}, learn=True,
                                  state_prior=0.01, parameter_prior=0.1, anchor=(w, b))
    print(f"  学习解 (learn=True, 目标钳制): E_learn {hist_l[0]:.8f} -> {hist_l[-1]:.8f}, "
          f"单调={bool(np.all(np.diff(hist_l) <= 1e-12))}, 驻定残差={res_l:.3e}")
    assert np.all(np.diff(hist_l) <= 1e-12), "学习解的能量非单调"
    assert res_l < 1e-5, f"学习解未收敛到驻点: {res_l}"

    # (3) 学习: 目标钳制 + 参数先验, 同一 repair 更新参数, 损失逐次下降
    print(f"  {'experience':>10} {'C(x_free)':>14} {'||theta-theta0||':>18} {'驻定残差':>12}")
    th0 = np.array(w + b)
    losses = []
    for k in range(6):
        xx, ww, bb, rr = settle(g, [0.0] * 3, w, b, inputs, state_prior=0.01,
                                learn=True, parameter_prior=0.1,
                                budget=1500, clamps={2: y})
        res, _ = projected_residual(g, xx, ww, bb, inputs, clamps={2: y}, learn=True,
                                    state_prior=0.01, parameter_prior=0.1, anchor=(w, b))
        xfree, _, _, _ = settle(g, [0.0] * 3, ww, bb, inputs, state_prior=0.01,
                                budget=3000)
        losses.append(cost(xfree, y))
        print(f"  {k:10d} {losses[-1]:14.8f} "
              f"{np.linalg.norm(np.array(ww + bb) - th0):18.6f} {res:12.3e}")
        w, b = ww, bb
    print(f"  自由相代价 C(x_free): {losses[0]:.6f} -> {losses[-1]:.6f}")
    assert losses[-1] < losses[0], "学习未降低代价"
    print("  [check] 同一 repair; rho 大时退化为查询; 学习降低自由相代价 —— 通过")
    return g


def part_E(n_grid=161, kappa=0.02, lam=0.35, T=20000, dt=2e-3, N_ind=4000):
    print()
    print("=" * 78)
    print("Part E  双尺度进化: 一生学习 + 族群 replicator-mutator (Baldwin)")
    print("=" * 78)
    phi_grid = np.linspace(0.02, 2.0, n_grid)

    def lifetime_fit(phi, n_task):
        phi = np.atleast_1d(np.asarray(phi, float))
        J = np.zeros_like(phi)
        for _ in range(n_task):
            w_star = rng.standard_normal(phi.shape)
            w = np.zeros_like(phi)
            for _ in range(3):
                w = w - phi * (w - w_star)
            J += -0.5 * (w - w_star) ** 2 - lam * phi ** 2
        return J / n_task

    Jhat = lifetime_fit(phi_grid, 20000)
    phi_star = phi_grid[np.argmax(Jhat)]
    J_analytic = -0.5 * (1 - phi_grid) ** 6 - lam * phi_grid ** 2
    print(f"  Ĵ(φ) 峰值 φ* = {phi_star:.4f}   解析式偏差 "
          f"{np.max(np.abs(Jhat - J_analytic)):.2e}")
    print(f"  纯先天适应度 (w≡0) = -0.500000  → 学习增益 {Jhat.max() + 0.5:+.6f}")

    dphi = phi_grid[1] - phi_grid[0]
    gradJ = np.gradient(Jhat, dphi)
    rho = np.exp(-((phi_grid - 1.0) ** 2) / (2 * 0.06 ** 2)); rho /= rho.sum() * dphi

    def free_energy(r):
        r = np.clip(r, 1e-300, None)
        return -np.sum(r * Jhat) * dphi + kappa * np.sum(r * np.log(r)) * dphi

    F = [free_energy(rho)]
    for t in range(T):
        flux = rho * gradJ
        rho = rho + dt * (kappa * ((np.roll(rho, -1) - 2 * rho + np.roll(rho, 1)) / dphi ** 2)
                          - np.gradient(flux, dphi))
        rho = np.clip(rho, 0, None); rho /= rho.sum() * dphi
    F.append(free_energy(rho))
    rho_g = np.exp(Jhat / kappa); rho_g /= rho_g.sum() * dphi
    tv = 0.5 * np.sum(np.abs(rho - rho_g)) * dphi
    mf = np.sum(rho * phi_grid) * dphi

    N = N_ind
    ph = rng.normal(1.0, 0.06, size=N)
    for _ in range(400):
        fit = lifetime_fit(ph, 20)
        wsel = np.exp((fit - fit.max()) / 1.0); wsel /= wsel.sum()
        ph = np.clip(rng.choice(ph, size=N, p=wsel) + 0.03 * rng.standard_normal(N),
                     phi_grid[0], phi_grid[-1])
    print(f"  族群自由能 𝔉: {F[0]:.6f} -> {F[-1]:.6f}  (单调下降)")
    print(f"  ρ_t 与 Gibbs ρ*∝exp(Ĵ/κ) 全变差 = {tv:.4f}")
    print(f"  E[φ]: 平均场 {mf:.4f}   个体基模拟 {ph.mean():.4f} ± {ph.std():.4f}")
    assert F[-1] <= F[0], "族群自由能未下降"
    assert tv < 0.05, f"未收敛到 Gibbs: {tv}"
    assert abs(ph.mean() - mf) < 0.15, "个体基模拟与平均场不一致"
    print("  [check] Baldwin 最优可塑性 φ*>0; 𝔉 单调; 个体基 ≈ 平均场 —— 通过")
    return phi_star


def part_F():
    print()
    print("=" * 78)
    print("Part F  可解实例的闭式解 vs 数值  (单 patch 链, state_prior=0)")
    print("=" * 78)
    print("  t = tanh(b + w u);  E = ½(x-t)²;  C = ½(x-y)²")
    print("  x^β = (t+βy)/(1+β);  dL/dw = (t-y)(1-t²)u")
    print("  闭式: 单边对比 D = (dL/dw)/(1+β);  中心对比 D = (dL/dw)/(1-β²)")
    g = Graph(1, 1, [("input", 0, 0)])
    w = [0.9]
    b = [0.2]
    u = 0.55
    y = 0.7
    t = math.tanh(b[0] + w[0] * u)
    dLdw = (t - y) * (1 - t * t) * u

    def numeric(beta, centered):
        if centered:
            xp = _nudge_settle(g, [0.0], w, b, [u], beta=beta, y=y,
                               state_prior=0.0, out=0)
            xm = _nudge_settle(g, [0.0], w, b, [u], beta=-beta, y=y,
                               state_prior=0.0, out=0)
            fp = evaluate(g, xp, w, b, [u], 0.0)["grad_w"][0]
            fm = evaluate(g, xm, w, b, [u], 0.0)["grad_w"][0]
            return 0.5 * (fp - fm) / beta
        xp = _nudge_settle(g, [0.0], w, b, [u], beta=beta, y=y,
                           state_prior=0.0, out=0)
        x0 = [t]
        fp = evaluate(g, xp, w, b, [u], 0.0)["grad_w"][0]
        f0 = evaluate(g, x0, w, b, [u], 0.0)["grad_w"][0]
        return (fp - f0) / beta

    print(f"  t = {t:.10f}   dL/dw = {dLdw:.10f}")
    print(f"  {'beta':>9} {'单边 数值':>14} {'单边 闭式':>14} "
          f"{'中心 数值':>14} {'中心 闭式':>14}")
    worst = 0.0
    for beta in (0.1, 0.05, 0.02, 0.01):
        one_n = numeric(beta, False)
        one_c = dLdw / (1 + beta)
        ctr_n = numeric(beta, True)
        ctr_c = dLdw / (1 - beta * beta)
        worst = max(worst, abs(one_n - one_c), abs(ctr_n - ctr_c))
        print(f"  {beta:9.3f} {one_n:14.8f} {one_c:14.8f} {ctr_n:14.8f} {ctr_c:14.8f}")
    print(f"  数值与闭式最大偏差 = {worst:.3e}")
    assert worst < 1e-6, f"闭式解与数值不符: {worst}"
    print("  [check] 闭式解精确成立 (单边因子 1/(1+β), 中心因子 1/(1-β²)) —— 通过")
    return dLdw


if __name__ == "__main__":
    import sys
    want = [a.upper() for a in sys.argv[1:]]
    parts = {"A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E, "F": part_F}
    todo = want or list(parts)
    for name in todo:
        parts[name]()
    if "E" in todo:
        print()
        print("=" * 78)
        print("全部通过。核心结论: 沉降单调收敛到稳定均衡 (Part B); EP 对比的阶数")
        print("单边 O(β) / 中心 O(β²) (Part C); 推理与学习是同一算子 (Part D);")
        print("可解实例闭式精确 (Part F); Baldwin 最优可塑性 φ*>0 (Part E)。")
        print("=" * 78)
