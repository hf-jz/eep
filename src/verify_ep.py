#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ep.py  --  数值验证：均衡传播 (Equilibrium Propagation, EP) 及其双尺度进化延伸

   Part A  EP 基本引理 (fundamental lemma) 的数值验证： Δθ_EP = -∂L/∂θ + O(β)
   Part B  沉降即计算：能量单调性 + 对比结构 (E 升, F 降)
   Part C  EP 训练一个非线性损失 (softmax 交叉熵) 的能量模型
   Part D  双时间尺度 (进化延伸) 的 averaging / replicator-mutator 验证

仅依赖 numpy。运行： /usr/bin/python3 src/verify_ep.py
"""

import numpy as np

rng = np.random.default_rng(20261010)

# ----------------------------------------------------------------------------
# 能量模型 (energy-based model, EBM)
#   x (n_in) 钳制输入; h (n_h) 隐层; o (n_out) 输出层
#   E(h,o; θ, x) = ½|h|² + ½|o|² - (W1 x)·h - (W2 h)·o - b·h - c·o
#   θ = {W1, W2, b, c}
#   Hessian = [[I, -W2ᵀ], [-W2, I]] ⪰ (1-‖W2‖)I  →  ‖W2‖<1 时严格凸 → 唯一极小
#   均衡: h = W1x + W2ᵀ o + b ;  o = W2 h + c     (含反馈 W2ᵀ 的循环网络)
# ----------------------------------------------------------------------------
class EBM:
    def __init__(self, n_in, n_h, n_out, scale=0.35):
        self.n_in, self.n_h, self.n_out = n_in, n_h, n_out
        u = lambda *s: scale * rng.standard_normal(s) / np.sqrt(s[0])
        self.W1, self.W2 = u(n_h, n_in), u(n_out, n_h)
        # 保证 ‖W2‖ < 1 (严格凸性 / 唯一均衡)
        s = np.linalg.svd(self.W2, compute_uv=False)[0]
        if s >= 0.9:
            self.W2 *= 0.9 / s
        self.b, self.c = u(n_h), u(n_out)

    # --- 能量与梯度 ---------------------------------------------------------
    def energy_grad(self, h, o, x, Cb=None):
        """E 与 ∇E; Cb = (softmax(o)-y) 时把 β*C 的梯度并入 (由调用方乘 β)"""
        grad_h = h - (self.W1 @ x) - (self.W2.T @ o) - self.b
        grad_o = o - (self.W2 @ h) - self.c
        if Cb is not None:
            grad_o = grad_o + Cb          # 调用方已把 β 乘进 Cb
        return grad_h, grad_o

    def energy(self, h, o, x):
        return (0.5 * h @ h + 0.5 * o @ o - (self.W1 @ x) @ h
                - (self.W2 @ h) @ o - self.b @ h - self.c @ o)

    def hess(self):
        """M = [[I, -W2ᵀ], [-W2, I]]  (严格正定, ‖W2‖<1)"""
        n_h, n_o = self.n_h, self.n_out
        M = np.eye(n_h + n_o)
        M[:n_h, n_h:] = -self.W2.T
        M[n_h:, :n_h] = -self.W2
        return M

    def settle(self, x, beta=0.0, y=None, h0=None, o0=None, steps=200, tol=1e-14):
        """精确求解 min_{h,o} F = E + β·C  (C = 交叉熵)
        自由相 (β=0): 闭式线性求解;  钳制相: 不动点迭代 (β 小 ⇒ 快速收缩)"""
        M = self.hess()
        r = np.concatenate([self.W1 @ x + self.b, self.c])
        s = np.linalg.solve(M, r)
        if beta > 0.0:
            for _ in range(steps):
                o = s[self.n_h:]
                r2 = r.copy(); r2[self.n_h:] -= beta * (_softmax(o) - y)
                s_new = np.linalg.solve(M, r2)
                if np.linalg.norm(s_new - s, np.inf) < tol:
                    s = s_new; break
                s = s_new
        return s[:self.n_h].copy(), s[self.n_h:].copy()

    def relax_trajectory(self, x, beta=0.0, y=None, steps=400, dt=0.4):
        """显式梯度流 (沉降动力学), 返回 F 的轨迹 —— 用于验证单调性"""
        h = np.zeros(self.n_h); o = np.zeros(self.n_out)
        Fs = []
        for _ in range(steps):
            Cb = beta * (_softmax(o) - y) if beta > 0 else None
            gh, go = self.energy_grad(h, o, x, Cb)
            Fs.append(self.energy(h, o, x)
                      + (beta * _ce(_softmax(o), y) if beta > 0 else 0.0))
            h, o = h - dt * gh, o - dt * go
        return np.array(Fs)

    # --- 参数梯度 ∂E/∂θ (供 EP 使用) ---------------------------------------
    def dEdtheta(self, h, o, x):
        return {'W1': -np.outer(h, x), 'W2': -np.outer(o, h),
                'b': -h, 'c': -o}

    def params(self):
        return {'W1': self.W1, 'W2': self.W2, 'b': self.b, 'c': self.c}

    def pack(self):
        return np.concatenate([self.W1.ravel(), self.W2.ravel(), self.b, self.c])

    def unpack(self, flat):
        i = 0
        for k, arr in (('W1', self.W1), ('W2', self.W2)):
            n = arr.size
            getattr(self, k)[:] = flat[i:i + n].reshape(arr.shape); i += n
        self.b[:] = flat[i:i + self.n_h]; i += self.n_h
        self.c[:] = flat[i:i + self.n_out]


def _softmax(o):
    z = o - o.max(); e = np.exp(z); return e / e.sum()

def _ce(p, y):
    return -np.sum(y * np.log(np.clip(p, 1e-300, None)))


# ============================================================================
# Part A : EP 基本引理的数值验证
# ============================================================================
def part_A():
    print("=" * 74)
    print("Part A  EP 基本引理:  Δθ_EP(β) = -∇L + O(β)   [L = C(o⁰, y)]")
    print("=" * 74)
    net = EBM(4, 6, 3)
    x = rng.standard_normal(4)
    y = np.zeros(3); y[rng.integers(3)] = 1.0

    def loss_at(theta_flat):
        net.unpack(theta_flat)
        h, o = net.settle(x, beta=0.0)
        return _ce(_softmax(o), y)

    def true_grad():
        th0 = net.pack(); g = np.zeros_like(th0); eps = 1e-6
        for i in range(th0.size):
            tp = th0.copy(); tp[i] += eps
            tm = th0.copy(); tm[i] -= eps
            g[i] = (loss_at(tp) - loss_at(tm)) / (2 * eps)
        net.unpack(th0)                      # 复原参数
        return g

    gL = true_grad()
    h0_, o0_ = net.settle(x, beta=0.0)

    print(f"{'β':>10} {'‖Δθ + ∇L‖':>14} {'‖err‖/β':>11} {'cos(Δθ,-∇L)':>13}")
    prev = None
    for beta in [4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3]:
        hb, ob = net.settle(x, beta=beta, y=y, h0=h0_, o0=o0_)
        d0, db = net.dEdtheta(h0_, o0_, x), net.dEdtheta(hb, ob, x)
        dtheta = np.concatenate([((db['W1'] - d0['W1']) / beta).ravel(),
                                 ((db['W2'] - d0['W2']) / beta).ravel(),
                                 (db['b'] - d0['b']) / beta,
                                 (db['c'] - d0['c']) / beta])
        update = -dtheta                        # EP 更新 Δθ = -(1/β)Δ(∂E/∂θ)
        err = np.linalg.norm(update + gL)
        cosv = float(update @ (-gL) / (np.linalg.norm(update) * np.linalg.norm(gL) + 1e-30))
        print(f"{beta:10.2e} {err:14.6e} {err/beta:11.6f} {cosv:13.10f}")
        prev = err
    # 收敛阶: err(β)/β 应趋于常数 (一阶 O(β))
    assert err / beta < 1e1, f"EP 残差不是 O(β): err/β = {err/beta}"
    assert cosv > 0.999, f"EP 更新方向错误: cos = {cosv}"
    print(f"  [check] 残差 O(β)（err/β = {err/beta:.3f}）且方向正确（cos = {cosv:.9f}）—— 通过")
    return net, x, y


# ============================================================================
# Part B : 沉降即计算 —— 能量单调 + 对比结构
# ============================================================================
def part_B():
    print()
    print("=" * 74)
    print("Part B  沉降即计算: 自由相 E 最小; 钳制相 E 升而总能量 F 降")
    print("=" * 74)
    net = EBM(4, 6, 3)
    x = rng.standard_normal(4)
    y = np.zeros(3); y[rng.integers(3)] = 1.0
    beta = 1e-2
    h0, o0 = net.settle(x, beta=0.0)
    hb, ob = net.settle(x, beta=beta, y=y, h0=h0, o0=o0)
    E0, Eb = net.energy(h0, o0, x), net.energy(hb, ob, x)
    C0, Cb = _ce(_softmax(o0), y), _ce(_softmax(ob), y)
    F0, Fb = E0 + beta * C0, Eb + beta * Cb
    print(f"  自由相   E⁰ = {E0: .10f}   C⁰ = {C0:.10f}")
    print(f"  钳制相   Eβ = {Eb: .10f}   Cβ = {Cb:.10f}")
    print(f"  ΔE = {Eb-E0:+.3e}  (>0 能量升高)   ΔF = {Fb-F0:+.3e}  (<0 总能量下降)")
    print(f"  损失下降 ΔC = {Cb-C0:+.3e}")
    Fs0 = net.relax_trajectory(x, beta=0.0, steps=400)
    mono = bool(np.all(np.diff(Fs0) <= 1e-15))
    print(f"  自由相沉降轨迹 F_t 单调下降: {mono}   (F 从 {Fs0[0]:.6f} → {Fs0[-1]:.6f})")
    assert Eb - E0 > 0, "钳制相应升高系统能量"
    assert Fb - F0 < 0, "钳制相未降低总能量"
    assert Cb - C0 < 0, "钳制相未降低损失"
    assert mono, "沉降轨迹非单调"
    print("  [check] ΔE>0, ΔF<0, ΔC<0, 轨迹单调 —— 通过")
    return Eb - E0 > 0 and Fb - F0 < 0


# ============================================================================
# Part C : 用 EP 训练 (非线性 softmax 交叉熵损失)
# ============================================================================
def part_C(steps=400, lr=3.0, beta=1e-3, n_class=3):
    print()
    print("=" * 74)
    print("Part C  EP 训练能量模型 (softmax 交叉熵),  演示: 训练/推理同一套物理")
    print("=" * 74)
    net = EBM(8, 12, n_class)
    # 合成任务: 3 类高斯团簇
    centers = rng.standard_normal((n_class, 8)) * 2.0
    def sample(n):
        ys = rng.integers(n_class, size=n)
        xs = centers[ys] + 0.8 * rng.standard_normal((n, 8))
        return xs, ys
    Xtr, Ytr = sample(600)
    Xte, Yte = sample(300)

    def acc(X, Y):
        ok = 0
        for x, yv in zip(X, Y):
            _, o = net.settle(x, beta=0.0)
            ok += int(np.argmax(o) == yv)
        return ok / len(Y)

    hist = []
    for t in range(steps):
        i = rng.integers(len(Xtr))
        x, yv = Xtr[i], Ytr[i]
        y = np.zeros(n_class); y[yv] = 1.0
        h0, o0 = net.settle(x, beta=0.0)
        hb, ob = net.settle(x, beta=beta, y=y, h0=h0, o0=o0)
        d0, db = net.dEdtheta(h0, o0, x), net.dEdtheta(hb, ob, x)
        for k in ('W1', 'W2', 'b', 'c'):
            getattr(net, k)[:] -= lr * (db[k] - d0[k]) / beta   # Δθ = -η/β·Δ(∂E/∂θ)
        if t % 50 == 0:
            hist.append((t, acc(Xte, Yte)))
    a = acc(Xte, Yte)
    for t, v in hist:
        print(f"    step {t:4d}   test acc = {v:.3f}")
    print(f"    最终 test acc = {a:.3f}  (3 类, 随机基线 0.333)")
    assert a > 0.9, f"EP 训练未收敛: acc = {a}"
    print("  [check] EP 训练收敛 —— 通过")
    return a


# ============================================================================
# Part D : 双时间尺度 (进化延伸) —— Baldwin 适应 + replicator-mutator
# ============================================================================
def part_D(n_grid=161, kappa=0.02, lam=0.35, T=20000, dt=2e-3, N_ind=4000):
    print()
    print("=" * 74)
    print("Part D  双尺度进化: 快尺度 = 一生内 EP 学习; 慢尺度 = replicator-mutator")
    print("=" * 74)

    phi_grid = np.linspace(0.02, 2.0, n_grid)

    def lifetime_fit(phi, n_task):
        """向量化: 一生 K 步学习的平均适应度 - 可塑性代谢代价 λφ²
        φ 可为标量或数组 (每个个体一个元参数)"""
        phi = np.atleast_1d(np.asarray(phi, float))
        J = np.zeros_like(phi)
        for _ in range(n_task):
            w_star = rng.standard_normal(phi.shape)
            w = np.zeros_like(phi)
            for _ in range(3):                 # 一生内 3 次局部 (EP 式) 更新
                w = w - phi * (w - w_star)
            J += -0.5 * (w - w_star) ** 2 - lam * phi ** 2
        return J / n_task

    # 有效 (后学习 / Baldwin) 适应度 Ĵ(φ): 在网格上以大量任务平均估计
    Jhat = lifetime_fit(phi_grid, 20000)
    phi_star = phi_grid[np.argmax(Jhat)]
    J_innate = -0.5 * 1.0                       # 无学习 (w≡0) 的期望适应度
    print(f"  Ĵ(φ) 峰值  φ* = {phi_star:.4f}   Ĵ(φ*) = {Jhat.max():.6f}")
    print(f"  纯先天适应度 (w≡0) = {J_innate:.6f}   → 学习带来的适应度增益 "
          f"{Jhat.max()-J_innate:+.6f}")
    print(f"  最优可塑性 φ* > 0  ⇒ 进化保留'会学习'的表型 (Baldwin, 非 Lamarck)")
    # 解析核对: w = w*(1-(1-φ)³) ⇒ Ĵ(φ) = -½(1-φ)⁶ - λφ²
    J_analytic = -0.5 * (1 - phi_grid) ** 6 - lam * phi_grid ** 2
    print(f"  与解析式 Ĵ=-½(1-φ)⁶-λφ² 的最大偏差 = {np.max(np.abs(Jhat-J_analytic)):.2e}")

    # 慢尺度 PDE: ∂_t ρ = κ ∂²ρ - ∂_φ(ρ·∂_φ(-Ĵ))   (Wasserstein 梯度流)
    dphi = phi_grid[1] - phi_grid[0]
    gradJ = np.gradient(Jhat, dphi)
    def free_energy(rho):
        rho = np.clip(rho, 1e-300, None)
        return -np.sum(rho * Jhat) * dphi + kappa * np.sum(rho * np.log(rho)) * dphi
    rho = np.exp(-((phi_grid - 1.0) ** 2) / (2 * 0.06 ** 2))
    rho /= rho.sum() * dphi
    Fpath = [free_energy(rho)]
    for t in range(T):
        flux = rho * gradJ
        dflux = np.gradient(flux, dphi)
        lap = (np.roll(rho, -1) - 2 * rho + np.roll(rho, 1)) / dphi ** 2
        lap[0] = lap[-1] = 0.0
        rho = rho + dt * (kappa * lap - dflux)      # ∂_tρ = κρ'' - ∂_φ(ρ·Ĵ')
        rho = np.clip(rho, 0, None); rho /= rho.sum() * dphi
        if t % (T // 8) == 0:
            Fpath.append(free_energy(rho))
    Fpath.append(free_energy(rho))
    F = np.array(Fpath); dF = np.diff(F)
    print(f"  自由能 𝓕(ρ) 单调下降: {bool(np.all(dF <= 1e-14))}   "
          f"(𝓕: {F[0]:.6f} → {F[-1]:.6f})")
    rho_gibbs = np.exp(Jhat / kappa); rho_gibbs /= rho_gibbs.sum() * dphi
    tv = 0.5 * np.sum(np.abs(rho - rho_gibbs)) * dphi
    print(f"  ρ_t 与 Gibbs ρ*∝exp(Ĵ/κ) 全变差距离 = {tv:.4f}   "
          f"E[φ]: ρ_t={np.sum(rho*phi_grid)*dphi:.4f}   Gibbs={np.sum(rho_gibbs*phi_grid)*dphi:.4f}")

    # 个体基随机模拟: 验证平均场 PDE 是真实种群动力学的极限
    N = N_ind
    ph = rng.normal(1.0, 0.06, size=N)
    for gen in range(400):
        fit = lifetime_fit(ph, 20)
        w = np.exp((fit - fit.max()) / 1.0); w /= w.sum()
        ph = rng.choice(ph, size=N, p=w)
        ph = np.clip(ph + 0.03 * rng.standard_normal(N), phi_grid[0], phi_grid[-1])
    ph_mean, ph_sd, mf_mean = ph.mean(), ph.std(), np.sum(rho * phi_grid) * dphi
    print(f"  个体基模拟 (N={N}, 400 代) 稳态 E[φ] = {ph_mean:.4f} ± {ph_sd:.4f}  "
          f"(平均场 {mf_mean:.4f})")
    Fmono = bool(np.all(np.diff(F) <= 1e-14))
    assert Fmono, "族群自由能非单调下降"
    assert tv < 0.05, f"ρ_t 未收敛到 Gibbs: TV = {tv}"
    assert abs(ph_mean - mf_mean) < 0.15, \
        f"个体基模拟与平均场不一致: {ph_mean:.4f} vs {mf_mean:.4f}"
    print(f"  [check] 𝔉 单调；TV={tv:.4f}<0.05；个体基与平均场差 "
          f"{abs(ph_mean-mf_mean):.4f}<0.15 —— 通过")
    return phi_star


if __name__ == "__main__":
    part_A()
    part_B()
    acc = part_C()
    phi_star = part_D()
    print()
    print("=" * 74)
    print(f"结论: EP 基本引理数值成立；双尺度延伸的 Baldwin 最优可塑性 φ*={phi_star:.3f}；")
    print(f"      自由能单调下降且 ρ_t → Gibbs；个体基模拟与平均场一致。")
    print("=" * 74)
