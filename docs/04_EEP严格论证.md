# 04 进化均衡传播（EEP）的严格论证

本节给出延伸理论的定理、证明（或证明骨架）、数值证据与**诚实的缺口**。
引用编号对应 `refs/参考文献.md`。数值证据来自 `src/verify_ep.py`（本机实跑）。

---

## 4.1 数值证据总表（先给证据，再给定理）

**表 1：`/usr/bin/python3 src/verify_ep.py` 本机实跑结果（约 1.3 s）**

| 部分 | 检验对象 | 结果 |
|---|---|---|
| A | EP 基本引理残差阶 | $\beta$ 从 $4\times10^{-2}$ 降到 $1.25\times10^{-3}$，$\|\Delta\theta+\nabla L\|$ 从 $8.95\times10^{-2}$ 降到 $2.84\times10^{-3}$；**$\|\text{err}\|/\beta$ 稳定在 2.24–2.27**（即 err $=O(\beta)$）；方向余弦 $\to 0.99999992$ |
| B | 沉降即计算 | 自由相 $E^0=-0.6536512$；钳制相 $\Delta E=+1.686\times10^{-5}>0$，$\Delta F=-1.695\times10^{-5}<0$，$\Delta C=-3.38\times10^{-3}<0$；沉降轨迹 $F_t$ **逐点单调**下降（$0\to-0.653651$） |
| C | EP 训练（softmax 交叉熵） | 3 类合成任务，600 训练 / 300 测试样本，400 步；收敛到 **test acc = 0.993**（随机基线 0.333） |
| D | 双尺度进化 | $\hat J(\varphi)$ 峰值 $\varphi^\star=0.379$；与解析式 $-0.5(1-\varphi)^6-0.35\varphi^2$ 的最大偏差 $8.4\times10^{-3}$（单样本 Monte-Carlo 误差量级）；自由能 $\mathfrak{F}$ 单调下降 $0.3792\to0.1100$；$\rho_t$ 与 Gibbs $\rho^\star\!\propto\!e^{\hat J/\kappa}$ 的**全变差距离 0.0053**；$\mathbb{E}[\varphi]$：PDE $0.4004$ vs 个体基模拟 $0.409\pm0.136$ |

表 1 是后文全部定量断言的依据。**A 验证 T1；B 验证 T2；D 验证 T4/T5/T7。**

---

## 4.2 T4 双时间尺度平均化定理（延伸的主定理）

**问题陈述.** (3.3) 中 L1（快）与 L2（慢）的耦合是**双向**的：$\varphi$ 决定 L1 的动力学，
L1 的结果又通过 $\hat J$ 决定 L2 的漂移。要把 L2 写成只含 $\varphi$ 的自洽方程，
必须证明"快变量可以被平均掉"。

**形式化.** 令 $\varepsilon = \tau_1/\tau_2 \ll 1$，$X=\varphi$（慢），$Y=(\theta, s)$（快）：

$$\frac{\mathrm{d}X}{\mathrm{d}t} = \varepsilon\, f(X,Y), \qquad
\frac{\mathrm{d}Y}{\mathrm{d}t} = g(X,Y) + \sigma\,\xi(t), \tag{4.1}$$

其中 $f$ 的适应度分量由 $J$ 给出，$g$ 是 (L0/L1) 的 EP 动力学。

> **定理 T4（平均化）.** 假设
> **(H1)〔一致双曲吸引〕** 对每个紧集 $\Phi_0 \Subset \Phi$ 中任一点 $\varphi$，
> 快子系统 $\dot Y = g(\varphi, Y)$ 存在吸引子 $\mathcal{A}(\varphi)$，
> 以**一致速率** $\lambda>0$ 指数吸引，且吸引域一致包含初值集 $\mathcal{Y}_0$；
> **(H2)〔正则性〕** $f, g$ Lipschitz；$J$ 有界；映射 $\Lambda(\varphi,\omega)$ 可测；
> $\hat J \in C^1(\Phi)$（或至少 $\hat J \in W^{1,\infty}_{\text{loc}}$）；
> **(H3)〔变异核〕** 变异算子 $M$ 是保守、有谱隙的有界核。
>
> 则当 $\varepsilon \to 0$：
> **(i)** 慢轨迹 $X^\varepsilon(t)$ 在任意有限区间 $[0,T]$ 上**一致收敛**到平均化方程
> $$\frac{\mathrm{d}\bar X}{\mathrm{d}t} = \varepsilon\, \bar f(\bar X), \qquad
> \bar f(\varphi) := \lim_{T\to\infty}\frac{1}{T}\int_0^T
>   \mathbb{E}\big[f(\varphi, Y_t^{\varphi})\big]\,\mathrm{d}t
>   \;=\; \nabla_\varphi \hat J(\varphi); \tag{4.2}$$
> **(ii)** 族群经验测度 $\rho^{\varepsilon}_t$ 弱收敛到 replicator–mutator 方程 (3.3)-L2 的解，
> 其适应度恰为 (3.1) 定义的**后学习适应度** $\hat J$。

**证明骨架.**
*(i)* 把 (4.1) 看成奇异摄动系统。(H1) 保证对每个冻结的 $\varphi$，
快流在时间 $\sim 1/\lambda$ 内进入不变的慢流形 $\mathcal{M} = \{(\varphi, Y): Y\in\mathcal{A}(\varphi)\}$。
由 Tikhonov 定理（或 Freidlin–Wentzell 第 7 章的平均化定理），当 $\varepsilon\to0$，
解在 $[0,T]$ 上一致收敛到受限方程。受限漂移正是 $f$ 沿快流的遍历平均。
关键一步：由 **定义 3.4**，$\hat J$ 的**定义本身就是这个平均**：
$$\hat J(\varphi) = \mathbb{E}_{\omega}[J(\Lambda(\varphi,\omega),\omega)]
= \lim_{T\to\infty}\frac1T\int_0^T \mathbb{E}\big[\,J(\theta_\tau^{\varphi},\omega)\,\big]\mathrm{d}\tau ,$$
（第一个等号是定义，第二个等号由 (H1) 的遍历性/指数混合给出）。
故 $\bar f = \nabla\hat J$。

*(ii)* 把族群看作 (4.1) 的 $N$ 个耦合副本。由 (H1)–(H3) 与交换性，
经验测度是 Wasserstein 空间中的紧序列，其极限满足弱收敛意义下的连续性方程；
$\hat J\in C^1$ 使漂移项在弱拓扑下连续（$\nabla\cdot(\rho\nabla\hat J)$ 对 $\rho$ 连续）。
扩散项由 $M$ 的谱隙保证不为退化。极限方程即 (3.3)-L2。$\blacksquare$

**注 4.1（T4 的强度与代价）.** T4 的价值在于：它把"进化作用于 $\hat J$（后学习适应度）而非
$J$（先天适应度）"这一 Baldwin 直觉**变成定理**——条件 (H2) 明确要求 $\hat J$ 可微，
而 $\hat J$ 的定义 (3.1) **内含**了一生学习的平均。它同时给出了 $\varepsilon\to0$ 的**误差阶**
（$O(\varepsilon)$，来自 Tikhonov 的边界层估计），这是"时间尺度分离有多快才算分离"的定量答案。

**注 4.2（T4 的直接前驱）.** Zucchet & Sacramento (2022) 已表明 EP 适用于任意 **bilevel 优化**
（含元学习）[c10 引述]。T4 是把这一观察放到**族群/测度**层面并给出收敛性条件。

### 4.2.1 数值验证（表 1 Part D）

- 平均化方程预测的稳态：Gibbs $\rho^\star \propto e^{\hat J/\kappa}$，$\mathbb{E}[\varphi] = 0.4004$。
- 真实个体基种群模拟（$N=4000$，400 代，每个体一生 3 次 EP 式更新，适应度按 20 个任务估计）：
  $\mathbb{E}[\varphi] = 0.409 \pm 0.136$。**一致。**
- 平均场 PDE 的 $\rho_t$ 与 Gibbs 的**全变差距离 = 0.0053**（趋零）。

---

## 4.3 T5 层级耗散：单一自由能账

> **定理 T5（耗散）.** 设 $\mathfrak{F}(\rho) = -\langle\hat J\rangle_\rho + \kappa H(\rho)$，
> $H$ 为相对参考测度的微分熵。则 (3.3)-L2 的解满足
> $$\frac{\mathrm{d}}{\mathrm{d}t}\mathfrak{F}(\rho_t) = -\int_\Phi \rho\,\big|\nabla_\varphi(
>   \tfrac{\delta\mathfrak{F}}{\delta\rho})\big|^2\,\mathrm{d}\varphi \;\le\; 0, \tag{4.3}$$
> 等号成立当且仅当 $\rho = \rho^\star \propto e^{\hat J/\kappa}$。
> 进一步，四层耗散可加：整个 EEP 级联是**单一 Lyapunov 泛函族**
> $\{\,E,\; L,\; \mathfrak{F}_{\text{投影}},\; \mathfrak{F}\,\}$ 的**分层梯度流**。

**证明.** (4.3)：Wasserstein 梯度流的能量恒等式。把 $\partial_t\rho = \nabla\cdot(\rho\nabla\frac{\delta\mathfrak{F}}{\delta\rho})$
与 $\frac{\mathrm d}{\mathrm dt}\mathfrak{F} = \int \frac{\delta\mathfrak{F}}{\delta\rho}\,\partial_t\rho$ 代入，
分部积分即得。平衡条件 $\nabla\frac{\delta\mathfrak{F}}{\delta\rho}=0 \Rightarrow
-\nabla\hat J + \kappa\nabla\log\rho = 0 \Rightarrow \rho^\star\propto e^{\hat J/\kappa}$。
可加性由 T2（L0）、T1（L1）、(3.2) 的信息投影不等式（L1.5）、上式（L2）逐层相加得到。$\blacksquare$

**表 1 Part D 证据**：$\mathfrak{F}$ 沿 $t$ 单调下降 $0.3792\to0.1100$（8 个采样点逐点递减）。

**注 4.3（与热力学代价的接口）.** T5 是**结构**断言（自由能单调），不是**能量账**断言。
真正的能量账需要附上 Landauer 型下界：L1 层的训练代价有
$E_{\text{train}} \gtrsim 2ND\,k_BT$ 的下界，与数字 $6ND$ FLOP × Landauer 同阶 [g3]；
而 L0 层（推理）在准静态极限下**可逆、无热力学下界** [g3]。热力学计算的实验数据是
推理 ~$10^5\,k_BT$/样本、热力学优势 $>10^7$ [g1]，且梯度下降训练生成式热力学计算机
**最小化训练期散热** [g2]。**结论：EEP 的能量优势集中在推理，不在学习/进化。**

---

## 4.4 T6 信息几何结构：一个诚实的非均匀结论

延伸理论最容易被过度声张的地方。这里给出精确的陈述，包括**一条否定性结果**。

> **命题 T6(a)〔慢层：自然梯度，已证，见文献〕.** replicator 方程是 KL 散度在
> Shahshahani（Fisher）度量 $g_{ij} = \delta_{ij}/\rho_i$ 下的**自然梯度流**，
> KL 是其严格 Lyapunov 函数；等价于乘性权重/熵镜像更新的连续极限，收敛到 softmax 平衡 [f1][f2]。

> **命题 T6(b)〔快层的状态子纤维：H-自然梯度，本文由 T1 立即可得〕.**
> 由 (2.6)，第二相的状态扰动在 $\beta=0$ 处恰为代价在度量 $H=\nabla_s^2E$ 下的**自然梯度方向**：
> $$\partial_\beta s^\beta\big|_{\beta=0} = -\,H^{-1}\nabla_s C(s^0,y). \tag{4.4}$$
> 这正是"第二相编码误差导数"的几何解释：误差信号是**度量为能量 Hessian** 的最速下降方向。
> 在 Gibbs 状态模型 (T3b) 中，由涨落–耗散关系 $H \propto (k_BT)^{-1}\Sigma^{-1}$（$\Sigma$ 为状态协方差），
> 故 (4.4) 是**统计白化后的 Newton 方向**，即 Fisher 意义下的自然方向。

> **命题 T6(c)〔参数层：不是自然梯度，除非度量为各向同性〕.**
> 定义参数空间上的拉回度量 $G(\theta) = (\partial_\theta s^\star)^{\!\top} H\,(\partial_\theta s^\star)$，
> 记 $B := \partial^2_{s\theta}E$，则 $G = B^{\!\top}H^{-1}B$，而
> $$\Delta\theta_{\text{EP}} = \nabla_\theta L = -B^{\!\top}H^{-1}\nabla_s C,
> \qquad G^{-1}\nabla_\theta L = -(B^{\!\top}H^{-1}B)^{-1}B^{\!\top}H^{-1}\nabla_s C . \tag{4.5}$$
> 二者一般**不等**；它们相等当且仅当 $B^{\!\top}H^{-1}B \propto I$（各向同性/白化条件）。

**证明.** (4.4) 直接来自 (2.6)。(4.5)：由 (2.8) $\nabla_\theta L = -B^{\!\top}H^{-1}\nabla_sC$，
代入 T1 得 $\Delta\theta_{\text{EP}} = \nabla_\theta L$；另一项为 $G^{-1}\nabla_\theta L$，
由 $G$ 的定义展开即得。二者相等 $\iff B^{\!\top}H^{-1}B \propto I$。$\blacksquare$

**这条否定性结果的正面价值.** 它给出一个**可操作的设计目标**：若把参数化选在
"白化坐标"下使 $G \propto I$，则 EP 更新**精确等于** L 的自然梯度。这是一个
**可检验的架构约束**，而不是修辞。它同时说明：EEP 的"升维"不是"统一在一个度量下"，
而是**每个纤维携带自己的度量** $\{H,\; G,\; g_{\text{Shahshahani}},\; W_2\}$，
层间的耦合只能通过 $\hat J$ 的链式法则传递。

**修正后的升维表述.** 总空间上的**块对角度量**
$\mathbb{G} := \mathrm{diag}\big(\tfrac{1}{\varepsilon^2}\,g_{\text{慢}},\; G_{\text{快}}\big)$
使投影 $\pi: (\Theta\times\Phi, \mathbb{G}) \to (\Phi, g_{\text{慢}})$ 成为
Riemannian 浸没（垂直方向 = 纤维 $\Theta$），**但** $G_{\text{快}}$ 下的 EP 流是
L 的欧氏梯度而非自然梯度（T6(c)）。故升维的准确表述是：
**一个纤维化的、逐纤维赋度量的分层梯度流**，而非"一个统一度量的梯度流"。

---

## 4.5 T7 Price–Baldwin 分解

> **命题 T7.** 分解 $\hat J = \hat J_0 + \Delta\hat J_{\text{学}}$，其中
> $\hat J_0(\varphi) = \mathbb{E}_\omega[J(\theta_0,\omega)]$ 是"冻结学习"（$\theta\equiv\theta_0$）的适应度，
> $\Delta\hat J_{\text{学}} \ge 0$ 是一生学习的适应度增益。则族群总适应度的代际变化满足
> Price 型分解
> $$\overline{\Delta\hat J} \;=\;
>   \underbrace{\frac{\mathrm{Cov}_\rho(w,\hat J_0)}{\bar w}}_{\text{选择（先天）}}
>   + \underbrace{\frac{\mathrm{Cov}_\rho(w,\Delta\hat J_{\text{学}})}{\bar w}}_{\text{选择（可塑性）}}
>   + \underbrace{\mathbb{E}_\rho[\Delta\hat J_{\text{学}}]_{\text{代际}}}_{\text{Baldwin/遗传容纳}}
>   + \underbrace{\mathcal{M}}_{\text{变异}} . \tag{4.6}$$

**(i) 稳态性.** 在 (3.3)-L2 的平衡 $\rho^\star$ 处 $\nabla\hat J = \kappa\nabla\log\rho^\star$，
故 $\mathrm{Cov}_{\rho^\star}(w,\hat J)=0$：**选择项在稳态消失，只剩 Baldwin 项与变异项。**
这是"可塑性一旦被遗传固定，就不再被选择"（遗传容纳完成）的形式化。

**(ii) 内部极值.** 可塑性有代谢代价 $\lambda\varphi^2$。在本项目的可解实例中
（一生 $K=3$ 步、凸二次任务、$\mathbb{E}[w_\star^2]=1$）：
$$w = w_\star\big(1-(1-\varphi)^3\big) \;\Longrightarrow\;
\hat J(\varphi) = -\tfrac12(1-\varphi)^6 - \lambda\varphi^2,
\qquad \frac{\mathrm d\hat J}{\mathrm d\varphi} = 3(1-\varphi)^5 - 2\lambda\varphi = 0 .$$
$\lambda = 0.35$ 时 $\varphi^\star = 0.379$——**与表 1 Part D 的数值峰值 $0.379$ 一致**，
且与解析式最大偏差仅 $8.4\times10^{-3}$（Monte-Carlo 误差量级）。$\varphi^\star>0$ 即
**"会学习"这一表型被保留**（Baldwin，非 Lamarck）。

**(iii) 环境稳定性依赖.** 代价项 $-\lambda\varphi^2$ 与学习增益项 $-(1/2)(1-\varphi)^6$ 的相对权重
决定 $\varphi^\star$：任务分布 $\nu$ **稳定**时学习增益趋于饱和，选择压力把 $\varphi^\star$ 推向
更小的可塑性（遗传容纳）；$\nu$ **快速变化**时 $\Delta\hat J_{\text{学}}$ 的方差项主导，
$\varphi^\star$ 保持在更大值。这与 Baldwin 元学习的实验对照一致：
**分布宽/快速变化时 Baldwin 优于 Lamarck，分布窄时相反** [h1]。

**注 4.4** (ii) 的推导用到了"一生内学习是凸问题的精确梯度步"这一可解性假设，
故 $\hat J$ 有闭式。一般情形下 $\hat J$ 只能通过采样估计（定义 3.4），
此时 T4 需要随机近似版本（见 §4.6 G2）。

---

## 4.6 缺口清单（**这些是未证或证据不足的，必须与定理分开读**）

| 编号 | 缺口 | 影响 | 严重度 |
|---|---|---|---|
| **G1** | **一致双曲性 (H1) 对 EP 不普遍成立。** T2 只保证收敛到**驻点集**；非凸深层网络上 EP 已知收敛困难、梯度消失，需要弱反馈 + 残差连接才可训练 10–20 层 [c13]。非保守扩展 [c8] 与全纯/热系综版本 [c6][c9] 是补丁，但都改变了原动力学。 | T4 的适用性 | **高** |
| **G2** | **(H2) 对离散元参数不成立。** $\varphi$ 含连接掩码 $A_\varphi$（组合结构），$\hat J$ 一般**不可微**。T4 需要 (1+λ)-ES / 选择梯度 / 随机近似的版本，其收敛性条件比确定性 Tikhonov 弱得多。 | T4 的严格性 | **高** |
| **G3** | **$\rho^\star \propto e^{\hat J/\kappa}$ 的推导假设 $\hat J$ 已确定。** 当 $\hat J$ 本身由有限样本估计时，选择-估计噪声耦合会产生跨代相关，"无偏选择"不再成立。 | T4(ii)、T7(i) | 中 |
| **G4** | **T6(c) 的白化条件 $B^{\!\top}H^{-1}B \propto I$ 是未解决的开放设计问题**，只有形式表述，无构造性算法。 | 升维表述的完整性 | 中 |
| **G5** | **L1.5（巩固）只在形式层面给出**（(3.2) 的约束信息投影）。其与 L1 的时间尺度分离（$\tau_1 \ll \tau_{1.5}$）未纳入 T4 的统一平均化证明。 | 层级完整性 | 中 |
| **G6** | **规模化的能力上限无任何证据。** 现有 EP 结果的上限是 MNIST（≈1% 错误）[c4]、ImageNet 32×32 [c6][c12]；Cadence 的界是 302 神经元线虫与 128 通道 / 8192 记录单元的 AMEN [a2][a5]。**没有任何证据**表明该层级可扩展到语言尺度或长程逻辑推理。 | 与 AGI 诉求的关系 | **高** |
| **G7** | **无完整 EEP 三层在真实生物系统上的实验验证。** 表 1 Part D 是合成玩具（标量元参数、凸二次任务）。 | 实证支持 | 高 |
| **G8** | **$\hat J$ 的定义依赖对"一生"的切分**（何为一个个体、一生何时结束），在持续学习系统中没有自然边界。 | 定义的良定性 | 中 |

**缺口与定理的关系**：T4 的**结论形式**不依赖 G1–G8；但 T4 的**适用性**依赖它们。
因此本项目把 T4 定性为"骨架完整、条件待补"的定理（见 `docs/06` 判语）。
