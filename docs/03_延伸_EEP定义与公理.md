# 03 延伸：进化均衡传播（EEP）的定义与公理

本节提出扩展对象。**设计原则：延伸必须同时满足两个外部约束**
（1）在数学上是 EP 的严格推广（去掉延伸即退回 EP）；
（2）符合自然界生物进化的基本规律（`docs/05` 给出形式化检验）。

---

## 3.1 为什么必须延伸

EP（`docs/02`）是**单层**变分对象：它回答了"给定能量函数 $E$ 与代价 $C$，
如何用局域信息做梯度下降"。它**没有**回答三件事：

| EP 的空白 | 生物界的对应事实 |
|---|---|
| 能量函数 $E$ 的形式（含连接拓扑）从哪来？ | 连接组由进化塑造；线虫 302 神经元、密度 ~12%、小世界 [i2] |
| 学习参数 $\theta$ 与元参数（可塑性规则、学习率）如何被选择？ | 进化塑造可塑性本身（Baldwin 遗传容纳）[h1][h2] |
| 一生持续学习如何避免灾难性遗忘？ | 快/慢双记忆系统（海马–皮层）、睡眠巩固 |

同时，EP 的两相结构（[c1]）本身已经是**三因子**形态（两细胞活动 + 一个广播标量），
与生物资格迹 + 神经调制的实验图景一致 [d3][d4]。这说明"把 EP 提升为进化层级"
不是类比，而是**把已经存在于 EP 中的结构性线索补全**。

**升维的方向**：从

$$\text{单一能量面 } E : \mathcal{S}\to\mathbb{R}$$

升到

$$\text{丛 } \pi: (\mathcal{S}\times\Theta\times\Phi) \to \Phi
\quad\text{上带两层度量 } (G_{\text{快}},\, g_{\text{慢}}) \text{ 的双尺度自然梯度流}.$$

即：**推理、学习、进化是同一个变分操作作用在丛的不同纤维上**。

---

## 3.2 层级（四个时间尺度）

| 层 | 变量 | 物理/生物对应 | 时间尺度 | Lyapunov 泛函 |
|---|---|---|---|---|
| L0 沉降 | 状态 $s\in\mathcal{S}$ | 神经活动、感知/推理 | $\tau_0$（ms） | 能量 $E$（T2） |
| L1 学习 | 参数 $\theta\in\Theta$（突触权） | 一生内可塑性、资格迹 | $\tau_1 = K\tau_0$（s–min） | 损失 $L$（T1） |
| L1.5 巩固 | 记忆测度 $m$ | 睡眠/做梦、离线重放 | $\tau_{1.5}$（小时） | 自由能（信息投影，§3.5） |
| L2 进化 | 元参数 $\varphi\in\Phi$，族群 $\rho$ | 变异–选择–遗传 | $\tau_2 = \tau_1/\varepsilon$（代） | 族群自由能 $\mathfrak{F}$（T5） |

时间尺度分离 $\tau_0 \ll \tau_1 \ll \tau_{1.5} \ll \tau_2$ 是整个结构可分析的前提，
也正是生物进化的基本事实（躯体可塑快、种系遗传慢）。

---

## 3.3 断言：**元参数**才是遗传单位

这是延伸最关键的取舍，也是它符合生物规律的**充要结构选择**。

**定义 3.1（元参数空间）.** 个体携带元参数
$$\varphi = \big(\,E_\varphi,\; A_\varphi,\; \eta_\varphi,\; \beta_\varphi,\; K_\varphi,\;
\text{neuron/synapse model}_\varphi\,\big) \in \Phi ,$$
其中 $A_\varphi$ 是**连接掩码**（哪些神经元-突触对存在，可类比线虫的解剖连接组 [a5][i2]），
$E_\varphi$ 是能量函数的形式（激活、泄漏、耦合符号），$\eta_\varphi,\beta_\varphi,K_\varphi$
是可塑性超参。

**$\varphi$ 显式地不包含一生所学的突触权 $\theta$。** 这是 Weismann 屏障（体细胞–种系分离）
的形式化，也是与 Lamarck 式继承划清界限的地方。

> **对照**：Cadence 的线虫把解剖连接组做成"代码里不可更改的刚性掩码"，学习只改权重
> [a5]。在 EEP 的记号里，**掩码 $A_\varphi$ 属于 $\varphi$（进化层），权 $\theta$ 属于 $\Theta$
> （学习层）**。Cadence 因此是 EEP 的一个**固定 $\varphi$ 截面**（$A$ 冻结，只跑 L0/L1）。

---

## 3.4 三条公理（**由生物规律反推得到**，不是事后添加）

> **公理 A1（局域物理可实现性）.** 每一层的动力学只使用局域可得信息：
> L0 只用局域势/相邻状态；L1 只用接缝两端活动加**一个广播标量**（三因子 [d3][d4]）；
> L2 的变异**不读取**适应度 $\hat J$。
>
> **公理 A2（盲变异）.** 变异算子 $M(\varphi'\,|\,\varphi)$ 与任务分布 $\nu$、
> 与适应度 $\hat J$、与环境反馈均**统计独立**。
>
> **公理 A3（事后因果选择）.** 适应度只在个体完成一生（L1 结束后）测量；
> 选择只能通过族群测度 $\rho$ 的重加权实现，绝不回写个体内部的 $\theta$。

> **公理 A4（Weismann 传递限制，可关闭）.** 遗传只传递 $\varphi$（与变异后的 $\varphi'$），
> 不传递 $\theta$。

A1–A3 分别对应生物进化的**三条基本规律**：变异（盲、无目的）、选择（事后、基于实际表现）、
遗传（受限保真传递）。A4 对应魏斯曼屏障。**延伸理论的全部定理都建在这四条之上**，
因此合规性不是事后检查，而是构造性保证（`docs/05` 的 T8 给出充要条件的证明）。

**为什么不能反过来"先有理论，再论证它符合生物规律"**：因为那样无法排除"理论其实隐含了
Lamarck 或全局梯度"这一可能。A1（局域性）排除了反向传播，A2 排除了"变异朝目标定向"，
A3 排除了"用未来信息选择"——三者共同把可容许的动力学类缩小到一个**可分类**的集合。

---

## 3.5 定义：EEP 系统

**定义 3.2（生境）.** $(\Omega, \mathcal{B}, \nu)$ 为任务/环境概率空间；$\omega\in\Omega$ 是一次
"一生"中遭遇的任务序列。

**定义 3.3（一生学习映射）.** 给定 $\varphi$ 与 $\omega$，由 EP 动力学迭代 $K$ 步定义
$$\Lambda : \Phi\times\Omega \to \Theta,\qquad
\Lambda(\varphi,\omega) = \theta_K, \qquad
\theta_{k+1} = \theta_k - \eta_\varphi\, D(\theta_k, \beta_\varphi)$$
其中 $D$ 是 (2.9) 的 EP 对比量。由 **T1**，每步都是对 $L$ 的 SGD（误差 $O(\beta_\varphi)$）；
由 **T2**，L0 的沉降在每个 $\theta_k$ 处以速率 $\ge 1-\|W_2\|$ 指数收敛。

**定义 3.4（后学习 / Baldwin 适应度）.** 结果适应度 $J: \Theta\times\Omega\to\mathbb{R}$，
$$\hat J(\varphi) \;=\; \mathbb{E}_{\omega \sim \nu}\big[\, J\big(\Lambda(\varphi,\omega),\,\omega\big)\,\big]. \tag{3.1}$$
$\hat J$ 是**后学习适应度**：它不是个体出生时的能力，而是"带着这套可塑性活过一生后"的
期望表现。这是 Baldwin 效应的形式化定义 [h1][h2]。

**定义 3.5（巩固层 L1.5）.** 快记忆 $m$ 与慢参数 $\theta$ 分离：
$$\frac{\partial m}{\partial \tau} = \text{(快写：一次经历即写入)},\qquad
\theta \leftarrow \Pi_{\mathcal{K}}\!\big(\mathbb{E}_{m}[\cdot]\big)
\;=\;\arg\min_{\theta} \sum_i w_i\, D_{\mathrm{KL}}\big(m_i \,\|\, p_\theta\big). \tag{3.2}$$
(3.2) 是**约束信息投影**——与 OPH 的 A3（未约束部分取最大随机性、约束信息投影 [b3][b5]）
**同一数学形式**。Cadence 的 "Records（快） + 慢权重 + 做梦重塑" 就是这一层的工程实现 [a6][a7]。
L1.5 的作用是灾难性遗忘的对策：把散点经验投影为可泛化的慢参数。

**定义 3.6（EEP 系统）.** 满足公理 A1–A4 的五元组
$$\mathfrak{E} = \big(\Omega,\nu;\ \Phi,\Theta;\ \Lambda;\ \hat J;\ \rho\big), \qquad \rho\in\mathcal{P}_2(\Phi).$$

**定义 3.7（层级动力学）.** EEP 的完整动力学为
$$
\begin{aligned}
\textbf{(L0)}\quad & \dot s = -\nabla_s E_{\varphi}(\theta, s), &&\text{(沉降 / 推理)}\\
\textbf{(L1)}\quad & \dot\theta = -\eta_\varphi\, \nabla_\theta L(\theta,\omega), &&\text{(一生学习，经 EP)}\\
\textbf{(L1.5)}\quad & \theta \leftarrow \Pi_{\mathcal{K}}\big(\mathbb{E}_m[\cdot]\big), &&\text{(巩固 / 做梦)}\\
\textbf{(L2)}\quad & \partial_t \rho = \varepsilon\Big[\kappa\,\Delta_\varphi \rho
   - \nabla_\varphi\!\cdot\big(\rho\,\nabla_\varphi (-\hat J)\big)\Big]. &&\text{(进化)}
\end{aligned} \tag{3.3}
$$

L2 是 **replicator–mutator 方程**的 Fokker–Planck 形式，等价于自由能
$\mathfrak{F}(\rho) = -\langle \hat J\rangle_\rho + \kappa\,H(\rho)$
在 Wasserstein-2 空间 $\mathcal{P}_2(\Phi)$ 中的梯度流 [f1][f3][f4]。
$\kappa$ 是变异强度（"温度"），$\varepsilon$ 是时间尺度分离参数。

---

## 3.6 命题：EEP 严格推广 EP

> **命题 3.8.** 取 $|\Phi| = 1$（元参数唯一）、$\varepsilon = 0$（无进化）、$m\equiv\delta$
> （无巩固）。则 (3.3) 退化为
> $$\dot s = -\nabla_s E(\theta,s), \qquad \dot\theta = -\eta\,\nabla_\theta L(\theta,\omega),$$
> 即标准 EP 的沉降 + 局域梯度下降（`docs/02` T1、T2）。

**证明.** 逐项代入即可：L1.5 的投影退化为恒等（$m$ 为点测度），L2 冻结（$\varepsilon=0$
时 $\rho$ 不演化），L0/L1 正是 EP 两相。$\blacksquare$

所以 EEP 不是 EP 的替代，而是**EP 的一个包络**：EP 是 EEP 在 $\Phi$ 为单点、$m$ 为
点测度时的截面。

---

## 3.7 升维的形式化：EEP 是一个双尺度 Riemannian 浸没

这是把"延伸"说成"升维"的**严格含义**（不是修辞）。

**构造.** 在 $\mathcal{S}\times\Theta\times\Phi$ 上放**逐纤维的度量**：
- **状态纤维度量** $H(\theta) = \nabla_s^2 E(\theta, s^\star)$：由 T6(b)，第二相的状态扰动
  $\partial_\beta s^\beta|_0 = -H^{-1}\nabla_s C$ 正是该度量下代价的自然梯度方向
  （在 Gibbs 状态模型中即统计白化后的 Newton 方向）。
- **参数纤维度量** $G(\theta) = (\partial_\theta s^\star)^{\!\top} H\,(\partial_\theta s^\star)$：
  由 **T1**，L1 的 EP 更新是 $L$ 的**欧氏梯度**；它在 $G$ 下退化为自然梯度
  **当且仅当**白化条件 $B^{\!\top}H^{-1}B \propto I$ 成立（**T6(c)**，一般情形**不成立**）。
- **慢度量** $g_{\text{慢}}$：$\Phi$ 上的 Wasserstein-2 度量（等价地，单纯形上
  Shahshahani 度量）；由 **T6(a)**，L2 是其上 $\mathfrak{F}$ 的自然梯度流。

**投影.** 取块对角度量 $\mathbb{G} := \mathrm{diag}\big(\varepsilon^{-2}g_{\text{慢}}, G\big)$，
则 $\pi: (\Theta\times\Phi, \mathbb{G}) \to (\Phi, g_{\text{慢}})$ 是 Riemannian 浸没
（垂直方向 = 纤维 $\Theta$）。

**语义.**
- **进化（L2）= 基空间 $\Phi$ 上的自然梯度流**：改变"这一族个体是怎样的族"。
- **学习（L1）= 纤维 $\Theta$ 上的流**：在族内把一个个体调整到适应；
  它是 $L$ 的欧氏梯度（T1），而**不是**通用意义下的自然梯度（T6(c)）。
- **推理（L0）= 状态纤维上的更内层流**，其扰动方向是 $H$-自然梯度（T6(b)）。

> **这就是"升维"的严格表述**：EP 在一个**能量面**上做梯度下降；
> EEP 在一个**纤维化的、逐纤维赋度量的分层梯度流**上做梯度下降，且外层流的参数
> （$\hat J$）由内层流的结果定义（(3.1)）。层级之间不是"调旋钮"，而是**同一个变分操作
> 作用在不同纤维上**。
>
> **必须避免的过度声张**：不能把 EEP 说成"单一度量下的自然梯度流"。T6(c) 是一条
> **否定性结果**——参数层的 EP 更新一般不是拉回度量下的自然梯度。准确的表述是
> "逐纤维赋度量的分层流"，其统一性来自**同一个变分原理**，而不是同一个度量。

**注 3.9（与 OPH 的结构同构）.** OPH 的"有限 patch + 重叠读回 + 修复 + 失配泛函 Φ 严格下降 +
共识标准型 + 对细化的稳定" [b1][b3][b5] 与 EEP 的"有限个体/突触/元参数 + 局域读回 +
EP 修复 + 各层 Lyapunov 严格下降 + 稳态 + 时间尺度分离"**是同一种数学骨架的两个实例**。
Pragma 自己也是这么描述 Cadence 的（"same observer patches", "same rules"）[a2][b5]。
EEP 的贡献是把这一骨架在**学习与进化**这一侧写全（含元参数层、巩固层与合规性定理）。
