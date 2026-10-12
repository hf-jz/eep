
你的质疑触及了 EP 理论真正值得深入研究的地方。不过，在开始一般化之前，我需要先纠正一个重要的学术判断：原论文的梯度等式并不是没有数学证明；其真正值得进一步发展的，是定理适用范围、动力系统假设、可实现性和误差界限还可以被更系统地公理化。

我核对了原论文第 3 节和附录 A。作者确实使用了隐函数定理与混合偏导数交换证明核心梯度公式。原论文还讨论了参数相关的损失函数，并在附录 C 推导了随机 Gibbs 分布下的对应结果。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2

与此同时，EP 已经存在名为 Generalized EP 的后续工作。Laborieux 和 Zenke 在 ICLR 2024 的工作研究了不要求权重对称的广义 EP，并分析了有限扰动和非对称性带来的不同偏差；另有一篇截至 2026 年 8 月已更新的预印本研究了热力学系综中的有限扰动 EP。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

因此，我们不应把研究目标简单设为“第一次给 EP 加上 Generalized 一词”。更有价值的目标是：

建立一套统一的、可验证的 EP 数学公理体系：从最小的平衡与可微性假设出发，严格证明何时两阶段状态差分等于真实梯度；再证明哪些条件可以放宽，以及放宽后必须付出什么代价。

下面先建立理论的第一个版本。这个版本将区分已经能够严格证明的结论、需要额外公理的推广，以及不能在一般情况下成立的主张。

# 一、重新界定 EP 的数学问题

## 1.1 原 EP 真正证明的是什么？

原论文的中心数学对象不是神经元，也不是特定的 Hopfield 网络，而是一个依赖参数、状态和引导变量的标量函数：

F(θ,β,s)F(\theta,\beta,s)F(θ,β,s)

其中：

* θ∈Rp\theta\in\mathbb R^pθ∈Rp：待训练的模型参数。

* s∈Rns\in\mathbb R^ns∈Rn：系统内部状态。

* β∈R\beta\in\mathbb Rβ∈R：控制扰动强度的变量。

* FFF：总势能或总能量函数。

系统在每个固定的 (θ,β)(\theta,\beta)(θ,β) 下，满足平衡条件：

∇sF(θ,β,sθβ)=0.\nabla_s F(\theta,\beta,s^\beta_\theta)=0.∇sF(θ,β,sθβ)=0.

定义自由平衡状态：

sθ0=sθβ∣β=0.s^0_\theta=s^\beta_\theta\big|_{\beta=0}.sθ0=sθββ=0.

再假定任务损失由总能量在 β=0\beta=0β=0 时对扰动变量的偏导数给出：

C(θ,s)=∂F∂β(θ,0,s).C(\theta,s) = \frac{\partial F}{\partial\beta}(\theta,0,s).C(θ,s)=∂β∂F(θ,0,s).

于是训练目标是：

J(θ)=C(θ,sθ0).J(\theta)=C(\theta,s^0_\theta).J(θ)=C(θ,sθ0).

EP 的定理表明：

∇θJ(θ)=lim⁡β→0∇θ partialF(θ,β,sθβ)−∇θ partialF(θ,0,sθ0)β\boxed{ \nabla_\theta J(\theta) = \lim_{\beta\rightarrow0} \frac{ \nabla_\theta^{\,\mathrm{partial}} F(\theta,\beta,s^\beta_\theta) - \nabla_\theta^{\,\mathrm{partial}} F(\theta,0,s^0_\theta) }{\beta} }∇θJ(θ)=β→0limβ∇θpartialF(θ,β,sθβ)−∇θpartialF(θ,0,sθ0)

这里的 ∇θ partial\nabla_\theta^{\,\mathrm{partial}}∇θpartial 强调：求偏导时固定 sss，随后才比较两个平衡状态。

这条等式是原论文的核心结论。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

## 1.2 为什么这条证明仍然留下了重要问题？

需要严格区分四种完全不同的命题：

|
命题

|

原论文的理论状态

|
| --- | --- |
|

两个局部平衡状态之差是否能给出真实梯度？

|

在规定的光滑性、平衡分支等条件下，可以证明

|
|

状态演化是否一定收敛到这些平衡状态？

|

需要另外的稳定性与收敛性条件

|
|

任意动力系统都能通过两阶段差分得到真实梯度吗？

|

不能直接推出

|
|

有限的、非零的扰动是否必然给出精确梯度？

|

原始确定性 EP 的极限公式不作此保证

|

原论文在一般框架中要求平衡点对应的状态 Hessian 非奇异，具体稳定局部极小值的情形使用正定 Hessian。这为局部可微的平衡状态分支提供了条件。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

我的判断是：EP 的核心梯度定理有严格的局部数学基础，但“平衡存在”“平衡稳定”“梯度对应关系”和“算法实际收敛”是四个不同的命题，不能用同一个定理一并证明。

一般化 EP 必须把这四个层次拆开。

# 二、建立第一套可公理化的 EP 框架

我建议将一般化分成两层：

* 第一层：变分型广义 EP（Variational GEP）。不改变原始 EP 的数学本质，但把假设、平衡存在性、稳定性和梯度定理明确分开，形成严谨的基础理论。

* 第二层：动力系统型广义 EP（Dynamical GEP）。进一步放弃必须存在统一标量能量函数的限制，研究一般平衡动力系统在什么附加条件下，仍然能通过两阶段扰动得到真实梯度。

第一层是第二层的基础。第二层的关键问题不是能否写出一个更一般的公式，而是：当系统不再来自势能函数时，EP 原先依赖的混合偏导对称性由什么条件替代？

## 2.1 公理组 A：变分型广义 EP

这里的“公理”指明确规定理论对象与允许使用的假设，而不是把需要证明的结论直接写成公理。

## Axiom A1 — 参数与状态空间

模型参数 θ\thetaθ 和内部状态 sss 分别属于有限维欧氏空间中的开集：

θ∈Θ⊂Rp,s∈S⊂Rn.\theta\in\Theta\subset\mathbb R^p,\qquad s\in S\subset\mathbb R^n.θ∈Θ⊂Rp,s∈S⊂Rn.

引导变量 β\betaβ 属于原点附近的开区间。总能量函数满足：

F:Θ×B×S→R,F∈C2.F:\Theta\times B\times S\to\mathbb R, \qquad F\in C^2.F:Θ×B×S→R,F∈C2.

这里 C2C^2C2 意味着函数具有连续的二阶偏导数。这足以支持下面的局部梯度定理及混合偏导交换。

## Axiom A2 — 局部平衡分支

存在平衡状态 sθβs^\beta_\thetasθβ，满足：

∇sF(θ,β,sθβ)=0.\nabla_s F(\theta,\beta,s^\beta_\theta)=0.∇sF(θ,β,sθβ)=0.

在研究点处，状态 Hessian

H=∇s2F(θ,0,sθ0)H=\nabla_s^2F(\theta,0,s^0_\theta)H=∇s2F(θ,0,sθ0)

必须非奇异。根据隐函数定理，这保证附近存在局部可微的平衡状态分支。

如果还要求该状态是稳定的局部能量极小值，则进一步要求：

H≻0.H\succ0.H≻0.

非奇异是局部隐函数分支的条件；正定是此处采用的稳定极小值条件，两者不应混为一谈。

## Axiom A3 — 目标函数的引导编码

定义任务损失为：

C(θ,s)=∂βF(θ,0,s).C(\theta,s) = \partial_\beta F(\theta,0,s).C(θ,s)=∂βF(θ,0,s).

损失不是任意附加在理论外部的量，而是总能量对引导变量的一阶响应。

## Axiom A4 — 训练目标

自由平衡状态为 sθ0s^0_\thetasθ0，训练目标定义为：

J(θ)=C(θ,sθ0).J(\theta)=C(\theta,s^0_\theta).J(θ)=C(θ,sθ0).

因为平衡状态本身依赖 θ\thetaθ，所以求 JJJ 的梯度必须考虑状态随参数变化的间接影响，不能只计算损失对参数的直接偏导。

## Axiom A5 — 两阶段观测规则

在相同参数与外部条件下，分别记录：

Q(θ,β)=∂θF(θ,β,sθβ).Q(\theta,\beta) = \partial_\theta F(\theta,\beta,s^\beta_\theta).Q(θ,β)=∂θF(θ,β,sθβ).

这里的参数偏导固定状态求取；两个阶段的状态变化则通过两次平衡求解体现。

这套公理组描述的是局部理论。它既不假定平衡点是全局唯一的，也不假定所有初始状态都会收敛到同一个平衡点。

## 2.2 广义 EP 定理 A：两阶段差分梯度定理

在公理 A1–A5 下，有：

∇θJ(θ)=lim⁡β→0Q(θ,β)−Q(θ,0)β\boxed{ \nabla_\theta J(\theta) = \lim_{\beta\to0} \frac{Q(\theta,\beta)-Q(\theta,0)}{\beta} }∇θJ(θ)=β→0limβQ(θ,β)−Q(θ,0)

即：

∇θJ(θ)=lim⁡β→0∂θF(θ,β,sθβ)−∂θF(θ,0,sθ0)β\boxed{ \nabla_\theta J(\theta) = \lim_{\beta\to0} \frac{ \partial_\theta F(\theta,\beta,s^\beta_\theta) - \partial_\theta F(\theta,0,s^0_\theta) }{\beta} }∇θJ(θ)=β→0limβ∂θF(θ,β,sθβ)−∂θF(θ,0,sθ0)

这个定理不是通过神经网络层数、激活函数或某一种连接结构推导出来的，而是由能量函数的光滑性、局部平衡条件和引导变量对目标函数的编码共同决定的。

它适用于满足这些条件的广义变分系统。

### 严格证明

定义约化势能：

Φ(θ,β)=F(θ,β,sθβ).\Phi(\theta,\beta) = F(\theta,\beta,s^\beta_\theta).Φ(θ,β)=F(θ,β,sθβ).

约化势能把内部状态取为平衡状态后的能量写成只依赖 θ\thetaθ 和 β\betaβ 的函数。

第一步，对 θ\thetaθ 求导。根据链式法则：

∇θΦ=∂θF+(Dθsθβ)T∇sF.\nabla_\theta\Phi = \partial_\theta F + \left(D_\theta s^\beta_\theta\right)^{\mathsf T} \nabla_sF.∇θΦ=∂θF+(Dθsθβ)T∇sF.

由平衡条件 ∇sF=0\nabla_s F=0∇sF=0，第二项严格为零，因此：

∇θΦ=∂θF\boxed{\nabla_\theta\Phi=\partial_\theta F}∇θΦ=∂θF

第二步，对 β\betaβ 求导：

∂βΦ=∂βF+(∇sF)T∂βsθβ.\partial_\beta\Phi = \partial_\beta F + (\nabla_sF)^{\mathsf T} \partial_\beta s^\beta_\theta.∂βΦ=∂βF+(∇sF)T∂βsθβ.

再次利用同一个平衡条件：

∂βΦ=∂βF\boxed{\partial_\beta\Phi=\partial_\beta F}∂βΦ=∂βF

第三步，使用公理 A3 和 A4：

J(θ)=C(θ,sθ0)=∂βF(θ,0,sθ0)=∂βΦ(θ,0).\begin{aligned} J(\theta) &=C(\theta,s^0_\theta)\\ &=\partial_\beta F(\theta,0,s^0_\theta)\\ &=\partial_\beta\Phi(\theta,0). \end{aligned}J(θ)=C(θ,sθ0)=∂βF(θ,0,sθ0)=∂βΦ(θ,0).

于是：

∇θJ(θ)=∇θ∂βΦ(θ,0).\nabla_\theta J(\theta) = \nabla_\theta\partial_\beta\Phi(\theta,0).∇θJ(θ)=∇θ∂βΦ(θ,0).

由于约化势能具有连续二阶偏导数，可以交换混合偏导：

∇θJ(θ)=∂β∇θΦ(θ,0).\nabla_\theta J(\theta) = \partial_\beta\nabla_\theta\Phi(\theta,0).∇θJ(θ)=∂β∇θΦ(θ,0).

最后使用第一步得到的关系：

∇θJ(θ)=ddβ∂θF(θ,β,sθβ)∣β=0.\nabla_\theta J(\theta) = \left. \frac{d}{d\beta} \partial_\theta F(\theta,\beta,s^\beta_\theta) \right|_{\beta=0}.∇θJ(θ)=dβd∂θF(θ,β,sθβ)β=0.

将导数写成差商的极限，就得到定理 A。证毕。

这一证明的关键不是技巧性的代数变换，而是一个明确的数学结构：

平衡条件+约化势能+混合偏导交换⟹EP 梯度恒等式\boxed{ \text{平衡条件} + \text{约化势能} + \text{混合偏导交换} \Longrightarrow \text{EP 梯度恒等式} }平衡条件+约化势能+混合偏导交换⟹EP 梯度恒等式

它也说明为什么不能在没有其他条件的情况下，将该定理推广到完全任意的动力系统：一般动力系统可能根本不存在这里使用的标量约化势能。

# 三、从势能系统推广到一般动力系统

这一步才是我们需要重点研究的新方向。

在原 EP 中，平衡条件具有特殊形式：

G(θ,s)=∇sE(θ,s)=0.G(\theta,s)=\nabla_s E(\theta,s)=0.G(θ,s)=∇sE(θ,s)=0.

也就是说，平衡残差 GGG 是某个标量能量 EEE 对状态的梯度。

若要求一般化 EP 适用于不满足这一结构的系统，就必须允许更一般的平衡方程：

G(θ,s)=0\boxed{G(\theta,s)=0}G(θ,s)=0

其中 GGG 是一般向量值函数，而不一定是任何标量势能的梯度。

这里存在一个重要问题：

即使 G(θ,s)=0G(\theta,s)=0G(θ,s)=0 存在稳定解，也不能仅凭这一事实保证原 EP 的两阶段差分就是正确的梯度。

为了解决这一问题，我们先从一般平衡系统的精确梯度入手。

## 3.1 公理组 B：一般平衡系统

## Axiom B1 — 一般平衡方程

令：

G:Θ×S→Rn,G∈C2.G:\Theta\times S\to\mathbb R^n, \qquad G\in C^2.G:Θ×S→Rn,G∈C2.

自由平衡状态 sθ0s^0_\thetasθ0 满足：

G(θ,sθ0)=0.G(\theta,s^0_\theta)=0.G(θ,sθ0)=0.

并且平衡 Jacobian：

A=DsG(θ,sθ0)A=D_sG(\theta,s^0_\theta)A=DsG(θ,sθ0)

非奇异。由隐函数定理，可以局部定义 sθ0s^0_\thetasθ0 对参数的可微依赖。

## Axiom B2 — 一般目标函数

令任务损失 ℓ(θ,s)\ell(\theta,s)ℓ(θ,s) 为至少具有连续一阶偏导数的函数，目标定义为：

J(θ)=ℓ(θ,sθ0).J(\theta)=\ell(\theta,s^0_\theta).J(θ)=ℓ(θ,sθ0).

如要比较二阶响应或建立更高阶梯度理论，则需要相应提高 GGG 和 ℓ\ellℓ 的光滑性要求。

## Axiom B3 — 可微的扰动平衡分支

使用一般状态扰动 b(θ,s)b(\theta,s)b(θ,s)，使引导状态满足：

G(θ,sθβ)+βb(θ,sθβ)=0.G(\theta,s^\beta_\theta) +\beta b(\theta,s^\beta_\theta)=0.G(θ,sθβ)+βb(θ,sθβ)=0.

在自由平衡点附近，要求对应 Jacobian 保持非奇异，以便该扰动分支局部存在。

在 B1–B3 下，我们首先可以严格推导出一般系统的真实梯度。

## 3.2 定理 B：一般平衡系统的精确梯度

为简化符号，所有以下矩阵均在自由平衡点 (θ,sθ0)(\theta,s^0_\theta)(θ,sθ0) 处求值。

由：

G(θ,sθ0)=0G(\theta,s^0_\theta)=0G(θ,sθ0)=0

对 θ\thetaθ 求导：

DθG+DsG Dθsθ0=0.D_\theta G+D_sG\,D_\theta s^0_\theta=0.DθG+DsGDθsθ0=0.

因此：

Dθsθ0=−A−1DθG\boxed{ D_\theta s^0_\theta=-A^{-1}D_\theta G }Dθsθ0=−A−1DθG

对目标函数应用链式法则：

∇θJ=∇θℓ+(Dθsθ0)T∇sℓ.\nabla_\theta J = \nabla_\theta\ell+ (D_\theta s^0_\theta)^{\mathsf T}\nabla_s\ell.∇θJ=∇θℓ+(Dθsθ0)T∇sℓ.

代入状态灵敏度关系后，得到：

∇θJ=∇θℓ−(DθG)TA−T∇sℓ\boxed{ \nabla_\theta J = \nabla_\theta\ell - (D_\theta G)^{\mathsf T} A^{-\mathsf T} \nabla_s\ell }∇θJ=∇θℓ−(DθG)TA−T∇sℓ

这就是一般可微平衡系统的精确梯度公式。

其重要含义是：即使不存在标量能量函数，只要平衡 Jacobian 可逆，目标函数的梯度依然可以通过隐函数微分得到。

但是，这条定理本身还不是 EP。它告诉我们真实梯度是什么，却没有保证可以用简单的自由相与引导相差分计算出它。

这两个问题需要分别证明。

# 四、广义 EP 的关键公理：广义互易性与引导匹配

我们现在已经知道，任意可微平衡系统的精确梯度为：

∇θJ=∇θℓ−(DθG)TA−T∇sℓ.\nabla_\theta J = \nabla_\theta\ell - (D_\theta G)^{\mathsf T} A^{-\mathsf T}\nabla_s\ell.∇θJ=∇θℓ−(DθG)TA−T∇sℓ.

下一步是寻找一种两阶段规则，能够在 β→0\beta\to0β→0 时重现这个梯度。

这要求引入两个新概念：

1. 互易性条件：系统的参数相关观测必须与平衡方程的参数响应相容。

2. 引导匹配条件：施加的扰动必须与真实梯度中需要的状态响应相匹配。

## 4.1 为什么直接使用损失梯度引导不一定成立？

对一般扰动方程：

G(θ,sθβ)+βb(θ,sθβ)=0,G(\theta,s^\beta_\theta) + \beta b(\theta,s^\beta_\theta)=0,G(θ,sθβ)+βb(θ,sθβ)=0,

在 β=0\beta=0β=0 处对引导强度求导：

A∂sθβ∂β∣0+b=0.A\left.\frac{\partial s^\beta_\theta} {\partial\beta}\right|_0+b=0.A∂β∂sθβ0+b=0.

因此：

∂sθβ∂β∣0=−A−1b\boxed{ \left.\frac{\partial s^\beta_\theta} {\partial\beta}\right|_0 =-A^{-1}b }∂β∂sθβ0=−A−1b

原始 EP 在能量系统中使用损失对状态的梯度作为外部引导，即 b=∇sℓb=\nabla_s\ellb=∇sℓ。但对于一般系统，真实目标梯度所需要的是 A−T∇sℓA^{-\mathsf T}\nabla_s\ellA−T∇sℓ，而普通扰动产生的是 A−1∇sℓA^{-1}\nabla_s\ellA−1∇sℓ。

当 AAA 不对称时，两者通常不同：

A−1∇sℓ≠A−T∇sℓ.A^{-1}\nabla_s\ell \neq A^{-\mathsf T}\nabla_s\ell.A−1∇sℓ=A−T∇sℓ.

这不是小的技术细节，而是把 EP 推广到一般动力系统时必须解决的数学障碍。

## 4.2 公理 B4：引入共轭参数观测

为一般系统定义一个向量值的参数观测函数：

P(θ,β,s)∈Rp.P(\theta,\beta,s)\in\mathbb R^p.P(θ,β,s)∈Rp.

要求在自由平衡附近满足：

DsP(θ,0,s)=(DθG(θ,s))T\boxed{ D_sP(\theta,0,s) = (D_\theta G(\theta,s))^{\mathsf T} }DsP(θ,0,s)=(DθG(θ,s))T

以及：

∂βP(θ,β,s)∣β=0=∇θℓ(θ,s)\boxed{ \left.\partial_\beta P(\theta,\beta,s) \right|_{\beta=0} = \nabla_\theta\ell(\theta,s) }∂βP(θ,β,s)∣β=0=∇θℓ(θ,s)

第一个条件规定状态变化如何影响参数观测；第二个条件规定扰动对参数观测产生的直接响应。

这两个条件可以看作原 EP 中 P=∂θFP=\partial_\theta FP=∂θF 的一般化。

为什么原 EP 自然满足第一个条件？

若：

G=∇sE,P(θ,0,s)=∇θE,G=\nabla_sE, \qquad P(\theta,0,s)=\nabla_\theta E,G=∇sE,P(θ,0,s)=∇θE,

则：

DsP=∇s∇θE=(DθG)T.D_sP = \nabla_s\nabla_\theta E = (D_\theta G)^{\mathsf T}.DsP=∇s∇θE=(DθG)T.

这里利用了同一个标量能量 EEE 的混合偏导对称性。

对一般 GGG，则需要将这一相容关系明确作为公理，并验证它是否成立。

## 4.3 公理 B5：广义引导匹配

在 A=DsGA=D_sGA=DsG 非奇异时，定义扰动方向：

b=AA−T∇sℓ\boxed{ b=A A^{-\mathsf T}\nabla_s\ell }b=AA−T∇sℓ

由此得到：

∂sθβ∂β∣0=−A−1b=−A−T∇sℓ.\left.\frac{\partial s^\beta_\theta}{\partial\beta} \right|_0 = -A^{-1}b = -A^{-\mathsf T}\nabla_s\ell.∂β∂sθβ0=−A−1b=−A−T∇sℓ.

这使状态响应与一般目标梯度要求的响应完全一致。

但必须指出：与经典 EP 的 b=∇sℓb=\nabla_s\ellb=∇sℓ 相比，这个广义引导方向通常不再是简单的损失梯度。它依赖系统 Jacobian，因此需要额外的线性代数运算，或者一种能够实现相同作用的辅助动力学。

## 4.4 定理 C：动力系统型广义 EP

在公理 B1–B5 下，定义两阶段差分：

gβ(θ)=P(θ,β,sθβ)−P(θ,0,sθ0)β.g_\beta(\theta) = \frac{ P(\theta,\beta,s^\beta_\theta) - P(\theta,0,s^0_\theta) }{\beta}.gβ(θ)=βP(θ,β,sθβ)−P(θ,0,sθ0).

那么：

lim⁡β→0gβ(θ)=∇θJ(θ)\boxed{ \lim_{\beta\to0}g_\beta(\theta) = \nabla_\theta J(\theta) }β→0limgβ(θ)=∇θJ(θ)

### 证明

对参数观测沿引导平衡分支求导：

ddβP(θ,β,sθβ)∣0=∂βP∣0+DsP(θ,0,sθ0)∂sθβ∂β∣0.\begin{aligned} \left.\frac{d}{d\beta} P(\theta,\beta,s^\beta_\theta) \right|_0 ={}& \left.\partial_\beta P\right|_0\\ &+ D_sP(\theta,0,s^0_\theta) \left.\frac{\partial s^\beta_\theta} {\partial\beta}\right|_0. \end{aligned}dβdP(θ,β,sθβ)0=∂βP∣0+DsP(θ,0,sθ0)∂β∂sθβ0.

由公理 B4：

∂βP∣0=∇θℓ,\left.\partial_\beta P\right|_0 =\nabla_\theta\ell,∂βP∣0=∇θℓ,

且：

DsP=(DθG)T.D_sP=(D_\theta G)^{\mathsf T}.DsP=(DθG)T.

结合公理 B5：

∂sθβ∂β∣0=−A−T∇sℓ.\left.\frac{\partial s^\beta_\theta} {\partial\beta}\right|_0 =-A^{-\mathsf T}\nabla_s\ell.∂β∂sθβ0=−A−T∇sℓ.

代入后：

dPdβ∣0=∇θℓ−(DθG)TA−T∇sℓ=∇θJ.\begin{aligned} \left.\frac{dP}{d\beta}\right|_0 &=\nabla_\theta\ell -(D_\theta G)^{\mathsf T} A^{-\mathsf T}\nabla_s\ell\\ &=\nabla_\theta J. \end{aligned}dβdP0=∇θℓ−(DθG)TA−T∇sℓ=∇θJ.

最后将导数写成差商极限：

lim⁡β→0gβ(θ)=∇θJ(θ).\lim_{\beta\to0}g_\beta(\theta) =\nabla_\theta J(\theta).β→0limgβ(θ)=∇θJ(θ).

证毕。

这个定理与原始 EP 的实质区别在于： 它不要求平衡方程本身必须是标量能量的梯度，但以更强的互易性条件和更一般的引导匹配条件为代价。

这给出了一条可能的数学推广路径，但不能因此宣称所有非保守动力系统都满足这些公理。

## 4.5 互易性公理什么时候可以成立？

公理 B4 的第一个条件并不自动成立。对每一个参数分量 θk\theta_kθk，它要求存在一个状态函数 PkP_kPk，使：

∂Pk∂si=∂Gi∂θk.\frac{\partial P_k}{\partial s_i} = \frac{\partial G_i}{\partial\theta_k}.∂si∂Pk=∂θk∂Gi.

假定状态空间局部单连通，且有关导数足够光滑，则局部存在这种 PkP_kPk 的一个相应可积性条件是：

∂2Gi∂sj ∂θk=∂2Gj∂si ∂θk\boxed{ \frac{\partial^2G_i} {\partial s_j\,\partial\theta_k} = \frac{\partial^2G_j} {\partial s_i\,\partial\theta_k} }∂sj∂θk∂2Gi=∂si∂θk∂2Gj

对所有状态指标 i,ji,ji,j 和参数指标 kkk 成立。

这一条件意味着，DθGD_\theta GDθG 关于状态变量的相关分量必须满足互易性，才能积分出具有所需性质的参数观测。

对于经典变分系统 G=∇sEG=\nabla_sEG=∇sE，该条件自然由同一能量函数的高阶偏导对称性得到；对于一般系统，它需要单独验证。

这里还有一个重要的研究问题：如果这一条件不成立，是否可以扩大观测状态、引入辅助变量或使用非局部的学习规则来恢复精确梯度？这正是进一步一般化可以探索的方向，而不是已经证明的普遍结论。

# 五、这套广义 EP 与原 EP 的关系

## 统一理论的两个层次

第一层：变分型 EP

G=∇sE,A=ATG=\nabla_sE,\qquad A=A^{\mathsf T}G=∇sE,A=AT

使用标量能量、对称 Hessian 和损失梯度引导。原论文的确定性 EP 就属于这一层。

放宽能量结构，增加相容性公理

第二层：动力系统型 EP

G(θ,s)=0G(\theta,s)=0G(θ,s)=0

允许非变分平衡方程，但要求参数观测相容，并使用匹配真实梯度的引导方向。

这个关系是有条件的，而不是简单的包含关系：第二层放宽了能量结构，却增加了其他条件。经典 EP 是其中一个特殊情形。

### 5.1 如果坚持使用最简单的损失引导呢？

若坚持使用：

b=∇sℓb=\nabla_s\ellb=∇sℓ

且参数观测满足 DsP=(DθG)TD_sP=(D_\theta G)^{\mathsf T}DsP=(DθG)T，那么差分极限为：

∇θℓ−(DθG)TA−1∇sℓ.\nabla_\theta\ell - (D_\theta G)^{\mathsf T} A^{-1}\nabla_s\ell.∇θℓ−(DθG)TA−1∇sℓ.

而真正的梯度是：

∇θℓ−(DθG)TA−T∇sℓ.\nabla_\theta\ell - (D_\theta G)^{\mathsf T} A^{-\mathsf T}\nabla_s\ell.∇θℓ−(DθG)TA−T∇sℓ.

两者之间的偏差为：

bgrad=(DθG)T(A−T−A−1)∇sℓ\boxed{ b_{\mathrm{grad}} = (D_\theta G)^{\mathsf T} \left(A^{-\mathsf T}-A^{-1}\right) \nabla_s\ell }bgrad=(DθG)T(A−T−A−1)∇sℓ

这条式子值得保留：它直接量化了在该类规则下，由非对称 Jacobian 导致的梯度偏差。

如果 AAA 对称且可逆，偏差为零。若 AAA 不对称，偏差可能为零，也可能非零，具体取决于参数耦合方向和损失梯度方向。

因此，不能笼统地说“权重不对称一定失败”，也不能说“取消对称性后原定理仍然自动成立”。需要分析实际系统的 Jacobian 及其与目标梯度的相互关系。2024 年的广义 EP 研究也将权重非对称与有限引导产生的偏差作为不同的问题研究。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

# 六、要把它真正发展成数学理论，还缺少哪些定理？

上面的两个定理建立了一个基础框架，但如果目标是形成一篇具有严格数学贡献的论文，不能停留在“给出一组充分条件，证明公式成立”。

还需要进一步回答以下问题。

## 6.1 定理 D：广义 EP 的必要条件与充分条件

我们需要研究一个更严格的问题：

给定一般平衡系统 G(θ,s)=0G(\theta,s)=0G(θ,s)=0，指定扰动规则 bbb 和参数观测函数 PPP，什么条件能够保证两阶段差分对所有允许的损失函数和参数方向都等于真实梯度？

这比只证明某一组人为设计的条件足以保证梯度正确更有理论价值。

初步研究应当分成两部分。

第一部分，建立充分条件。例如上文的互易性公理、引导匹配公理、局部平衡可微性，已经构成一组可行的充分条件。

第二部分，研究必要条件。需要针对预先定义的算法类别，证明如果要求两阶段差分对一类足够丰富的损失函数都精确，那么状态响应和参数观测必须满足哪些约束。

这里有一个重要细节：必要条件取决于我们允许的算法类别。如果允许任意设计参数观测函数，问题可能退化为直接把真实梯度编码进观测函数；如果限制观测必须由特定局部动力学产生，那么问题才具有更强的物理与算法意义。

所以，我们不应该宣称上文的 B4、B5 就是所有广义 EP 的唯一公理。它们是一个可以严格证明的候选公理系统，下一步需要研究其最小性、必要性以及可实现性。

## 6.2 定理 E：平衡存在、稳定与收敛的独立理论

原 EP 的梯度公式建立在平衡分支存在且可微的前提下。然而，这并不意味着具体动力系统一定能在有限时间内到达相应状态。

对于变分型 EP，可以考虑更一般的动力学：

s˙=−M(s,θ,β)∇sF(θ,β,s)\dot s=-M(s,\theta,\beta)\nabla_sF(\theta,\beta,s)s˙=−M(s,θ,β)∇sF(θ,β,s)

其中 MMM 是对称正定的状态迁移矩阵。

此时：

dFdt=−∇sFTM∇sF≤0.\frac{dF}{dt} = -\nabla_sF^{\mathsf T} M\nabla_sF \leq0.dtdF=−∇sFTM∇sF≤0.

这给出了能量单调不增的条件。如果平衡点处的 Hessian 正定，且 MMM 在该点附近正定，那么该平衡点在常规光滑性条件下是局部渐近稳定的。

但要证明从任意初始状态全局收敛，还需要进一步的条件，例如适当的能量下界、紧致的等能集合、排除其他不变集等。单凭能量下降，不能自动保证全局收敛到指定的平衡点。

因此，建议将广义 EP 的证明体系分成三个独立的定理层次：

|
定理层次

|

要证明的内容

|
| --- | --- |
|

存在性与正则性

|

平衡状态是否存在，以及是否依赖参数可微

|
|

动力学稳定性

|

选定的状态演化是否趋向相应平衡状态

|
|

梯度一致性

|

两阶段差分是否等于目标函数的真实梯度

|

只有这三个层次被分别说明，才可以把数学上的梯度恒等式与可执行的 EP 算法联系起来。

## 6.3 定理 F：有限引导强度下的误差界

实际计算无法使用真正的 β→0\beta\to0β→0，所以还应当研究：

gβ(θ)=Q(θ,β)−Q(θ,0)βg_\beta(\theta) = \frac{Q(\theta,\beta)-Q(\theta,0)}{\beta}gβ(θ)=βQ(θ,β)−Q(θ,0)

与真实梯度之间的偏差。

如果引导分支上的函数 Q(θ,β)Q(\theta,\beta)Q(θ,β) 关于 β\betaβ 具有足够的二阶光滑性，且相关导数在所考虑的邻域内有界，那么由 Taylor 展开：

Q(θ,β)=Q(θ,0)+βQβ(θ,0)+O(β2).Q(\theta,\beta) = Q(\theta,0) +\beta Q_\beta(\theta,0) +O(\beta^2).Q(θ,β)=Q(θ,0)+βQβ(θ,0)+O(β2).

所以：

gβ(θ)=∇θJ(θ)+O(∣β∣)\boxed{ g_\beta(\theta) = \nabla_\theta J(\theta)+O(|\beta|) }gβ(θ)=∇θJ(θ)+O(∣β∣)

这给出了局部有限引导误差的阶数。它不是说所有 EP 实现都自动满足同样的误差常数；该界限依赖所选的函数正则性、平衡分支及其导数的有界性。

还值得研究使用正负两个引导强度构造中心差分：

gβcentral=Q(θ,β)−Q(θ,−β)2β.g_\beta^{\mathrm{central}} = \frac{Q(\theta,\beta)-Q(\theta,-\beta)}{2\beta}.gβcentral=2βQ(θ,β)−Q(θ,−β).

在额外的三阶光滑性和相关导数有界条件下，可以进一步得到二阶的截断误差。不过，这需要进行正、负两个引导相的计算，还要验证负引导在物理系统中的可实现性及平衡分支的稳定性，因此不能仅凭数值精度就认定这种方案更优。

## 6.4 随机 EP 应当如何纳入公理体系？

原论文已经讨论过随机扩展：使用 Langevin 动力学，将确定性平衡点推广为 Gibbs–Boltzmann 平稳分布，并建立相应的期望梯度关系。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

进一步发展时，必须明确区分：

* 确定性平衡点与随机平稳分布；

* 状态期望与自由能梯度；

* 期望意义下的精确梯度与有限样本的估计误差；

* 无穷小扰动极限与有限扰动下的关系。

特别值得跟踪的是 Litman 的 Finite-Nudge Equilibrium Propagation in Thermal Ensembles。该预印本在 2026 年 8 月更新的版本中，研究了热力学系综中的有限扰动，并通过 Helmholtz 自由能差及损失—能量协方差积分，建立与有限扰动相关的梯度结果。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

这意味着，有限扰动、随机系综和精确梯度不能仅仅作为原 EP 的附加实验，而应成为统一广义 EP 理论的组成部分。 不过，相关结论具有特定的系综模型、统计假设与目标定义，不能未经推导就推广到一般非平衡系统。

# 七、现有研究已经做到什么，我们还可以研究什么？

这里必须特别谨慎，因为 Generalized EP 并不是一个尚无相关成果的研究空白。

![Paul C. Bressloff Et Al - Classical Localization and Percolation in Random Environments On Trees | PDF | Phase Transition | Probability Theory](https://images.openai.com/static-rsc-4/zjuUcj0swK5U8nZ-AzYWUOMIB-D6WSb9DSoXRtOTAn7tL9ZS2wzzoPFwPeffncLAiYxZqNXGaMXtpQxWbmMcF_90XB0t1lJ-rIEiRc8zTLH-VI6S4VLnIdVCzdXgmBZdRiSd9Rf7-17Nur1_bTBaOuWGB1Ngm0CKEqXyTWQQXgw?purpose=inline)

2017 — 原始 Equilibrium Propagation

Scellier & Bengio

建立能量模型上的两阶段梯度定理，证明在适当条件下可得到真实目标梯度，并给出确定性和随机两类推导。

阅读原论文 

![ICLR Poster Improving equilibrium propagation without weight symmetry through Jacobian homeostasis](https://images.openai.com/static-rsc-4/PtuVWPt2v5QuBw6VCYUyyNA-DVz59eZNvscapkRI2fJ5lHaZbUAw9mwfhu1PNiFM2rsSwBmptDPrkz6UA6DrBN2IYwzYo1vMSg802r-VIVXUQxf3m-iJ43i4G2p4BNnicejLq10xs8xw6SyHXgzbVQbIpwxLQLNg0fvjEDIfkdM?purpose=inline)

2024 — 非对称权重下的广义 EP

Laborieux & Zenke

研究非对称网络中的梯度估计偏差，并提出 Jacobian homeostasis 来改善功能 Jacobian 的对称性相关性质。

阅读论文 

![NeurIPS  Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines](https://images.openai.com/static-rsc-4/MA_0nHtfl7UkFR-_ISyIrY0JCVWFGCAFgNrJ3Dhm5AhPnY3mMEBG58iPss2gbJGxNbletbK2D7FqJ9il16PAUAr-gOympvSk6YSiTmC7kuEITNcoA39V1SLs9oClaQ8QPLjOnJ8izIgXmiH-ICJqR-kIjrgYvZOpj3ailL3Uxwg?purpose=inline)

2025–2026 — 热力学系综与有限扰动

Elon Litman，预印本，2026 年 8 月更新

研究有限扰动情况下的自由能梯度、对比式更新与协方差积分表达。

阅读最新预印本 

基于这些工作，我建议将研究目标具体化为：

建立“变分 EP—非变分平衡 EP—随机与有限扰动 EP”的统一公理体系，并研究精确性条件、必要性、稳定性和误差界。

其中，非变分系统上的互易性与引导匹配定理，是上文提出的候选拓展方向；其学术新颖性及其与已有广义 EP 理论的区别，仍需要通过更全面的文献和数学比较来确定。

# 八、建议采用的正式理论结构

我不建议直接把上面内容包装成“已经完成证明的新 EP 理论”。更可靠的做法，是将其作为一个可验证的研究纲领，逐项完成以下工作：

1. 定义阶段： 明确参数空间、状态空间、平衡关系、任务目标、引导机制与参数观测；区分变分系统与一般动力系统。

2. 基础定理： 先证明局部平衡分支存在、状态对参数的灵敏度公式，以及变分型 EP 梯度定理。

3. 一般化定理： 建立一般平衡系统的精确梯度公式，证明广义互易性与匹配引导条件的充分性，并研究对指定算法类别的必要条件。

4. 稳定性与收敛： 单独建立平衡的局部稳定性、全局收敛性或其他可验证的动力学收敛条件。

5. 误差理论： 推导有限引导强度误差、数值平衡误差、随机估计误差及多类误差之间的关系。

6. 计算验证： 在可解析的线性平衡系统和非线性小模型上，与隐函数微分或自动微分基准进行逐项对照。

这个顺序很重要：如果直接从复杂网络实验开始，即使性能变好，也很难知道增益到底来自一般化 EP 的理论结构，还是来自某种额外的 Jacobian 修正或优化器。

## 最终判断

我们现在可以建立一个明确的数学起点：

* 变分型 EP 的核心是约化势能的混合偏导恒等式。

* 一般平衡系统的核心是隐函数定理给出的精确梯度公式。

* 两阶段学习要在一般动力系统中保持梯度精确性，则需要额外的互易性、观测相容性和引导匹配条件。

* 平衡的存在与梯度公式成立，并不等价于实际动态过程一定收敛；有限引导、随机性与数值误差也必须独立处理。

我建议下一步首先围绕“定理 B 与定理 C”展开：严格审查广义互易性条件是否必要、匹配引导是否可以放宽，以及在什么最弱条件下才能得到精确梯度。 这将决定我们最终得到的究竟只是原 EP 的一个重新表述，还是一个真正具有一般性、能够处理更广泛动力系统的数学理论。


# 九、下一步的核心：先建立“精确性判据”，而不是继续增加公理

经过对 2018 年论文的原文核对，我发现了一个必须纳入研究设计的关键事实：Scellier 等人已经在 Generalization of Equilibrium Propagation to Vector Field Dynamics 中，推导了一般向量场平衡点的精确梯度，并明确指出其广义 EP 更新通常只是梯度的近似。梯度误差与向量场 Jacobian 的非对称性有关。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

这也要求我们修正上一轮的研究表述：此前的“广义互易性”与“匹配引导”可以构造出一个有条件的精确梯度定理，但它还不能被视为新的、最一般的 EP 公理体系。特别是，利用平衡 Jacobian 的转置逆来设计引导方向，虽然可以恢复精确梯度，却可能把原本希望通过局部动力学实现的工作转化成了一次隐式微分或伴随线性方程求解。

因此，真正值得推进的研究问题是：一般动力系统上的两阶段局部学习，何时与真实梯度一致？不能一致时，偏差的精确表达式、必要充分条件与可计算上界分别是什么？

我们现在先把这个问题彻底数学化。

## 九点一、统一符号与定义

令参数、状态分别为：

θ∈Rp,s∈Rn.\theta\in\mathbb R^p,\qquad s\in\mathbb R^n.θ∈Rp,s∈Rn.

一般动力系统为：

s˙=f(θ,s).\dot s=f(\theta,s).s˙=f(θ,s).

在给定参数 θ\thetaθ 时，假设系统有局部平衡状态 sθs_\thetasθ，满足：

f(θ,sθ)=0.f(\theta,s_\theta)=0.f(θ,sθ)=0.

定义平衡点处的两个 Jacobian：

A:=Dsf(θ,sθ)∈Rn×n,A:=D_s f(\theta,s_\theta) \in\mathbb R^{n\times n},A:=Dsf(θ,sθ)∈Rn×n,

B:=Dθf(θ,sθ)∈Rn×p.B:=D_\theta f(\theta,s_\theta) \in\mathbb R^{n\times p}.B:=Dθf(θ,sθ)∈Rn×p.

假设 AAA 非奇异，且 fff 至少二阶连续可微。根据隐函数定理，自由平衡状态局部可微，并且：

Dθsθ=−A−1B.\boxed{D_\theta s_\theta=-A^{-1}B.}Dθsθ=−A−1B.

设损失函数为：

ℓ:Rp×Rn→R,\ell:\mathbb R^p\times\mathbb R^n\to\mathbb R,ℓ:Rp×Rn→R,

定义：

q:=∇sℓ(θ,sθ).q:=\nabla_s\ell(\theta,s_\theta).q:=∇sℓ(θ,sθ).

我们的目标函数是：

J(θ)=ℓ(θ,sθ).J(\theta)=\ell(\theta,s_\theta).J(θ)=ℓ(θ,sθ).

这里要始终区分三个对象：真实目标梯度、平衡点对扰动的响应，以及算法通过局部参数更新得到的方向。后面会证明，它们只有在特定条件下才是同一个向量。

## 九点二、定理 1：任意可微平衡系统的真实梯度

先考虑目标函数：

J(θ)=ℓ(θ,sθ),J(\theta)=\ell(\theta,s_\theta),J(θ)=ℓ(θ,sθ),

其中 sθs_\thetasθ 满足：

f(θ,sθ)=0.f(\theta,s_\theta)=0.f(θ,sθ)=0.

由隐函数定理：

Dθsθ=−A−1B.D_\theta s_\theta=-A^{-1}B.Dθsθ=−A−1B.

于是，对 JJJ 使用链式法则：

∇θJ=∇θℓ+(Dθsθ)Tq=∇θℓ−BTA−Tq.\begin{aligned} \nabla_\theta J &=\nabla_\theta\ell +(D_\theta s_\theta)^{\mathsf T}q\\ &=\nabla_\theta\ell -B^{\mathsf T}A^{-\mathsf T}q. \end{aligned}∇θJ=∇θℓ+(Dθsθ)Tq=∇θℓ−BTA−Tq.

因此：

∇θJ=∇θℓ−BTA−Tq(T1)\boxed{ \nabla_\theta J = \nabla_\theta\ell -B^{\mathsf T}A^{-\mathsf T}q } \tag{T1}∇θJ=∇θℓ−BTA−Tq(T1)

这里的 ∇θℓ\nabla_\theta\ell∇θℓ 是保持状态 sss 不变时的直接参数梯度，BTA−TqB^{\mathsf T}A^{-\mathsf T}qBTA−Tq 则包含了参数通过改变平衡状态而间接影响损失的部分。

定理 1 不需要系统存在全局能量函数。 它只需要局部平衡点正则、系统可微。这正是 2018 年一般向量场 EP 推导真实梯度时所使用的核心结构。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2

## 九点三、定理 2：经典两阶段 EP 在一般向量场下计算的是什么？

考虑带有目标引导的动力系统：

s˙=f(θ,s)−β∇sℓ(θ,s)\boxed{ \dot s=f(\theta,s)-\beta\nabla_s\ell(\theta,s) }s˙=f(θ,s)−β∇sℓ(θ,s)

令 sθβs_\theta^\betasθβ 表示该系统的局部平衡状态：

f(θ,sθβ)−β∇sℓ(θ,sθβ)=0.f(\theta,s_\theta^\beta) -\beta\nabla_s\ell(\theta,s_\theta^\beta)=0.f(θ,sθβ)−β∇sℓ(θ,sθβ)=0.

当 β=0\beta=0β=0 时，恢复自由平衡点 sθs_\thetasθ。

对平衡方程关于 β\betaβ 求导，在 β=0\beta=0β=0 处得到：

A∂sθβ∂β∣β=0=q.A \left.\frac{\partial s_\theta^\beta} {\partial\beta}\right|_{\beta=0} =q.A∂β∂sθββ=0=q.

因此：

∂sθβ∂β∣β=0=A−1q(T2)\boxed{ \left.\frac{\partial s_\theta^\beta} {\partial\beta}\right|_{\beta=0} =A^{-1}q } \tag{T2}∂β∂sθββ=0=A−1q(T2)

这个结果非常关键：非对称动力系统对目标扰动的响应，由 A−1A^{-1}A−1 决定。

经典两阶段学习进一步利用参数与系统状态的局部耦合关系，把这种状态变化转化为参数更新。用下降方向的符号约定，可将其写为：

dEP=−∇θℓ+BTA−1q(T3)\boxed{ d_{\mathrm{EP}} = -\nabla_\theta\ell +B^{\mathsf T}A^{-1}q } \tag{T3}dEP=−∇θℓ+BTA−1q(T3)

这里 dEPd_{\mathrm{EP}}dEP 是参数更新方向，即采用 θ←θ+ηdEP\theta\leftarrow\theta+\eta d_{\mathrm{EP}}θ←θ+ηdEP 的约定。

对照真实的负梯度：

−∇θJ=−∇θℓ+BTA−Tq,-\nabla_\theta J = -\nabla_\theta\ell +B^{\mathsf T}A^{-\mathsf T}q,−∇θJ=−∇θℓ+BTA−Tq,

两者存在明确的差异。

## 九点四、定理 3：广义 EP 精确性的必要充分条件

定义梯度方向误差：

eEP:=dEP−(−∇θJ).e_{\mathrm{EP}} := d_{\mathrm{EP}}-(-\nabla_\theta J).eEP:=dEP−(−∇θJ).

代入 T1 和 T3，直接得到：

eEP=BT(A−1−A−T)q(T4)\boxed{ e_{\mathrm{EP}} = B^{\mathsf T} \left(A^{-1}-A^{-\mathsf T}\right)q } \tag{T4}eEP=BT(A−1−A−T)q(T4)

由此可以给出最直接的判据。

定理 3。 在前述正则性和局部平衡条件下，对于给定参数点、参数耦合矩阵 BBB 和损失梯度 qqq，经典两阶段向量场 EP 的无穷小扰动更新等于真实负梯度，当且仅当：

BT(A−1−A−T)q=0.\boxed{ B^{\mathsf T} \left(A^{-1}-A^{-\mathsf T}\right)q=0. }BT(A−1−A−T)q=0.

### 证明

根据 T1，真实负梯度为：

−∇θJ=−∇θℓ+BTA−Tq.-\nabla_\theta J = -\nabla_\theta\ell +B^{\mathsf T}A^{-\mathsf T}q.−∇θJ=−∇θℓ+BTA−Tq.

根据 T3，两阶段 EP 的无穷小扰动更新为：

dEP=−∇θℓ+BTA−1q.d_{\mathrm{EP}} = -\nabla_\theta\ell+B^{\mathsf T}A^{-1}q.dEP=−∇θℓ+BTA−1q.

二者相等，当且仅当消去相同的直接参数项后，剩余项满足：

BTA−1q=BTA−Tq.B^{\mathsf T}A^{-1}q = B^{\mathsf T}A^{-\mathsf T}q.BTA−1q=BTA−Tq.

这等价于 T4 右侧为零。证毕。

### 这个定理比简单要求对称权重更一般

T4 揭示了三个层次的结论。

第一，对某个具体任务、参数点和损失函数，即使 AAA 不对称，EP 也可能恰好给出正确梯度。这取决于 BBB 与 qqq 是否能够感知到那部分非对称性。

第二，如果要求对所有可能的损失梯度 qqq 都精确，则对固定 BBB，需要：

BT(A−1−A−T)=0.B^{\mathsf T}(A^{-1}-A^{-\mathsf T})=0.BT(A−1−A−T)=0.

如果 BBB 具有满行秩，这进一步推出：

A−1=A−T,A^{-1}=A^{-\mathsf T},A−1=A−T,

也就是：

A=AT.\boxed{A=A^{\mathsf T}.}A=AT.

第三，如果进一步要求在任意充分丰富的参数耦合结构下都精确，那么 AAA 的对称性不仅是一个方便的充分条件，也是保证这一类两阶段规则普遍精确的必要条件。

这说明，我们真正应当研究的结构不是“整个网络的权重是否对称”这一单一属性，而是平衡点处的动力系统 Jacobian 如何影响梯度估计，以及这种影响能否被特定任务或参数化结构抵消。2018 年的原始广义 EP 工作也明确指出，真实梯度与其更新方向的关系取决于向量场 Jacobian 的对称性。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

## 九点五、推论 1：可以精确量化非对称性导致的梯度偏差

令 Jacobian 的反对称部分为：

Aa=A−AT2.A_{\mathrm{a}} =\frac{A-A^{\mathsf T}}{2}.Aa=2A−AT.

由于：

A−1−A−T=A−1(AT−A)A−T,A^{-1}-A^{-\mathsf T} = A^{-1}(A^{\mathsf T}-A)A^{-\mathsf T},A−1−A−T=A−1(AT−A)A−T,

因此：

A−1−A−T=−2A−1AaA−T.(T5)\boxed{ A^{-1}-A^{-\mathsf T} = -2A^{-1}A_{\mathrm a}A^{-\mathsf T}. } \tag{T5}A−1−A−T=−2A−1AaA−T.(T5)

代入 T4：

eEP=−2BTA−1AaA−Tq.(T6)\boxed{ e_{\mathrm{EP}} = -2B^{\mathsf T} A^{-1}A_{\mathrm a}A^{-\mathsf T}q. } \tag{T6}eEP=−2BTA−1AaA−Tq.(T6)

这就是一个直接的误差分解式。

在谱范数下，它给出：

∥eEP∥2≤2∥B∥2∥A−1∥22∥Aa∥2∥q∥2.(T7)\boxed{ \|e_{\mathrm{EP}}\|_2 \leq 2\|B\|_2 \|A^{-1}\|_2^2 \|A_{\mathrm a}\|_2 \|q\|_2. } \tag{T7}∥eEP∥2≤2∥B∥2∥A−1∥22∥Aa∥2∥q∥2.(T7)

这个上界表明：

* Jacobian 越接近对称，非对称性产生的梯度偏差上界越小。

* 系统越接近奇异，∥A−1∥\|A^{-1}\|∥A−1∥ 越大，偏差可能被显著放大。

* 即使 Jacobian 的非对称部分很小，也不能忽略系统的条件数。

最后一点尤其重要。只约束 Jacobian 的反对称部分，并不足以全面保证广义 EP 的学习稳定性；还必须同时研究平衡点的正则性、Jacobian 的条件数，以及损失梯度和参数耦合的尺度。

## 九点六、推论 2：稳定平衡不意味着精确梯度

考虑一个二维线性系统：

s˙=As+θ,A=\(−1a0−1\),a≠0.\dot s=As+\theta, \qquad A= \begin{bmatrix} -1&a\\ 0&-1 \end{bmatrix}, \qquad a\ne0.s˙=As+θ,A=\(−10​a−1​\),a=0.

由于 AAA 的两个特征值均为 −1-1−1，系统是渐近稳定的。令 B=IB=IB=I，并考察 θ=0\theta=0θ=0、s=0s=0s=0 附近的损失：

ℓ(s)=12(s1+1)2.\ell(s)=\frac12(s_1+1)^2.ℓ(s)=21(s1+1)2.

此时：

q=\(10\).q=\begin{bmatrix}1\\0\end{bmatrix}.q=\(10​\).

矩阵逆为：

A−1=\(−1−a0−1\),A−T=\(−10−a−1\).A^{-1}= \begin{bmatrix} -1&-a\\ 0&-1 \end{bmatrix}, \qquad A^{-\mathsf T}= \begin{bmatrix} -1&0\\ -a&-1 \end{bmatrix}.A−1=\(−10​−a−1​\),A−T=\(−1−a​0−1​\).

于是：

dEP=A−1q=\(−10\),d_{\mathrm{EP}} =A^{-1}q = \begin{bmatrix}-1\\0\end{bmatrix},dEP=A−1q=\(−10​\),

而真实负梯度是：

−∇θJ=A−Tq=\(−1−a\).-\nabla_\theta J =A^{-\mathsf T}q = \begin{bmatrix}-1\\-a\end{bmatrix}.−∇θJ=A−Tq=\(−1−a​\).

两者的差为：

eEP=\(0a\).e_{\mathrm{EP}} = \begin{bmatrix}0\\a\end{bmatrix}.eEP=\(0a​\).

这个反例严格说明：动力系统即使拥有稳定的平衡点，经典广义 EP 仍可能产生偏离真实梯度的更新方向。

因此，“动力系统能够稳定收敛”与“EP 的学习方向准确”必须是两个独立的理论问题。

# 十、能否通过修改引导机制，恢复一般系统中的精确梯度？

现在我们遇到一个真正重要的理论分岔。

在一般系统中，如果保持经典 EP 的自然引导方式：

fβ=f−β∇sℓ,f_\beta=f-\beta\nabla_s\ell,fβ=f−β∇sℓ,

状态响应是 A−1qA^{-1}qA−1q，而真实梯度需要 A−TqA^{-\mathsf T}qA−Tq。如果 AAA 不对称，这两种响应通常不一样。

那么是否可以设计另一种扰动，使系统自身的状态变化恰好携带真实梯度需要的信息？

答案是可以从数学上构造，但必须明确其代价。

## 10.1 定理 4：伴随匹配引导

定义一个新的引导向量场：

fβ(θ,s)=f(θ,s)−βr(θ,s)\boxed{ f_\beta(\theta,s) = f(\theta,s) -\beta r(\theta,s) }fβ(θ,s)=f(θ,s)−βr(θ,s)

令 rrr 在自由平衡点满足：

r(θ,sθ)=AA−Tq.\boxed{ r(\theta,s_\theta)=AA^{-\mathsf T}q. }r(θ,sθ)=AA−Tq.

这不同于经典 EP 直接使用 qqq 作为引导方向。

在平衡点处对扰动强度求导：

A∂sθβ∂β−r(θ,sθ)=0.A\frac{\partial s_\theta^\beta}{\partial\beta} -r(\theta,s_\theta)=0.A∂β∂sθβ−r(θ,sθ)=0.

因此：

∂sθβ∂β∣0=A−1r(θ,sθ)=A−1AA−Tq=A−Tq.\begin{aligned} \left.\frac{\partial s_\theta^\beta} {\partial\beta}\right|_0 &=A^{-1}r(\theta,s_\theta)\\ &=A^{-1}AA^{-\mathsf T}q\\ &=A^{-\mathsf T}q. \end{aligned}∂β∂sθβ0=A−1r(θ,sθ)=A−1AA−Tq=A−Tq.

于是，如果使用同样的参数响应学习规则：

dmatch=−∇θℓ+BT∂sθβ∂β∣0,\boxed{ d_{\mathrm{match}} = -\nabla_\theta\ell+ B^{\mathsf T} \left.\frac{\partial s_\theta^\beta} {\partial\beta}\right|_0, }dmatch=−∇θℓ+BT∂β∂sθβ0,

就有：

dmatch=−∇θJ.\boxed{d_{\mathrm{match}}=-\nabla_\theta J.}dmatch=−∇θJ.

也就是说，在前述可微性和平衡正则条件下，伴随匹配引导能够为一般非对称平衡系统恢复精确梯度。

### 10.2 这个结果为什么还不能直接称为一种新的 EP 算法？

因为要计算：

AA−Tq,AA^{-\mathsf T}q,AA−Tq,

一般需要先求解伴随线性系统：

ATv=q,A^{\mathsf T}v=q,ATv=q,

再计算：

r=Av.r=Av.r=Av.

这本身已经引入了隐式微分中常见的伴随计算。

经典 EP 的一个重要特点，是希望借助物理系统自身的状态松弛过程获得相关梯度信息，而不必单独计算完整的反向传播或伴随路径。上述构造恢复了数学上的精确性，却未必保留这一计算优势。

因此，我们应当区分：

* 精确性定理： 可以证明什么样的引导方向能够恢复真实梯度。

* 可实现性定理： 这一引导方向能否通过局部状态交互实现，而不需要显式求解昂贵的线性系统。

* 计算复杂度定理： 该实现相对自动微分、隐式微分和传统 EP 是否真的具有优势。

没有后两者，单独证明第一条还不足以形成具有明确实际价值的广义 EP 算法。

# 十一、初步公理体系应该怎样写？

到目前为止，我们可以整理出一套更精确的公理化框架。为避免将尚未证明的普遍性质当作公理，我建议将其表述为广义 EP 的基础假设系统，以及由其推出的精确性定理。

## 公理组 GEP-A：一般平衡学习系统

A1 — 平衡正则性

存在局部平衡状态 sθs_\thetasθ，满足 f(θ,sθ)=0f(\theta,s_\theta)=0f(θ,sθ)=0。系统函数足够光滑，且 A=DsfA=D_s fA=Dsf 非奇异。

A2 — 目标定义

任务目标为 J(θ)=ℓ(θ,sθ)J(\theta)=\ell(\theta,s_\theta)J(θ)=ℓ(θ,sθ)，损失至少具有连续一阶偏导数。

A3 — 目标引导规则

引导后的状态由一族动力系统 fβf_\betafβ 决定，且在 β=0\beta=0β=0 恢复原始自由动力学。具体引导向量场属于算法设计的一部分，而不是预先规定所有系统都必须采用损失梯度。

A4 — 参数响应规则

明确规定怎样从状态变化得到参数更新。例如：

d=−∇θℓ+BT∂βsβ∣0.d=-\nabla_\theta\ell+B^{\mathsf T}\partial_\beta s^\beta|_0.d=−∇θℓ+BT∂βsβ∣0.

是否精确，必须由定理判定，而不能预先假定。

A5 — 动力学实现条件

如果要求算法通过实际状态演化实现，则还必须规定平衡点稳定性、收敛性，以及扰动后平衡分支的可达性。这些条件独立于梯度公式本身。

从这个公理体系出发，可以建立至少三类不同的理论。

|
理论类别

|

主要目标

|

核心判据

|
| --- | --- | --- |
|

精确型 GEP

|

不限制系统必须存在全局能量，但要求梯度精确

|

T4 的投影条件，或伴随匹配引导

|
|

近似型 GEP

|

保留自然损失引导和局部状态学习

|

Jacobian 非对称性导致的误差界

|
|

动力学型 GEP

|

研究有限时间、未完全收敛及非平衡状态

|

状态轨迹与平衡梯度之间的误差界

|

注意，A1–A5 是一套待进一步完善的理论组织方式，并非声称它们已经是唯一、最小或最终的 EP 公理。真正的数学工作还包括证明哪些假设可以删除、哪些条件不可缺少，以及不同算法类别中的必要充分条件。

# 十二、现有文献对这条研究路线的约束

已有结果为我们的研究确定了三个明确的边界。

第一，2018 年的 Generalization of Equilibrium Propagation to Vector Field Dynamics 已经建立了一般向量场上的广义 EP，并通过平衡点 Jacobian 推导真实梯度与两阶段更新之间的关系。因此，不能把“将 EP 扩展到非能量向量场”本身作为全新的理论贡献。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

第二，Laborieux 和 Zenke 在 ICLR 2024 的论文中，进一步研究了非对称网络的广义 EP，区分有限扰动偏差和 Jacobian 非对称性带来的偏差，并通过 Jacobian homeostasis 改善学习表现。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

第三，2025–2026 年的预印本 Equilibrium Propagation Without Limits 探索了热力学系综中有限扰动下的精确梯度表达及协方差积分。它提示我们，随机平衡模型和有限扰动也值得纳入更完整的研究版图，但相关理论具有额外的统计与系综假设。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

因此，未来工作的创新点需要放在更加明确的问题上，例如：

* 对给定的局部学习规则，证明梯度精确性的必要充分条件，而不只列举充分条件。

* 建立包含 Jacobian 非对称性、病态性和有限扰动的统一误差分解。

* 研究如何在不显式计算完整伴随梯度的条件下，逼近伴随匹配的状态响应。

* 严格区分可达的稳定平衡、数学上存在的平衡分支，以及有限时间内实际观测到的状态。

这些方向是否足以形成新的学术贡献，还需要与已有定理逐项比较。

# 十三、下一步应当研究什么？

我建议不要马上加入随机 EP、热力学 EP 或复杂神经网络。首先把确定性的一般平衡系统研究透彻。

具体的推进顺序如下：

1. 完成精确性判据的完整定理化。 将 T1–T7 整理为统一记号下的定理、推论和证明，明确针对固定损失的条件与针对所有损失的条件之间的区别。

2. 研究必要条件。 对标准损失引导和局部参数更新规则，证明在何种参数耦合丰富性假设下，Jacobian 对称性是普遍精确性的必要条件，并找出更弱的任务相关条件。

3. 建立有限扰动误差理论。 将 Jacobian 非对称性误差与有限 β\betaβ 的截断误差分别界定，再研究二者的耦合及数值平衡误差。

4. 研究无需显式伴随求解的修正方法。 对伴随匹配引导的计算代价进行分析，探索局部近似、结构约束或动力学机制能否在可证明的条件下逼近真实梯度。

5. 最后才扩展到非平衡与随机系统。 在确定性理论稳定之后，再引入轨迹平均、随机平稳分布和有限时间的状态演化。

## 当前阶段的结论

我们已经得到一个可验证的数学起点：

真实梯度=∇θℓ−BTA−Tq\boxed{ \text{真实梯度} = \nabla_\theta\ell-B^{\mathsf T}A^{-\mathsf T}q }真实梯度=∇θℓ−BTA−Tq

而标准两阶段向量场 EP 的下降方向为：

dEP=−∇θℓ+BTA−1q.\boxed{ d_{\mathrm{EP}} = -\nabla_\theta\ell+B^{\mathsf T}A^{-1}q. }dEP=−∇θℓ+BTA−1q.

两者的区别由：

eEP=BT(A−1−A−T)q\boxed{ e_{\mathrm{EP}} = B^{\mathsf T}(A^{-1}-A^{-\mathsf T})q }eEP=BT(A−1−A−T)q

完全刻画。

这让我们不必仅凭“网络是否存在能量函数”来划定 EP 的理论边界。我们可以直接分析平衡点的局部 Jacobian、任务梯度和参数耦合之间的关系，判断某一种学习规则究竟何时精确、何时近似，以及偏差有多大。

下一阶段最有学术价值的工作，就是围绕这个精确性判据建立必要充分条件与统一误差理论。 这比直接提出一个更宽泛的 Generalized EP 定义更扎实，也更容易判断最终成果是否真正超越现有研究。

