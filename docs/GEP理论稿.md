
# 十四、关键文献更新：2026 年已经出现精确的非保守 EP

在继续推导必要充分条件之前，我检索到了一个会实质性改变研究路线的进展。

Scurria、Vanden Abeele、Mognetti 和 Massar 的论文 Equilibrium Propagation for Non-Conservative Systems 已于 2026 年发表于 ICML 2026。它不只讨论近似梯度，而是直接提出了适用于一般非保守动力系统的精确 EP 扩展，并给出了两种方法：Asymmetric EP（AsymEP）和 Dyadic EP。

![](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press\&sz=32)

Proceedings of Machine Learning Research+1

这项工作与我们上一轮的推导高度相关。因此，我需要明确修正研究定位：“把 EP 扩展到非保守动力系统，并恢复精确梯度”已经有直接的先行成果，不能再作为我们未来工作的主要原创主张。

但这并不意味着研究应该停止。相反，它让真正需要证明的问题更加清楚：怎样建立覆盖不同平衡机制、局部学习规则、有限扰动和有限时间动力学的统一理论，并明确每一种理论保证的边界。

## 14.1 先对照三条路线

路线 A：2018 年 Vector Field EP

允许非对称的向量场，但原始学习规则一般只能近似真实梯度。偏差与平衡点处的 Jacobian 非对称性有关。

原论文 

路线 B：2026 年 AsymEP

保留原始自由动力学，在引导阶段加入与 Jacobian 反对称部分有关的修正力，使引导后 Jacobian 在自由平衡点处等于原 Jacobian 的转置，从而恢复无穷小扰动极限下的精确梯度。

论文及数学推导 

路线 C：2026 年 Dyadic EP

把原系统扩展到双倍状态空间，在扩展空间构造一个标量能量，使原先的非保守动力学能够嵌入变分框架，再在该扩展系统中使用 EP。

ICML 2026 正式会议论文 

这两条新路线分别对应两个不同的思想：

* AsymEP：不改变自由推理时的系统，而是修正学习阶段的局部动力学。

* Dyadic EP：改变状态空间，在扩大后的空间重新建立变分结构。

  ![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

  arXiv+2

接下来，我们以这些最新成果为已知起点，继续完成真正有价值的数学工作：先证明 AsymEP 的精确性和适用条件，再找出有限扰动与有限收敛时间下仍然缺少的理论保证。

# 十五、把最新 EP 结果写成严格的定理体系

## 15.1 定理 GEP-1：一般平衡系统的真实梯度

设系统满足：

x˙=F(x,θ),F(x0(θ),θ)=0.\dot x=F(x,\theta), \qquad F(x_0(\theta),\theta)=0.x˙=F(x,θ),F(x0(θ),θ)=0.

定义自由平衡点处的 Jacobian 和参数耦合矩阵：

J:=DxF(x0,θ),P:=DθF(x0,θ).J:=D_xF(x_0,\theta),\qquad P:=D_\theta F(x_0,\theta).J:=DxF(x0,θ),P:=DθF(x0,θ).

任务损失为 C(x,θ)C(x,\theta)C(x,θ)，并记：

q:=∇xC(x0,θ).q:=\nabla_x C(x_0,\theta).q:=∇xC(x0,θ).

假设 FFF 和 CCC 可微，且 JJJ 非奇异。由隐函数定理：

Dθx0=−J−1P.D_\theta x_0=-J^{-1}P.Dθx0=−J−1P.

因此：

∇θC(x0,θ)=∇θC−PTJ−Tq(1)\boxed{ \nabla_\theta C(x_0,\theta) = \nabla_\theta C -P^{\mathsf T}J^{-\mathsf T}q } \tag{1}∇θC(x0,θ)=∇θC−PTJ−Tq(1)

式中，第一项表示损失对参数的直接依赖，第二项表示参数通过改变平衡状态而影响损失的间接作用。

为了简化后面的比较，先假设损失不显式依赖参数，即 ∇θC=0\nabla_\theta C=0∇θC=0。此时真实负梯度方向是：

d∗=PTJ−Tq(2)\boxed{ d_* =P^{\mathsf T}J^{-\mathsf T}q } \tag{2}d∗=PTJ−Tq(2)

这是后续各种 EP 更新规则都需要匹配的参照量。

## 15.2 定理 GEP-2：经典向量场 EP 的必要充分条件

在同一平衡点附近，经典 Vector Field EP 使用：

F(xβ,θ)−β∇xC(xβ)=0.F(x_\beta,\theta)-\beta\nabla_x C(x_\beta)=0.F(xβ,θ)−β∇xC(xβ)=0.

对 β\betaβ 在零点求导：

Jdxβdβ∣0=q.J\left.\frac{dx_\beta}{d\beta}\right|_0=q.Jdβdxβ0=q.

所以：

dxβdβ∣0=J−1q.\left.\frac{dx_\beta}{d\beta}\right|_0 =J^{-1}q.dβdxβ0=J−1q.

相应的下降方向为：

dVF=PTJ−1q.d_{\mathrm{VF}}=P^{\mathsf T}J^{-1}q.dVF=PTJ−1q.

与真实下降方向相减：

eVF=PT(J−1−J−T)q(3)\boxed{ e_{\mathrm{VF}} = P^{\mathsf T} \left(J^{-1}-J^{-\mathsf T}\right)q } \tag{3}eVF=PT(J−1−J−T)q(3)

由此得到：

定理 GEP-2。 对给定参数化、损失和自由平衡点，经典向量场 EP 在无穷小引导极限下精确的必要充分条件为：

PT(J−1−J−T)q=0.(4)\boxed{ P^{\mathsf T} \left(J^{-1}-J^{-\mathsf T}\right)q=0. } \tag{4}PT(J−1−J−T)q=0.(4)

这一条件比要求 J=JTJ=J^{\mathsf T}J=JT 更一般。即使 Jacobian 不对称，特定损失方向与参数耦合也可能使误差恰好为零。

但如果要求对所有 qqq 都精确，则必须满足：

PT(J−1−J−T)=0.P^{\mathsf T} \left(J^{-1}-J^{-\mathsf T}\right)=0.PT(J−1−J−T)=0.

当参数耦合具有足够丰富的方向，例如 PPP 满行秩时，这进一步要求：

J=JT.J=J^{\mathsf T}.J=JT.

因此，对这类普遍精确性要求而言，对称 Jacobian 是必要条件；它并不是每一个特定任务获得精确梯度的必要条件。

这一结论与 2018 年向量场 EP 的梯度推导及 2026 年 AsymEP 对 Jacobian 非对称性误差的分析一致。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

## 15.3 定理 GEP-3：精确引导的局部唯一性

现在允许一般的学习阶段修正：

Fβ(x)=F(x,θ)+Γ(x)−β∇xC(x),F_\beta(x) = F(x,\theta)+\Gamma(x)-\beta\nabla_x C(x),Fβ(x)=F(x,θ)+Γ(x)−β∇xC(x),

其中：

Γ(x0)=0.\Gamma(x_0)=0.Γ(x0)=0.

这保证自由平衡点 x0x_0x0 不因修正项而移动。

定义：

K:=DxΓ(x0).K:=D_x\Gamma(x_0).K:=DxΓ(x0).

学习阶段的平衡 Jacobian 为：

Jlearn=J+K.J_{\mathrm{learn}}=J+K.Jlearn=J+K.

因此，只要它非奇异，状态对引导变量的一阶响应就是：

dxβdβ∣0=(J+K)−1q.\left.\frac{dx_\beta}{d\beta}\right|_0 =(J+K)^{-1}q.dβdxβ0=(J+K)−1q.

### 精确性定理

如果要求状态响应对所有损失梯度 qqq 都等于真实梯度所需的伴随响应，那么：

(J+K)−1q=J−Tq对所有 q.(J+K)^{-1}q=J^{-\mathsf T}q \quad\text{对所有 }q.(J+K)−1q=J−Tq对所有 q.

这等价于：

(J+K)−1=J−T.(J+K)^{-1}=J^{-\mathsf T}.(J+K)−1=J−T.

进而：

K=JT−J.(5)\boxed{ K=J^{\mathsf T}-J. } \tag{5}K=JT−J.(5)

令 Jacobian 的对称与反对称部分分别为：

SJ=J+JT2,AJ=J−JT2.S_J=\frac{J+J^{\mathsf T}}2,\qquad A_J=\frac{J-J^{\mathsf T}}2.SJ=2J+JT,AJ=2J−JT.

则：

K=−2AJ.(6)\boxed{K=-2A_J.} \tag{6}K=−2AJ.(6)

这意味着：对于这一类采用线性化平衡响应的学习算法，要想对任意损失梯度恢复精确的伴随响应，学习阶段的 Jacobian 必须从 JJJ 修正到 JTJ^{\mathsf T}JT。

对于给定任务，如果只要求最终参数梯度正确，则必要充分条件可以更弱：

PT\((J+K)−1−J−T\)q=0.P^{\mathsf T} \left\((J+K)^{-1}-J^{-\\mathsf T}\\right\)q=0.PT\((J+K)−1−J−T\)q=0.

因此，式（6）是普遍状态响应精确性的唯一 Jacobian 修正，而不是所有特定任务精确学习的唯一可能机制。

## 15.4 这如何严格对应 AsymEP？

2026 年的 AsymEP 使用以下学习阶段动力学：

FA(x,θ,β)=F(x,θ)−β∇xC(x)−2AJ(x−x0)(7)\boxed{ F_A(x,\theta,\beta) = F(x,\theta) -\beta\nabla_x C(x) -2A_J(x-x_0) } \tag{7}FA(x,θ,β)=F(x,θ)−β∇xC(x)−2AJ(x−x0)(7)

其中 AJA_JAJ 在自由平衡点计算后固定，不随引导阶段的状态变化重新定义。

由于修正项在 x=x0x=x_0x=x0 时为零，自由平衡点保持不变。而在该点处：

DxFA(x0,θ,0)=J−2AJ=J−(J−JT)=JT.\begin{aligned} D_xF_A(x_0,\theta,0) &=J-2A_J\\ &=J-(J-J^{\mathsf T})\\ &=J^{\mathsf T}. \end{aligned}DxFA(x0,θ,0)=J−2AJ=J−(J−JT)=JT.

因此：

dxAβdβ∣0=J−Tq.(8)\boxed{ \left.\frac{dx_A^\beta}{d\beta}\right|_0 = J^{-\mathsf T}q. } \tag{8}dβdxAβ0=J−Tq.(8)

使用正负两个引导阶段的对称差分：

dA(β)=PTxAβ−xA−β2β,(9)\boxed{ d_A(\beta) = P^{\mathsf T} \frac{x_A^\beta-x_A^{-\beta}}{2\beta}, } \tag{9}dA(β)=PT2βxAβ−xA−β,(9)

在 β→0\beta\to0β→0 时：

lim⁡β→0dA(β)=PTJ−Tq=d∗.\lim_{\beta\to0}d_A(\beta) =P^{\mathsf T}J^{-\mathsf T}q =d_*.β→0limdA(β)=PTJ−Tq=d∗.

这就证明了 AsymEP 在相应假设下的无穷小扰动梯度精确性。该修正恰好实现了定理 GEP-3 给出的 Jacobian 匹配条件。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

这里有一个需要在正式理论中写清楚的边界：式（5）唯一确定的是修正项在自由平衡点处的一阶导数。满足同一个导数条件的修正函数不止一个。AsymEP 选择线性修正 −2AJ(x−x0)-2A_J(x-x_0)−2AJ(x−x0)，是其中一个具体实现，而不是唯一可能的非线性动力学。

# 十六、有限扰动误差：精确梯度不等于有限步算法无误差

上面的定理处理的是：

lim⁡β→0dA(β)=d∗.\lim_{\beta\to0}d_A(\beta)=d_*.β→0limdA(β)=d∗.

但实际计算使用的是有限的 β\betaβ。因此，我们必须单独研究截断误差。

## 16.1 定理 GEP-4：对称差分的有限扰动误差

假设 AsymEP 的正、负引导平衡分支 xAβx_A^\betaxAβ 在区间 \(−β0,β0\)\(-\\beta\_0,\\beta\_0\)\(−β0​,β0​\) 上三次连续可微，而且：

M3:=sup⁡∣β∣≤β0∥d3xAβdβ3∥<∞.M_3:= \sup_{|\beta|\le\beta_0} \left\| \frac{d^3x_A^\beta}{d\beta^3} \right\|<\infty.M3:=∣β∣≤β0supdβ3d3xAβ<∞.

令：

v∗=dxAβdβ∣0.v_*=\left.\frac{dx_A^\beta}{d\beta}\right|_0.v∗=dβdxAβ0.

对平衡分支在 β=0\beta=0β=0 处作 Taylor 展开：

xAβ=xA0+βv∗+β22xA′′(0)+O(β3).x_A^\beta = x_A^0+\beta v_* +\frac{\beta^2}{2}x_A''(0) +O(\beta^3).xAβ=xA0+βv∗+2β2xA′′(0)+O(β3).

相应地：

xA−β=xA0−βv∗+β22xA′′(0)+O(β3).x_A^{-\beta} = x_A^0-\beta v_* +\frac{\beta^2}{2}x_A''(0) +O(\beta^3).xA−β=xA0−βv∗+2β2xA′′(0)+O(β3).

相减后，二阶项抵消。因此：

∥xAβ−xA−β2β−v∗∥≤β26M3.(10)\boxed{ \left\| \frac{x_A^\beta-x_A^{-\beta}}{2\beta} -v_* \right\| \le \frac{\beta^2}{6}M_3. } \tag{10}2βxAβ−xA−β−v∗≤6β2M3.(10)

进而，参数更新方向的误差满足：

∥dA(β)−d∗∥≤β26∥PT∥M3.(11)\boxed{ \|d_A(\beta)-d_*\| \le \frac{\beta^2}{6} \|P^{\mathsf T}\|M_3. } \tag{11}∥dA(β)−d∗∥≤6β2∥PT∥M3.(11)

这条界限说明，采用正负对称差分之后，在上述光滑性条件下，有限引导的截断误差是 O(β2)O(\beta^2)O(β2)，而不是 O(β)O(\beta)O(β)。

如果只使用单边差分：

xAβ−xA0β,\frac{x_A^\beta-x_A^0}{\beta},βxAβ−xA0,

在二阶导数有界时，一般只能得到 O(∣β∣)O(|\beta|)O(∣β∣) 的截断误差。中心差分的优势来自对称项抵消，而不是它能够在任意有限 β\betaβ 下自动保持精确。

## 16.2 有限收敛时间会引入另一类误差

上面的误差界假设我们已经得到真正的引导平衡点。但实际算法通常只演化有限时间 TTT。

设正负引导阶段计算得到：

x^+,x^−,\widehat x_+,\qquad \widehat x_-,x+,x−,

其与真实引导平衡点的误差分别为：

ε+=∥x^+−xAβ∥,ε−=∥x^−−xA−β∥.\varepsilon_+ = \|\widehat x_+-x_A^\beta\|, \qquad \varepsilon_- = \|\widehat x_--x_A^{-\beta}\|.ε+=∥x+−xAβ∥,ε−=∥x−−xA−β∥.

那么实际梯度估计满足：

∥d^A−d∗∥≤∥PT∥\(β2M36+ε++ε−2∣β∣\).(12)\boxed{ \|\widehat d_A-d_*\| \le \|P^{\mathsf T}\| \left\( \\frac{\\beta^2M\_3}{6} + \\frac{\\varepsilon\_++\\varepsilon\_-}{2|\\beta|} \\right\). } \tag{12}∥dA−d∗∥≤∥PT∥\(6β2M3​​+2∣β∣ε+​+ε−​​\).(12)

这个式子把两种误差分开了：

* 第一项是有限扰动导致的截断误差。

* 第二项是状态未完全收敛导致的误差。

这里出现了一个很重要的实际问题：当 β\betaβ 变小时，第二项中的 1/∣β∣1/|\beta|1/∣β∣ 会放大状态求解误差。

因此，不能只通过不断减小引导强度来提高 EP 精度。如果平衡状态的求解容差保持不变，状态误差反而可能在差分阶段被显著放大。

若系统在所考虑的平衡点附近局部指数稳定，且两种引导状态都从自由平衡附近启动，那么有限时间状态误差通常随时间指数衰减；但要使差分估计误差持续减小，必须将收敛时间与引导强度、数值积分误差共同考虑。

# 十七、稳定性：AsymEP 与 Dyadic EP 分别需要什么条件？

这一部分关系到数学定理能否变成实际算法。

## 17.1 AsymEP 的局部稳定性

假设原始自由动力学在平衡点处渐近稳定，即其 Jacobian JJJ 是 Hurwitz 矩阵：

Re⁡λi(J)<0.\operatorname{Re}\lambda_i(J)<0.Reλi(J)<0.

由于 JTJ^{\mathsf T}JT 和 JJJ 具有完全相同的特征值，故：

JTJ^{\mathsf T}JT

同样是 Hurwitz 矩阵。

而 AsymEP 的学习阶段 Jacobian 在自由平衡点处正好是 JTJ^{\mathsf T}JT。所以，在光滑性与局部平衡分支存在条件下，正负引导足够小时，学习阶段的平衡点仍可保持局部渐近稳定。

这构成了一个有用的局部稳定性结论：

AsymEP 的一阶 Jacobian 修正并不改变自由平衡点的线性稳定特征值，只是把 Jacobian 转置。

但它不意味着任意初始状态都一定收敛，也不意味着较大的引导强度下仍能保持稳定。非正规 Jacobian 还可能产生显著的瞬态放大，即使全部特征值的实部都是负数。

## 17.2 Dyadic EP 的双状态变分结构

Dyadic EP 在扩展状态空间 (z,z′)(z,z')(z,z′) 中定义：

m=z+z′2,d=z−z′.m=\frac{z+z'}2,\qquad d=z-z'.m=2z+z′,d=z−z′.

并构造标量函数：

H(z,z′,θ)=−dTF(m,θ).(13)\boxed{ H(z,z',\theta) = -d^{\mathsf T}F(m,\theta). } \tag{13}H(z,z′,θ)=−dTF(m,θ).(13)

引入任务损失后：

HT=H+βC(m).H_T=H+\beta C(m).HT=H+βC(m).

其相应的状态方程可写成：

m˙=F(m,θ),\dot m=F(m,\theta),m˙=F(m,θ),

d˙=JF(m,θ)Td−β∇mC(m).\dot d=J_F(m,\theta)^{\mathsf T}d -\beta\nabla_m C(m).d˙=JF(m,θ)Td−β∇mC(m).

因此，在平衡点处：

F(m)=0,F(m)=0,F(m)=0,

JF(m,θ)Td=β∇mC(m).J_F(m,\theta)^{\mathsf T}d =\beta\nabla_m C(m).JF(m,θ)Td=β∇mC(m).

若原系统平衡点为 x0x_0x0，则：

m=x0,m=x_0,m=x0,

而且：

d=βJF(x0,θ)−Tq.(14)\boxed{ d=\beta J_F(x_0,\theta)^{-\mathsf T}q. } \tag{14}d=βJF(x0,θ)−Tq.(14)

这非常关键：在这个理想平衡方程中，平均状态 mmm 保持为原自由平衡，而差分状态 ddd 精确携带了所需的伴随梯度信息。上述结构来自 2026 年 Dyadic EP 论文的双状态变分构造。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

进一步，假设 JJJ 是 Hurwitz 矩阵，那么在 β=0\beta=0β=0、d=0d=0d=0 附近，扩展系统的线性化在 (m,d)(m,d)(m,d) 坐标中具有 JJJ 与 JTJ^{\mathsf T}JT 作为对角块。因此，原系统的局部指数稳定性可以支持扩展系统的局部稳定性。对于非零引导，仍然需要保证所研究的平衡分支存在，并验证相应 Jacobian 的稳定性。

这里必须强调：构造了一个标量变分函数，并不自动证明该扩展系统的实际演化一定收敛。 变分形式、平衡点的存在、局部稳定和全局收敛依然是不同的问题。

# 十八、一个值得进一步审计的推论：Dyadic EP 的有限引导是否可以精确？

这一步尤其值得认真处理，因为它可能进一步强化我们对已有结果的理解。

在刚才的 Dyadic EP 推导中，理想平衡状态满足：

m=x0,d=βJ−Tq.m=x_0, \qquad d=\beta J^{-\mathsf T}q.m=x0,d=βJ−Tq.

假设损失不显式依赖参数，则：

H=−dTF(m,θ).H=-d^{\mathsf T}F(m,\theta).H=−dTF(m,θ).

在保持状态固定时，对参数求偏导：

∂θH=−dTP.\partial_\theta H = -d^{\mathsf T}P.∂θH=−dTP.

代入式（14）：

∂θH=−βqTJ−1P.\partial_\theta H = -\beta q^{\mathsf T} J^{-1}P.∂θH=−βqTJ−1P.

因此，单阶段更新：

Δθ=−ηβ∂θH\Delta\theta = -\frac{\eta}{\beta}\partial_\theta HΔθ=−βη∂θH

给出：

Δθ=ηPTJ−Tq=−η∇θC(x0).(15)\boxed{ \Delta\theta = \eta P^{\mathsf T}J^{-\mathsf T}q = -\eta\nabla_\theta C(x_0). } \tag{15}Δθ=ηPTJ−Tq=−η∇θC(x0).(15)

从这些平衡方程直接看，右侧的状态差分对 β\betaβ 呈线性关系。因此，在平衡解精确获得、JJJ 非奇异且损失不显式依赖参数的理想条件下，这个单阶段更新似乎对任意非零 β\betaβ 都具有精确梯度性质，而不只是在 β→0\beta\to0β→0 时成立。

2026 年论文的式（64）–（71）给出了这一推导涉及的平衡关系与参数更新公式。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

我建议把这个结果明确列为一个需要单独核查和形式化的命题，而不是直接宣布发现了新的有限扰动 EP 算法。尤其需要区分：

* Dyadic EP 的理想单阶段驻点更新；

* 对正负引导状态进行有限差分的标准 EP 更新；

* 有限时间内实际得到的近似驻点更新。

三者的误差结构并不相同。

如果损失显式依赖参数，那么式（15）还需要补上直接参数梯度项。更准确地说，应使用：

−ηβ∂θHT-\frac{\eta}{\beta}\partial_\theta H_T−βη∂θHT

或者单独加入 −η∇θC-\eta\nabla_\theta C−η∇θC 这一项，才能对应完整目标函数的负梯度。

# 十九、现在的研究空白应当如何重新定义？

经过这一轮推导，我们应当将研究重点从笼统的“建立 Generalized EP”调整为几个更具体的问题。

### 方向 A：精确性理论

已建立的基础

通过真实梯度公式、投影精确性条件和 Jacobian 转置匹配条件，可以严格区分一般向量场 EP 的近似精确性与 AsymEP 的局部精确性。

### 方向 B：有限引导与有限时间的统一误差理论

值得继续发展

需要把有限扰动截断误差、状态求解误差和数值离散误差纳入同一个可计算的误差上界，并证明适当的引导强度和收敛容差如何选择。

### 方向 C：一般平衡动力学的稳定性理论

需要额外证明

不能把“修正后的 Jacobian 符合伴随结构”当作收敛定理。还需要建立局部稳定、有限引导稳定以及有限时间误差之间的关系。

### 方向 D：统一变分原理

已有直接先行成果

Dyadic EP 已经在双倍状态空间中构造了非保守系统的变分形式。因此，新工作需要进一步明确什么样的状态空间扩展具有可证明性、稳定性和可实现性的统一条件，而不能简单重复其构造。

2026 年论文已经在 MNIST、Fashion-MNIST 与 CIFAR-10 上比较了 AsymEP、Dyadic EP、VF 和反向传播；它报告了 AsymEP 与 Dyadic EP 在相应实验设置中的有效性。该结果支持算法可行性，但不替代上面所说的有限时间收敛与误差定理。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

## 二十、下一步正式研究任务

我建议下一阶段不再增加抽象公理，而是完成一份以定理为中心的理论稿，至少包含以下成果：

1. 把定理 GEP-1 至 GEP-4 的假设、证明、必要性范围与反例整理成统一的数学文本。

2. 严格验证 Dyadic EP 单阶段更新在有限非零 β\betaβ 下的精确性条件，并与该方法正式表述的无穷小扰动理论相互核对。

3. 推导 AsymEP 在有限引导、有限收敛时间与数值积分条件下的统一误差界。

4. 分析局部指数稳定性、非正规瞬态放大与梯度误差之间的联系，明确何种情况下算法虽然梯度公式正确，却无法在合理时间内获得稳定的学习信号。

5. 将每个理论结论配备可解析的线性反例、数值验证，以及与自动微分真实梯度的比较。

本阶段最重要的结论是：我们已经能够对一般 EP 的精确性给出明确判据，并找到与 AsymEP 对应的唯一局部 Jacobian 修正；但真正决定这套理论能否进一步成为严格、可执行的数学物理框架的，是有限扰动、有限时间及平衡稳定性的联合分析。

这应当成为下一步正式研究的核心，而不是继续用更多定义扩充“Generalized EP”的名称。


已经完成第一阶段的理论研究与可复现实验。按照前面确定的五项正式研究任务，我建立了统一的数学记号、定理与证明，编写了 PyTorch/NumPy 验证程序，并实际运行了自动微分对照、有限扰动、有限时间及非正规动力系统实验。

## 一、研究成果文件

GEP 第一阶段研究包

中文理论稿、Python 源码、测试程序、误差数据、JSON 结果及三张实验图。

下载完整研究包 ZIP 

中文理论研究稿

包含定理 1–4、完整推导、有限扰动误差界、局部稳定性分析、数值结果与下一阶段研究问题。

阅读或下载理论稿 

可复现验证程序

使用非线性平衡系统，将理论梯度与 PyTorch 自动微分直接比较。

查看 Python 源码  查看验证摘要 JSON 

## 二、五项正式研究任务的完成情况

任务 1：定理 GEP-1 至 GEP-4 的严格化

已给出一般平衡系统的真实梯度、Vector Field EP 的精确性必要充分条件、AsymEP 的 Jacobian 修正，以及相应证明与反例。

任务 2：Dyadic EP 的有限扰动性质

已从扩展状态空间的驻点方程推导有限非零 β\betaβ 下的单阶段精确梯度条件，并通过数值求解扩展驻点方程验证。

任务 3：AsymEP 的有限扰动、有限时间与积分误差

已推导中心差分的 O(β2)O(\beta^2)O(β2) 截断误差界，并建立包含状态求解误差的统一误差界；还对有限时间松弛与数值积分误差分别进行了实验。

任务 4：非正规系统的稳定性与误差分析

已构造特征值全部稳定、但存在明显瞬态放大和梯度偏差的非正规矩阵反例，并给出基于 Jacobian 反对称部分的梯度误差上界。

任务 5：解析、自动微分和数值计算交叉验证

验证程序已实际运行，全部预设断言通过。结果表、误差曲线、非正规瞬态增益图均已生成。

## 三、最重要的数学结果

首先，必须以已有文献为起点。2018 年的 Vector Field EP 已经研究了非保守向量场，并指出标准更新一般不能得到精确梯度。2026 年 ICML 正式论文 Equilibrium Propagation for Non-Conservative Systems 则提出了 AsymEP 和 Dyadic EP，分别通过 Jacobian 修正和扩展状态空间恢复精确梯度。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

因此，我们现在要研究的不能再是泛泛地提出“Generalized EP”，而是厘清这些方法各自精确到什么程度、适用条件是什么，以及有限计算会产生哪些误差。

### 1. 一般平衡系统的真实梯度

对于：

F(x0(θ),θ)=0,F(x_0(\theta),\theta)=0,F(x0(θ),θ)=0,

令：

A=DxF,B=DθF,q=∇xC.A=D_xF,\qquad B=D_\theta F,\qquad q=\nabla_x C.A=DxF,B=DθF,q=∇xC.

在 AAA 非奇异的条件下：

∇θC(x0(θ),θ)=∇θC−BTA−Tq\boxed{ \nabla_\theta C(x_0(\theta),\theta) = \nabla_\theta C-B^{\mathsf T}A^{-\mathsf T}q }∇θC(x0(θ),θ)=∇θC−BTA−Tq

这为所有后续 EP 方法提供了统一的真实梯度参照。

### 2. AsymEP 的有限扰动误差

在光滑平衡分支上，AsymEP 的引导阶段通过反对称 Jacobian 修正，把局部动力学 Jacobian 从 AAA 调整为 ATA^{\mathsf T}AT，从而在无穷小扰动极限下恢复真实梯度。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

但实际采用有限 β\betaβ 时，中心差分仍有截断误差：

∥d^A−d∗∥≤∥BT∥\(M36β2+r++r−2∣β∣\)\boxed{ \|\hat d_A-d_*\| \le \|B^{\mathsf T}\| \left\( \\frac{M\_3}{6}\\beta^2 + \\frac{r\_++r\_-}{2|\\beta|} \\right\) }∥d^A−d∗∥≤∥BT∥\(6M3​​β2+2∣β∣r+​+r−​​\)

其中 M3M_3M3 控制平衡分支的三阶导数，r+r_+r+、r−r_-r− 是两侧扰动状态的求解误差。

这个结果揭示了一个工程关键点：减小 β\betaβ 可以降低截断误差，却可能放大状态求解误差。扰动强度不能单独优化，必须和求解容差、积分时间及积分精度共同考虑。

### 3. Dyadic EP 的有限扰动精确性

通过扩展状态空间的驻点方程：

F(m,θ)=0,ATd=βq,F(m,\theta)=0,\qquad A^{\mathsf T}d=\beta q,F(m,θ)=0,ATd=βq,

在跟踪同一局部平衡分支时有：

m=x0,d=βA−Tq.m=x_0,\qquad d=\beta A^{-\mathsf T}q.m=x0,d=βA−Tq.

因此，该结构下的单阶段参数更新在精确驻点处可以代数地恢复真实负梯度，且这个等式对有限非零 β\betaβ 成立。需要区别的是：这是 Dyadic 扩展系统的特定驻点性质，并不意味着有限时间数值积分也能无误差地得到同一结果。相关驻点与更新结构可在 ICML 2026 论文的附录中核对。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

## 四、实际数值验证结果

## 梯度对照

<table class="_6IUVGW_Table" data-d-column-sizing="auto" data-d-dividers="" style="table-layout: auto;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">方法</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">相对误差</span></p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">隐函数梯度 vs. PyTorch 自动微分</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium" data-d-tabular-nums="true" style="color: var(--color-text-success);">3.10 × 10⁻¹⁶</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Dyadic 驻点更新 vs. 自动微分</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium" data-d-tabular-nums="true" style="color: var(--color-text-success);">3.10 × 10⁻¹⁶</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Vector Field EP vs. 自动微分</td><td data-d-component="table-cell" data-d-valign="start">1.555</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">自动微分 vs. 中心有限差分校验</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium" data-d-tabular-nums="true" style="color: var(--color-text-success);">4.47 × 10⁻¹¹</p></td></tr></tbody></table>

VF 的误差来自本次构造的非对称示例，不代表对所有 VF 模型都有相同误差。

AsymEP 的中心差分实验进一步得到：

* β=0.4\beta=0.4β=0.4：相对误差约 6.21%；

* β=0.2\beta=0.2β=0.2：约 1.54%；

* β=0.1\beta=0.1β=0.1：约 0.385%；

* β=0.05\beta=0.05β=0.05：约 0.0963%。

对全部测试点进行 log-log 拟合，中心差分的经验误差阶为 2.002，单边差分的经验误差阶为 0.946，与理论预测的二阶和一阶相符。

有限扰动误差

打开有限 nudge 误差图 

有限时间误差

打开有限时间松弛图 

另外，稳定非正规矩阵示例中，两个特征值均为 −1-1−1，但矩阵二范数条件数约为 101.99，区间 0≤t≤50\le t\le50≤t≤5 内的最大瞬态增益约为 3.716。这说明仅证明局部稳定并不足以保证数值状态响应容易计算，也不足以保证未经修正的 VF 梯度准确。

## 五、如何复现

下载 ZIP 后，在项目根目录运行：

Bash

```
python -m pip install numpy scipy torch matplotlib
python src/run_validation.py
```

成功结束标记是：

```
ALL ASSERTIONS PASSED
```

然后可运行：

Bash

```
python tests/test_gep.py
```

全部理论稿、源代码和实验输出都已经包含在研究包中。

## 六、下一步研究路线

当前交付物完成了第一轮理论审计和数值验证，但还没有完成证明助手形式化验证，也没有覆盖随机 EP、噪声扰动、平衡分支跳变或实际神经形态硬件。

下一阶段应集中解决三个更难的问题：最弱精确性条件、有限时间与数值误差的紧致统一上界、非正规系统的瞬态放大和梯度误差之间的定量关系。 这些工作能够在已有 AsymEP 与 Dyadic EP 成果之上继续推进，而不重复已发表的核心结论。


## 本阶段先锁定三个理论结论

已有文献已覆盖非保守系统的精确 EP 扩展，因此这次研究将不重复提出相同算法，而是研究其精确性边界、计算误差与稳定性之间的定量关系。2018 年的 Vector Field EP、2024 年的 Jacobian Homeostasis，以及 2026 年的 AsymEP/Dyadic EP，是本阶段对照的主要工作。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2

结论 1：对一组特定任务，EP 精确性不必要求整个 Jacobian 对称。

定义

D=A−1−A−T.D=A^{-1}-A^{-\mathsf T}.D=A−1−A−T.

对状态损失梯度 qqq，标准 Vector Field EP 的误差为

e=BTDq.e=B^{\mathsf T}Dq.e=BTDq.

若允许的损失梯度集合为 Q\mathcal QQ，令 S=span⁡(Q)S=\operatorname{span}(\mathcal Q)S=span(Q)，那么对集合中的所有任务都精确的必要充分条件是

BTDS={0}.\boxed{B^{\mathsf T}D S=\{0\}.}BTDS={0}.

这比笼统要求 A=ATA=A^{\mathsf T}A=AT 更弱，因为某些 Jacobian 的非对称方向可能根本不会通过参数耦合 BBB 影响指定任务。

结论 2：存在有限扰动、状态误差与计算精度之间的定量权衡。

当中心差分截断误差为 aβ2a\beta^2aβ2，而求根、观测或计算误差产生 d/∣β∣d/|\beta|d/∣β∣ 项时，可使用：

E(β)≤aβ2+d∣β∣+E0.E(\beta)\le a\beta^2+\frac{d}{|\beta|}+E_0.E(β)≤aβ2+∣β∣d+E0.

忽略约束时，该上界的最优扰动尺度为

β∗=(d2a)1/3.\boxed{\beta_*=\left(\frac{d}{2a}\right)^{1/3}.}β∗=(2ad)1/3.

这里的 E0E_0E0 汇总与扰动无关的误差项。这是误差预算下的最优尺度，不代表对所有非线性系统都存在同一个最优扰动值。

结论 3：稳定性必须使用比特征值更丰富的指标来评估。

对于非正规矩阵，即使所有特征值实部都为负，∥eAt∥\|e^{At}\|∥eAt∥ 仍可能在一段时间内显著大于 1。由此产生的瞬态放大可能影响有限时间状态误差及其转化为梯度估计的过程。因此，本阶段将同时分析 Jacobian 的条件数、矩阵指数增益，以及有限时间下的实际梯度误差。

第二阶段研究已完成，理论稿、实验代码、数据和测试结果均已打包。基线程序和第二阶段程序都成功运行，5 项自动测试全部通过。

## 一、下载研究成果

## GEP 第二阶段完整研究包

约 976 KB，包含第一阶段基线、第二阶段理论稿、源代码、自动测试、实验数据、图表和 SHA-256 文件清单。

下载完整 ZIP 研究包 

第二阶段中文理论报告

打开完整定理、证明和误差分析 

实验与自动验证

第二阶段 Python 程序 

实验结果汇总 JSON 

测试运行记录 

## 二、本阶段获得的核心结果

### 1. 找到了比“Jacobian 必须对称”更弱的任务相关精确性条件

对于标准 Vector Field EP，记 D=A−1−A−TD=A^{-1}-A^{-\mathsf T}D=A−1−A−T，允许的损失梯度构成集合 Q\mathcal QQ，并定义 S=span⁡(Q)S=\operatorname{span}(\mathcal Q)S=span(Q)。

对固定的局部平衡点和参数耦合，VF 对所有指定任务精确的必要充分条件为：

BTDS={0}\boxed{B^{\mathsf T}DS=\{0\}}BTDS={0}

数值反例中，AAA 明显不对称，但在受限的损失梯度子空间上，精确性算子的范数为 0；当允许所有状态梯度方向时，该范数为 6。

这说明，对称性并不总是特定任务精确性的必要条件。真正起作用的是 Jacobian 的非对称方向能否通过参数耦合映射到任务的梯度方向。

### 2. 建立了误差预算下的最优 nudge 尺度

采用中心差分时，有限扰动截断误差在适当光滑条件下为 O(β2)O(\beta^2)O(β2)。如果每个阶段还存在不随 β\betaβ 缩小的绝对状态误差，则差分会将它放大为 O(1/∣β∣)O(1/|\beta|)O(1/∣β∣)。

因此，对误差代理

E(β)=aβ2+dβ+E0E(\beta)=a\beta^2+\frac{d}{\beta}+E_0E(β)=aβ2+βd+E0

可以严格得到：

β∗=(d2a)1/3\boxed{\beta_*=\left(\frac{d}{2a}\right)^{1/3}}β∗=(2ad)1/3

在本次测试模型中，误差上界给出的最优值为 β∗≈0.1068\beta_*\approx0.1068β∗≈0.1068；400 次随机状态扰动实验中，测试网格上的最小 RMS 误差出现在 β=0.085\beta=0.085β=0.085。

两者并不完全相同，因为前者优化的是保守误差上界，后者优化的是指定随机误差分布下的经验 RMS。该结果不能直接当作其他网络的最佳超参数。

### 3. 将四类误差统一起来

研究稿将 AsymEP 的梯度误差分解为：

etotal=enudge+etime+eintegration+estate\boxed{ e_{\mathrm{total}}= e_{\mathrm{nudge}}+ e_{\mathrm{time}}+ e_{\mathrm{integration}}+ e_{\mathrm{state}} }etotal=enudge+etime+eintegration+estate

分别对应有限扰动、未完全收敛、数值积分与状态或测量误差。

在 β=0.1\beta=0.1β=0.1、Euler 步长为 0.01 的实验中，实际总梯度误差从 T=0.25T=0.25T=0.25 时约 0.1909 降至 T=8T=8T=8 时约 0.001207。随着松弛时间增加，有限时间误差与积分误差明显下降，最终误差接近有限 nudge 引入的误差底限。

这说明，单纯延长松弛时间不能消除所有梯度误差；必须同时考虑扰动强度、数值求解和有限时间收敛。

### 4. 量化了非正规动力学的瞬态放大

下图对比相同稳定特征值下，不同非正规耦合强度造成的有限时间梯度响应放大。

查看非正规梯度响应图 

在矩阵族

Ak=\(−1k0−0.1\)A_k=\begin{bmatrix}-1&k\\0&-0.1\end{bmatrix}Ak=\(−10​k−0.1​\)

中，所有系统的特征值都固定为 −1,−0.1-1,-0.1−1,−0.1，但非正规性随 kkk 增强。

|
非对角耦合 kkk

|

条件数 κ2(Ak)\kappa_2(A_k)κ2(Ak)

|

半群最大增益

|

梯度响应峰值／初始值

|
| --- | --- | --- | --- |
|

2

|

50.08

|

1.738

|

1.518 倍

|
|

5

|

260.10

|

3.949

|

3.856 倍

|
|

10

|

1010.10

|

7.782

|

7.735 倍

|
|

20

|

4010.10

|

15.505

|

15.481 倍

|

这些结果使用特意选择的应力测试方向，不代表所有损失任务的平均行为。但它们清楚地表明：特征值稳定不等于有限时间响应不会被放大。

本阶段进一步用 Lyapunov 方程建立了有效的指数包络，并用程序检查所有采样时间点的数值界限。不过，该界限可能偏保守，实验也不等同于证明助手完成的形式化证明。

## 三、研究边界

这里还需要保持学术定位准确：非保守系统上的精确 EP 已经由 AsymEP 和 Dyadic EP 等工作推进，并已发表于 ICML 2026。因此，本阶段不把“首次提出一般化精确 EP”作为贡献。

![](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press\&sz=32)

Proceedings of Machine Learning Research+1

当前取得的是对任务相关精确性、有限计算误差和非正规瞬态响应的定量细化，以及与自动微分和数值积分相对照的可复现实验。完整报告也明确列出了尚未覆盖的随机系综、分支跳变、真实硬件和形式化证明助手等边界。

下一阶段最值得推进的是：建立更紧的非正规梯度误差界，并在这个界的基础上构造可以自适应选择 β\betaβ、松弛时间 TTT 和积分步长 hhh 的算法。 这将把目前的分析性结论进一步转化成可执行的广义 EP 数值方案。


## 第三阶段的关键进展：自适应控制器已经跑通

本阶段已在前两阶段的非线性平衡系统基准上实现并运行自适应原型，不再只是推导控制策略。

选定 nudging 强度

# β = 0.04

最终松弛时间

# T = 7.078

相对真实梯度误差

# 0.0618%

总误差指示器

# 5.995 × 10⁻⁴

以上均为当前指定的三维非线性基准模型、给定误差预算和候选 β 集合下的实测结果，不是对所有 EP 系统的普遍性能保证。

控制器先通过线性化方向误差选择初始松弛时间，再根据有限 β 的实际输出变化检查是否需要延长时间；随后综合中心差分的尺度变化、积分误差指示器和给定的状态不确定性预算选择 β。步长则由 RK4 步长加倍误差估计进行调整。

最重要的发现是：不能只根据有限 nudging 的截断误差选 β。 本次实验中，β 越小，截断误差越低，但状态不确定性经过差分放大后的上界越高，因此在指定的不确定性预算下，控制器选择了 β = 0.04，而不是最小候选值 0.02。

## 二、下载第三阶段完整成果

GEP 第三阶段完整研究包

约 1.5 MB。包含完整中文理论报告、可复现代码、自动测试、第二阶段基线、误差数据、实验图表及 SHA-256 校验清单。

下载 GEP_Research_Stage3.zip 

理论与证明

第三阶段中文理论报告 

核心算法

自适应 GEP Python 实现 

实验数据

完整实验汇总 JSON 

自适应参数选择详情 

不同积分器的对照结果 

运行与测试记录

程序运行日志 

自动测试记录 

## 三、关键实验图

1. 自适应 nudging 强度选择

展示截断误差、状态不确定性和综合误差分数如何共同影响 β 的选择。

查看自适应 β 曲线 

2. 自适应松弛时间

展示时间控制器如何根据方向性尾项与实际梯度变化延长松弛时间。

查看 T 选择曲线 

3. 非正规系统中的梯度误差界

比较任务方向的精确线性化尾项、全空间算子界、乘积范数界和 Lyapunov 包络。

查看非正规误差界比较图 

## 四、对第三阶段的学术判断

本阶段取得了可运行的自适应原型，并通过了 9 项程序内部断言和 5 项自动测试。但这还不是最终的、具备形式化认证的广义 EP 算法。

需要特别区分三种结果：

* 已经严格推导的部分： 一般平衡系统的真实梯度、任务方向有限时间线性化误差，以及给定状态误差半径下的三角不等式上界。

* 本次数值验证的部分： 在指定模型和误差预算下，自适应控制器的实际梯度误差、nudge 选择与积分器对照结果。

* 还需要继续证明的部分： 从步长加倍估计得到的全局误差认证，以及非线性有限 nudging 下、有限时间剩余梯度误差的严格上界。

这一区分尤其重要。现有研究已经提出了非保守系统上的 AsymEP 和 Dyadic EP，因此不能把它们本身当作尚未解决的问题。

![](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press\&sz=32)

Proceedings of Machine Learning Research+2

我建议第四阶段直接进入“认证型自适应 GEP（Certified Adaptive GEP）”： 将现有的经验误差指示器替换为可验证的局部截断误差界和非线性剩余误差界，并证明控制器在明确假设下能够满足给定的梯度误差容差。这样才能把目前经过实验验证的自适应原型，进一步推进为具有严格数学保证的算法。


# 第四阶段完成：认证型自适应 GEP

这一阶段已经从“经验误差估计”推进到具有明确数学误差证书的自适应算法原型。完整研究包已经生成，数值实验成功运行，5 项自动测试全部通过。

## 核心验证结果

要求满足的误差容差

# 5.0 × 10⁻⁴

认证得到的总误差上界

# 4.3911 × 10⁻⁴

选择的 nudging 强度

# β = 0.005

选择的松弛时间

# T = 6.0

结果适用于本阶段明确定义的三维非线性基准模型，不代表对任意 EP 系统的普遍保证。

## 一、下载完整研究包

GEP_Research_Stage4.zip

完整代码、中文理论报告、证明常数、精确残差包络、实验结果、自动测试和第三阶段基线。

下载第四阶段完整研究包 

中文理论报告

GEP_Stage4_Theory_Report_CN.md 

核心算法源代码

certified_gep.py 

完整认证结果及精确有理数上界

certified_run.json 

运行和测试记录

stage4_run.log 

tests_final.log 

MANIFEST.sha256 

## 二、第四阶段最重要的理论成果

### 1. 证明了基准系统的全局强收缩性

对修正后的引导动力系统 gβ(x)g_\beta(x)gβ(x)，通过 Jacobian 对称部分的 Gershgorin 行界，得到了：

λmax⁡(Dgβ+DgβT2)≤−0.6575<−0.64\lambda_{\max}\left( \frac{Dg_\beta+Dg_\beta^{\mathsf T}}{2} \right) \le -0.6575<-0.64λmax(2Dgβ+DgβT)≤−0.6575<−0.64

在整个 ∣β∣≤0.32|\beta|\le 0.32∣β∣≤0.32 范围内成立。

因此，可以取统一的强收缩常数：

α=0.64\boxed{\alpha=0.64}α=0.64

这个结果不是从一组数值采样点估算出来的，而是由模型系数和导数范围导出的全局解析上界。在本阶段假设下，它还支持平衡点存在性、唯一性和相应的残差距离定理。

### 2. 建立了独立于 ODE 积分器的平衡点距离证书

设 xβx_\betaxβ 为精确平衡点，x^\hat xx^ 为数值算法返回的候选状态。利用强收缩性，可以严格推出：

∥x^−xβ∥2≤∥gβ(x^)∥2α\boxed{ \|\hat x-x_\beta\|_2 \le \frac{\|g_\beta(\hat x)\|_2}{\alpha} }∥x^−xβ∥2≤α∥gβ(x^)∥2

这是本阶段最关键的改进。

数值积分器只负责生成候选状态。即使积分过程存在离散误差，也不需要把步长加倍估计误认为真实误差上界；最终候选点是否足够接近平衡点，由独立的残差计算决定。

程序进一步使用精确有理数、显式的 sin⁡\sinsin 和 tanh⁡\tanhtanh Taylor 余项界，以及平方根的整数上取整，计算残差范数的有理数上界。相关理论与经典的 ODE 后验误差分析有联系，但这里采用的是针对特定强收缩平衡问题的专门证书。

![](https://www.google.com/s2/favicons?domain=https://epubs.siam.org\&sz=32)

SIAM Journal on Numerical Analysis+1

### 3. 将有限 nudging 与平衡求解误差合并为总证书

对于中心差分梯度估计，得到了：

∥d^β−d∗∥2≤CBβ2X36⏟有限 nudging 截断误差+CBρ++ρ−2αβ⏟平衡残差误差\boxed{ \|\hat d_\beta-d_*\|_2 \le \underbrace{C_B\frac{\beta^2X_3}{6}}_{\text{有限 nudging 截断误差}} + \underbrace{C_B\frac{\rho_++\rho_-}{2\alpha\beta}}_{\text{平衡残差误差}} }∥d^β−d∗∥2≤有限 nudging 截断误差CB6β2X3+平衡残差误差CB2αβρ++ρ−

其中 X3X_3X3 是通过隐式微分、Jacobian 逆范数界和高阶导数界得到的第三阶平衡分支导数上界；ρ+\rho_+ρ+、ρ−\rho_-ρ− 是两个引导相的认证残差。

这条定理把两种不同来源的误差分开，并在明确条件下给出了可以直接计算的上界。

## 三、实际结果：误差预算全部通过

下面的图展示总梯度误差证书中两个误差分项与目标容差的关系。

查看认证误差预算图 

|
误差分项

|

认证上界

|
| --- | --- |
|

有限 nudging 截断误差

|

3.6560×10−43.6560\times10^{-4}3.6560×10−4

|
|

两相平衡残差导致的梯度误差

|

7.3508×10−57.3508\times10^{-5}7.3508×10−5

|
|

总梯度误差上界

|

4.3911×10−44.3911\times10^{-4}4.3911×10−4

|
|

目标误差容差

|

5.0×10−45.0\times10^{-4}5.0×10−4

|

由于：

4.3911×10−4<5.0×10−4,4.3911\times10^{-4}<5.0\times10^{-4},4.3911×10−4<5.0×10−4,

控制器接受了 β=0.005\beta=0.005β=0.005、T=6.0T=6.0T=6.0、最大积分步长 hmax⁡=1.0h_{\max}=1.0hmax=1.0 这一配置。

实际数值梯度与基准真实梯度的差约为 1.0764×10−51.0764\times10^{-5}1.0764×10−5，低于上述保守上界。不过，实测误差小于证书只是交叉验证；数学保证来自推导的误差界和残差包络，而不是实测结果本身。

查看随松弛时间变化的认证误差图 

## 四、可复现运行

解压 ZIP 后进入项目根目录：

Bash

```
python -m pip install -r requirements.txt
python src/certified_gep.py
python -m unittest discover -s tests -v
```

算法成功时会输出：

```
STAGE 4 CERTIFICATE PASSED
```

该程序的核心证书以精确有理数形式保存在 `certified_run.json` 中，同时提供便于阅读的十进制数值。

## 五、必须明确的证明边界

本阶段已经将终端状态的平衡误差从经验估计提升为针对指定数学模型的、基于显式假设的严格误差证书。但仍需区分几个层次。

第一，证书针对本报告定义的精确十进制有理数模型及 ∣β∣≤0.32|\beta|\le0.32∣β∣≤0.32 的区域。

第二，证书证明候选端点距离唯一平衡点不超过所给上界；它并没有证明整个 ODE 轨迹在每个时刻都落在某个验证型包络内。

第三，它不包含真实测量噪声、模型参数不确定性或实际物理系统与基准模型之间的偏差，也不是由 Lean、Isabelle 等证明助手独立完成的形式化机器证明。

这些边界并不否定当前证书的用途，而是限定了数学保证的适用范围。2026 年的 AsymEP 和 Dyadic EP 已经处理非保守 EP 的精确梯度问题；本阶段的工作是在此基础上研究数值求解结果如何获得可复核的误差保证，而不是重新声称提出一种首次实现精确梯度的 EP 规则。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

下一步最值得做的，是把这一证书从当前全局强收缩模型推广到只在局部收缩、甚至具有多个平衡分支的一般 GEP 系统，并研究区间 Newton 或验证型局部根求解怎样取代全局收缩假设。 这将决定认证型自适应 GEP 能否从当前基准原型迈向更一般的神经网络和非线性物理模型。

