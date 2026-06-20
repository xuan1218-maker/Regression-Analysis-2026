# 线性模型期末复习与标准解答

本文档基于提供的考试题目，结合复习纲要，整理了相关的知识点回顾以及对应的标准解答过程。

## 第一题

### 第一小问

#### 1. 可估计函数
**定义**：在线性模型 $y = X\beta + \epsilon$ 中，参数的线性组合 $c'\beta$ 称为**可估计的 (estimable)**，如果存在一个常数向量 $a$，使得线性估计量 $a'y$ 是 $c'\beta$ 的无偏估计。即 $E(a'y) = c'\beta$ 对所有 $\beta$ 成立。
**等价条件**：由 $E(a'y) = a'X\beta = c'\beta$ 恒成立，可知 $c' = a'X$ 或 $c = X'a$。这表示 $c$ 必须属于 $X'$ 的列空间，即 $c \in \mathcal{M}(X')$。

#### 2. 广义逆
**为什么需要广义逆？**
在求解正规方程 $X'X \beta = X'y$ 时，如果设计矩阵 $X$ 列降秩（即 $\text{rank}(X) = r < p$），那么 $X'X$ 不满秩，其常规的逆矩阵 $(X'X)^{-1}$ 不存在。为表达方程的解，我们需要引入广义逆。

**广义逆的定义**：
对于任意矩阵 $A$，满足 $AGA = A$ 的矩阵 $G$ 称为 $A$ 的广义逆，记作 $A^-$（或称为 $\{1\}$-逆）。
Moore-Penrose 广义逆 $A^+$ 是唯一满足以下 4 个条件的矩阵：
1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)' = AA^+$
4. $(A^+A)' = A^+A$

**MP 广义逆的性质**：
- **唯一性**：MP 广义逆 $A^+$ 是唯一的，而一般的广义逆 $A^-$ 不唯一。
- **范数**：$x = A^+y$ 是方程 $Ax=y$ 的最小范数最小二乘解。
- **秩**：$\text{rank}(A^+) = \text{rank}(A)$。
- **广义逆的广义逆**：$(A^+)^+ = A$。

**用广义逆解决 $(X'X)$ 秩亏问题**：
正规方程 $X'X \hat{\beta} = X'y$ 始终是相容的。利用广义逆，它的解可以表示为 $\hat{\beta} = (X'X)^- X'y$。对于任意满足 $X'X(X'X)^-X'X = X'X$ 的广义逆，由于投影矩阵 $P = X(X'X)^-X'$ 是唯一且对称幂等的，因此 $X\hat{\beta}$ 的估计结果总是唯一的。

#### 3. BLUE 估计量
**定义**：BLUE (Best Linear Unbiased Estimator，最佳线性无偏估计) 指的是在所有满足**线性**（是 $y$ 的线性组合）和**无偏**（期望等于被估参数）的估计量中，具有**最小方差**的估计量。

#### 4. 第一小问的实际求解
**证明：**
**(1) 线性与无偏性：**
对于可估计函数 $c'\beta$，存在向量 $a$ 使得 $c' = a'X$。
其最小二乘估计为 $c'\hat{\beta} = c'(X'X)^-X'y$。显然它是 $y$ 的线性组合。
求其期望：
$$ E(c'\hat{\beta}) = E(c'(X'X)^-X'y) = c'(X'X)^-X'X\beta $$
代入 $c' = a'X$：
$$ E(c'\hat{\beta}) = a'X(X'X)^-X'X\beta $$
由广义逆性质 $X(X'X)^-X'X = X$，可得：
$$ E(c'\hat{\beta}) = a'X\beta = c'\beta $$
因此 $c'\hat{\beta}$ 是 $c'\beta$ 的线性无偏估计。

**(2) 最小方差（最佳性）：**
设 $d'y$ 是 $c'\beta$ 的任意一个线性无偏估计。无偏性要求 $E(d'y) = d'X\beta = c'\beta$ 对任意 $\beta$ 成立，这蕴含 $d'X = c'$。
计算它的方差：
$$ Var(d'y) = d' Cov(y) d = \sigma^2 d'd $$
另一方面，最小二乘估计 $c'\hat{\beta} = a'X(X'X)^-X'y = a'Py$，其中 $P = X(X'X)^-X'$ 是到 $\mathcal{M}(X)$ 的正交投影矩阵，具有对称且幂等 ($P=P', P^2=P$) 的性质。
注意到 $c' = d'X$，所以 $a'X = d'X$，进而在两边右乘 $(X'X)^-X'$ 得：
$$ a'X(X'X)^-X' = d'X(X'X)^-X' \implies a'P = d'P $$
因此 $c'\hat{\beta} = d'Py$。其方差为：
$$ Var(c'\hat{\beta}) = Var(d'Py) = d'P Cov(y) P d = \sigma^2 d'P P d = \sigma^2 d'Pd $$
现在比较 $Var(d'y)$ 和 $Var(c'\hat{\beta})$：
$$ Var(d'y) - Var(c'\hat{\beta}) = \sigma^2 d'd - \sigma^2 d'Pd = \sigma^2 d'(I-P)d $$
由于 $I-P$ 也是对称幂等矩阵（因而是半正定矩阵），所以 $d'(I-P)d \ge 0$ 恒成立。
即 $Var(d'y) \ge Var(c'\hat{\beta})$，等号成立当且仅当 $(I-P)d = 0$，即 $d = Pd$，此时 $d'y = d'Py = c'\hat{\beta}$。
这证明了 $c'\hat{\beta}$ 具有最小方差，并且这个最佳线性无偏估计是**唯一**的。

---

### 第二小问

#### 1. 二次型的期望
**如何计算二次型的期望？**
设随机向量 $y$ 具有期望 $\mu = E(y)$ 和协方差矩阵 $\Sigma = Cov(y)$。对于任意非随机的对称矩阵 $A$，二次型 $y'Ay$ 的期望公式为：
$$ E(y'Ay) = \text{tr}(A\Sigma) + \mu'A\mu $$

#### 2. 第二小问的实际求解
**证明：**
残差向量为 $e = y - X\hat{\beta} = y - X(X'X)^-X'y = (I - P)y$，其中 $P = X(X'X)^-X'$。
残差平方和可以表示为二次型：
$$ ||y - X\hat{\beta}||^2 = y'(I-P)'(I-P)y = y'(I-P)y $$
这里用到了 $I-P$ 是对称幂等矩阵的性质。
由于 $E(y) = X\beta, Cov(y) = \sigma^2 I_n$，应用二次型期望公式：
$$ E(||y - X\hat{\beta}||^2) = E(y'(I-P)y) = \text{tr}((I-P)(\sigma^2 I_n)) + (X\beta)'(I-P)(X\beta) $$
分别计算两部分：
1. $X$ 的列向量在 $P$ 所表示的投影空间内，因此 $(I-P)X = X - PX = X - X = 0$。所以 $(X\beta)'(I-P)(X\beta) = \beta'X'(I-P)X\beta = 0$。
2. $\text{tr}((I-P)\sigma^2 I_n) = \sigma^2 \text{tr}(I-P)$。因为投影矩阵的迹等于它的秩，$\text{tr}(P) = \text{rank}(X) = r$，所以 $\text{tr}(I-P) = n - r$。
因此：
$$ E(||y - X\hat{\beta}||^2) = \sigma^2(n - r) + 0 = \sigma^2(n - r) $$
所以，$\hat{\sigma}^2 = \frac{||y - X\hat{\beta}||^2}{n - r}$ 的期望为：
$$ E(\hat{\sigma}^2) = E\left(\frac{||y - X\hat{\beta}||^2}{n - r}\right) = \frac{\sigma^2(n - r)}{n - r} = \sigma^2 $$
这就证明了 $\hat{\sigma}^2$ 是 $\sigma^2$ 的无偏估计。

---

### 第三小问

#### 1. 联合分布用向量形式的表出
在正态性假定下 $\epsilon \sim N(0, \sigma^2 I_n)$，有 $y \sim N(X\beta, \sigma^2 I_n)$。
$y$ 的似然函数（即多元正态分布的联合密度函数）可以表示为：
$$ L(\beta, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{n/2}} \exp\left( -\frac{1}{2\sigma^2} (y - X\beta)'(y - X\beta) \right) $$

#### 2. 实际求解
取对数似然函数：
$$ \ell(\beta, \sigma^2) = \ln L(\beta, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}(y - X\beta)'(y - X\beta) $$
为了求 $\sigma^2$ 的极大似然估计，我们首先固定 $\sigma^2$，对 $\beta$ 求极大化。这等价于最小化 $(y - X\beta)'(y - X\beta)$，其解即为最小二乘解 $\hat{\beta} = (X'X)^-X'y$。
将 $\hat{\beta}$ 代回对数似然函数中，得到关于 $\sigma^2$ 的剖面似然：
$$ \ell(\hat{\beta}, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}||y - X\hat{\beta}||^2 $$
令其对 $\sigma^2$ 的导数为 0：
$$ \frac{\partial \ell}{\partial (\sigma^2)} = -\frac{n}{2\sigma^2} + \frac{||y - X\hat{\beta}||^2}{2(\sigma^2)^2} = 0 $$
解得：
$$ \hat{\sigma}^2_{MLE} = \frac{||y - X\hat{\beta}||^2}{n} $$
这就是 $\sigma^2$ 的极大似然估计。

---

## 第二题

### 1. 广义最小二乘估计 (GLS)
**背景与普通 OLS 的区别**：
普通的最小二乘法 (OLS) 假定误差项具有同方差且互不相关，即 $Cov(\epsilon) = \sigma^2 I$。当误差项的协方差矩阵为 $Cov(\epsilon) = \sigma^2 \Sigma$ ($\Sigma \neq I$) 时，存在异方差或自相关现象，此时 OLS 估计虽然仍是无偏的，但不再是最小方差的 (不再是 BLUE)。此时需要使用广义最小二乘法 (GLS)。

**广义最小二乘的构造**：
因为已知 $\Sigma > 0$（正定矩阵），所以它存在非奇异的对称平方根矩阵，设为 $\Sigma^{1/2}$，使得 $\Sigma = \Sigma^{1/2}\Sigma^{1/2}$。
令变换矩阵为 $W = \Sigma^{-1/2}$。在模型两边同乘 $W$：
$$ \Sigma^{-1/2}y = \Sigma^{-1/2}X\beta + \Sigma^{-1/2}\epsilon $$
记 $\tilde{y} = \Sigma^{-1/2}y$, $\tilde{X} = \Sigma^{-1/2}X$, $\tilde{\epsilon} = \Sigma^{-1/2}\epsilon$。此时新误差项的协方差为：
$$ Cov(\tilde{\epsilon}) = Cov(\Sigma^{-1/2}\epsilon) = \Sigma^{-1/2} Cov(\epsilon) (\Sigma^{-1/2})' = \Sigma^{-1/2} (\sigma^2 \Sigma) \Sigma^{-1/2} = \sigma^2 I $$
这就将原模型转化为了满足 OLS 基本假设的标准形式。

**性质**：
对转换后的模型应用 OLS 得到的估计量即为广义最小二乘估计，它具有无偏性，并且是变换后模型的最佳线性无偏估计 (BLUE)。

### 2. 实际求解
对变换后的模型 $\tilde{y} = \tilde{X}\beta + \tilde{\epsilon}$ 应用普通最小二乘法。
由于 $\text{rank}(X) = r$，且 $\Sigma^{-1/2}$ 满秩，因此 $\text{rank}(\tilde{X}) = r$。若 $r < p$，我们需要使用广义逆来写出估计量：
$$ \beta^* = (\tilde{X}'\tilde{X})^- \tilde{X}'\tilde{y} $$
将 $\tilde{X}$ 和 $\tilde{y}$ 替换回原变量：
$$ \beta^* = ((\Sigma^{-1/2}X)' (\Sigma^{-1/2}X))^- (\Sigma^{-1/2}X)' (\Sigma^{-1/2}y) $$
$$ \beta^* = (X'\Sigma^{-1/2}\Sigma^{-1/2}X)^- X'\Sigma^{-1/2}\Sigma^{-1/2}y $$
$$ \beta^* = (X'\Sigma^{-1}X)^- X'\Sigma^{-1}y $$
这就是 $\beta$ 的广义最小二乘估计。

---

## 第三题

### 1. 线性假设
**回忆形式和条件**：
一般的线性假设形式为 $H\beta = d$。本题中考察齐次线性假设 $H\beta = 0$。
约束条件 $\mathcal{M}(H^\top) \subset \mathcal{M}(X^\top)$ 等价于 $H$ 的每一行都在 $X$ 的行空间内。这保证了假设 $H\beta$ 是**可估计的**，即存在矩阵 $M$ 使得 $H = MX$。

**回忆验证方法**：
通常使用 F 检验。构造一个服从 F 分布的统计量，该统计量由约束下的残差平方和 $SS_{H_e}$ 与无约束下的残差平方和 $SS_e$ 的差构成。

### 2. 所有小问的求解（投影矩阵与二次型观点）

为了方便表述，引入无约束最小二乘正交投影矩阵 $P = X(X^\top X)^+X^\top$。

#### (1) 证明 $SS_e \sim \sigma^2 \chi^2_{n-r}$
**证明**：
残差平方和可以写为二次型：$SS_e = ||y - X\hat{\beta}||^2 = y^\top (I-P) y$。
因为 $y \sim N_n(X\beta, \sigma^2 I)$，考察二次型 $y^\top \frac{I-P}{\sigma^2} y$。
注意到 $\frac{I-P}{\sigma^2} \cdot Cov(y) = \frac{I-P}{\sigma^2} (\sigma^2 I) = I-P$，且 $I-P$ 是对称幂等矩阵。
根据正态分布二次型定理，$y^\top \frac{I-P}{\sigma^2} y$ 服从卡方分布。
其中自由度为 $\text{tr}(I-P) = n - \text{rank}(P) = n - r$。
非中心参数 $\lambda = \frac{1}{\sigma^2} E(y)^\top (I-P) E(y) = \frac{1}{\sigma^2} (X\beta)^\top (I-P) (X\beta) = 0$（因为 $(I-P)X = 0$）。
所以：
$$ \frac{SS_e}{\sigma^2} \sim \chi^2_{n-r} \implies SS_e \sim \sigma^2 \chi^2_{n-r} $$

#### (2) 证明 $SS_{H_e} - SS_e \sim \sigma^2 \chi^2_{m, \delta}$
**证明**：
记 $A = H (X^\top X)^+ X^\top$。无约束最小二乘解满足 $\hat{\beta} = (X^\top X)^+ X^\top y$，因此 $H\hat{\beta} = Ay$。
给定的平方和之差为：
$$ SS_{H_e} - SS_e = (H\hat{\beta})^\top \left( H (X^\top X)^+ H^\top \right)^{-1} (H\hat{\beta}) $$
可以改写成 $y$ 的二次型 $y^\top P_H y$，其中：
$$ P_H = A^\top \left( H (X^\top X)^+ H^\top \right)^{-1} A = X(X^\top X)^+ H^\top \left( H (X^\top X)^+ H^\top \right)^{-1} H (X^\top X)^+ X^\top $$
由于 $\mathcal{M}(H^\top) \subset \mathcal{M}(X^\top)$，存在 $M$ 使得 $H = MX$。利用 $X(X^\top X)^+ X^\top X = X$，可以验证 $H (X^\top X)^+ X^\top X (X^\top X)^+ H^\top = H (X^\top X)^+ H^\top$。
这证明了 $P_H$ 是幂等矩阵 ($P_H^2 = P_H$)。
由二次型定理：$\frac{SS_{H_e} - SS_e}{\sigma^2} = y^\top \frac{P_H}{\sigma^2} y$ 服从卡方分布。
自由度为 $\text{tr}(P_H)$。利用迹的循环不变性：
$$ \text{tr}(P_H) = \text{tr} \left( \left( H(X^\top X)^+H^\top \right)^{-1} H(X^\top X)^+X^\top X(X^\top X)^+H^\top \right) = \text{tr} \left( I_m \right) = m $$
非中心参数 $\delta$ 为：
$$ \delta = \frac{1}{\sigma^2} (X\beta)^\top P_H (X\beta) $$
注意到 $AX\beta = H(X^\top X)^+X^\top X\beta = H\beta$（因为 $H=MX$），所以：
$$ \delta = \frac{1}{\sigma^2} (H\beta)^\top \left( H (X^\top X)^+ H^\top \right)^{-1} (H\beta) $$
因此 $\frac{SS_{H_e} - SS_e}{\sigma^2} \sim \chi^2_{m, \delta}$，即 $SS_{H_e} - SS_e \sim \sigma^2 \chi^2_{m, \delta}$。

#### (3) 证明 $SS_{H_e} - SS_e$ 与 $SS_e$ 相互独立
**证明**：
这两个统计量对应的二次型矩阵分别为 $P_H$ 和 $I-P$。
根据 Craig 定理，证明这两个二次型相互独立等价于证明它们矩阵乘积为 0，即 $(I-P)P_H = 0$。
因为 $(I-P)X = 0$，而 $P_H$ 的结构中左边第一项包含 $X$，故有：
$$ (I-P) P_H = (I - X(X^\top X)^+X^\top) X \dots = (X - X) \dots = 0 $$
这就证明了二次型矩阵正交，因此 $SS_{H_e} - SS_e$ 与 $SS_e$ 在统计上相互独立。

#### (4) 证明当线性假设 $H\beta = 0$ 为真时, $F \sim F_{m, n-r}$
**证明**：
当线性假设 $H\beta = 0$ 为真时，子问题(2)中的非中心参数变为：
$$ \delta = \frac{1}{\sigma^2} (0)^\top \left( H (X^\top X)^+ H^\top \right)^{-1} (0) = 0 $$
此时 $\frac{SS_{H_e} - SS_e}{\sigma^2}$ 退化为中心卡方分布 $\chi^2_m$。
又根据子问题(1)和(3)，$\frac{SS_e}{\sigma^2} \sim \chi^2_{n-r}$，并且这两个卡方统计量相互独立。
根据 F 分布的定义（两个独立的卡方分布变量除以它们各自自由度的商服从 F 分布），得出：
$$ F = \frac{\left(\frac{SS_{H_e} - SS_e}{\sigma^2}\right) \Big/ m}{\left(\frac{SS_e}{\sigma^2}\right) \Big/ (n-r)} = \frac{(SS_{H_e} - SS_e)/m}{SS_e / (n-r)} \sim F_{m, n-r} $$
证明完毕。