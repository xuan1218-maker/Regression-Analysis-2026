1. (30 分) 给定线性模型: $y = X\beta + \epsilon$, 其中:
   - $X$ 为 $n \times p$ 设计矩阵, 其维度满足 $n \leq p$, $\text{rank}(X) = r$;
   - $y$ 为 $n \times 1$ 随机向量,
   - $\beta$ 为 $p \times 1$ 待估计参数向量,
   - $\epsilon$ 为 $n \times 1$ 随机误差向量。随机误差向量满足 $E(\epsilon) = 0, \text{Cov}(\epsilon) = \sigma^2 I_n$, 其中 $I_n$ 为 $n \times n$ 单位矩阵。

那么，

1. 证明: 对于任何的可估计函数 $c'\beta$, 它的最小二乘估计 $c'\hat{\beta}$ 为其唯一 BLUE(Best Linear Unbiased Estimator), 其中 $\hat{\beta} = (X'X)^{-}X'y$.
2. 证明: $\hat{\sigma}^2 = \frac{||y - X\hat{\beta}||^2}{n - r}$ 为 $\sigma^2$ 的无偏估计;
3. 给定线性模型的正态性假定, 即 $\epsilon \sim N(0, \sigma^2 I_n)$, 求 $\sigma^2$ 的极大似然估计。

---

2. (30 分) 给定线性模型: $y = X\beta + \epsilon$, 其中:
   - $X$ 为 $n \times p$ 设计矩阵, 且 $n \geq p$, $\text{rank}(X) = r$,
   - $y$ 为 $n \times 1$ 随机向量,
   - $\beta$ 为 $p \times 1$ 待估未知参数向量,
   - $\epsilon$ 为 $n \times 1$ 随机误差向量, 且 $E(\epsilon) = 0, \text{Cov}(\epsilon) = \sigma^2 \Sigma$, 其中 $\Sigma > 0$。

求: $\beta$ 的广义最小二乘估计 $\beta^*$.

---

3. (40 分) 给定正态线性模型: $y = X\beta + e$, 其中:
   - $X$ 为 $n \times p$ 设计矩阵 ($n \geq p$), $\text{rank}(X) = r$,
   - $y$ 为 $n \times 1$ 随机向量,
   - $\beta$ 为 $p \times 1$ 待估未知参数向量,
   - 随机误差向量 $\epsilon \sim N_n(0, \sigma^2 I)$。

齐次线性假设 $H\beta = 0$, $\text{rank}(H_{m \times p}) = m$, $\mathcal{M}(H^\top) \subset \mathcal{M}(X^\top)$, 证明:

1. $SS_e \sim \sigma^2 \chi^2_{n-r}$, 其中 $SS_e = |y - X\hat{\beta}|^2$ 为残差平方和;
2. $SS_{H_e} - SS_e = (H\hat{\beta})^\top \left( H (X^\top X)^+ H^\top \right)^{-1} (H\hat{\beta}) \sim \sigma^2 \chi^2_{m, \delta}$, 其中:
   - 非中心参数 $\delta = (H\beta)^\top \left( H (X^\top X)^+ H^\top \right)^{-1} (H\beta) / \sigma^2$,
   - $SS_{H_e} = |y - X\hat{\beta}_H|^2$ 为在约束 $H\beta = 0$ 下的残差平方和,
   - $\hat{\beta}_H$ 为范数最小的约束最小二乘估计;
3. $SS_{H_e} - SS_e$ 与 $SS_e$ 相互独立;
4. 当线性假设 $H\beta = 0$ 为真时, $F \sim F_{m, n-r}$, 其中 $F = \frac{(SS_{H_e} - SS_e)/m}{SS_e / (n-r)}$。