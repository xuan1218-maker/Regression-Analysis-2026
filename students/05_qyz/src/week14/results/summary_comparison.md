# Summary Comparison: Lasso vs PCR

## 1. 选择与压缩的本质差异
- Lasso 解决的是“谁留下”，它通过对原始变量施加 L1 惩罚，产生稀疏系数。
- PCR 解决的是“保留多少个方向”，它先对特征空间做 PCA，再在低维主成分上回归。

## 2. 何时更自然
- 当数据是真正的 sparse truth 时，只有少数原始变量直接驱动 y，Lasso 更自然。
- 当数据更像 latent-factor truth，原始变量之间存在重复信息，PCR 更自然。

## 3. 经营问题视角
- Lasso 回答的是“哪个原始特征被保留下来”；
- PCR 回答的是“用多少个稳定方向来表示原始特征”。
- 如果业务方要求短名单，优先 Lasso；
- 如果业务方要求稳健预测器，PCR 更值得考虑。

## 4. 为什么不把前向 / 后向选择拉回主舞台
- 本周主线在于 selection vs compression，而前向/后向选择仍然属于变量筛选范式。
- 如果一定要加，前向/后向选择更接近 selection 路线，而不是 compression。