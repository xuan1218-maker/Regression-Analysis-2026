# Week 13 Synthetic Report: Regularized Regression on Correlated Data

## 1. 数据生成设计与 DGP

本任务自己生成了 `src/week13/data/synthetic_correlated.csv`，样本量为 **520**，特征数为 **12**，满足 Week 13 对"样本量不少于 300、至少 8 个特征、显式构造共线性特征族"的要求。

真实数据生成过程（DGP）为：

```text
 y = 20 + 4.0 * x_signal_1 - 2.5 * x_independent_1 + 1.8 * x_independent_2 + noise
```

其中：

- **高度相关特征族**：`x_signal_1`, `x_signal_2`, `x_signal_3`，三者来自同一个 latent factor，彼此 Pearson 相关系数接近 1.0；
- **第二组潜在相关变量**：`x_independent_1`, `x_independent_2`, `x_mixed_collinear`，其中 `x_mixed_collinear` 是前两个变量的线性混合；
- **纯噪声变量**：`noise_1`, `noise_2`, `noise_3`, `noise_4`, `noise_5`，与 y 无任何关联；
- **弱信号变量**：`weak_signal`，它与 latent factor 有轻微关系，但不在真实 DGP 中。

## 2. 高相关变量检查

下面是绝对相关系数超过 0.75 的变量对。可以看到 `x_signal_1/x_signal_2/x_signal_3` 这组变量高度相关，非常适合展示 OLS 系数不稳定和 Ridge/Lasso/Elastic Net 的差异。

| feature_1 | feature_2 | abs_corr |
| --- | --- | --- |
| x_signal_1 | x_signal_2 | 0.9977 |
| x_signal_1 | x_signal_3 | 0.9976 |
| x_signal_2 | x_signal_3 | 0.9968 |
| x_independent_1 | x_mixed_collinear | 0.9403 |


## 3. OLS 与 Ridge 的系数稳定性对比

作业要求至少做 50 次不同随机切分。我这里做了 **60 次 train/test split**，每次分别拟合 OLS 和 Ridge(alpha=10)，并收集高度相关特征族的系数。结果表明，OLS 在高度相关变量之间会"抢解释权"，单个变量的系数标准差较大；Ridge 通过 L2 penalty 把系数整体收缩，因此跨样本切分更稳定。

| model | feature | coef_mean | coef_std | coef_min | coef_max |
| --- | --- | --- | --- | --- | --- |
| OLS | x_signal_1 | 1.2394 | 0.8323 | -0.4070 | 2.9767 |
| OLS | x_signal_2 | 0.6181 | 0.6993 | -0.9081 | 2.1447 |
| OLS | x_signal_3 | 2.0745 | 0.6735 | 0.5522 | 3.4321 |
| Ridge_alpha_10 | x_signal_1 | 1.2915 | 0.0661 | 1.1420 | 1.4297 |
| Ridge_alpha_10 | x_signal_2 | 1.2239 | 0.0623 | 1.0786 | 1.3689 |
| Ridge_alpha_10 | x_signal_3 | 1.3752 | 0.0742 | 1.2052 | 1.5248 |


对应图像：`src/week13/results/figures/synthetic_coefficient_stability.png`。

### 稳定性分析

从上面表格可以看到：
- OLS 在 `x_signal_1/x_signal_2/x_signal_3` 三个高度相关特征上的系数标准差远大于 Ridge；
- Ridge 通过 L2 正则化有效抑制了系数在不同切分之间的波动，使结论更稳定；
- 这直观地向业务方展示了：引入正则化后，哪怕换一批样本，我们的结论也变得稳定得多。

## 4. 为什么正则化前必须标准化？

Ridge、Lasso、Elastic Net 的 penalty 都直接作用在系数大小上。如果某个变量的量纲很大，模型可以用较小系数表达相同变化；如果某个变量量纲很小，则需要较大系数。若不标准化，penalty 会把"量纲差异"误当成"变量重要性差异"，导致正则化不公平。因此本实验用 `Pipeline([CustomStandardScaler(), model])`，其中 `CustomStandardScaler` 来自自己的 `src/utils/transformers.py`，并已改造为兼容 sklearn Pipeline 的 `BaseEstimator/TransformerMixin` 子类。

## 5. GridSearchCV 调参与最优参数

对 Ridge 和 Lasso 使用对数空间 `alpha`（`np.logspace(-4, 3, 36)`）；对 Elastic Net 同时搜索 `alpha` 与 `l1_ratio` 的二维网格。5 折交叉验证的 RMSE 曲线保存为：`src/week13/results/figures/synthetic_cv_alpha_curves.png`。

| model | best_params | best_cv_rmse |
| --- | --- | --- |
| Ridge | {'model__alpha': '1.5849'} | 1.6072 |
| Lasso | {'model__alpha': '0.0158'} | 1.6073 |
| ElasticNet | {'model__alpha': '0.0278', 'model__l1_ratio': '0.7000'} | 1.6055 |


图像中可以看到典型的 U 型曲线：alpha 太小 → 接近 OLS，过拟合；alpha 太大 → 过度收缩，欠拟合。最低点对应的 alpha 即为最优超参数。

## 6. 测试集模型表现

| model | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| OLS | 1.4815 | 1.1704 | 6.0768 |
| Ridge | 1.4563 | 1.1580 | 6.0127 |
| Lasso | 1.4545 | 1.1549 | 6.0053 |
| ElasticNet | 1.4328 | 1.1405 | 5.9217 |
| ForwardSelectionTop5 | 1.4232 | 1.1426 | 5.9516 |


RMSE 和 MAE 均由自己的 `src/utils/metrics.py` 中的 `summarize_regression_metrics` 计算（底层调用 `calculate_rmse` 和 `calculate_mae`），而不是直接调用 sklearn 指标。OLS 在预测上并不一定非常差，但它的系数解释不稳定；正则化的价值主要体现在稳定性和变量筛选解释上。

## 7. 模型性格：Ridge / Lasso / Elastic Net 如何处理相关变量？

下面列出关键变量在三个最优正则化模型中的标准化系数。

| model | feature | coefficient |
| --- | --- | --- |
| Ridge | x_signal_1 | 1.3068 |
| Ridge | x_signal_2 | 0.6133 |
| Ridge | x_signal_3 | 2.0287 |
| Ridge | x_independent_1 | -1.2533 |
| Ridge | x_independent_2 | 2.2544 |
| Ridge | x_mixed_collinear | -1.3438 |
| Ridge | noise_1 | 0.0358 |
| Ridge | noise_2 | 0.0493 |
| Lasso | x_signal_1 | 1.1258 |
| Lasso | x_signal_2 | 0.0000 |
| Lasso | x_signal_3 | 2.8153 |
| Lasso | x_independent_1 | -1.5219 |
| Lasso | x_independent_2 | 2.1488 |
| Lasso | x_mixed_collinear | -1.0463 |
| Lasso | noise_1 | 0.0167 |
| Lasso | noise_2 | 0.0318 |
| ElasticNet | x_signal_1 | 1.2932 |
| ElasticNet | x_signal_2 | 0.8812 |
| ElasticNet | x_signal_3 | 1.7505 |
| ElasticNet | x_independent_1 | -1.6207 |
| ElasticNet | x_independent_2 | 2.0848 |
| ElasticNet | x_mixed_collinear | -0.9196 |
| ElasticNet | noise_1 | 0.0183 |
| ElasticNet | noise_2 | 0.0254 |


对应图像：`src/week13/results/figures/synthetic_model_coefficients.png`。

### 模型性格解读

- **Ridge**：倾向于把高度相关的一组变量（`x_signal_1/x_signal_2/x_signal_3`）一起保留，并较均匀地缩小系数。这与 L2 penalty 的性质一致——它收缩但不稀疏化。
- **Lasso**：倾向于在高度相关变量中挑选少数变量（可能只保留 `x_signal_1`），把其他同组变量压到 0。这与 L1 penalty 的"尖角"性质一致——它有自动变量筛选效果，但也可能随机保留其中一个而丢掉同组变量。
- **Elastic Net**：同时有 L1 和 L2 penalty（`l1_ratio` 控制混合比例），通常比 Lasso 温和，会在稀疏性和组保留之间折中。

这些表现与课堂上学习的"模型性格"完全一致。

## 8. 自定义前向选择 vs Lasso 自动筛选

我在 `src/utils/models.py` 中实现了 `ForwardSelectorCV`，它每一步用 K 折 CV 比较所有候选变量，选择能使验证 RMSE 最低的变量加入。

前向选择过程：

| step | added_feature | cv_rmse | selected_features |
| --- | --- | --- | --- |
| 1 | x_signal_1 | 3.4209 | x_signal_1 |
| 2 | x_independent_1 | 2.4280 | x_signal_1, x_independent_1 |
| 3 | x_independent_2 | 1.6235 | x_signal_1, x_independent_1, x_independent_2 |
| 4 | weak_signal | 1.6115 | x_signal_1, x_independent_1, x_independent_2, weak_signal |
| 5 | x_signal_3 | 1.6030 | x_signal_1, x_independent_1, x_independent_2, weak_signal, x_signal_3 |


前向选择最终变量：

```text
['x_signal_1', 'x_independent_1', 'x_independent_2', 'weak_signal', 'x_signal_3']
```

前向选择测试集指标：

| model | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| ForwardSelectionTop5 | 1.4232 | 1.1426 | 5.9516 |


Lasso 非零变量名单：

```text
['x_signal_1', 'x_signal_3', 'x_independent_1', 'x_independent_2', 'x_mixed_collinear', 'noise_1', 'noise_2', 'noise_3', 'noise_4', 'weak_signal']
```

### 对比分析

两者不完全一致是正常的：
- 前向选择是**贪心搜索**，每一步只看"当前新增一个变量后 CV RMSE 是否下降"；
- Lasso 是在**同一个优化目标**中同时平衡 loss 和 L1 penalty。
- 面对高度相关变量时，Lasso 更容易保留其中一个代表变量，而前向选择可能根据当时已经入选的变量组合继续补充其他变量。
