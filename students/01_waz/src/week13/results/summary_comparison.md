# Week 13 Summary Comparison: Theory and Practice

## 1. Lasso 面对高度相关变量组的业务风险，Elastic Net 如何缓解？

在模拟数据中，`x_signal_1`, `x_signal_2`, `x_signal_3` 几乎表达同一类信息（来自同一个 latent factor）。Lasso 的 L1 penalty 会产生稀疏解，因此它可能**只保留其中一个变量**，把其他同组变量压到 0。

从预测角度这未必是坏事（去掉了冗余信息），但从**业务解释角度有风险**：业务方可能误以为被压为 0 的变量完全不重要，但事实上它们与保留下来的变量代表同一维度，只是因为共线性被"牺牲"了。

**Elastic Net 的缓解机制**：Elastic Net 同时包含 L1 与 L2 penalty。L1 带来变量筛选能力，L2 让相关变量可以成组保留并共同收缩。因此 Elastic Net 通常比 Lasso 更适合"变量高度相关但都代表同一业务维度"的场景——它不会像 Lasso 那样"狠心"地只留一个。

## 2. GridSearchCV 的最低验证误差，与"越稀疏越好/越稳越好"有什么异同？

三者从不同维度评价模型：

| 维度 | 关注点 | 方法 |
|------|--------|------|
| **最低验证误差** | 预测准不准 | GridSearchCV 选择 CV RMSE 最低的 alpha |
| **越稀疏越好** | 变量名单短不短 | 偏好 Lasso 的大 alpha 或后向剔除 |
| **越稳越好** | 样本变化时结论是否可靠 | 看 Ridge 或多次切分的系数标准差 |

三者相关但不完全相同：
- 如果任务是**纯预测**（如 Kaggle 上分），优先看 GridSearchCV 的最低验证误差；
- 如果任务是**向业务方解释关键因素**，则还要综合考虑系数稳定性、变量共线性和 Lasso/Elastic Net 的筛选差异；
- 最低验证误差的超参数不一定产生最稀疏的模型——GridSearchCV 追求的是泛化能力，而非解释简洁性。

## 3. 前向选择/后向剔除与 Lasso 的效率和结果差异

本作业实现的是前向选择 Top-K（`ForwardSelectorCV`）。

### 效率对比

| 方法 | 计算复杂度 | 特点 |
|------|-----------|------|
| **前向选择** | 每一步遍历剩余候选变量并做 K 折 CV，O(p × k × K × n) | 贪心搜索，早期选择可能影响后续结果 |
| **Lasso** | 一个凸优化问题中同时完成拟合和筛选 | 计算更统一，借助坐标下降等算法效率更高 |

### 结果差异

- 前向选择的优点是过程**直观透明**，每一步都能解释"为什么加入这个变量"；
- Lasso 的优点是**全局优化**，在高度相关变量中自动保留代表变量；
- 但 Lasso 在高度相关变量组中可能表现**不稳定**——稍微修改 alpha 或样本切分，可能保留组内不同的变量。

### 实际体会

在前向选择的每一步需要做 `(剩余特征数 × K 折)` 次模型拟合，当特征数较多时计算压力明显。而 Lasso 只需一次 `GridSearchCV`，计算效率更高。但前向选择的过程记录（`history_frame`）对业务解释非常友好。

## 4. 模拟数据结果概览

| model | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| OLS | 1.4815 | 1.1704 | 6.0768 |
| Ridge | 1.4563 | 1.1580 | 6.0127 |
| Lasso | 1.4545 | 1.1549 | 6.0053 |
| ElasticNet | 1.4328 | 1.1405 | 5.9217 |
| ForwardSelectionTop5 | 1.4232 | 1.1426 | 5.9516 |


模拟数据的主要结论：
- **OLS** 预测不一定很差，但高度相关变量的单个系数不稳定；
- **Ridge** 更稳定，系数标准差显著小于 OLS；
- **Lasso** 更稀疏，自动将噪声变量和冗余变量压缩为 0；
- **Elastic Net** 介于二者之间，兼具稀疏性和组保留能力。

## 5. 本周文件与代码位置

- 入口：`src/week13/main.py`
- 自定义指标：`src/utils/metrics.py`（含 `calculate_rmse`, `calculate_mae`, `calculate_mape`, `summarize_regression_metrics`）
- 自定义标准化器：`src/utils/transformers.py`（含 `CustomStandardScaler`，已兼容 sklearn Pipeline）
- 自定义前向选择：`src/utils/models.py` 中的 `ForwardSelectorCV`
- 模拟数据：`src/week13/data/synthetic_correlated.csv`
- 图像目录：`src/week13/results/figures/`
- 合成报告：`src/week13/results/synthetic_report.md`
- 总结报告：`src/week13/results/summary_comparison.md`

## 6. 技术要点总结

1. **目标函数 = loss + penalty**：正则化的本质是在拟合数据和约束复杂度之间寻求平衡；
2. **系数收缩与变量筛选**：L2 收缩、L1 筛选、Elastic Net 折中；
3. **交叉验证与超参数寻优**：GridSearchCV 是科学选择 alpha 的标准方法；
4. **算法对比与稳定性验证**：多次随机切分 + 箱线图是展示稳定性的有力手段。
