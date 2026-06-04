# Week 13 Summary Comparison

## 1. Lasso 面对高度相关变量组的业务风险，Elastic Net 如何缓解？

在模拟数据中，`x_signal_1`, `x_signal_2`, `x_signal_3` 几乎表达同一类信息。Lasso 的 L1 penalty 会产生稀疏解，因此它可能只保留其中一个变量，把其他同组变量压到 0。从预测角度这未必是坏事，但从业务解释角度有风险：业务方可能误以为被压为 0 的变量完全不重要。

Elastic Net 同时包含 L1 与 L2 penalty。L1 带来变量筛选，L2 让相关变量可以成组保留并共同收缩。因此 Elastic Net 通常比 Lasso 更适合“变量高度相关但都代表同一业务维度”的场景。

## 2. GridSearchCV 的最低验证误差，与“越稀疏越好/越稳越好”有什么异同？

GridSearchCV 的目标是选择验证 RMSE 最低的超参数，它直接服务于预测泛化表现。但“越稀疏越好”是解释性偏好，“越稳越好”是系数稳定性偏好。三者相关但不完全相同：

- 最低验证误差：关注预测准不准；
- 稀疏：关注变量名单短不短；
- 稳定：关注样本变化时结论是否可靠。

因此，如果任务是纯预测，可以优先看 GridSearchCV；如果任务是向业务方解释关键因素，则还要看系数稳定性、变量共线性和 Lasso/Elastic Net 的筛选差异。

## 3. 前向选择/后向剔除与 Lasso 的效率和结果差异

本作业实现的是前向选择 Top-K。它的优点是过程直观，每一步都能解释“为什么加入这个变量”；缺点是计算成本较高，因为每一步要遍历剩余候选变量并做交叉验证，而且它是贪心算法，早期选择可能影响后续结果。

Lasso 则是在一个优化目标中同时完成拟合和筛选，计算上通常更统一、更适合高维数据。但在高度相关变量组中，Lasso 的选择可能不稳定，可能只保留其中一个代表变量。

## 4. 模拟数据结果概览

| model | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| OLS | 1.4815 | 1.1704 | 6.0768 |
| Ridge | 1.4563 | 1.1580 | 6.0127 |
| Lasso | 1.4545 | 1.1549 | 6.0053 |
| ElasticNet | 1.4328 | 1.1405 | 5.9217 |
| ForwardSelectionTop5 | 1.4232 | 1.1426 | 5.9516 |


模拟数据的主要结论是：OLS 预测不一定很差，但高度相关变量的单个系数不稳定；Ridge 更稳定，Lasso 更稀疏，Elastic Net 介于二者之间。

## 5. Kaggle 真实数据结果概览

| model | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| OLS | 0.1479 | 0.1079 | 0.9047 |
| Ridge | 0.1476 | 0.1060 | 0.8895 |
| Lasso | 0.1484 | 0.1067 | 0.8953 |
| ElasticNet | 0.1483 | 0.1065 | 0.8942 |
| ForwardSelectionTop8 | 0.1534 | 0.1124 | 0.9429 |


Kaggle House Prices 数据的主要结论是：正则化方法提供了更稳健的系数解释框架。即使测试误差提升不巨大，Ridge/Lasso/Elastic Net 仍能帮助我们理解高维、共线性特征下的变量收缩与筛选。

## 6. 本周文件与代码位置

- 入口：`week13/main.py`
- 自定义指标：`src/utils/metrics.py`
- 自定义标准化、填补、预处理：`src/utils/transformers.py`
- 自定义前向选择：`src/utils/models.py` 中的 `ForwardSelectorCV`
- 模拟数据：`week13/data/synthetic_correlated.csv`
- Kaggle 数据：`week13/data/kaggle_house_prices.csv`
- 图像目录：`week13/results/figures/`
