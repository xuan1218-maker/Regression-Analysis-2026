# Week 13 Kaggle Report: House Prices Regularization Study

## 1. 数据来源与业务背景

本部分完成 Week13 的选做加分任务。数据来自 Kaggle 竞赛 **House Prices - Advanced Regression Techniques**，目标是根据 Ames, Iowa 房屋属性预测 `SalePrice`。本提交将原始训练集保存为：

```text
week13/data/kaggle_house_prices.csv
```

该数据集适合练习正则化和变量筛选，因为它有 **1460** 行、**81** 列，特征数量较多；同时房屋面积、楼层面积、地下室面积、车库面积、质量评分等变量之间存在明显业务相关性，容易产生共线性。

本实验为了保持模型解释清楚，使用数值型变量建模，共使用 **36** 个特征。目标变量使用 `log1p(SalePrice)`，因此 RMSE/MAE 是对数价格误差，不是美元误差。

## 2. 数据清洗与自定义预处理

- 删除 `Id`，因为它只是样本编号；
- 仅保留数值型特征，保证线性模型系数含义清晰；
- 删除缺失率过高的数值列；
- 用自己的 `CustomNumericImputer` 在训练集上拟合中位数并填补缺失；
- 在 Ridge/Lasso/Elastic Net 的 `Pipeline` 中使用自己的 `CustomStandardScaler`；
- 训练/测试划分后，测试集只使用训练集学到的填补值和标准化参数，避免数据泄露。

## 3. GridSearchCV 最优参数

| model | best_params | best_cv_rmse |
| --- | --- | --- |
| Ridge | {'model__alpha': '100.0000'} | 0.1620 |
| Lasso | {'model__alpha': '0.0073'} | 0.1587 |
| ElasticNet | {'model__alpha': '0.0142', 'model__l1_ratio': '0.5000'} | 0.1586 |


CV 曲线图：`week13/results/figures/kaggle_cv_alpha_curves.png`。

## 4. 测试集表现

| model | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| OLS | 0.1479 | 0.1079 | 0.9047 |
| Ridge | 0.1476 | 0.1060 | 0.8895 |
| Lasso | 0.1484 | 0.1067 | 0.8953 |
| ElasticNet | 0.1483 | 0.1065 | 0.8942 |
| ForwardSelectionTop8 | 0.1534 | 0.1124 | 0.9429 |


解释：如果正则化方法没有显著优于 OLS，可能原因有三点：第一，当前只使用数值变量，维度虽高但样本量也有 1460 行；第二，OLS 在预测层面未必崩溃，但系数解释可能不稳定；第三，真正能提升 Kaggle 排名的特征工程通常还包括大量类别变量编码、异常值处理和非线性模型，本作业重点是正则化回归的推测比较。

## 5. Lasso 删除了哪些特征？业务上是否合理？

Lasso 非零变量预览：

```text
['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', '3SsnPorch', 'ScreenPorch', 'PoolArea']
```

Lasso 压缩为 0 的变量预览：

```text
['LotFrontage', 'MasVnrArea', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtHalfBath', 'BedroomAbvGr', 'GarageYrBlt', 'OpenPorchSF', 'EnclosedPorch', 'MiscVal', 'MoSold', 'YrSold']
```

从业务上看，被压为 0 并不意味着这些房屋属性“没有用”。很多面积类变量高度相关，例如 `TotalBsmtSF`, `1stFlrSF`, `GrLivArea`, `GarageArea`；质量类变量也可能互相替代。Lasso 面对相关变量时倾向于保留一个代表变量，因此被剔除的变量更准确地说是“在当前 penalty 和其他变量共同存在的条件下，边际贡献不足”。

## 6. 如果业务方要最关键的 5 个影响因素，我会以什么方法为准？

我不会直接只看 Lasso。Lasso 的优点是稀疏，但它在高度相关变量组内可能随机保留其中一个。业务解释时，我更倾向于综合：

1. Elastic Net 的绝对系数排名；
2. Lasso 的非零变量名单；
3. 前向选择的 Top-K 结果；
4. 房地产业务常识。

Elastic Net 的前 10 个标准化系数如下：

| model | feature | coefficient | abs_coef |
| --- | --- | --- | --- |
| ElasticNet | OverallQual | 0.1207 | 0.1207 |
| ElasticNet | GrLivArea | 0.0892 | 0.0892 |
| ElasticNet | YearBuilt | 0.0709 | 0.0709 |
| ElasticNet | GarageCars | 0.0610 | 0.0610 |
| ElasticNet | OverallCond | 0.0369 | 0.0369 |
| ElasticNet | Fireplaces | 0.0319 | 0.0319 |
| ElasticNet | BsmtFullBath | 0.0317 | 0.0317 |
| ElasticNet | YearRemodAdd | 0.0272 | 0.0272 |
| ElasticNet | MSSubClass | -0.0241 | 0.0241 |
| ElasticNet | TotRmsAbvGrd | 0.0172 | 0.0172 |


前向选择过程如下：

| step | added_feature | cv_rmse | selected_features |
| --- | --- | --- | --- |
| 1 | OverallQual | 0.2308 | OverallQual |
| 2 | GrLivArea | 0.2066 | OverallQual, GrLivArea |
| 3 | YearBuilt | 0.1882 | OverallQual, GrLivArea, YearBuilt |
| 4 | GarageCars | 0.1792 | OverallQual, GrLivArea, YearBuilt, GarageCars |
| 5 | OverallCond | 0.1703 | OverallQual, GrLivArea, YearBuilt, GarageCars, OverallCond |
| 6 | BsmtFullBath | 0.1625 | OverallQual, GrLivArea, YearBuilt, GarageCars, OverallCond, BsmtFullBath |
| 7 | MSSubClass | 0.1565 | OverallQual, GrLivArea, YearBuilt, GarageCars, OverallCond, BsmtFullBath, MSSubClass |
| 8 | Fireplaces | 0.1536 | OverallQual, GrLivArea, YearBuilt, GarageCars, OverallCond, BsmtFullBath, MSSubClass, Fireplaces |


最终前向选择变量：

```text
['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageCars', 'OverallCond', 'BsmtFullBath', 'MSSubClass', 'Fireplaces']
```

图像：`week13/results/figures/kaggle_top_coefficients.png`。

## 7. 真实决策风险

该模型可以帮助理解哪些房屋属性与价格相关，但不能直接当作因果结论。例如 `OverallQual` 重要，并不等价于“只要人为提高评分就能同比例提高售价”。此外，房价受地段、宏观周期、供需、学校和社区等影响，单纯线性模型只能提供一个可解释基准。
