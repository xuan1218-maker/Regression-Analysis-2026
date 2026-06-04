# Task B：Kaggle 真实数据报告

## B1. 数据集信息

- **数据集名称**: House Prices: Advanced Regression Techniques (with engineered features)
- **来源**: Kaggle (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **数据文件**: `train_with_engineered_features.csv`（已清洗并添加工程特征）
- **样本量**: 1458 行
- **特征数**: 45 个数值特征
- **目标变量**: SalePrice

### 为什么适合练习正则化

1. 特征数量较多（>= 15），存在高维场景；

2. 面积相关特征（GrLivArea, 1stFlrSF, 2ndFlrSF, TotalBsmtSF, TotalSF）之间存在天然的共线性；

3. 工程特征（TotalSF, TotalBath, HouseAge 等）与原始特征高度相关，适合观察正则化对共线性的处理效果。

## B2. 模型对比

| 模型 | RMSE | MAE | 最优 alpha | l1_ratio |
|------|------|-----|-----------|----------|
| OLS | 27295.80 | 20053.50 | — | — |
| Ridge | 26971.97 | 19997.88 | 51.7947 | — |
| Lasso | 26925.51 | 19986.10 | 719.6857 | — |
| ElasticNet | 26968.27 | 20006.54 | 0.6158 | 0.9 |

### 特征重要度（系数）

#### Lasso Top-5 最重要特征

| 排名 | 特征 | Lasso 系数 |
|------|------|-----------|
| 1 | TotalSF | 28927.74 |
| 2 | OverallQual | 19837.10 |
| 3 | BsmtFinSF1 | 9889.49 |
| 4 | GrLivArea | 9816.50 |
| 5 | BedroomAbvGr | -9120.03 |

#### Lasso 剔除的特征 (16 个)

['YearRemodAdd', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'FullBath', 'HalfBath', 'GarageYrBlt', 'EnclosedPorch', '3SsnPorch', 'MoSold', 'YrSold', 'HasFireplace', 'HasPool']

## B3. 推测与解释

### 正则化是否提升了验证集表现？

与 OLS 相比：
- Ridge RMSE 降低了 1.19%
- Lasso RMSE 降低了 1.36%
- ElasticNet RMSE 降低了 1.20%

如果改善幅度不大，可能原因：
1. 数据已经过清洗和工程特征处理，共线性已被部分缓解；
2. 特征数量相对于样本量不算特别多（不存在 "p >> n" 的极端高维场景）；
3. OLS 在特征数适中时本身表现就不错，正则化的边际收益有限。

### Lasso 剔除了哪些特征？

Lasso 共剔除了 16 个特征（系数为 0）。

从业务逻辑看：
- Lasso 倾向于在相关特征组中只保留一个。例如 TotalSF 和 GrLivArea 高度相关，Lasso 可能只保留其中一个，这是合理的——两者携带类似信息；
- 被剔除的低方差或弱相关特征（如 PoolArea, MiscVal 等）在业务上确实可能对房价影响有限。

### 最关键的 5 个影响因素

以 **Lasso** 的结果为准，原因：
1. Lasso 具有内置的变量选择能力（L1 惩罚产生稀疏解），能自动剔除不重要的特征；
2. 相比 Ridge（保留所有特征）和 OLS（无惩罚），Lasso 的特征重要度排序更清晰；
3. ElasticNet 的 l1_ratio 接近 1 时行为类似 Lasso，可作为交叉验证。

Top-5 最关键因素：

| 排名 | 特征 | 系数 | 业务含义 |
|------|------|------|----------|
| 1 | TotalSF | 28927.74 | 总面积（含地下室和车库） |
| 2 | OverallQual | 19837.10 | 整体材料和装修质量 |
| 3 | BsmtFinSF1 | 9889.49 | 地下室中经过装修的区域面积 |
| 4 | GrLivArea | 9816.50 | 地上居住面积 |
| 5 | BedroomAbvGr | -9120.03 | 卧室数量 |
