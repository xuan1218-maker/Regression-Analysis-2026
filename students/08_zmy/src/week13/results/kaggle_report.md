# Kaggle 艾姆斯房价预测报告

## 数据集说明
- 名称: 艾姆斯房价数据集
- 来源: Kaggle 房价预测竞赛
- 目标变量: SalePrice (美元)
- 原始特征: 79 个 (我们仅使用了数值特征以便聚焦正则化)

## 数据预处理
- 删除 Id 列，剔除缺失率超过 50% 的数值列，使用 CustomImputer 以均值填补剩余缺失值。
- 未使用类别编码，以保持对正则化效果的关注。

## 模型性能 (5折交叉验证 + 测试集)
- 岭回归: RMSE=37231.84, MAE=22283.27, alpha=188.7392
- Lasso: RMSE=37239.42, MAE=22767.45, alpha=1000.0000
- ElasticNet: RMSE=37239.42, MAE=22767.45, alpha=1000.0000, l1_ratio=1.00

## 变量选择结果
- Lasso 选择了 23 个特征: ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF']...
- 前向选择 (p<0.05): ['OverallQual', 'GrLivArea', 'BsmtFinSF1', 'GarageCars', 'MSSubClass', 'YearRemodAdd', 'LotArea', 'YearBuilt', 'BsmtFullBath', 'Fireplaces']...
- 逐步回归: ['OverallQual', 'GrLivArea', 'BsmtFinSF1', 'GarageCars', 'MSSubClass', 'YearRemodAdd', 'LotArea', 'YearBuilt', 'BsmtFullBath', 'Fireplaces']...

## 业务解释
正则化提高了模型的泛化能力。Lasso 减少了特征数量，可以简化模型部署。但在房地产领域，许多特征都可能重要，因此岭回归可能更合适。
