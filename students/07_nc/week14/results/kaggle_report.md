# 第 14 周 Kaggle 报告：House Prices 房价数据

## 1. 数据集和目标变量

选做真实数据实验使用的是：

```text
week14/data/kaggle_house_prices.csv
```

该数据来自 House Prices 回归任务，目标变量是 `SalePrice`。我对目标变量做了 `log1p(SalePrice)` 变换，用来减弱房价右偏分布的影响。

这一版作业只使用数值型特征。即使如此，这份数据仍然适合 Week14，因为房屋属性天然存在相关性：居住面积、地下室面积、车库容量、房间数、建造年份和质量评分往往一起变化。

## 2. 模型对比

| model | test_RMSE | test_MAE | complexity_metric | complexity_value |
| --- | --- | --- | --- | --- |
| OLS | 0.1489 | 0.1090 | numeric_coefficients | 36.0000 |
| LassoCV | 0.1519 | 0.1097 | nonzero_coefficients | 19.0000 |
| PCR | 0.1512 | 0.1112 | retained_components | 30.0000 |

本次运行中，测试集 RMSE 最低的模型是 **OLS**，其在 `log1p(SalePrice)` 上的 RMSE 为 **0.1489**。

图像：`week14/results/figures/kaggle_model_rmse.png`

- **横轴**：模型名称。
- **纵轴**：`log1p(SalePrice)` 上的测试集 RMSE。
- **柱子**：OLS、LassoCV 和 PCR。
- **结论**：真实数据中不一定有某一个方法压倒性胜出，因此解释时不能只看误差，还要考虑模型稳定性和复杂度。

图像：`week14/results/figures/kaggle_pcr_cv_curve.png`

- **横轴**：保留的主成分数量 `k`。
- **纵轴**：PCR 五折交叉验证 RMSE。
- **曲线**：随着压缩程度减弱，PCR 验证误差如何变化。
- **结论**：这条曲线帮助判断保留多少个主成分已经足够，避免盲目保留所有高维原始变量。

## 3. 高维与共线性的证据

数值特征中绝对相关系数最高的若干对是：

| feature_1 | feature_2 | abs_corr |
| --- | --- | --- |
| GarageCars | GarageArea | 0.8825 |
| GrLivArea | TotRmsAbvGrd | 0.8255 |
| TotalBsmtSF | 1stFlrSF | 0.8195 |
| YearBuilt | GarageYrBlt | 0.7772 |
| 2ndFlrSF | GrLivArea | 0.6875 |
| BedroomAbvGr | TotRmsAbvGrd | 0.6766 |
| BsmtFinSF1 | BsmtFullBath | 0.6492 |
| GrLivArea | FullBath | 0.6300 |

这些高相关在房价数据中是合理的，因为很多变量都在描述相近概念，例如总面积、房间数量、地下室面积和车库面积。

标准化后 OLS 系数绝对值较大的变量是：

| feature | OLS_standardized_coef |
| --- | --- |
| OverallQual | 0.1203 |
| YearBuilt | 0.0932 |
| GarageCars | 0.0617 |
| GrLivArea | 0.0507 |
| OverallCond | 0.0469 |
| BsmtFullBath | 0.0376 |
| MSSubClass | -0.0356 |
| 1stFlrSF | 0.0346 |
| Fireplaces | 0.0339 |
| 2ndFlrSF | 0.0296 |

OLS 在这里并没有完全失败，但解释时需要谨慎。原因是房地产特征之间经常相关，一个变量的系数可能取决于模型中是否同时包含了其他相近变量。

## 4. Lasso 和 PCR 哪个更合适？

对于这份只使用数值变量的房价数据，数据结构既有稀疏信号，也有潜因子结构。一些变量本身就很有业务含义，例如 `OverallQual` 和 `GrLivArea`，这支持 Lasso 式变量选择；同时，很多变量又共同反映房屋规模、质量和年代等潜在概念，这也支持 PCR 式信息压缩。

如果业务方希望得到一个短变量清单，我会先使用 Lasso；如果目标是构建在相关特征下更稳定的预测基线，我会同时比较 PCR 和 Ridge 类模型。
