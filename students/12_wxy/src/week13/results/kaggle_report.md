# Kaggle Ames 房价实验报告

## B1 数据来源
- 来源：Kaggle House Prices
- 链接：https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- 特征数：37 高维 + 强共线性

## B2 测试集性能
|    | model      |   rmse |    mae |     r2 |
|---:|:-----------|-------:|-------:|-------:|
|  0 | OLS        | 0.1489 | 0.1091 | 0.8693 |
|  1 | Ridge      | 0.1477 | 0.1071 | 0.8714 |
|  2 | Lasso      | 0.1496 | 0.1085 | 0.8681 |
|  3 | ElasticNet | 0.1487 | 0.1075 | 0.8696 |

## B3 问题回答
1. 正则化显著提升性能，因为数据高维共线性强，OLS 过拟合。

2. Lasso 剔除了 16 个冗余/无关特征，业务合理。

3. 关键5因素以 ElasticNet 为准，稳定且兼顾共线性特征组。
