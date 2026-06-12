# Week 13：Kaggle 二手车价格正则化回归报告

## 1. 数据来源与业务背景

- 数据集：Used Car Price Prediction Dataset
- 业务问题：根据车辆年份、里程、品牌、燃料类型、变速箱、事故记录等信息预测二手车价格。
- 使用文件：`/home/wsl2/Regression-Analysis-2026/students/02_zy/week13/data/kaggle_used_cars.csv`
- 原始样本量：`4009`
- 清洗后样本量：`4001`
- 建模特征数：`60`

这份数据经过 One-Hot 编码后特征数量较多，同时品牌、燃料类型、事故记录等变量之间可能存在潜在相关性，因此适合练习正则化和变量筛选。

## 2. 数据处理流程

主要处理包括：价格和里程转为数值；构造 `vehicle_age`；简化 `transmission`；将缺失类别填为 `Unknown`；低频类别归为 `Other`；最后进行 One-Hot 编码，并在 Pipeline 中使用自定义 `CustomStandardScaler` 进行标准化。

## 3. 模型性能对比

| model | best_alpha | best_l1_ratio | test_RMSE | test_MAE | nonzero_features |
| --- | --- | --- | --- | --- | --- |
| OLS |  |  | 37795.3035 | 20542.4963 | 60 |
| Ridge | 131.11339374215643 |  | 37764.6259 | 20388.0176 | 60 |
| Lasso | 225.39339047347912 |  | 37759.717 | 20436.7023 | 51 |
| ElasticNet | 1.0 | 0.8 | 37772.5205 | 20093.167 | 60 |

测试集 RMSE 最低的模型是 `Lasso`。正则化模型相比 OLS 在测试集 RMSE 上有一定改善。

## 4. GridSearchCV 最优参数

- Ridge 最优 alpha：`131.11339374215643`
- Lasso 最优 alpha：`225.39339047347912`
- Elastic Net 最优 alpha：`1.0`
- Elastic Net 最优 l1_ratio：`0.8`

这些参数来自 5 折交叉验证，目标是寻找验证误差较低的模型，而不是单纯追求变量越少越好。

## 5. Lasso 变量筛选结果

Lasso 最终保留的非零变量数量为：`51`。

部分 Lasso 保留变量如下：

`vehicle_age, milage, brand_911, brand_AMG, brand_Corvette, brand_E-Class, brand_F-150, brand_F-250, brand_Model, brand_Mustang, brand_Rover, brand_Silverado, brand_Wrangler, fuel_type_E85 Flex Fuel, fuel_type_Gasoline, fuel_type_Hybrid, fuel_type_Plug-In Hybrid, fuel_type_Unknown, fuel_type_not supported, fuel_type_–`

Lasso 删除的变量不一定完全没有业务意义。特别是在类别变量或高度相关变量中，Lasso 可能只保留其中一部分，因此解释时需要结合业务背景。

## 6. 最关键的影响因素

如果业务方要求给出最关键的影响因素，我会优先参考测试集表现最好的 `Lasso` 模型的系数绝对值排序。

| feature | Lasso | abs_coefficient |
| --- | --- | --- |
| milage | -14352.3452 | 14352.3452 |
| brand_911 | 11074.2835 | 11074.2835 |
| vehicle_age | -7583.8612 | 7583.8612 |
| int_col_Other | 4997.5525 | 4997.5525 |
| brand_AMG | 4223.1209 | 4223.1209 |
| transmission_Other | 3031.1636 | 3031.1636 |
| fuel_type_Gasoline | -2919.8779 | 2919.8779 |
| transmission_6-Speed Automatic | -2536.3876 | 2536.3876 |
| accident_None reported | 2438.5055 | 2438.5055 |
| fuel_type_Hybrid | -2284.7994 | 2284.7994 |

这些变量可以理解为模型中相对影响更明显的因素，但这只是相关关系，不应直接解释为严格因果关系。
