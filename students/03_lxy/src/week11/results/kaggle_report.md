# 第11周 Kaggle 真实数据报告

## 1. 数据集选择
- Kaggle 数据集名称：Medical Cost Personal Datasets
- Kaggle 链接：https://www.kaggle.com/datasets/mirichoi0218/insurance
- 下载日期：2026-05-19
- 本地原始文件：`kaggle_insurance_raw.csv`
- 本地建模工作副本：`kaggle_insurance_working.csv`
- 目标变量：`charges`
- 每一行样本表示一位被保险人的人口属性和保费相关特征。

选择这份数据的原因：
- 目标变量是连续变量，适合做回归。
- 同时包含数值变量和类别变量。
- 原始数据虽然比较整洁，但仍然有偏态和明显的组间差异。
- 每一条样本都对应一位真实保险客户，因此业务含义清楚，不是单纯为了演示算法而拼出来的教学型数据。
- 为了符合 Week 11 的训练要求，我保留了原始 Kaggle 文件，并额外构造了一份工作副本，在不破坏原始结构的前提下加入少量缺失值和 BMI 异常值，模拟更真实的数据质量问题。
- 相比一份几乎不用清洗的“演示型数据”，这份数据至少同时包含类别变量、偏态目标、组间差异和可模拟的缺失值/异常值，更能体现完整的数据处理流程。

## 2. 清洗与预处理
- 数值变量缺失填补：自定义 `CustomImputer(strategy="mean")`
- 类别变量缺失填补：自定义 `CustomImputer(strategy="most_frequent")`
- 异常值处理：自定义 `Winsorizer`
- 标准化：自定义 `CustomStandardScaler`
- 类别变量编码：自定义 `CustomOneHotEncoder`
- 主模型：自定义 `GradientDescentOLS`
- baseline：`sklearn.linear_model.LinearRegression`

工作副本中的缺失情况：

```text
age          0
sex          0
bmi         28
children    18
smoker      16
region       0
charges      0
```

描述性统计：

```text
           age      bmi  children   charges
count  1338.00  1310.00   1320.00   1338.00
mean     39.21    30.75      1.09  13270.42
std      14.05     6.26      1.20  12110.01
min      18.00    15.96      0.00   1121.87
25%      27.00    26.32      0.00   4740.29
50%      39.00    30.40      1.00   9382.03
75%      51.00    34.79      2.00  16639.91
max      64.00    57.50      5.00  63770.43
```

图形检查：

![Kaggle 目标变量分布](kaggle_target_distribution.png)
![Kaggle 真实值与预测值](kaggle_actual_vs_pred.png)
![Kaggle 残差图](kaggle_residuals.png)

## 3. 无泄露 5 折交叉验证
自定义流程结果：

| fold | rmse | mae | mape |
| --- | --- | --- | --- |
| 1 | 6354.6890 | 4316.8106 | 43.8261 |
| 2 | 6036.3133 | 4237.7082 | 41.3979 |
| 3 | 5739.8612 | 3934.2844 | 43.7942 |
| 4 | 7227.3145 | 4808.2814 | 39.6059 |
| 5 | 5981.6326 | 4147.5748 | 46.6716 |

自定义模型平均指标：
- RMSE: 6267.9621
- MAE: 4288.9319
- MAPE: 43.0592%

baseline 结果：

| fold | rmse | mae | mape |
| --- | --- | --- | --- |
| 1 | 6354.6884 | 4316.8111 | 43.8261 |
| 2 | 6036.3136 | 4237.7087 | 41.3980 |
| 3 | 5739.8606 | 3934.2841 | 43.7942 |
| 4 | 7227.3158 | 4808.2826 | 39.6060 |
| 5 | 5981.6324 | 4147.5749 | 46.6716 |

baseline 平均指标：
- RMSE: 6267.9622
- MAE: 4288.9323
- MAPE: 43.0592%

## 4. 推断结果与业务解释
全样本解析解 OLS 中影响较大的系数如下：

```text
         feature  coefficient direction
      smoker_yes 23707.647895  positive
       intercept  9086.139177  positive
             age  3563.270895  positive
             bmi  2017.959113  positive
region_southwest  -938.321970  negative
region_southeast  -928.427958  negative
        children   620.442032  positive
region_northwest  -314.234548  negative
```

解释：
- `smoker_yes` 是最稳定、最强的正向信号，对医疗费用影响最明显。
- `age` 和 `bmi` 也呈现出与常识一致的正向影响。
- `children` 的稳定性明显弱一些，这也符合它对费用影响通常没那么直接的直觉。
- 与吸烟状态和 BMI 相比，地区变量的影响相对较小。
- 因此，我最信任的是 `smoker_yes`、`age`、`bmi` 这几类结果；而 `children` 和部分地区变量虽然直觉上可能有影响，但模型中的稳定性明显更弱。

## 5. 诊断与风险
VIF 较高的变量如下：

| feature | vif |
| --- | --- |
| region_southeast | 1.6413 |
| region_southwest | 1.5284 |
| region_northwest | 1.5186 |
| bmi | 1.0988 |
| age | 1.0157 |
| smoker_yes | 1.0123 |
| sex_male | 1.0081 |
| children | 1.0044 |

如果把这个模型用于真实业务，主要风险包括：
- `charges` 明显右偏，少数高费用样本会显著拉高绝对误差。
- 吸烟状态信号很强，但也可能对数据质量和样本漂移比较敏感。
- 这份工作副本中故意加入了缺失值和异常值，说明稳健预处理是必要步骤。
- 从业务角度看，当前 MAE 表示模型对个体医疗费用仍可能有几千金额单位的误差，因此更适合作为定价参考，而不是精确到个人报销金额的预测工具。
- 从 VIF 看，这份数据没有特别明显的严重共线性，说明真实数据上的主要风险更多来自偏态、异常样本和业务分布漂移，而不是共线性主导。
- 如果真的上线，我最担心的是未来样本结构变化，例如吸烟比例、地区结构或医疗成本分布变化，导致模型外推能力下降。


## 6. 工程实现与答辩准备
### 6.1 Kaggle 流程分阶段说明
1. `load_kaggle_data()`：读取工作副本并检查必要字段是否存在。
2. `evaluate_with_cv()`：完成无泄露 5 折交叉验证，输出自定义模型与 baseline 的每折指标。
3. `preprocess_full()`：在全样本上重新做一次预处理，用于生成解释性系数和 VIF。
4. `write_kaggle_report()`：写出数据理解、指标、推断、风险与答辩说明。

### 6.2 真实数据上的无泄露保证
- 真实数据的均值填补、众数填补、截尾、标准化和编码结构，全部在每一折训练集内部学习。
- 验证集从不参与这些参数的计算，所以没有把未来信息泄露回训练阶段。

### 6.3 utils 真实调用顺序
`CustomImputer` -> `Winsorizer` -> `CustomStandardScaler` -> `CustomOneHotEncoder` -> `GradientDescentOLS` -> `calculate_rmse` / `calculate_mae` / `calculate_mape` -> `summarize_vif`

### 6.4 如果老师让我现场改代码
- 想切换 Kaggle 文件：改 `KAGGLE_RAW_PATH` 或 `KAGGLE_WORKING_PATH`
- 想改异常值截尾规则：改 `Winsorizer(lower_quantile=..., upper_quantile=...)`
- 想换 baseline：把 `LinearRegression` 换成 `Ridge` 等其他 sklearn 回归器

