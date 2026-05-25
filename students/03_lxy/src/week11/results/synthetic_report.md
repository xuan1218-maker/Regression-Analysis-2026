# 第11周模拟数据报告

## 1. 场景设定与 DGP
这份模拟数据描述的是连锁门店的每周销售额场景。

- 目标变量：`weekly_sales`
- 连续变量：`tv_budget_k`、`digital_budget_k`、`discount_rate`、`inventory_index`
- 类别变量：`store_tier`
- 显式构造的共线性：`digital_budget_k = 0.82 * tv_budget_k + noise`
- 在 DGP 中应当正向影响销售额的变量：电视广告投入、数字广告投入、库存指数、较高等级门店
- 在 DGP 中应当负向影响销售额的变量：折扣率
- 主动加入的真实世界问题：缺失值、异常值、量纲差异、共线性

目标变量生成公式如下：

```text
weekly_sales = 45000
             + 185 * tv_budget_k
             + 140 * digital_budget_k
             - 92000 * discount_rate
             + 18 * inventory_index
             + store_tier_effect
             + random_noise
```

## 2. 数据概览
- 样本量：360
- `discount_rate` / `inventory_index` / `store_tier` 的缺失数量：10 / 8 / 12
- `tv_budget_k` 与 `digital_budget_k` 的相关系数：0.8509
- 在解释层面最冗余、最高相关的变量：`tv_budget_k` 和 `digital_budget_k`

```text
       tv_budget_k  digital_budget_k  discount_rate  inventory_index  weekly_sales
count       360.00            360.00         350.00           352.00        360.00
mean        183.52            149.26           0.16           811.79     103626.61
std          41.31             31.83           0.07           168.61      16369.55
min          48.84             23.66           0.03           361.18      53939.52
25%         156.84            128.52           0.10           705.86      93482.12
50%         182.26            148.91           0.17           806.90     104662.93
75%         206.38            169.43           0.22           895.33     114802.01
max         423.75            240.12           0.28          1913.19     151270.61
```

图形检查：

![模拟数据高相关变量图](synthetic_feature_relationships.png)
![模拟数据真实值与预测值](synthetic_actual_vs_pred.png)
![模拟数据残差图](synthetic_residuals.png)

## 3. 无泄露 5 折交叉验证
主模型：自定义 `GradientDescentOLS`

预处理顺序：
- 数值变量：均值填补 -> winsorization 截尾 -> 标准化
- 类别变量：众数填补 -> one-hot 编码
- 上述所有会学习参数的步骤都在每一折训练集上 fit，再对验证集 transform，避免数据泄露

| fold | rmse | mae | mape |
| --- | --- | --- | --- |
| 1 | 8134.4512 | 6450.6862 | 6.6458 |
| 2 | 6957.8889 | 5570.6153 | 5.4865 |
| 3 | 7803.4650 | 6249.5791 | 6.4071 |
| 4 | 7119.5427 | 5706.5597 | 5.9635 |
| 5 | 8874.5020 | 7099.0660 | 7.0107 |

自定义模型平均指标：
- RMSE: 7777.9700
- MAE: 6215.3013
- MAPE: 6.3027%

对照组：`sklearn.linear_model.LinearRegression`

| fold | rmse | mae | mape |
| --- | --- | --- | --- |
| 1 | 8134.4486 | 6450.6844 | 6.6458 |
| 2 | 6957.8869 | 5570.6159 | 5.4865 |
| 3 | 7803.4454 | 6249.5693 | 6.4071 |
| 4 | 7119.5422 | 5706.5680 | 5.9635 |
| 5 | 8874.5170 | 7099.0835 | 7.0107 |

baseline 平均指标：
- RMSE: 7777.9680
- MAE: 6215.3042
- MAPE: 6.3027%

## 4. 推断结果检查
与 DGP 设定的方向对照如下：
- `tv_budget_k`：理论方向 `positive`，模型识别方向 `positive`
- `digital_budget_k`：理论方向 `positive`，模型识别方向 `positive`
- `discount_rate`：理论方向 `negative`，模型识别方向 `negative`
- `inventory_index`：理论方向 `positive`，模型识别方向 `positive`
- `store_tier_premium`：理论方向 `negative`，模型识别方向 `negative`
- `store_tier_standard`：理论方向 `negative`，模型识别方向 `negative`

全样本解析解 OLS 中，绝对值较大的系数如下：

```text
            feature   coefficient direction
          intercept 113105.733832  positive
store_tier_standard -14309.863724  negative
 store_tier_premium  -6482.333513  negative
   digital_budget_k   6427.987190  positive
      discount_rate  -6420.459009  negative
        tv_budget_k   4656.262981  positive
    inventory_index   2842.273092  positive
```

解释：
- 模型成功识别出 `discount_rate` 为负向影响，广告预算变量为正向影响。
- `tv_budget_k` 和 `digital_budget_k` 的方向与设定一致，但由于二者被故意设计为高度相关，所以系数大小稳定性较弱。
- `store_tier` 的系数是相对被省略的参考组 `flagship` 来解释的，因此 `premium` 和 `standard` 为负值是符合 DGP 设定的。
- 这说明模型识别出来的变量方向总体与 DGP 一致，没有出现关键变量方向反转的问题。
- 在这组数据里最难稳定识别的变量就是 `tv_budget_k` 和 `digital_budget_k`，因为它们被刻意构造成高相关变量，容易共享解释力。
- 如果后续某次实验出现二者系数波动甚至大小变化，主要原因更可能是共线性和噪声，而不是预处理逻辑错误。

## 5. 诊断结果
VIF 较高的变量如下：

| feature | vif |
| --- | --- |
| digital_budget_k | 5.1010 |
| tv_budget_k | 5.0921 |
| store_tier_premium | 1.9227 |
| store_tier_standard | 1.9122 |
| discount_rate | 1.0190 |
| inventory_index | 1.0068 |

结论：
- 最明显的多重共线性集中在两个广告预算变量上，VIF 大约在 5 左右，属于比较明显的共线性风险。
- 即使预测误差还可以接受，高相关变量的系数解释仍然需要谨慎。
- 这也说明在模拟数据中，因为我们知道真实 DGP，所以能更清楚地区分“系数不稳定”和“变量方向真的错了”。


## 6. 工程实现与答辩准备
### 6.1 main.py 主流程分阶段说明
1. `ensure_directories()`：输入为空，输出是已经创建好的 `data/` 与 `results/` 目录。
2. `generate_synthetic_data()`：输入是随机种子，输出是模拟数据 DataFrame，并写出 `synthetic_regression.csv`。
3. `write_synthetic_report()`：输入是模拟数据 DataFrame，输出是交叉验证结果、VIF、系数方向分析，并写出 `synthetic_report.md`。
4. `main()`：串联以上阶段，确保通过单一入口执行完整流程。

### 6.2 缺失值、异常值、标准化、编码分别在哪一层完成
- 缺失值处理：`preprocess_split()` 与 `preprocess_full()` 中调用 `CustomImputer`
- 异常值处理：`preprocess_split()` 与 `preprocess_full()` 中调用 `Winsorizer`
- 标准化：`preprocess_split()` 与 `preprocess_full()` 中调用 `CustomStandardScaler`
- 编码：`preprocess_split()` 与 `preprocess_full()` 中调用 `CustomOneHotEncoder`

### 6.3 为什么 5 折交叉验证没有数据泄露
- 因为 `preprocess_split()` 里所有会学习参数的步骤都先在训练集上 `fit`，再应用到验证集上 `transform`。
- 如果有一行代码写错，最危险的位置就是把 `fit_transform()` 错用到验证集，或者在交叉验证之前先对全量数据做 `preprocess_full()`。

### 6.4 这次真正被调用到的 utils 组件及顺序
`CustomImputer` -> `Winsorizer` -> `CustomStandardScaler` -> `CustomOneHotEncoder` -> `GradientDescentOLS` -> `calculate_rmse` / `calculate_mae` / `calculate_mape` -> `summarize_vif`

### 6.5 如果老师现场让我改参数或路径
- 替换数据路径：改 `SYNTHETIC_PATH` 或 Kaggle 数据路径常量
- 调整样本量：改 `generate_synthetic_data()` 里的 `n_samples`
- 调整异常值比例或缺失值比例：改 `missing_numeric_idx`、`missing_categorical_idx`、`outlier_idx` 的大小
- 调整模型训练参数：改 `GradientDescentOLS` 的 `learning_rate`、`max_iter`、`tol`

