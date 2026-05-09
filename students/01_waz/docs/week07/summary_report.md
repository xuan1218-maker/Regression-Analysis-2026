# 第七周作业：优化引擎与泛化能力评估

## 1. 交付物自查

- [x] 规范的 Python 工程代码
- [x] 明确可运行入口：`uv run python main.py`
- [x] 自动生成 `results/summary_report.md`
- [x] 自动生成 `results/learning_curve_full_vs_mini.png`
- [x] 控制台输出包含交叉验证、调参与 Test 集结果表

## 2. 5 折交叉验证结果

- 平均 CV R²: 0.9072
- 平均 CV RMSE: 72.4134

各折详细结果如下：

- Fold 1: R²=0.8981, RMSE=73.8816
- Fold 2: R²=0.8914, RMSE=77.8810
- Fold 3: R²=0.9076, RMSE=72.1901
- Fold 4: R²=0.9310, RMSE=67.3914
- Fold 5: R²=0.9079, RMSE=70.7232

## 3. GradientDescentOLS 的实现

本次 `GradientDescentOLS` 的核心实现思路如下：

1. 参数初始化：将回归系数初始化为零向量，并准备 `loss_history_` 与 `full_loss_history_` 两个列表。
2. 梯度计算：对当前参与更新的数据计算预测值、残差以及 MSE 的梯度。
3. 参数更新：使用 `coef_ -= learning_rate * gradient` 执行梯度下降更新。
4. 模式支持：
   - `full_batch` 每轮使用全部训练样本更新一次。
   - `mini_batch` 每轮随机抽取一部分样本更新一次。
5. 损失记录：
   - `loss_history_` 记录当前这轮实际参与更新的数据对应的训练 loss。
   - `full_loss_history_` 额外记录全训练集 MSE，用于更稳定的早停判断。
6. 早停策略：如果相邻两轮的全训练集 MSE 变化小于 `tol=1e-5`，则提前停止。

### 截距项如何处理

- 先对特征做标准化，再手动在最前面拼接一列全 1 作为截距项。
- 这样截距项不会被标准化，同时模型仍能显式学习偏置项。

## 4. 学习曲线分析

- 已生成 `learning_curve_full_vs_mini.png`。
- 图中比较了 `full_batch` 和 `mini_batch` 两种模式下 loss 随 epoch 下降的轨迹。
- 纵轴使用对数刻度，以避免前期较大的 loss 把后期差异压缩得看不清。
- 为了更清楚展示 mini-batch 的随机波动，绘图时将 mini-batch 比例设为 2%。

### full_batch 与 mini_batch 的差异

- `full_batch` 曲线整体更平滑，因为每轮都使用全部样本，梯度方向更加稳定。
- `mini_batch` 曲线整体同样向下，但波动更明显，因为每轮只观察一个随机小批次，梯度带有随机性。
- 两者最终仍会收敛到相近区域，因为线性回归在该设置下是一个凸优化问题。

## 5. 学习率调参结果

| 学习率 | Validation R² | Validation RMSE |
|--------|---------------|-----------------|
| 0.1 | 0.9009 | 72.3167 |
| 0.01 | 0.9003 | 72.5033 |
| 0.001 | 0.5970 | 145.8038 |
| 0.0001 | -9.0004 | 726.2848 |
| 1e-05 | -13.2044 | 865.5850 |

- 最优学习率：**0.1**
- 选择理由：该学习率在 Validation 集上取得了最高的 R²。
- `0.001` 的表现明显较差，说明学习率过小，有限迭代内收敛不足。
- `0.0001` 和 `1e-5` 的结果很差，说明这组超参数下优化几乎没有得到有效结果，可以视为失败案例并在报告中说明。

## 6. 标准化与数据泄露

本次实验严格避免了数据泄露，做法如下：

1. 先将数据划分为 Train / Validation / Test。
2. 只在 Train 集上 `fit` 标准化器，得到均值和标准差。
3. 使用同一个变换分别去 `transform` Validation 和 Test。

为什么不能在全部数据上先做标准化？

- 如果先在全部数据上标准化，那么 Validation 和 Test 的统计信息就已经泄露给模型了。
- 这样会让评估结果偏乐观，不能真实反映模型对未见数据的泛化能力。
- 因此，标准化只能依赖 Train 集统计量，之后再把同一变换应用到 Validation 和 Test。

## 7. Test 集性能对比

| 模型 | Test R² | Test RMSE |
|------|---------|-----------|
| GradientDescentOLS | 0.8909 | 78.5333 |
| AnalyticalOLS | 0.8902 | 78.7816 |

### 结果解释

- 两种方法在 Test 集上的表现非常接近。
- 这是合理的，因为它们本质上都在求解同一个线性回归最优化问题。
- `AnalyticalOLS` 通过闭式解直接求参数；`GradientDescentOLS` 则通过迭代逐步逼近同一个最优解。
- 因此，只要学习率与训练过程设置合理，二者结果接近是符合预期的。

## 8. 展示时可重点说明的内容

- `GradientDescentOLS` 如何初始化、计算梯度、更新参数以及记录 loss。
- 为什么 `full_batch` 更平滑，而 `mini_batch` 更容易波动。
- 最终选择的最佳学习率是 **0.1**，并说明选择依据。
- 为什么标准化不能在全部数据上先做，这和数据泄露之间的关系是什么。
- 为什么 Test 集上 `GradientDescentOLS` 与 `AnalyticalOLS` 的结果会非常接近。

## 9. 最低完成标准对照

- [x] AnalyticalOLS 可以正常 fit 和 predict
- [x] GradientDescentOLS 支持 full_batch 与 mini_batch
- [x] 已记录 loss_history_
- [x] 已完成 5-Fold CV
- [x] 已完成 Train / Validation / Test 划分
- [x] 已进行至少 5 个学习率调参
- [x] 已正确执行标准化且避免数据泄露
- [x] 已输出学习曲线图
- [x] 已在 Test 集上比较 GradientDescentOLS 与 AnalyticalOLS
