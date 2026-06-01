# Kaggle Real-World Regression Report

## 数据集概况
- Shape: (500, 6)
- 特征数量: 7
- 特征列表: age, bmi, children, smoker_yes, region_northwest, region_southeast, region_southwest

## 交叉验证指标对比

### CustomOLS (个人实现)

| Metric | Mean | Std |
|--------|------|-----|
| RMSE | 21.8670 | 2.1218 |
| MAE  | 16.0483 | 0.7721 |
| MAPE | 3.15% | 0.22% |

### Ridge Baseline (sklearn, alpha=1.0)

| Metric | Mean | Std |
|--------|------|-----|
| RMSE | 21.8652 | 2.1153 |
| MAE  | 16.0479 | 0.7712 |
| MAPE | 3.15% | 0.22% |

## CustomOLS 全量系数 (Intercept=517.8158)

| Feature | Coefficient |
|---------|-------------|
| age | 46.1370 |
| bmi | 6.3822 |
| children | 9.8235 |
| smoker_yes | 49.4121 |
| region_northwest | -7.5254 |
| region_southeast | 6.5757 |
| region_southwest | -7.1114 |

## VIF 共线性诊断

| Feature | VIF |
|---------|-----|
| age | 1.0331 |
| bmi | 1.0114 |
| children | 1.0162 |
| smoker_yes | 1.0216 |
| region_northwest | 1.6134 |
| region_southeast | 1.6034 |
| region_southwest | 1.6215 |

---

## 业务误差解读

### RMSE / MAE 的业务意义
- RMSE 表示预测值与真实值的均方根偏差，受异常值影响较大；
- MAE 给出平均绝对偏差，更直观反映典型预测误差的规模；
- MAPE 以百分比形式呈现相对误差，便于跨量纲比较。

### 模型对比
- CustomOLS 与 Ridge Baseline 的指标对比可以揭示：
  - 若 Ridge 显著优于 OLS，说明数据中存在多重共线性，正则化有效压制了方差；
  - 若两者接近，说明特征间共线性不严重，OLS 的无偏性得以保留。

## 上线风险评估
1. **数据泄漏风险**: 本流程严格采用逐折 fit-transform，杜绝了全量预处理的数据泄漏；
2. **共线性风险**: 若 VIF 诊断存在高共线性特征，模型系数解释性下降；
3. **泛化能力**: CV 指标的标准差反映了模型在不同数据切片上的稳定性。
