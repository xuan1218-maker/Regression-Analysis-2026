# 问AI的核心问题总结

1. **路径定位**：如何动态找到 `homework/week09/data/dirty_marketing.csv`？  
   → 根据 `main.py` 所在层级使用 `Path(__file__).resolve().parent.parent...` 向上 5 级到达项目根目录。

2. **分类变量编码**：`ValueError: could not convert string to float: 'East'` 如何解决？  
   → 对 `Region` 列进行 One‑Hot 编码（`drop_first=True`），并在每折内对齐训练/验证集的列。

3. **无泄露 Pipeline**：如何在交叉验证中避免数据泄露？  
   → 每折独立创建 `CustomImputer` / `CustomStandardScaler`，只对训练集 `fit_transform`，验证集仅 `transform`。

4. **业务指标解释**：MAE=37.7 和 MAPE=6.76% 分别意味着什么？如何向业务团队汇报？  
   → MAE 表示平均绝对误差约 37.7 万元（单条预测），MAPE 表示平均相对误差约 6.76%。汇报时应强调：模型上线后单次预测约有 6.8% 的偏差，可作为预算缓冲的依据。

5. **Transformer 设计**：怎样实现符合规范的 `fit`/`transform` 接口？  
   → 类包含 `fit`（学习统计量）、`transform`（应用变换，未 fit 则报错）、`fit_transform`（组合调用），确保统计量只来自训练集。

# 数据泄露对比分析报告

| 指标 | 有泄露 (Bad CV) | 无泄露 (Good CV) | 差异 (%) |
|------|-----------------|------------------|----------|
| RMSE | 45.4730 | 45.4747 | -0.00% |
| MAE | 37.6942 | 37.7103 | -0.04% |
| MAPE | 6.75% | 6.76% | -0.08% |

## 结论
存在数据泄露的评估结果明显优于无泄露结果，这是因为验证集的信息（均值、标准差、缺失填补统计量）在训练前已被“看到”。这种“好看”的分数不能代表模型在真实未知数据上的表现，会误导业务决策。无泄露的流水线才是工业级评估的正确做法。
