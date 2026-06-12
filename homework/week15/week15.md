# Week 15 Assignment: Logistic Regression and Binary Classification
**（第十五周实操：逻辑回归、分类指标与阈值权衡）**

## 背景与目标 (Background)
前两周，我们一直在讨论线性模型在高维、共线性和稳定性问题下会遇到什么困难，以及 `Ridge`、`Lasso`、`PCR` 分别在解决什么问题。

本周，问题发生了更本质的变化：  
我们不再预测一个连续数值，而是预测“会不会发生”，也就是一个 `0/1` 结果。

因此，这周的任务不是把“线性回归稍微改一改”，而是认真回答下面几个问题：

1. 为什么不能直接把 `OLS` 拿来做二分类？
2. 为什么逻辑回归的输出更适合解释成概率？
3. 为什么分类任务的训练目标会自然走到 `Bernoulli + MLE + log loss`？
4. 为什么分类模型不能只看 `accuracy`？
5. 在高维分类里，`L1` 和 `L2` 正则化分别在解决什么问题？

本周的关键词是：
- **binary classification**
- **logistic regression**
- **sigmoid**
- **Bernoulli likelihood**
- **log loss**
- **threshold / recall / precision / ROC-AUC**

---

## 目录规范 (Directory Architecture)
请继续维护你的个人算法库，并为 Week 15 新增一个完整工作区：

```text
students/<your_name>/
├── pyproject.toml
└── src/
    ├── utils/
    │   ├── models.py
    │   ├── metrics.py
    │   ├── transformers.py
    │   └── diagnostics.py
    └── week15/
        ├── data/
        │   ├── synthetic_binary.csv
        │   └── real_binary_*.csv          <-- 真实数据（可选）
        ├── results/
        │   ├── synthetic_report.md
        │   ├── threshold_report.md
        │   ├── regularization_report.md
        │   └── real_data_report.md        <-- 可选
        └── main.py
```

---

## 工程总要求 (Engineering Rules)

### Rule 1: 继续复用你自己的 `utils/`
允许使用 `sklearn` 的逻辑回归和评估工具，但数据清洗、实验组织、结果汇总、部分指标计算或图形解释，仍然应尽量通过你自己的 `src/utils/` 来管理。

### Rule 2: `sklearn` 的边界
允许使用的 `sklearn` 功能：
- `LogisticRegression`
- `LinearRegression`（仅作为错误示范或基准对照）
- `train_test_split`
- `Pipeline`
- `StandardScaler`
- `GridSearchCV`
- `KFold`
- `confusion_matrix`
- `roc_auc_score`
- `log_loss`

你需要自己清楚组织或自己实现的部分包括：
- 图表生成与解释；
- threshold 扫描流程；
- 至少一个你自己整理的分类指标汇总表；
- 对实验结果的业务化解读。

### Rule 3: 单一入口
必须仅通过：

```text
uv run src/week15/main.py
```

完成本周全部流程。

### Rule 4: 需要画图时，请把“图里展示什么”写清楚
凡是题目要求“画图”的地方，你都必须在报告里明确说明：
- 横轴是什么；
- 纵轴是什么；
- 每条线 / 每组柱 / 每个点 / 每个颜色分别代表什么；
- 这张图想支持的结论是什么。

我们不规定你必须用哪一种图，但会要求你把**比较对象**和**展示变量**交代清楚。

---

## Task A：自己生成二分类数据，比较“线性回归式思路”和逻辑回归
本任务对应课堂前两幕：为什么不能直接拿 `OLS` 做分类，以及为什么 `sigmoid` 很自然。

### A1. 生成一份带有明确概率结构的模拟二分类数据
请自己生成一份二分类数据。

要求：
1. 样本量不少于 `400`；
2. 至少 `4` 个特征；
3. 其中至少 `2` 个特征对类别概率有明显影响；
4. 目标变量 `y` 必须是通过“先生成概率，再按 Bernoulli 抽样”的方式得到，而不是直接手写一个硬阈值标签。

提示：你可以先构造

$$
\eta = X\beta
$$

再通过

$$
p = \frac{1}{1 + e^{-\eta}}
$$

生成概率，最后从 `Bernoulli(p)` 采样得到 `y`。

### A2. 保存数据并说明 DGP
把数据保存到：

```text
src/week15/data/synthetic_binary.csv
```

在 `synthetic_report.md` 中说明：
- 你的样本量和特征数；
- 哪些变量会提高正类概率，哪些会降低正类概率；
- 你的真实数据生成机制（DGP）是什么。

### A3. 用 `LinearRegression` 和 `LogisticRegression` 做并排对比
请至少训练以下两个模型：
- `LinearRegression`
- `LogisticRegression`

要求：
1. 两个模型都在同一份训练集上拟合；
2. 都在同一份测试集上评估；
3. 对 `LinearRegression`，你需要额外说明：如果把它的输出硬解释成概率，会出现什么问题。

### A4. 画出核心对比图
请至少完成 1 张核心图，展示 `LinearRegression` 与 `LogisticRegression` 输出行为的差别。

你可以选择以下两种方式之一：

1. 如果你使用 1 个主要特征做展示：
   - 横轴：该特征；
   - 纵轴：模型输出；
   - 图中至少同时展示：
     - `0/1` 标签散点；
     - `LinearRegression` 的预测曲线；
     - `LogisticRegression` 的预测概率曲线。

2. 如果你使用 2 个主要特征做展示：
   - 画二维散点图；
   - 用颜色表示真实标签；
   - 额外展示两个模型的预测边界或概率热力图。

无论你选哪一种方式，报告里都必须明确写出：
- 图中横轴和纵轴是什么；
- 哪条线或哪种颜色代表哪个模型；
- 图中最想说明的现象是什么。

### A5. 回答本任务的核心问题
在 `synthetic_report.md` 中回答：
1. `LinearRegression` 在这个任务里最不自然的地方是什么？
2. 为什么逻辑回归的输出更容易解释成概率？
3. 这里的关键区别，到底是“能不能分类”，还是“输出是否有概率意义”？

---

## Task B：从 Bernoulli 概率走到 log loss
本任务对应课堂第三幕：为什么逻辑回归自然会走到 `Bernoulli + MLE`。

### B1. 写出本周必须出现的三个公式
请在 `threshold_report.md` 或 `synthetic_report.md` 中，写出并解释以下对象：

1. `Bernoulli` 分布：
   $$
   Y \sim Bernoulli(p)
   $$
2. 单样本 likelihood：
   $$
   L(p;y)=p^y(1-p)^{1-y}
   $$
3. 单样本负对数似然（或 log loss）的形式。

要求：  
每个公式后面都要配 2 到 4 句自己的解释，不能只贴公式。

### B2. 画“损失如何随预测概率变化”的图
请至少画 1 张图，比较当真实标签固定时，不同预测概率会带来怎样的损失变化。

建议方式：
- 画两组曲线：
  - 当 `y = 1` 时；
  - 当 `y = 0` 时。
- 每组里至少包含两条损失曲线：
  - squared error
  - log loss

图形要求：
- 横轴：预测为正类的概率 `p`
- 纵轴：loss value
- 每条线必须有图例

报告中必须明确解释：
- 哪两种 loss 在比较；
- 当模型“错得很自信”时，哪一种 loss 惩罚更重；
- 这张图想支持的结论是什么。

### B3. 把图像现象和统计建模对应起来
在报告里回答：
1. 为什么二分类里“错得很自信”需要被重罚？
2. 为什么说 `log loss` 不是凭空指定的，而是来自 `Bernoulli likelihood`？
3. 如果我们已经把输出解释成概率，那么为什么 `log loss` 比 `MSE` 更自然？

---

## Task C：分类指标、混淆矩阵与阈值权衡
本任务对应课堂第五幕：为什么分类不能只看 `accuracy`。

### C1. 先给出混淆矩阵和基础指标
请基于你的逻辑回归模型，在测试集上计算并展示：
- `TP`
- `TN`
- `FP`
- `FN`
- `accuracy`
- `precision`
- `recall`
- `F1`

要求：
1. 你可以使用 `sklearn` 计算，也可以自己封装；
2. 最终必须在报告中以表格形式展示；
3. 表格列名必须清楚，读者不需要猜每一列是什么意思。

### C2. 做一次 threshold 扫描
请至少扫描一组阈值，例如：

```text
0.1, 0.2, 0.3, ..., 0.9
```

对每一个阈值，重新计算：
- `accuracy`
- `precision`
- `recall`
- `F1`

### C3. 画 threshold 曲线
请至少画 1 张图展示 threshold 改变时，指标如何变化。

图形要求：
- 横轴：classification threshold
- 纵轴：metric value
- 图中至少包含两条曲线，推荐包含四条：
  - `accuracy`
  - `precision`
  - `recall`
  - `F1`

报告中必须明确说明：
- 每条曲线分别代表什么；
- 当阈值升高时，哪些指标通常会上升或下降；
- 你在这张图里观察到了怎样的 trade-off。

### C4. 结合一个业务场景解释指标选择
请在以下三个场景中任选一个：
- 信用违约预测
- 用户流失预警
- 疾病初筛

回答：
1. 这个场景里，你最在意的是 `precision`、`recall`、`accuracy` 还是 `F1`？
2. 为什么？
3. 如果让你给业务方推荐一个阈值，你会如何说明理由？

---

## Task D：正则化逻辑回归（L1 vs L2）
本任务对应课堂第六幕：为什么逻辑回归在工业界长期不过时，以及正则化在高维分类里的意义。

### D1. 构造一份“特征较多或带共线性”的二分类数据
你可以继续使用自己的模拟数据，并把它扩展成更复杂版本；也可以单独生成一份新数据。

要求：
1. 特征数不少于 `20`；
2. 至少包含一组明显相关的特征；
3. 可以额外加入一些噪声特征。

### D2. 比较 `L1` 和 `L2` 逻辑回归
请比较：
- `penalty='l1'`
- `penalty='l2'`

要求：
1. 使用合理的标准化流程；
2. 用交叉验证或验证集选择超参数；
3. 至少比较测试集上的：
   - `accuracy`
   - `recall`
   - `ROC-AUC`
   - `log loss`

### D3. 画出“性能 + 模型复杂度”的对比结果
你至少需要完成以下二者之一：

1. 结果表
   - 行：不同模型（`L1`, `L2`）
   - 列：`accuracy`, `recall`, `ROC-AUC`, `log loss`, 非零系数个数

2. 对比图
   - 一张图展示性能指标；
   - 另一张图展示非零系数个数或系数大小分布。

如果画图，必须明确说明：
- 横轴是什么；
- 纵轴是什么；
- 每根柱子、每个颜色或每个模型分别代表什么；
- 这张图要支持的结论是什么。

### D4. 回答核心比较问题
在 `regularization_report.md` 中回答：
1. `L1` 和 `L2` 的预测表现差很多吗？
2. 哪一个模型更稀疏？
3. 哪一个模型更适合“给出一个更短的变量名单”？
4. 如果业务方更在意模型稳定性而不是变量筛选，你更偏向哪一个？

---

## Task E：真实数据挑战（可选 / Optional）
*本部分为选做题，适合希望把逻辑回归真正落到真实分类任务上的同学。完成可获得额外加分。*

### E1. 自选真实二分类数据
你可以从 Kaggle、UCI 或其他公开平台选择一份二分类数据，例如：
- 信用评分
- 电信流失
- 医疗筛查
- 广告点击预测

### E2. 跑一遍完整逻辑回归流程
至少完成：
1. 数据清洗与必要预处理；
2. 训练普通逻辑回归；
3. 做一次 threshold 分析；
4. 如有余力，再比较 `L1` / `L2`。

### E3. 回答真实业务问题
在 `real_data_report.md` 中回答：
1. 这个数据里，单看 `accuracy` 会不会误导判断？
2. 你最后更信任哪个指标？为什么？
3. 如果你要向业务方解释模型输出，你会强调“类别”还是“概率”？为什么？

---

## Task F：总结
在 `results/summary.md` 中，结合本周实验回答下面几个问题：

1. 为什么逻辑回归不是“线性回归后面接一个 sigmoid”这么简单？
2. `sigmoid`、`Bernoulli likelihood`、`log loss` 三者之间是什么关系？
3. 为什么分类模型不能只看 `accuracy`？
4. `L1` 和 `L2` 逻辑回归分别更适合什么目标？
5. 如果业务方要的是“一个能输出稳定概率、还能解释变量方向”的模型，逻辑回归为什么仍然是一个很强的 baseline？

---

## 交付物要求 (Deliverables)
你最终至少应提交并产出以下内容：

1. `src/week15/main.py`
2. `src/week15/data/synthetic_binary.csv`
3. `src/week15/results/synthetic_report.md`
4. `src/week15/results/threshold_report.md`
5. `src/week15/results/regularization_report.md`
6. `src/week15/results/summary.md`
7. `src/week15/results/real_data_report.md`（可选）
