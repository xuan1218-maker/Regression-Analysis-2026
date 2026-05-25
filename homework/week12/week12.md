# 🎭 Week 12 Assignment: The Bias-Variance Visual Lab
**（第十二周实操：用 Python 脚本把偏差-方差权衡“演出来”）**

## 🎯 背景与目标 (Background)
从本周开始，我们将采取一个新的视角：
- 现象先于定义；
- 图像先于公式；
- 判断先于术语。

但是，**课程提交规范仍然以 `Python 源代码 + Markdown 报告 + 清晰运行入口` 为准**。

也就是说：
- 你可以在本地用 notebook 进行探索；
- 但最终提交时，**请整理为可复现的 `.py` 文件与 `.md` 报告**；
- 不要只提交 notebook 实验过程。

因此，本周作业的重点不是“抄写 bias-variance 的定义”，而是：

> **用 Python 脚本生成图和表，让现象发生；再用 Markdown 报告解释你看到了什么。**

本周你将围绕以下三个核心问题完成一份可复现的可视化实验作业：

1. 模型复杂度增加时，为什么训练误差下降而测试误差不一定下降？
2. 什么叫 `high variance`，它在图上长什么样？
3. 为什么异常值会让 `RMSE` 和 `MAE` 表现出不同的性格？

---

## 📂 目录规范 (Directory Architecture)
请在你自己的工作区中新增 Week 12 目录。推荐结构如下：

```text
students/<your_name>/
├── pyproject.toml
└── src/
    ├── utils/
    │   ├── metrics.py          <-- 继续维护你的 RMSE / MAE / MAPE 等函数
    │   ├── models.py           <-- 可继续复用已有模型封装（可选）
    │   └── transformers.py     <-- 本周不强制新增，但可以复用
    └── week12/
        ├── results/
        │   ├── figures/
        │   │   ├── candidate_models.png
        │   │   ├── error_curves.png
        │   │   ├── variance_demo.png
        │   │   └── loss_outlier_comparison.png
        │   └── report.md
        └── main.py             <-- 本周唯一执行入口
```

### 最低提交要求
必须提交：
1. `src/week12/main.py`
2. `src/week12/results/report.md`
3. 至少若干张关键图到 `src/week12/figures/`

### 说明
- 本周**不要求提交 notebook**；
- 如果你本地先用 notebook 探索，请最终整理为 `.py + .md`；
- 评价时以 `main.py` 能否跑通、图表是否清晰、报告是否能解释现象为准。

---

## 🧰 工程规则 (Engineering Rules)

### Rule 1: 本周主交付物是 Python 脚本，不是 Notebook 文件
本周的主要成果不是一篇长篇 PDF，也不是只含实验过程的 `.ipynb`。

你需要提交一份**可运行的 Python 脚本工作流**，它应该完成：
- 数据生成；
- 模型拟合；
- 指标计算；
- 图形输出；
- Markdown 总结输出。

### Rule 2: 可以使用 `sklearn`，但最好不要直接调用现成的 bias-variance demo
本周允许使用：
- `sklearn.preprocessing.PolynomialFeatures`
- `sklearn.pipeline.Pipeline`
- `sklearn.linear_model.LinearRegression`
- `train_test_split`

因为本周重点是：
- 看见现象；
- 组织可复现实验；
- 建立 bias-variance 的直觉；
- 比较 `RMSE` 和 `MAE` 的行为。

也就是说，本周不要求你重新手写一整套多项式拟合优化器。

### Rule 3: 评估指标请优先复用你自己的 `utils/metrics.py`
你在 Week 10 / Week 11 已经维护了评估函数，本周**优先复用自己的实现**，尤其是：
- `calculate_rmse(...)`
- `calculate_mae(...)`

如果你额外用 `sklearn.metrics` 做交叉核对，也可以，但请明确说明：
- 哪些结果来自你的 `utils`；
- 哪些是外部库用于校验。

### Rule 4: 必须提供清晰入口 `main.py`
你需要确保以下命令可以作为唯一执行入口：

```bash
uv run src/week12/main.py
```

运行后至少应完成：
1. 生成图像到 `figures/`
2. 生成总结报告到 `results/summary.md`

### Rule 5: 图和表要能投屏讲清楚
请假设你要把输出图投到教室大屏幕上。

因此：
- 图标题必须明确；
- 坐标轴必须标注；
- 默认字号不能太小；
- 一张图最好只承载一个核心判断。

### Rule 6: 可复现优先于“手工过程”
你不能依赖：
- 手工点 notebook cell；
- 临时变量残留；
- 先运行某个脚本、再运行另一个脚本但 README 不写清楚。

本周要求的是：
> **一个入口，产出完整结果。**

---

## 🎬 Task A：构造“会过拟合”的可视化舞台
你需要先生成一份模拟回归数据，用它来展示模型复杂度变化下的行为。

### A1. 自定义真实函数与噪声数据
请自己生成一份一维回归数据，要求：
1. 样本量不少于 `100`；
2. 自定义一个非线性真实函数，例如：
   - `sin(x)`
   - `sin(x) + 线性趋势`
   - 你自己设计的其他平滑函数
3. 加入随机噪声；
4. 使用 `train/test split` 划分训练集和测试集。

### A2. 比较三位候选模型
至少比较以下三种多项式复杂度：
- `degree = 1`
- `degree = 4`
- `degree = 15`

你需要输出一张图，例如：`figures/candidate_models.png`，其中至少展示：
- 训练点；
- 测试点；
- 真实函数（或近似真实曲线）；
- 各自的拟合曲线；
- 对应的 `train RMSE` 与 `test RMSE`。

### A3. 在 `summary.md` 中回答
请回答：
1. 谁最像欠拟合？
2. 谁最像过拟合？
3. 如果今天必须选一个上线，你会先押谁？为什么？

---

## 📉 Task B：画出完整的复杂度-误差曲线
现在不要只看三个例子，而要把整个复杂度范围拉出来。

### B1. 扫描多个复杂度
请至少扫描：
- `degree = 1` 到 `degree = 18`

对每个复杂度，计算并记录：
- `train RMSE`
- `test RMSE`

### B2. 画出误差曲线
请输出一张图，例如：`figures/error_curves.png`，要求：
- 横轴：模型复杂度（degree）
- 纵轴：误差（RMSE）
- 两条曲线：`train RMSE` 和 `test RMSE`

### B3. 给出一个“成绩单”表格
请在 `summary.md` 中至少写出一个表格，包含：
- `degree`
- `train RMSE`
- `test RMSE`
- `generalization gap = test - train`

并明确指出：
1. 测试误差最低的复杂度是多少；
2. 泛化 gap 最大的大概落在哪些复杂度附近；
3. 为什么训练误差最低的模型不一定是最好的模型。

---

## 🌪 Task C：用 repeated sampling 把 variance 画出来
很多同学会机械地背：
- high bias
- high variance

但真正难的是：
> **你能不能把“variance 大”变成一张学生一眼就看懂的图？**

### C1. 固定真实函数，重复抽样训练集
请保持真实函数不变，只改变训练样本。

至少做：
- `degree = 2`
- `degree = 15`

对每个复杂度，重复抽样并拟合至少 `10` 次。

### C2. 把多条拟合曲线画在同一张图上
请输出一张图，例如：`figures/variance_demo.png`，要求：
- 同一复杂度下，多次训练得到的曲线叠加绘制；
- 同时画出真实函数；
- 两个复杂度分别成图，方便对比。

### C3. 至少输出一个定量 summary
除了图，还请在 `summary.md` 中给出一个小表格或数值 summary，例如：
- 平均预测标准差 (`mean prediction std`)
- 最大预测标准差 (`max prediction std`)

### C4. 回答一句话问题
请在 `summary.md` 中用一句话补全：

> high variance model 的危险，不是它不会拟合训练集，而是它对 ______ 过于敏感。

---

## 💥 Task D：让异常值攻击 `RMSE` 与 `MAE`
本周第二条主线是：
> 不同损失函数代表不同的风险偏好。

### D1. 构造一个“干净预测”场景
请构造一组：
- `y_true`
- `y_pred_clean`

使得预测误差较小且分布正常。

### D2. 人为加入一个明显离群点
再构造：
- `y_pred_outlier`

要求：
- 只改动少量样本；
- 但制造一个非常大的预测误差。

### D3. 比较 `RMSE` 与 `MAE`
你至少需要输出：
1. 一个比较图，例如：`figures/loss_outlier_comparison.png`
2. 一个比较表格写入 `summary.md`：
   - clean prediction
   - one large outlier
   - 分别对应的 `RMSE` 和 `MAE`

### D4. 业务解释
请在 `summary.md` 中用短文字回答：
1. 为什么 `RMSE` 更容易被大错拉高？
2. 如果线上系统偶尔一次大错的代价极高，你更想看哪个指标？
3. 如果数据天然包含较多异常值，你会不会重新考虑指标选择？

---

## 🧩 Task E：把脚本写成“可讲授的实验工作流”
虽然本周最终提交不是 notebook，但你的 `main.py` 不应只是“堆一坨代码”。

### E1. 推荐你把 `main.py` 组织成若干函数
建议你至少拆成：
- `generate_data()`
- `run_model_complexity_demo()`
- `run_variance_demo()`
- `run_loss_comparison_demo()`
- `write_summary_report()`
- `main()`

### E2. 推荐你让输出顺序具有叙事性
也就是说，程序执行结果应该像这样推进：
1. 先生成三位候选模型图；
2. 再生成完整复杂度误差曲线；
3. 再生成 variance 图；
4. 最后生成 RMSE vs MAE 对比；
5. 收束到 `summary.md`。

### E3. 可以在代码中加少量“内部暴露式打印信息”
例如：
- `print("[Stage 1] Comparing candidate polynomial models...")`
- `print("[Stage 2] Sweeping model complexity...")`

这样你的 `main.py` 运行起来更像一个有结构的实验流程，而不是无声脚本。

---

## 📝 Task F：输出简短总结 `results/summary.md`
请在：
- `src/week12/results/summary.md`

中总结以下内容：

### 必答 1：三条结论
请写出你本周最重要的三条结论。

### 必答 2：一个最能代表过拟合的图
请说明：
- 你生成的哪一张图最能代表“过拟合不是抽象概念，而是可见现象”？
- 为什么？

### 必答 3：指标选择判断
请回答：
- 在什么情况下你更愿意报告 `RMSE`？
- 在什么情况下你更愿意报告 `MAE`？

### 必答 4：与下一周的连接
请回答：

> 如果模型复杂度过高会带来 high variance，那么下一步我们为什么自然会想到正则化（Ridge / Lasso）？

---

## 📦 交付物要求 (Deliverables)
你最终至少需要提交：
1. `src/week12/main.py`
2. `src/week12/results/summary.md`
3. 至少若干张关键图到 `src/week12/figures/`

可选但不作主提交要求：
- 你本地用于探索的 notebook
- 你额外写的辅助脚本

但请注意：
> **批改以 `.py + .md + 清晰入口` 为准。**

---

## ✅ 建议验证命令 (Validation)
建议至少验证：

```bash
uv run src/week12/main.py
```

如果你的项目启用了 `ruff`，也推荐执行：

```bash
uvx ruff format src
uvx ruff check src
```

---

## 🚨 验收红线 (Red Lines)
- 如果没有 `main.py` 唯一入口，工程部分不能算完成。
- 如果程序运行后没有产生图或没有 `summary.md`，视为作业不完整。
- 如果图没有标题、没有坐标轴、无法投屏辨认，视为可视化不合格。
- 如果你只贴外部资料解释 bias-variance，而没有自己生成图或实验现象，视为作业不完整。
- 如果代码无法复现结果、依赖隐藏状态或手工操作，工程部分视为未完成。
- 如果你在作业中使用 AI，但自己无法解释关键函数在做什么，课堂追问时会扣除相应分数。

---

## 🌟 加分方向 (Bonus)
- 你的 `main.py` 组织得非常清晰，像一个分阶段实验流程；
- 你输出的图在讲授场景下非常清楚，适合直接投屏；
- 你给出更清晰的 repeated sampling 定量 summary；
- 你把 `RMSE` / `MAE` 的业务解释写得非常具体，而不是只说“一个更敏感，一个更稳健”；
- 你在保证 `py + md + main.py` 的前提下，还做了额外的本地 notebook 探索，并把结果规范地整理回正式提交物。
