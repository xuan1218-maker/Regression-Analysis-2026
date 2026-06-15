# Task A: Synthetic Binary Classification Report

## A2. Data Generation Process (DGP)

- **样本量**: 500
- **特征数**: 4 (X1, X2, X3, X4)
- **正类比例**: 0.424
- **DGP**: 先构造线性预测值 $\eta = -0.5 + 1.5 \cdot X_1 + -1.0 \cdot X_2$，再通过 sigmoid 转换为概率 $p = 1/(1+e^{-\eta})$，最后从 Bernoulli(p) 采样得到 y。
- **X1** 对正类概率有正向影响（系数 1.5）；
- **X2** 对正类概率有负向影响（系数 -1.0）；
- **X3、X4** 是纯噪声特征，不影响类别概率。

## A3. 模型对比

| 模型 | Accuracy | Log Loss |
|------|----------|----------|
| LinearRegression (threshold=0.5) | 0.7933 | 0.6846 |
| LogisticRegression | 0.7933 | 0.4545 |

LinearRegression 的输出范围为 [-0.783, 1.211]，
出现了超出 [0, 1] 的值，这些值无法被解释为概率。

## A4. 核心对比图

**图 task_a_comparison.png**：
- 横轴：X1（一个有信息量的特征），其他特征固定在均值
- 纵轴：模型输出
- 灰色散点：真实标签 (0/1)
- 蓝色曲线：LinearRegression 的预测值（线性，可能超出 [0,1]）
- 红色曲线：LogisticRegression 的预测概率（S 形，始终在 [0,1] 内）
- **结论**：LinearRegression 的输出是无界的直线，不能合理解释为概率；LogisticRegression 通过 sigmoid 将输出压缩到 (0,1)，天然适合概率解释。

## A5. 核心问题

### Q1. LinearRegression 在这个任务里最不自然的地方是什么？
LinearRegression 的输出是无界的连续值，可能小于 0 或大于 1。在二分类问题中，我们需要的是"属于正类的概率"，而概率必须在 [0,1] 区间内。LinearRegression 无法保证这一点，因此将其输出硬解释为概率在数学上是不合理的。

### Q2. 为什么逻辑回归的输出更容易解释成概率？
逻辑回归通过 sigmoid 函数 $\sigma(\eta) = 1/(1+e^{-\eta})$ 将线性组合映射到 (0,1) 区间，输出天然满足概率的基本性质：有界性。同时，逻辑回归的训练目标（最大化 Bernoulli 似然）保证了输出在统计意义上是对 $P(Y=1|X)$ 的一致估计。

### Q3. 关键区别是"能不能分类"还是"输出是否有概率意义"？
关键区别是**输出是否有概率意义**。LinearRegression 也能做分类（通过设定阈值），但它的输出没有概率解释。逻辑回归的输出不仅能在 (0,1) 内，而且经过 MLE 训练后，输出值与真实的条件概率 $P(Y=1|X)$ 有明确的对应关系。这使得我们可以对输出做更有意义的决策——比如调整阈值、校准概率等。
