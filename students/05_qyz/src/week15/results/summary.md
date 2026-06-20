# Week 15 Summary

---

## 1. 为什么逻辑回归不是“线性回归 + sigmoid”这么简单？

虽然形式上：

$$
p = sigmoid(X\beta)
$$

看起来只是线性回归后接一个 sigmoid，但本质区别在于：

---

### （1）建模对象不同

- 线性回归：预测连续数值
- 逻辑回归：预测概率

---

### （2）统计假设不同

| 模型 | 分布假设 |
|------|----------|
| Linear Regression | Gaussian noise |
| Logistic Regression | Bernoulli distribution |

---

### （3）优化目标不同

- 线性回归：MSE 最小化
- 逻辑回归：log loss（MLE）

---

### 结论：

> 逻辑回归是一种概率模型，并非对线性回归做后置处理得到的模型。

---

## 2. sigmoid、Bernoulli likelihood、log loss 的关系

三者构成一个完整概率建模闭环：

---

### （1）sigmoid（映射函数）

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

作用：

> 将线性输出映射到概率空间 (0,1)

---

### （2）Bernoulli likelihood（生成模型）

$$
Y \sim Bernoulli(p)
$$

作用：

> 定义数据生成机制

---

### （3）log loss（优化目标）

$$
-\log L = -[y\log p + (1-y)\log(1-p)]
$$

作用：

> 最大化数据在模型下的概率（MLE）

---

### 三者关系总结：

> sigmoid 提供概率 → Bernoulli 定义数据 → log loss 用于学习参数

---

## 3. 为什么分类模型不能只看 accuracy？

---

### （1）accuracy 忽略错误类型

混淆矩阵中：

- FP（误报）
- FN（漏报）

accuracy 无法区分

---

### （2）阈值敏感性

在本实验中：

- threshold ↑ → precision ↑ recall ↓
- threshold ↓ → recall ↑ precision ↓

说明：

> accuracy 不是稳定指标

---

### （3）类别不平衡问题

accuracy 在不平衡数据中会失真

---

### 结论：

> 分类模型的性能必须使用多项指标综合评估。

---

## 4. L1 vs L2 逻辑回归适用目标

---

### L1（Lasso）

特点：

- 稀疏性强
- 自动特征选择

适用：

- 高维数据
- 特征筛选
- 可解释模型

---

### L2（Ridge）

特点：

- 参数稳定
- 不做变量删除

适用：

- 预测任务
- 共线性问题
- 工业建模

---

### 本实验结论：

| Model | Role |
|------|------|
| L1 | feature selection |
| L2 | stable prediction |

---

## 5. 为什么 Logistic Regression 仍然是强 baseline？

---

### （1）概率可解释性

输出：

> P(y=1 | x)

可直接用于业务决策

---

### （2）稳定性强

- 对小数据鲁棒
- 不容易过拟合（配合正则化）

---

### （3）可解释性

系数 β 可解释变量方向：

- β > 0 → 正相关
- β < 0 → 负相关

---

### （4）工业适用性强

- 计算成本低
- 可扩展性好
- 易部署

---

## FINAL CONCLUSION

---

Logistic Regression 的核心优势是：

> 简单性 + 可解释性 + 概率意义

---