# Week04 Homework Report：线性回归求解器性能对比

## 一、实验设置
- 低维数据：N=10000，P=10
- 高维数据：N=10000，P=2000
- 测试求解器共 5 种：
  1. AnalyticalSolver
  2. GradientDescentSolver
  3. statsmodels.api.OLS
  4. sklearn.linear_model.LinearRegression
  5. sklearn.linear_model.SGDRegressor

---

## 二、耗时与误差结果（来自实际运行输出）
### 低维场景（N=10000, P=10）
|Custom Solvers         |Time (s) | MSE    |
|-------------------    |---------|--------|
| AnalyticalSolver      | 0.0209  | 0.2471 |
| GradientDescentSolver | 7.3323  | 0.2471 |

### 高维场景（N=10000, P=2000）
|Custom Solvers            |Time (s)  | MSE    |
|--------------------------|----------|--------|
| AnalyticalSolver         | 1.0590   | 0.2103 |
| GradientDescentSolver    | 293.9919 | 0.2103 |
| statsmodels.OLS          | 115.1201 |   --   |
| sklearn.LinearRegression | 19.7164  |   --   |
| sklearn.SGDRegressor     | 294.8730 |   --   |

---

## 三、思考题回答

### 1. 高维场景下，哪个 API 极其缓慢？为什么？
**在高维场景下，运行极其缓慢的 API 有：**
- **GradientDescentSolver（手写）**：293.99s
- **SGDRegressor（sklearn）**：294.87s
- **statsmodels.OLS**：115.12s

**原因：**
- **梯度下降类求解器（GD/SGD）** 需要多次迭代（3000 轮），每一轮都要遍历全部数据计算梯度。高维下特征数量大，迭代计算量巨大，因此耗时极长。
- **statsmodels.OLS** 是解析解方法，需要计算 \(X^TX\) 并求解逆矩阵，时间复杂度为 \(O(P^3)\)。当特征维度 P 从 10 上升到 2000 时，计算量呈立方级爆炸，速度大幅下降。

---

### 2. 为什么 SGDRegressor 理论上能在极短时间完成任务？

原因如下：
1. **不需要构造大矩阵 \(X^TX\)**
2. **不需要矩阵求逆或分解**
3. **只进行向量级别的运算**
4. **时间复杂度是 O(N·P)，与维度呈线性关系**，而非解析解的立方级复杂度

在真实工业场景中，SGD 只需很少迭代即可收敛，远快于解析解方法。
本次实验中 SGDRegressor 运行较慢，是因为使用了较多迭代次数，并非算法本身缺陷。

---
