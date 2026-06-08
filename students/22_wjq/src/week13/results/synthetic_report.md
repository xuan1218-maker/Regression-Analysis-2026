# 模拟数据正则化回归报告

## 1. DGP
y = 5 + 3X1 + 2X2 + X3 + 1.5X5 + X6 + ε
X1-X4高度相关（由潜变量驱动），X7-X10为纯噪声。

## 2. 系数稳定性
- X1: OLS std=0.0917, Ridge std=0.0895
- X2: OLS std=0.0873, Ridge std=0.0864
- X3: OLS std=0.0650, Ridge std=0.0643
- X4: OLS std=0.0366, Ridge std=0.0361
- X5: OLS std=0.0476, Ridge std=0.0474
- X6: OLS std=0.0391, Ridge std=0.0390
- X7: OLS std=0.0282, Ridge std=0.0280
- X8: OLS std=0.0312, Ridge std=0.0311
- X9: OLS std=0.0300, Ridge std=0.0299
- X10: OLS std=0.0249, Ridge std=0.0248

## 3. 测试集性能
| 模型 | RMSE | MAE | R2 |
|------|------|-----|-----|
| OLS | 0.9846 | 0.8084 | 0.9724 |
| Ridge | 0.9841 | 0.8065 | 0.9724 |
| Lasso | 0.9862 | 0.8104 | 0.9723 |
| ElasticNet | 0.9863 | 0.8098 | 0.9723 |

## 4. 变量筛选
- 前向选择: ['X1', 'X5', 'X2', 'X6', 'X3', 'X7', 'X9', 'X4']
- Lasso选择: ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X9', 'X10']
