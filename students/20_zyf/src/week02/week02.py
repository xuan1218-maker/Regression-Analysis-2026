import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ======================
# 1. 生成数据
# ======================
np.random.seed(42)

N = 100
beta_0 = 1
beta_1 = 2

x = np.random.uniform(0, 10, N)
epsilon = np.random.normal(0, 1, N)
y = beta_0 + beta_1 * x + epsilon

# ======================
# 2. 公式估计
# ======================
x_mean = np.mean(x)
y_mean = np.mean(y)

beta_1_hat = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
beta_0_hat = y_mean - beta_1_hat * x_mean

# ======================
# 3. 方差估计（beta_1）
# ======================
y_hat = beta_0_hat + beta_1_hat * x
sigma2_hat = np.sum((y - y_hat)**2) / (N - 2)
var_beta_1_hat = sigma2_hat / np.sum((x - x_mean)**2)

# ======================
# 4. Bias
# ======================
bias_beta_0 = beta_0_hat - beta_0
bias_beta_1 = beta_1_hat - beta_1

# ======================
# 5. sklearn
# ======================
x_reshaped = x.reshape(-1, 1)
lr = LinearRegression().fit(x_reshaped, y)

# ======================
# 6. statsmodels
# ======================
import statsmodels.formula.api as smf
df = pd.DataFrame({'x': x, 'y': y})
model = smf.ols('y ~ x', data=df).fit()

# ======================
# 7. 假设检验 & ANOVA
# ======================
p_value = model.pvalues[1]
anova_table = sm.stats.anova_lm(model, typ=2)

# ======================
# 8. 输出结果
# ======================
print("=== 公式估计 ===")
print(beta_0_hat, beta_1_hat)

print("\n=== sklearn ===")
print(lr.intercept_, lr.coef_[0])

print("\n=== statsmodels ===")
print(model.params)

print("\n=== 方差 ===")
print(var_beta_1_hat)

print("\n=== bias ===")
print(bias_beta_0, bias_beta_1)

print("\n=== p-value ===")
print(p_value)

print("\n=== ANOVA ===")
print(anova_table)

# ======================
# 9. 绘图
# ======================
plt.scatter(x, y)
plt.plot(x, y_hat)
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("regression_plot.png")
plt.show()