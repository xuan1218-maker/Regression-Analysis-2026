# ===================== 生成 synthetic_correlated.csv =====================
import numpy as np
import pandas as pd

# 固定随机种子，保证结果可复现
np.random.seed(42)

# 样本数量（满足 >= 300）
n_samples = 300

# ---------------------- 构造【强共线性特征组】（3个）----------------------
base = np.random.randn(n_samples)  # 共同基础变量

x1 = base + np.random.randn(n_samples) * 0.05
x2 = base + np.random.randn(n_samples) * 0.08
x3 = base + np.random.randn(n_samples) * 0.10

# ---------------------- 纯噪声特征（与y无关）----------------------
x4 = np.random.randn(n_samples)
x5 = np.random.randn(n_samples)
x6 = np.random.randn(n_samples)
x7 = np.random.randn(n_samples)
x8 = np.random.randn(n_samples)

# ---------------------- 真实数据生成公式 DGP ----------------------
# y 只由 x1, x2 决定，其他都是干扰项
y = 3.0 * x1 + 2.0 * x2 + np.random.randn(n_samples) * 0.5

# ---------------------- 构造 DataFrame 并保存 ----------------------
X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8])
features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
df = pd.DataFrame(X, columns=features)
df["y"] = y

# 保存到 week13/data/ 下面
df.to_csv("students/17_jxx/src/week13/data/synthetic_correlated.csv", index=False)

print("✅ 成功生成：synthetic_correlated.csv")
print(f"✅ 样本数：{len(df)}")
print(f"✅ 特征数：{len(features)}")
print(f"✅ 强相关特征：x1, x2, x3")
print(f"✅ 真实DGP：y = 3.0*x1 + 2.0*x2 + noise")