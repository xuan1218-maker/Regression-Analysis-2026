import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from src.utils.diagnostics import calculate_vif
from src.utils.models import AnalyticalOLS


# =========================
# 读取数据
# =========================
df = pd.read_csv("src/week09/data/clean_marketing.csv")

print("清洗后的数据：")
print(df.head())

# =========================
# 划分特征和目标
# =========================
X = df.drop(columns=["Sales"])
y = df["Sales"]

feature_names = X.columns.tolist()

X = X.values
y = y.values

# =========================
# VIF检测
# =========================
print("\n========== VIF 检测 ==========")

vif_results = calculate_vif(X, feature_names)

for feature, vif in vif_results:

    print(f"{feature}: {vif:.2f}")

    if vif > 10:
        print(
            f"\033[91mWARNING: {feature} 存在严重多重共线性！\033[0m"
        )

# =========================
# 5折交叉验证
# =========================
print("\n========== 5-Fold CV ==========")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # =====================
    # 训练模型
    # =====================
    model = AnalyticalOLS()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    r2_scores.append(r2)

    print(f"Fold {fold + 1} R2: {r2:.4f}")

# =========================
# 平均R2
# =========================
mean_r2 = np.mean(r2_scores)

print("\n========== 最终结果 ==========")
print(f"平均 R2: {mean_r2:.4f}")