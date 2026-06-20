"""
Model Diagnostics Toolbox (Week15 Version)

功能：
1. VIF（多重共线性）
2. 条件数 / 秩检测
3. 残差图
4. QQ图
5. 相关性热力图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.models import AnalyticalOLS


# =========================
# 基础诊断指标
# =========================
def calculate_rank(X):
    return np.linalg.matrix_rank(X)


def calculate_condition_number(X):
    return np.linalg.cond(X)


# =========================
# VIF
# =========================
def calculate_vif(X: np.ndarray):
    vif_values = []
    n_features = X.shape[1]

    for j in range(n_features):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)

        X_design = np.column_stack([np.ones(len(X_j)), X_j])

        model = AnalyticalOLS(fit_intercept=False)
        model.fit(X_design, y_j)

        y_pred = model.predict(X_design)

        sse = np.sum((y_j - y_pred) ** 2)
        sst = np.sum((y_j - np.mean(y_j)) ** 2)

        r2 = 1 - sse / sst if sst != 0 else 0
        vif = 1 / (1 - r2) if r2 < 0.999 else np.inf

        vif_values.append(vif)

    return vif_values


def calculate_vif_dataframe(df: pd.DataFrame, cols: list):
    X = df[cols].values
    return pd.DataFrame({
        "feature": cols,
        "VIF": calculate_vif(X)
    })


def print_vif_warning(vif_df: pd.DataFrame, threshold=10):
    print("\n" + "=" * 50)
    print("VIF Diagnostic Report")
    print("=" * 50)

    severe = False

    for _, row in vif_df.iterrows():
        f = row["feature"]
        v = row["VIF"]

        if v > threshold:
            print(f"[HIGH] {f}: {v:.2f}")
            severe = True
        elif v > 5:
            print(f"[MID ] {f}: {v:.2f}")
        else:
            print(f"[OK  ] {f}: {v:.2f}")

    if severe:
        print("\n⚠ Severe multicollinearity detected.")
    else:
        print("\n✓ No severe multicollinearity.")

    return severe


# =========================
# Plot utilities
# =========================
def _fig_path():
    base = Path(__file__).resolve().parents[1]
    path = base / "week15" / "results" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================
# Residual Plot
# =========================
def plot_residuals(y_true, y_pred, name="residuals.png"):
    res = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, res, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.savefig(_fig_path() / name, dpi=150)
    plt.close()


# =========================
# QQ Plot
# =========================
def plot_qq_residuals(y_true, y_pred, name="qq.png"):
    import scipy.stats as stats

    res = y_true - y_pred

    plt.figure()
    stats.probplot(res, dist="norm", plot=plt)
    plt.title("QQ Plot")

    plt.savefig(_fig_path() / name, dpi=150)
    plt.close()


# =========================
# Correlation Matrix
# =========================
def plot_correlation_matrix(df: pd.DataFrame, name="corr.png"):
    corr = df.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig(_fig_path() / name, dpi=150)
    plt.close()