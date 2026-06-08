"""
Week 14: High-Dimensional Regression, PCA, and PCR
===================================================
单一入口：uv run src/week14/main.py

任务概览：
  Task A - 生成高维数据，观察 OLS 失稳
  Task B - PCA 与 PCR
  Task C - Lasso vs PCR：selection vs compression
  Task D - 真实数据挑战（Kaggle House Prices）
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LassoCV

# ── 将项目 src/ 加入 path，以便 import utils ─────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.models import AnalyticalOLS, PCR
from utils.metrics import calculate_rmse, calculate_mae, coefficient_cv
from utils.transformers import CustomStandardScaler
from utils.diagnostics import (
    calculate_condition_number,
    calculate_matrix_rank,
    coefficient_stability_analysis,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 路径常量 ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 全局绘图设置 ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# =====================================================================
#  工具函数
# =====================================================================

def save_fig(fig, name: str):
    """保存图片到 results/figures/"""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 已保存 -> {path}")


def add_intercept(X: np.ndarray) -> np.ndarray:
    """在矩阵前添加全1截距列"""
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_fit_predict(X_train, y_train, X_test):
    """OLS 拟合与预测的快捷函数"""
    model = AnalyticalOLS()
    model.fit(add_intercept(X_train), y_train)
    y_train_pred = model.predict(add_intercept(X_train))
    y_test_pred = model.predict(add_intercept(X_test))
    return model, y_train_pred, y_test_pred


def write_report(path: str, content: str):
    """写入报告文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [报告] 已保存 -> {path}")


# =====================================================================
#  Task A: 生成高维数据，观察 OLS 失稳
# =====================================================================

def generate_synthetic_data(
    n_samples: int = 150,
    n_features: int = 80,
    n_latent: int = 5,
    noise_std: float = 1.0,
    random_state: int = 42,
) -> tuple:
    """
    生成带有潜在低秩结构的高维回归数据。

    机制：
      Z (n × n_latent)  ~ N(0, I)           # 潜在因子
      W (n_latent × p)  ~ N(0, 1)           # 因子载荷
      X = Z @ W + ε_x                       # 原始变量
      y = Z @ γ + ε_y                       # 目标仅由潜在因子驱动

    """
    rng = np.random.default_rng(random_state)

    Z = rng.standard_normal((n_samples, n_latent))
    W = rng.standard_normal((n_latent, n_features))
    X = Z @ W + rng.standard_normal((n_samples, n_features)) * 0.5

    gamma = rng.standard_normal(n_latent) * 3.0
    y = Z @ gamma + rng.standard_normal(n_samples) * noise_std

    return X, y, Z, W, gamma


def task_a1_save_data():
    """A1 & A2: 生成并保存合成数据"""
    print("\n" + "=" * 70)
    print("  Task A1/A2: 生成高维合成数据")
    print("=" * 70)

    X, y, Z, W, gamma = generate_synthetic_data(
        n_samples=150, n_features=80, n_latent=5
    )

    # 保存 CSV
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    df["y"] = y
    csv_path = os.path.join(DATA_DIR, "synthetic_highdim.csv")
    df.to_csv(csv_path, index=False)
    print(f"  数据已保存 -> {csv_path}")
    print(f"  样本量 n={X.shape[0]}, 特征数 p={X.shape[1]}, 潜在因子数={Z.shape[1]}")

    return X, y, Z, W, gamma


def task_a3_ols_dimension_experiment(X, y):
    """A3: 不同特征维度下的 OLS 表现"""
    print("\n" + "=" * 70)
    print("  Task A3: OLS 随特征维度变化的误差与矩阵结构")
    print("=" * 70)

    n_total = X.shape[0]
    # 取前 p 列作为特征（数据生成时列是有序的）
    p_values = [10, 30, 60, 80]

    results = {
        "p": [], "train_rmse": [], "test_rmse": [],
        "rank": [], "cond_number": [],
    }

    for p in p_values:
        X_p = X[:, :p]
        X_train, X_test, y_train, y_test = train_test_split(
            X_p, y, test_size=0.3, random_state=42
        )

        model, y_train_pred, y_test_pred = ols_fit_predict(X_train, y_train, X_test)

        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        rank = calculate_matrix_rank(X_train)
        cond = calculate_condition_number(X_train)

        results["p"].append(p)
        results["train_rmse"].append(train_rmse)
        results["test_rmse"].append(test_rmse)
        results["rank"].append(rank)
        results["cond_number"].append(cond)

        print(f"  p={p:>3d}  |  train_RMSE={train_rmse:.4f}  test_RMSE={test_rmse:.4f}  "
              f"rank={rank}  cond={cond:.2e}")

    # ── 图1: 误差随特征维度变化 ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(results["p"], results["train_rmse"], "o-", label="Train RMSE", linewidth=2)
    ax.plot(results["p"], results["test_rmse"], "s--", label="Test RMSE", linewidth=2)
    ax.set_xlabel("Feature Dimension (p)")
    ax.set_ylabel("RMSE")
    ax.set_title("Task A3: OLS Error vs Feature Dimension p\n"
                 "(n=150 fixed, 70% train split)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "a3_rmse_vs_p.png")

    # ── 图2: 矩阵结构随特征维度变化 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar([str(p) for p in results["p"]], results["rank"], color="steelblue", alpha=0.8)
    ax1.axhline(y=n_total * 0.7, color="red", linestyle="--", label="n_train (upper bound)")
    ax1.set_xlabel("Feature Dimension (p)")
    ax1.set_ylabel("rank(X_train)")
    ax1.set_title("Matrix Rank vs Feature Dimension")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar([str(p) for p in results["p"]], results["cond_number"], color="coral", alpha=0.8)
    ax2.set_xlabel("Feature Dimension (p)")
    ax2.set_ylabel("Condition Number")
    ax2.set_title("Condition Number vs Feature Dimension\n(larger = more ill-conditioned)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task A3: X_train Matrix Structure Degrades as p Increases", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "a3_matrix_structure_vs_p.png")

    return results


def task_a4_coefficient_instability(X, y):
    """A4: 重复切分展示系数不稳定"""
    print("\n" + "=" * 70)
    print("  Task A4: 系数不稳定性分析（50次随机切分）")
    print("=" * 70)

    # 选取前 5 个特征进行追踪
    feature_indices = list(range(5))
    n_splits = 50

    result = coefficient_stability_analysis(
        X, y, n_splits=n_splits,
        feature_indices=feature_indices, random_state=0,
    )

    coef_matrix = result["coef_matrix"]
    coef_stds = result["coef_std"]

    for i, idx in enumerate(feature_indices):
        print(f"  X{idx}: mean={result['coef_mean'][i]:>10.4f}  "
              f"std={coef_stds[i]:>10.4f}  CV={coef_stds[i] / max(abs(result['coef_mean'][i]), 1e-10):.4f}")

    # ── 箱线图: 同一变量系数在不同切分下的波动 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [coef_matrix[:, i] for i in range(coef_matrix.shape[1])],
        labels=[f"X{idx}" for idx in feature_indices],
        patch_artist=True,
    )
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Feature")
    ax.set_ylabel("OLS Coefficient Value")
    ax.set_title(f"Task A4: OLS Coefficient Instability Across 50 Random Splits\n"
                 f"(p={X.shape[1]}, high-dimensional)")
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, "a4_coefficient_instability.png")

    return result


def write_synthetic_report(a3_results, a4_result):
    """生成 synthetic_report.md"""
    print("\n  撰写 synthetic_report.md ...")

    report = """# Synthetic Data Report — Task A & B

## 1. 数据生成机制 (A1/A2)

### 基本参数
| 参数 | 值 |
|------|-----|
| 样本量 n | 150 |
| 特征维度 p | 80 |
| 潜在因子数 (latent factors) | 5 |
| 噪声标准差 | 1.0 |

### 潜在因子结构

数据生成机制如下：

$$Z \\in \\mathbb{R}^{n \\times 5} \\sim N(0, I_5)$$
$$W \\in \\mathbb{R}^{5 \\times 80} \\sim N(0, 1)$$
$$X = Z W + \\varepsilon_X, \\quad \\varepsilon_X \\sim N(0, 0.5^2 I)$$
$$y = Z \\gamma + \\varepsilon_y, \\quad \\varepsilon_y \\sim N(0, 1)$$

- **Z** 是 5 个潜在因子，每个样本有 5 个独立的隐变量；
- **W** 是因子载荷矩阵，将 5 个因子线性组合成 80 个原始变量；
- **y** 仅由这 5 个潜在因子通过 $\\gamma$ 驱动，与 80 个原始变量没有直接独立关系。

### 为什么说这是"高维 + 信息冗余"？

1. **高维**：p = 80 接近甚至超过训练集样本量（约 105），属于 $p \\approx n$ 甚至 $p > n$ 的典型场景；
2. **信息冗余**：80 个原始变量实际上只是 5 个潜在因子的线性组合加噪声，真正的"有效维度"仅为 5；
3. 因此 X 的列之间存在强烈的多重共线性——它们都在"争夺"同一组潜在因子的解释权。

---

## 2. OLS 在高维下的失稳 (A3)

### 误差随特征维度变化

| p | Train RMSE | Test RMSE | rank(X_train) | Condition Number |
|---|-----------|-----------|---------------|-----------------|
"""
    for i, p in enumerate(a3_results["p"]):
        report += (f"| {p} | {a3_results['train_rmse'][i]:.4f} | "
                   f"{a3_results['test_rmse'][i]:.4f} | "
                   f"{a3_results['rank'][i]} | "
                   f"{a3_results['cond_number'][i]:.2e} |\n")

    report += """
**关键观察**：
- 当 p 较小时（p=10），train RMSE 和 test RMSE 差距不大，模型泛化尚可；
- 当 p 增大到接近甚至超过训练集样本量时，**train RMSE 持续下降甚至趋近于 0**，但 **test RMSE 开始上升或剧烈波动**；
- 这就是"虚假的低训练误差"——OLS 在高维下可以完美拟合训练数据（甚至拟合噪声），但这种拟合没有泛化能力。

**条件数的含义**：
条件数 $\\kappa(X) = \\sigma_{\\max} / \\sigma_{\\min}$ 衡量矩阵的病态程度。
当 p 增大时，X 的列之间共线性加剧，条件数急剧增长，说明 OLS 的系数估计对数据扰动极度敏感。

---

## 3. 系数不稳定性 (A4)

对同一份数据集进行 50 次不同随机切分，每次用 OLS 拟合，收集前 5 个特征的系数。

| 特征 | 系数均值 | 系数标准差 | 变异系数 (CV) |
|------|---------|-----------|-------------|
"""
    for i, idx in enumerate(a4_result["feature_indices"]):
        mean = a4_result["coef_mean"][i]
        std = a4_result["coef_std"][i]
        cv = std / max(abs(mean), 1e-10)
        report += f"| X{idx} | {mean:.4f} | {std:.4f} | {cv:.4f} |\n"

    report += """
### 核心发现

1. **系数在剧烈波动**：不同随机切分下，同一变量的 OLS 系数可以差异巨大，甚至符号翻转；
2. **不仅仅是误差在波动**：如果只是误差波动，说明模型预测不够精确；但系数波动说明模型的"解释"本身不稳定——同样的变量在不同子样本中可能被赋予完全不同的含义；
3. **为什么系数不稳定是重要风险？**
   - 在业务场景中，如果一个变量今天是正影响、明天是负影响，决策者将无法信任模型；
   - 系数不稳定是高维/共线性问题的直接后果，也是我们转向 PCA/PCR 的根本动机。

---

## 4. 公式与定义 (B4)

### 4.1 OLS 估计式

$$\\hat{\\beta}_{OLS} = (X^\\top X)^{-1} X^\\top y$$

当 $X^\\top X$ 接近奇异（高维/共线性）时，求逆过程会放大微小扰动，导致 $\\hat{\\beta}$ 剧烈波动。

### 4.2 第一主成分的方差最大化定义

第一主成分方向 $v_1$ 是使得投影后方差最大的方向：

$$v_1 = \\arg\\max_{\\|v\\|=1} \\mathrm{Var}(Xv) = \\arg\\max_{\\|v\\|=1} v^\\top X^\\top X v$$

其解为 $X^\\top X$ 最大特征值对应的特征向量（即 SVD 中 $V$ 的第一列）。

### 4.3 PCR 流程的符号表达

给定标准化后的数据 $\\tilde{X}$，做 SVD 分解：

$$\\tilde{X} = U S V^\\top$$

取前 k 个右奇异向量 $V_k = [v_1, \\ldots, v_k]$，投影得到主成分：

$$Z_k = \\tilde{X} V_k \\in \\mathbb{R}^{n \\times k}$$

然后在 $Z_k$ 上做 OLS 回归：

$$y = Z_k \\alpha + \\varepsilon, \\quad \\hat{\\alpha} = (Z_k^\\top Z_k)^{-1} Z_k^\\top y$$

最终预测：$\\hat{y} = \\tilde{X} V_k \\hat{\\alpha}$

**核心思想**：先将 p 维原始变量压缩到 k 维主成分空间（$k \\ll p$），再在低维空间中做稳定的回归。
"""

    write_report(os.path.join(RESULTS_DIR, "synthetic_report.md"), report)


# =====================================================================
#  Task B: PCA 与 PCR
# =====================================================================

def task_b1_pca_analysis(X):
    """B1: PCA 分析与累计解释方差曲线"""
    print("\n" + "=" * 70)
    print("  Task B1: PCA — 累计解释方差分析")
    print("=" * 70)

    scaler = CustomStandardScaler()
    X_std = scaler.fit_transform(X)

    # SVD
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    explained_var = (S ** 2) / (X_std.shape[0] - 1)
    total_var = np.sum(explained_var)
    cum_var_ratio = np.cumsum(explained_var) / total_var

    # 找到解释 90% 和 95% 方差所需的主成分数
    k_90 = int(np.searchsorted(cum_var_ratio, 0.90)) + 1
    k_95 = int(np.searchsorted(cum_var_ratio, 0.95)) + 1
    print(f"  前 5 个主成分解释方差比例: {cum_var_ratio[4]:.4f}")
    print(f"  解释 90% 方差需要 {k_90} 个主成分")
    print(f"  解释 95% 方差需要 {k_95} 个主成分")

    # ── 累计解释方差曲线 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, "b-o", markersize=3)
    ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.7, label="90% Variance Threshold")
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="95% Variance Threshold")
    ax.axvline(x=k_90, color="red", linestyle=":", alpha=0.5)
    ax.axvline(x=k_95, color="green", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("Task B1: PCA Cumulative Explained Variance\n"
                 f"(p={X.shape[1]}, true latent dimension = 5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "b1_cumulative_explained_variance.png")

    return cum_var_ratio, k_90, k_95


def task_b2_pcr_workflow(X, y, cum_var_ratio):
    """B2: PCR 工作流 — 不同 k 值的比较"""
    print("\n" + "=" * 70)
    print("  Task B2: PCR 工作流 — k 值选择")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    k_values = list(range(1, 21))
    train_rmses = []
    test_rmses = []
    cv_rmses = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        pcr = PCR(n_components=k)
        pcr.fit(X_train, y_train)

        train_rmse = calculate_rmse(y_train, pcr.predict(X_train))
        test_rmse = calculate_rmse(y_test, pcr.predict(X_test))

        # CV RMSE
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            pcr_cv = PCR(n_components=k)
            pcr_cv.fit(X_train[train_idx], y_train[train_idx])
            val_pred = pcr_cv.predict(X_train[val_idx])
            cv_scores.append(calculate_rmse(y_train[val_idx], val_pred))
        cv_rmse = np.mean(cv_scores)

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        cv_rmses.append(cv_rmse)

    # OLS 基线
    ols_model, ols_train_pred, ols_test_pred = ols_fit_predict(X_train, y_train, X_test)
    ols_train_rmse = calculate_rmse(y_train, ols_train_pred)
    ols_test_rmse = calculate_rmse(y_test, ols_test_pred)
    print(f"  OLS 基线: train_RMSE={ols_train_rmse:.4f}, test_RMSE={ols_test_rmse:.4f}")

    best_k_idx = int(np.argmin(cv_rmses))
    best_k = k_values[best_k_idx]
    print(f"  最优 k (by CV): {best_k}, CV_RMSE={cv_rmses[best_k_idx]:.4f}")

    # ── 图: PCR 误差随 k 变化 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, train_rmses, "o-", label="PCR Train RMSE", linewidth=2)
    ax.plot(k_values, test_rmses, "s--", label="PCR Test RMSE", linewidth=2)
    ax.plot(k_values, cv_rmses, "^-.", label="PCR CV RMSE (5-fold)", linewidth=2)
    ax.axhline(y=ols_train_rmse, color="blue", linestyle=":", alpha=0.5,
               label=f"OLS Train RMSE = {ols_train_rmse:.2f}")
    ax.axhline(y=ols_test_rmse, color="red", linestyle=":", alpha=0.5,
               label=f"OLS Test RMSE = {ols_test_rmse:.2f}")
    ax.axvline(x=best_k, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("RMSE")
    ax.set_title("Task B2: PCR Error vs Number of Components k\n"
                 "(vs OLS baseline, gray dashed = CV-optimal k)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "b2_pcr_rmse_vs_k.png")

    return k_values, train_rmses, test_rmses, cv_rmses, best_k


def write_pcr_section_in_report(k_values, train_rmses, test_rmses, cv_rmses, best_k):
    """将 B2/B3 内容追加到 synthetic_report.md"""
    print("\n  追加 PCR 分析到 synthetic_report.md ...")

    section = f"""

---

## 5. PCA 分析 (B1)

前 5 个主成分（对应真实潜在因子数）已经解释了大部分方差。
这验证了我们的数据生成机制：80 个原始变量中，真正的信息维度仅为 5。
其余 75 个维度只是 5 个因子的线性组合 + 噪声。

**"原始高维空间贴近一个更低维子空间"的解释**：
X 的 80 列并不是 80 个独立信息源，而是 5 个潜在因子的不同线性组合。
因此，X 的有效秩远小于 80——大部分"方向"只是噪声。
PCA 能够识别出这些主要方向（主成分），将数据从 80 维压缩到 5-10 维而不丢失关键信息。

---

## 6. PCR 分析 (B2/B3)

### PCR 不同 k 值的误差

| k | Train RMSE | Test RMSE | CV RMSE |
|---|-----------|-----------|---------|
"""
    for i, k in enumerate(k_values):
        section += (f"| {k} | {train_rmses[i]:.4f} | "
                    f"{test_rmses[i]:.4f} | {cv_rmses[i]:.4f} |\n")

    section += f"""
**最优 k = {best_k}**（由 5-fold CV 选择）。

### CV 曲线解释 (B3)

- **PCR CV RMSE** 代表：在训练集内部，通过 5 折交叉验证估计的泛化误差。它是对 test RMSE 的一种无偏估计；
- **与 train/test 曲线的关系**：CV RMSE 通常位于 train RMSE 和 test RMSE 之间。当 k 过小时，三者都较高（欠拟合）；当 k 增大，train RMSE 下降，但 CV 和 test RMSE 先降后升（过拟合）；
- **为什么 OLS 可以取得很低的训练误差但并不更好？**
  OLS 在原始 80 维空间中拟合，它可以利用所有 80 个方向（包括噪声方向）来完美拟合训练数据。
  但这相当于拟合噪声——训练误差为 0 并不意味着模型好，反而意味着严重过拟合。
  PCR 通过限制使用前 k 个主成分，强制模型只使用信号最强的方向，从而获得更好的泛化能力。
"""

    # 追加写入
    report_path = os.path.join(RESULTS_DIR, "synthetic_report.md")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(section)


# =====================================================================
#  Task C: Lasso vs PCR — selection vs compression
# =====================================================================

def generate_sparse_truth(n=200, p=60, n_informative=5, noise=1.0, seed=42):
    """场景1：Sparse Truth — 只有少数原始变量直接决定 y"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:n_informative] = rng.standard_normal(n_informative) * 5.0
    y = X @ beta + rng.standard_normal(n) * noise
    return X, y, beta


def generate_latent_factor_truth(n=200, p=60, n_latent=5, noise=1.0, seed=42):
    """场景2：Latent-Factor Truth — 潜在因子驱动"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, n_latent))
    W = rng.standard_normal((n_latent, p))
    X = Z @ W + rng.standard_normal((n, p)) * 0.5
    gamma = rng.standard_normal(n_latent) * 3.0
    y = Z @ gamma + rng.standard_normal(n) * noise
    return X, y


def compare_lasso_pcr(X, y, scenario_name, n_splits=30, seed=0):
    """在给定数据上比较 Lasso 和 PCR"""
    print(f"\n  场景: {scenario_name}")

    lasso_test_rmses, pcr_test_rmses = [], []
    lasso_n_nonzero, pcr_k_used = [], []
    lasso_coefs, pcr_coefs = [], []

    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed + i
        )

        # LassoCV
        lasso = LassoCV(cv=5, random_state=seed + i, max_iter=5000)
        scaler = CustomStandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        lasso.fit(X_train_std, y_train)
        lasso_pred = lasso.predict(X_test_std)
        lasso_test_rmses.append(calculate_rmse(y_test, lasso_pred))
        lasso_n_nonzero.append(int(np.sum(np.abs(lasso.coef_) > 1e-6)))
        lasso_coefs.append(lasso.coef_.copy())

        # PCR — 用 CV 选 k
        best_k, best_cv = 1, np.inf
        kf = KFold(n_splits=5, shuffle=True, random_state=seed + i)
        for k in range(1, 21):
            cv_scores = []
            for tr_idx, vl_idx in kf.split(X_train):
                pcr_cv = PCR(n_components=k)
                pcr_cv.fit(X_train[tr_idx], y_train[tr_idx])
                cv_scores.append(calculate_rmse(y_train[vl_idx], pcr_cv.predict(X_train[vl_idx])))
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv:
                best_cv = mean_cv
                best_k = k

        pcr = PCR(n_components=best_k)
        pcr.fit(X_train, y_train)
        pcr_pred = pcr.predict(X_test)
        pcr_test_rmses.append(calculate_rmse(y_test, pcr_pred))
        pcr_k_used.append(best_k)
        pcr_coefs.append(pcr.coef_.copy())

    # 汇总
    results = {
        "scenario": scenario_name,
        "lasso_rmse_mean": np.mean(lasso_test_rmses),
        "lasso_rmse_std": np.std(lasso_test_rmses),
        "pcr_rmse_mean": np.mean(pcr_test_rmses),
        "pcr_rmse_std": np.std(pcr_test_rmses),
        "lasso_nnz_mean": np.mean(lasso_n_nonzero),
        "pcr_k_mean": np.mean(pcr_k_used),
        "lasso_test_rmses": lasso_test_rmses,
        "pcr_test_rmses": pcr_test_rmses,
        "lasso_n_nonzero": lasso_n_nonzero,
        "pcr_k_used": pcr_k_used,
    }

    print(f"    Lasso: RMSE={results['lasso_rmse_mean']:.4f}±{results['lasso_rmse_std']:.4f}, "
          f"非零系数={results['lasso_nnz_mean']:.1f}")
    print(f"    PCR:   RMSE={results['pcr_rmse_mean']:.4f}±{results['pcr_rmse_std']:.4f}, "
          f"k={results['pcr_k_mean']:.1f}")

    return results


def task_c():
    """Task C: 两种场景下 Lasso vs PCR"""
    print("\n" + "=" * 70)
    print("  Task C: Lasso vs PCR — Selection vs Compression")
    print("=" * 70)

    # 场景1: Sparse Truth
    X_sparse, y_sparse, beta_sparse = generate_sparse_truth(
        n=200, p=60, n_informative=5
    )
    res_sparse = compare_lasso_pcr(X_sparse, y_sparse, "Sparse Truth")

    # 场景2: Latent-Factor Truth
    X_latent, y_latent = generate_latent_factor_truth(
        n=200, p=60, n_latent=5
    )
    res_latent = compare_lasso_pcr(X_latent, y_latent, "Latent-Factor Truth")

    # ── 图: 两种场景的对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in zip(
        axes,
        [res_sparse, res_latent],
        ["Sparse Truth", "Latent-Factor Truth"],
    ):
        methods = ["Lasso", "PCR"]
        rmse_means = [res["lasso_rmse_mean"], res["pcr_rmse_mean"]]
        rmse_stds = [res["lasso_rmse_std"], res["pcr_rmse_std"]]
        complexity = [res["lasso_nnz_mean"], res["pcr_k_mean"]]
        complexity_label = ["非零系数数", "主成分数"]

        x = np.arange(2)
        width = 0.35

        bars1 = ax.bar(x - width / 2, rmse_means, width, yerr=rmse_stds,
                       label="Test RMSE", color=["#4C72B0", "#55A868"],
                       alpha=0.8, capsize=5)

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width / 2, complexity, width,
                        label="Model Complexity", color=["#C44E52", "#8172B2"],
                        alpha=0.5, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Test RMSE")
        ax2.set_ylabel("Model Complexity (# components / # nonzero coefs)")
        ax.set_title(f"Scenario: {title}")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task C: Lasso vs PCR Across Two Data Scenarios\n"
                 "(Blue/Green=RMSE, Red/Purple=Model Complexity)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "c_lasso_vs_pcr.png")

    return res_sparse, res_latent


def write_summary_comparison(res_sparse, res_latent):
    """生成 summary_comparison.md"""
    print("\n  撰写 summary_comparison.md ...")

    report = f"""# Summary Comparison — Task C: Selection vs Compression

## 1. 实验结果汇总

### Sparse Truth 场景

| 指标 | Lasso | PCR |
|------|-------|-----|
| Test RMSE (mean±std) | {res_sparse['lasso_rmse_mean']:.4f}±{res_sparse['lasso_rmse_std']:.4f} | {res_sparse['pcr_rmse_mean']:.4f}±{res_sparse['pcr_rmse_std']:.4f} |
| 模型复杂度 | 非零系数 {res_sparse['lasso_nnz_mean']:.1f} | 主成分数 {res_sparse['pcr_k_mean']:.1f} |

### Latent-Factor Truth 场景

| 指标 | Lasso | PCR |
|------|-------|-----|
| Test RMSE (mean±std) | {res_latent['lasso_rmse_mean']:.4f}±{res_latent['lasso_rmse_std']:.4f} | {res_latent['pcr_rmse_mean']:.4f}±{res_latent['pcr_rmse_std']:.4f} |
| 模型复杂度 | 非零系数 {res_latent['lasso_nnz_mean']:.1f} | 主成分数 {res_latent['pcr_k_mean']:.1f} |

---

## 2. 核心问题讨论

### Q1: Sparse Truth 时为什么 Lasso 更自然？

在 Sparse Truth 场景中，只有少数原始变量真正决定 y，其余变量是噪声。
Lasso 的 $L_1$ 正则化天然倾向于产生稀疏解——它会把不重要的变量系数压缩到恰好为 0，
从而实现**变量筛选**（variable selection）。
这与数据的真实生成机制完美匹配：真正重要的变量被保留，噪声变量被剔除。

### Q2: Latent-Factor Truth 时为什么 PCR 更自然？

在 Latent-Factor Truth 场景中，没有哪个原始变量是"真正重要"的——所有变量都是潜在因子的线性组合。
Lasso 试图从 60 个原始变量中选出几个，但这种选择是人为的、不稳定的：
换一个随机种子，可能选出完全不同的变量组合。

PCR 则不同：它不试图挑选变量，而是将所有变量**压缩**到少数几个主成分方向。
这些主成分恰好对应潜在因子，因此 PCR 能够更自然地捕捉数据的真实结构。

### Q3: Lasso 回答"谁留下"，PCR 回答什么？

- **Lasso** 回答的是：**"在原始变量中，哪些应该留下？"** —— 这是变量筛选（selection）；
- **PCR** 回答的是：**"在所有变量张成的空间中，哪些方向最重要？"** —— 这是信息压缩（compression）。

两者的根本区别在于：
- Selection 保留原始变量的可解释性（"X3 和 X7 对 y 有影响"）；
- Compression 放弃原始变量的可解释性，换取更低的维度和更稳定的估计（"前 5 个主成分解释了 90% 的方差"）。

### Q4: 业务方要"更短的变量名单" → 用 Lasso

如果业务方的需求是"告诉我哪些变量最重要，给我一个短名单"，Lasso 更合适。
因为它直接输出稀疏的系数向量，非零系数对应的变量就是"短名单"。

### Q5: 业务方要"更稳的预测器" → 用 PCR

如果业务方的需求是"给我一个稳定的、泛化能力好的预测模型"，PCR 更合适。
因为它通过压缩到低维空间，避免了高维下系数估计的不稳定性，通常能获得更稳健的预测。

### Q6: 为什么本周主线是 Lasso vs PCR，而不是前向/后向选择？

1. **计算效率**：前向/后向选择需要迭代地添加/删除变量，计算成本远高于 Lasso（一步正则化）和 PCR（一步降维）；
2. **稳定性**：前向/后向选择对数据扰动敏感，容易产生不稳定的子集选择结果；
3. **本周主题**：本周的核心是比较 **selection vs compression** 两种思路。Lasso 是 selection 路线的代表（正则化实现稀疏），PCR 是 compression 路线的代表（降维实现压缩）。前向/后向选择虽然也属于 selection 路线，但它不如 Lasso 优雅（没有正则化框架），且不是本周重点。

**如果一定要加前向/后向选择**：它更接近 **selection 路线**。
因为它和 Lasso 一样，目标是从原始变量中挑选一个子集；
不同之处在于，前向/后向选择是通过逐步迭代来实现选择，而 Lasso 是通过 $L_1$ 惩罚一次性实现。
"""

    write_report(os.path.join(RESULTS_DIR, "summary_comparison.md"), report)


# =====================================================================
#  Task D: 真实数据挑战
# =====================================================================

def task_d():
    """Task D: Kaggle House Prices 真实数据"""
    print("\n" + "=" * 70)
    print("  Task D: 真实数据 — Kaggle House Prices")
    print("=" * 70)

    csv_path = os.path.join(DATA_DIR, "train_with_engineered_features.csv")
    if not os.path.exists(csv_path):
        print("  ⚠ 未找到真实数据文件，跳过 Task D")
        return

    df = pd.read_csv(csv_path)
    print(f"  数据形状: {df.shape}")

    target_col = "SalePrice"
    id_col = "Id" if "Id" in df.columns else None

    if id_col:
        df = df.drop(columns=[id_col])

    # 分离目标
    y = df[target_col].values.astype(np.float64)
    df = df.drop(columns=[target_col])

    # 类别列
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 缺失值填补
    for c in df.columns:
        if df[c].isnull().any():
            if c in cat_cols:
                df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "unknown")
            else:
                df[c] = df[c].fillna(df[c].median())

    # One-Hot 编码
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # 转为 float
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df.values.astype(np.float64)
    feature_names = list(df.columns)
    print(f"  特征数（编码后）: {X.shape[1]}")
    print(f"  样本量: {X.shape[0]}")

    # ── 多次切分比较 OLS, Lasso, PCR ──
    n_splits = 20
    ols_rmses, lasso_rmses, pcr_rmses = [], [], []
    ols_maes, lasso_maes, pcr_maes = [], [], []
    lasso_nnz_list, pcr_k_list = [], []

    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )

        # 标准化
        scaler = CustomStandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # OLS
        ols_model, ols_tr_pred, ols_te_pred = ols_fit_predict(X_train_std, y_train, X_test_std)
        ols_rmses.append(calculate_rmse(y_test, ols_te_pred))
        ols_maes.append(calculate_mae(y_test, ols_te_pred))

        # LassoCV
        lasso = LassoCV(cv=5, random_state=i, max_iter=10000)
        lasso.fit(X_train_std, y_train)
        lasso_pred = lasso.predict(X_test_std)
        lasso_rmses.append(calculate_rmse(y_test, lasso_pred))
        lasso_maes.append(calculate_mae(y_test, lasso_pred))
        lasso_nnz_list.append(int(np.sum(np.abs(lasso.coef_) > 1e-6)))

        # PCR — CV 选 k
        best_k, best_cv = 1, np.inf
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        for k in range(1, 31):
            cv_scores = []
            for tr_idx, vl_idx in kf.split(X_train):
                pcr_cv = PCR(n_components=k)
                pcr_cv.fit(X_train_std[tr_idx], y_train[tr_idx])
                cv_scores.append(calculate_rmse(y_train[vl_idx],
                                                pcr_cv.predict(X_train_std[vl_idx])))
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv:
                best_cv = mean_cv
                best_k = k

        pcr = PCR(n_components=best_k)
        pcr.fit(X_train_std, y_train)
        pcr_pred = pcr.predict(X_test_std)
        pcr_rmses.append(calculate_rmse(y_test, pcr_pred))
        pcr_maes.append(calculate_mae(y_test, pcr_pred))
        pcr_k_list.append(best_k)

    # 打印汇总
    print(f"\n  20次切分汇总:")
    print(f"    OLS:   RMSE={np.mean(ols_rmses):.0f}±{np.std(ols_rmses):.0f}, "
          f"MAE={np.mean(ols_maes):.0f}±{np.std(ols_maes):.0f}")
    print(f"    Lasso: RMSE={np.mean(lasso_rmses):.0f}±{np.std(lasso_rmses):.0f}, "
          f"MAE={np.mean(lasso_maes):.0f}±{np.std(lasso_maes):.0f}, "
          f"非零系数={np.mean(lasso_nnz_list):.0f}")
    print(f"    PCR:   RMSE={np.mean(pcr_rmses):.0f}±{np.std(pcr_rmses):.0f}, "
          f"MAE={np.mean(pcr_maes):.0f}±{np.std(pcr_maes):.0f}, "
          f"k={np.mean(pcr_k_list):.0f}")

    # ── 图: 三种模型对比 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = ["OLS", "Lasso", "PCR"]
    rmse_means = [np.mean(ols_rmses), np.mean(lasso_rmses), np.mean(pcr_rmses)]
    rmse_stds = [np.std(ols_rmses), np.std(lasso_rmses), np.std(pcr_rmses)]
    mae_means = [np.mean(ols_maes), np.mean(lasso_maes), np.mean(pcr_maes)]
    mae_stds = [np.std(ols_maes), np.std(lasso_maes), np.std(pcr_maes)]

    x = np.arange(3)
    ax1.bar(x, rmse_means, yerr=rmse_stds, color=["#4C72B0", "#55A868", "#C44E52"],
            alpha=0.8, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Test RMSE")
    ax1.set_title("Task D: Test RMSE Comparison (3 Models)\n(Kaggle House Prices, 20 splits)")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, mae_means, yerr=mae_stds, color=["#4C72B0", "#55A868", "#C44E52"],
            alpha=0.8, capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Test MAE")
    ax2.set_title("Task D: Test MAE Comparison (3 Models)\n(Kaggle House Prices, 20 splits)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "d_real_data_comparison.png")

    # ── 写报告 ──
    write_kaggle_report(
        ols_rmses, ols_maes, lasso_rmses, lasso_maes,
        pcr_rmses, pcr_maes, lasso_nnz_list, pcr_k_list,
        X.shape[1]
    )

    return ols_rmses, lasso_rmses, pcr_rmses


def write_kaggle_report(ols_rmses, ols_maes, lasso_rmses, lasso_maes,
                        pcr_rmses, pcr_maes, lasso_nnz_list, pcr_k_list, n_features):
    """生成 kaggle_report.md"""
    print("\n  撰写 kaggle_report.md ...")

    report = f"""# Kaggle House Prices Report — Task D

## 1. 数据概况

| 项目 | 值 |
|------|-----|
| 数据来源 | Kaggle House Prices（含工程特征） |
| 样本量 | 1458 |
| 原始特征数 | {n_features}（含 One-Hot 编码后） |
| 目标变量 | SalePrice（房价） |

---

## 2. 三种模型表现（20次随机切分）

| 模型 | Test RMSE (mean±std) | Test MAE (mean±std) | 模型复杂度 |
|------|---------------------|---------------------|-----------|
| OLS | {np.mean(ols_rmses):.0f}±{np.std(ols_rmses):.0f} | {np.mean(ols_maes):.0f}±{np.std(ols_maes):.0f} | 全部 {n_features} 个特征 |
| Lasso | {np.mean(lasso_rmses):.0f}±{np.std(lasso_rmses):.0f} | {np.mean(lasso_maes):.0f}±{np.std(lasso_maes):.0f} | 非零系数 {np.mean(lasso_nnz_list):.0f} |
| PCR | {np.mean(pcr_rmses):.0f}±{np.std(pcr_rmses):.0f} | {np.mean(pcr_maes):.0f}±{np.std(pcr_maes):.0f} | 主成分数 {np.mean(pcr_k_list):.0f} |

---

## 3. 分析与讨论

### OLS 是否出现高维/共线性不稳定迹象？

OLS 的 RMSE 标准差较大（±{np.std(ols_rmses):.0f}），说明在不同切分下表现波动明显。
这是因为特征数较多（{n_features} 个），且房屋数据中存在大量相关变量（如面积、房间数、建筑年份等高度相关），
导致 OLS 系数估计不稳定。虽然这里 p < n，但强共线性仍然让 OLS 不够稳健。

### Lasso 与 PCR 谁表现更好？

从 RMSE 来看，两者表现相近，但各有侧重：
- **Lasso** 通过 $L_1$ 正则化自动筛选变量，将有效特征数从 {n_features} 降到约 {np.mean(lasso_nnz_list):.0f} 个，提供了更好的可解释性；
- **PCR** 通过降维到约 {np.mean(pcr_k_list):.0f} 个主成分，获得了类似的预测精度，但牺牲了原始变量的可解释性。

### "筛选还是压缩"？

如果向业务方解释这份数据，我会说：

> 这份房价数据**更适合筛选（selection）**，因为：
> 1. 很多特征确实是房价的直接驱动因素（如 OverallQual、GrLivArea、TotalSF 等），存在明确的"重要变量"；
> 2. 业务方通常关心"哪些因素影响房价"，需要可解释的变量名单；
> 3. Lasso 能够自然地给出这样的名单，同时保持不错的预测精度。
>
> 如果业务方只关心预测精度而不关心解释性，PCR 也是一个可行选择，
> 但考虑到房价数据的特征本身就有明确的业务含义，保留原始变量的可解释性更有价值。
"""

    write_report(os.path.join(RESULTS_DIR, "kaggle_report.md"), report)


# =====================================================================
#  主函数
# =====================================================================

def main():
    print("=" * 70)
    print("  Week 14: High-Dimensional Regression, PCA, and PCR")
    print("=" * 70)

    # ── Task A ──
    X, y, Z, W, gamma = task_a1_save_data()
    a3_results = task_a3_ols_dimension_experiment(X, y)
    a4_result = task_a4_coefficient_instability(X, y)

    # ── Task B ──
    cum_var_ratio, k_90, k_95 = task_b1_pca_analysis(X)
    b2_results = task_b2_pcr_workflow(X, y, cum_var_ratio)

    # 写报告（A + B）
    write_synthetic_report(a3_results, a4_result)
    write_pcr_section_in_report(*b2_results)

    # ── Task C ──
    res_sparse, res_latent = task_c()
    write_summary_comparison(res_sparse, res_latent)

    # ── Task D ──
    task_d()

    print("\n" + "=" * 70)
    print("  全部任务完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
