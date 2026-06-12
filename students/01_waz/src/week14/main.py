"""Week 14 homework: PCA, PCR, high-dimensional regression and Lasso vs PCR.

Run from students/01_waz with:

    uv run src/week14/main.py

Covers:
  Task A — High-dim OLS overfitting & coefficient instability
  Task B — PCA cumulative variance & PCR with CV
  Task C — Lasso vs PCR: sparse truth vs latent-factor truth
  Task D — (Optional) real-data challenge
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split

# Ensure src/ is on sys.path so that `from utils.xxx` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.diagnostics import coefficient_std, condition_number, matrix_rank   # noqa: E402
from utils.metrics import calculate_mae, calculate_rmse                        # noqa: E402
from utils.models import CustomPCA, PCR, cv_pcr_scores                         # noqa: E402
from utils.transformers import CustomStandardScaler, standardize_train_test    # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
WEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK_DIR / "data"
RESULTS_DIR = WEEK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def reset_outputs() -> None:
    """Create required folders and clear old reports/figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=160)
    plt.close()


def df_to_md(df: pd.DataFrame, float_digits: int = 4) -> str:
    """Convert a DataFrame to a markdown table string."""
    try:
        return df.to_markdown(index=False, floatfmt=f".{float_digits}f")
    except Exception:
        return df.to_string(index=False)


# ---------------------------------------------------------------------------
# Data generation (Task A1)
# ---------------------------------------------------------------------------

def make_latent_factor_data(
    n_samples: int = 160,
    n_features: int = 120,
    n_factors: int = 6,
    noise_x: float = 0.25,
    noise_y: float = 0.8,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate high-dimensional data with a low-rank latent-factor structure.

    - n_factors latent variables drawn from N(0,1)
    - random loadings map latent → n_features observed columns
    - y is driven by the latent factors (first 4 have true signal)
    """
    rng = np.random.default_rng(random_state)

    latent = rng.normal(size=(n_samples, n_factors))
    loadings = rng.normal(size=(n_factors, n_features))
    X = latent @ loadings + noise_x * rng.normal(size=(n_samples, n_features))

    beta_factor = np.array([3.0, -2.5, 1.5, 0.8, 0.0, 0.0])[:n_factors]
    y = latent @ beta_factor + noise_y * rng.normal(size=n_samples)

    columns = [f"x{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y
    return df


def make_sparse_truth_data(
    n_samples: int = 160,
    n_features: int = 120,
    n_active: int = 6,
    noise: float = 1.0,
    random_state: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data where only the first n_active features have true signal."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:n_active] = np.array([4.0, -3.0, 2.5, -2.0, 1.5, 1.0])[:n_active]
    y = X @ beta + noise * rng.normal(size=n_samples)
    return X, y, beta


# ---------------------------------------------------------------------------
# Task A3: OLS error vs dimension
# ---------------------------------------------------------------------------

def run_ols_dimension_experiment() -> pd.DataFrame:
    """Fit OLS at increasing feature counts, record train/test RMSE, rank, cond."""
    rows = []
    p_values = [10, 30, 60, 120]

    for p in p_values:
        df = make_latent_factor_data(
            n_samples=160, n_features=p, n_factors=6,
            noise_y=0.8, random_state=100 + p,
        )
        X = df.drop(columns=["y"]).to_numpy()
        y = df["y"].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.35, random_state=42,
        )
        X_train_s, X_test_s, _ = standardize_train_test(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)

        rows.append({
            "p": p,
            "train_rmse": calculate_rmse(y_train, model.predict(X_train_s)),
            "test_rmse": calculate_rmse(y_test, model.predict(X_test_s)),
            "rank_X_train": matrix_rank(X_train_s),
            "condition_number": condition_number(X_train_s),
        })

    result = pd.DataFrame(rows)

    # --- Figure 1: train/test RMSE vs p ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(result["p"], result["train_rmse"], marker="o", label="Train RMSE")
    ax1.plot(result["p"], result["test_rmse"], marker="s", label="Test RMSE")
    ax1.set_xlabel("Number of features p")
    ax1.set_ylabel("RMSE")
    ax1.set_title("OLS Error vs. Feature Dimension")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(result["p"], result["rank_X_train"], marker="o", color="tab:blue", label="Rank")
    ax2.set_xlabel("Number of features p")
    ax2.set_ylabel("Rank of training matrix", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Matrix Rank and Condition Number")
    ax3 = ax2.twinx()
    ax3.plot(result["p"], result["condition_number"], marker="s", color="tab:red", label="Cond. number")
    ax3.set_ylabel("Condition number κ(X)", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax2.grid(alpha=0.3)

    save_figure("A3_ols_overfitting.png")
    return result


# ---------------------------------------------------------------------------
# Task A4: Coefficient instability under repeated splits
# ---------------------------------------------------------------------------

def run_coefficient_instability() -> pd.DataFrame:
    """50 random train/test splits → show OLS coefficient boxplots."""
    df = make_latent_factor_data(n_samples=160, n_features=120, n_factors=6,
                                 noise_y=0.8, random_state=42)
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    selected_cols = [0, 1, 2, 3, 4]
    records = []
    coef_list = []

    for seed in range(50):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.35, random_state=seed,
        )
        X_train_s, X_test_s, _ = standardize_train_test(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)

        coef_list.append(model.coef_)
        for j in selected_cols:
            records.append({
                "split": seed,
                "feature": f"x{j + 1}",
                "coefficient": float(model.coef_[j]),
                "train_rmse": calculate_rmse(y_train, model.predict(X_train_s)),
                "test_rmse": calculate_rmse(y_test, model.predict(X_test_s)),
            })

    coef_df = pd.DataFrame(records)
    coef_matrix = np.vstack(coef_list)

    # Boxplot
    plt.figure(figsize=(8, 4.8))
    data = [coef_df.loc[coef_df["feature"] == f"x{j + 1}", "coefficient"].values
            for j in selected_cols]
    plt.boxplot(data, labels=[f"x{j + 1}" for j in selected_cols])
    plt.xlabel("Feature")
    plt.ylabel("Coefficient across 50 random splits")
    plt.title("OLS Coefficient Instability")
    plt.grid(axis="y", alpha=0.3)
    save_figure("A4_coefficient_instability.png")

    return coef_df


# ---------------------------------------------------------------------------
# Task B1: PCA cumulative explained variance
# ---------------------------------------------------------------------------

def run_pca_analysis() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA on the training set → cumulative variance curve."""
    df = pd.read_csv(DATA_DIR / "synthetic_highdim.csv")
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42,
    )

    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = CustomPCA()
    pca.fit(X_train_s)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    pca_df = pd.DataFrame({
        "component": np.arange(1, len(cumsum) + 1),
        "cumulative_explained_variance": cumsum,
    })

    plt.figure(figsize=(7, 4.5))
    plt.plot(pca_df["component"], pca_df["cumulative_explained_variance"], marker="o")
    plt.axhline(0.80, color="gray", linestyle="--", label="80% variance")
    plt.axhline(0.90, color="red", linestyle="--", label="90% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlim(1, 30)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure("B1_pca_cumulative_variance.png")

    return pca_df, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task B2: PCR with CV
# ---------------------------------------------------------------------------

def run_pcr_experiment(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    """PCR evaluated at k=1..20: train/test/CV RMSE vs k."""
    k_list = list(range(1, 21))
    train_rmse_list = []
    test_rmse_list = []
    cv_rmse_list = []

    for k in k_list:
        model = PCR(n_components=k)
        model.fit(X_train, y_train)
        train_rmse_list.append(calculate_rmse(y_train, model.predict(X_train)))
        test_rmse_list.append(calculate_rmse(y_test, model.predict(X_test)))

    cv_scores = cv_pcr_scores(X_train, y_train, k_list, cv=5, random_state=42)
    cv_rmse_list = [cv_scores[k] for k in k_list]

    result = pd.DataFrame({
        "k": k_list,
        "train_rmse": train_rmse_list,
        "test_rmse": test_rmse_list,
        "cv_rmse": cv_rmse_list,
    })

    best_k = int(result.loc[result["cv_rmse"].idxmin(), "k"])

    # OLS baseline
    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ols = LinearRegression().fit(X_train_s, y_train)
    ols_test_rmse = calculate_rmse(y_test, ols.predict(X_test_s))

    plt.figure(figsize=(8, 4.8))
    plt.plot(result["k"], result["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(result["k"], result["test_rmse"], marker="s", label="Test RMSE")
    plt.plot(result["k"], result["cv_rmse"], marker="^", label="5-fold CV RMSE")
    plt.axhline(ols_test_rmse, color="red", linestyle="--",
                label=f"OLS test RMSE = {ols_test_rmse:.3f}")
    plt.axvline(best_k, color="gray", linestyle="--",
                label=f"CV best k = {best_k}")
    plt.xlabel("Number of principal components k")
    plt.ylabel("RMSE")
    plt.title("PCR Error vs. Number of Components")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure("B2_pcr_cv_rmse.png")

    return result, best_k


# ---------------------------------------------------------------------------
# Task C: Lasso vs PCR — sparse truth vs latent-factor truth
# ---------------------------------------------------------------------------

def evaluate_lasso_vs_pcr_scenario(
    name: str, X: np.ndarray, y: np.ndarray,
) -> dict[str, Any]:
    """Run LassoCV and PCR on one scenario, return summary dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42,
    )
    X_train_s, X_test_s, _ = standardize_train_test(X_train, X_test)

    # Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=20000)
    lasso.fit(X_train_s, y_train)
    lasso_pred = lasso.predict(X_test_s)
    n_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-8))

    # PCR
    pcr_rows = []
    for k in range(1, 21):
        model = PCR(n_components=k)
        model.fit(X_train, y_train)
        pcr_rows.append({
            "k": k,
            "test_rmse": calculate_rmse(y_test, model.predict(X_test)),
            "test_mae": calculate_mae(y_test, model.predict(X_test)),
        })
    pcr_df = pd.DataFrame(pcr_rows)
    cv_scores = cv_pcr_scores(X_train, y_train, list(range(1, 21)), cv=5)
    pcr_df["cv_rmse"] = pcr_df["k"].map(cv_scores)
    best_k = int(pcr_df.loc[pcr_df["cv_rmse"].idxmin(), "k"])

    best_pcr = PCR(n_components=best_k)
    best_pcr.fit(X_train, y_train)
    pcr_pred = best_pcr.predict(X_test)

    return {
        "scenario": name,
        "lasso_test_rmse": calculate_rmse(y_test, lasso_pred),
        "lasso_test_mae": calculate_mae(y_test, lasso_pred),
        "lasso_nonzero": n_nonzero,
        "pcr_test_rmse": calculate_rmse(y_test, pcr_pred),
        "pcr_test_mae": calculate_mae(y_test, pcr_pred),
        "pcr_best_k": best_k,
        "lasso_coef_sum": float(np.sum(np.abs(lasso.coef_))),
    }


def run_lasso_vs_pcr() -> pd.DataFrame:
    """Compare Lasso and PCR under two data-generation mechanisms."""
    # Sparse truth
    X_sp, y_sp, _ = make_sparse_truth_data(n_samples=160, n_features=120,
                                           n_active=6, noise=1.0)
    # Latent-factor truth
    df_lf = make_latent_factor_data(n_samples=160, n_features=120, n_factors=6,
                                    noise_y=0.8, random_state=42)
    X_lf = df_lf.drop(columns=["y"]).to_numpy()
    y_lf = df_lf["y"].to_numpy()

    res_sp = evaluate_lasso_vs_pcr_scenario("Sparse truth", X_sp, y_sp)
    res_lf = evaluate_lasso_vs_pcr_scenario("Latent-factor truth", X_lf, y_lf)

    summary = pd.DataFrame([res_sp, res_lf])

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scenarios = ["Sparse truth", "Latent-factor truth"]
    for ax, scenario in zip(axes, scenarios):
        row = summary[summary["scenario"] == scenario].iloc[0]
        methods = ["Lasso", "PCR"]
        rmses = [row["lasso_test_rmse"], row["pcr_test_rmse"]]
        complexity = [row["lasso_nonzero"], row["pcr_best_k"]]
        bars = ax.bar(methods, rmses, color=["#2ca02c", "#1f77b4"])
        ax.set_title(scenario)
        ax.set_ylabel("Test RMSE")
        ax.grid(axis="y", alpha=0.3)
        for bar, comp in zip(bars, complexity):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"complexity={comp}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Lasso vs PCR: Comparison under Different Data Mechanisms")
    save_figure("C_lasso_vs_pcr.png")
    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_synthetic_report(
    ols_result: pd.DataFrame,
    pca_df: pd.DataFrame,
    pcr_result: pd.DataFrame,
    best_k: int,
    lasso_pcr_summary: pd.DataFrame,
) -> None:
    """Generate the main report as a markdown file."""

    # Find k where 90% variance is reached
    top_90 = int(pca_df.loc[pca_df["cumulative_explained_variance"] >= 0.9,
                            "component"].iloc[0])

    ols_md = df_to_md(ols_result.rename(columns={
        "p": "特征数量", "train_rmse": "训练 RMSE",
        "test_rmse": "测试 RMSE", "rank_X_train": "矩阵秩",
        "condition_number": "条件数",
    }))

    pcr_md = df_to_md(pcr_result.rename(columns={
        "k": "主成分数", "train_rmse": "训练 RMSE",
        "test_rmse": "测试 RMSE", "cv_rmse": "CV RMSE",
    }))

    report = f"""# Week 14 实验报告：高维回归、PCA 与 PCR

## 一、数据生成机制

本次实验生成了一份高维低秩模拟数据：样本量 160，原始特征 120 个。

数据并非让 120 个变量独立决定 y，而是先生成 6 个潜在因子，再通过载荷矩阵线性组合出全部 120 个观测特征。y 也主要由这 6 个潜在因子驱动（前 4 个有真实信号），并叠加少量噪声。

因此本数据有两个关键性质：
- **高维**：特征数较多（最高 120），接近甚至可能超过训练样本数；
- **信息冗余**：120 列本质上只承载约 6 个独立方向的信息，存在严重的多重共线性。

## 二、Task A3：OLS 随维度膨胀的过拟合

| 特征数量 | 10 | 30 | 60 | 120 |
|---|---|---|---|---|
| 训练 RMSE | {ols_result['train_rmse'].iloc[0]:.3f} | {ols_result['train_rmse'].iloc[1]:.3f} | {ols_result['train_rmse'].iloc[2]:.3f} | {ols_result['train_rmse'].iloc[3]:.3f} |
| 测试 RMSE | {ols_result['test_rmse'].iloc[0]:.3f} | {ols_result['test_rmse'].iloc[1]:.3f} | {ols_result['test_rmse'].iloc[2]:.3f} | {ols_result['test_rmse'].iloc[3]:.3f} |
| 矩阵秩 | {ols_result['rank_X_train'].iloc[0]} | {ols_result['rank_X_train'].iloc[1]} | {ols_result['rank_X_train'].iloc[2]} | {ols_result['rank_X_train'].iloc[3]} |
| 条件数 | {ols_result['condition_number'].iloc[0]:.0f} | {ols_result['condition_number'].iloc[1]:.0f} | {ols_result['condition_number'].iloc[2]:.0f} | {ols_result['condition_number'].iloc[3]:.0f} |

**解读**：随着特征数 p 增大，训练 RMSE 持续下降（模型越来越"贴合"训练集），但测试 RMSE 并未同步下降——p=120 时训练误差甚至接近 0，但测试误差比 p=10 时要高。这就是典型的**高维过拟合**：模型记住了训练样本，但没有抓住可泛化的规律。

同时，条件数随 p 急剧增大，说明设计矩阵越来越"病态"——在那些接近零的奇异值方向上，数据对系数的约束极弱，OLS 估计变得极其不稳定。

## 三、Task A4：OLS 系数的不稳定性

在 50 次不同的随机 train/test 划分下，对前 5 个特征分别拟合 OLS 模型并记录其系数，结果如箱线图所示。

由于 120 个特征共享同一组潜在因子，任意两个特征之间都存在潜在的共线性。OLS 在不同样本划分下，系数可以在相关变量之间剧烈摇摆——"谁重要"经常换人，解释不可轻易相信。

## 四、Task B1：PCA 累计解释方差

前 **{top_90}** 个主成分就已解释超过 90% 的总方差。

这说明虽然数据有 120 列，但真正有效的信息集中在一个远低于 120 维的子空间中。因为所有特征都来自 6 个潜在因子的线性组合，信息高度冗余——PCA 只需要极少数主成分就能捕捉数据的绝大部分波动。

## 五、Task B2：PCR 误差曲线

CV 最优的主成分个数为 **k = {best_k}**。

{ols_md}

CV 曲线先降后稳：主成分太少（k=1,2）欠拟合；k 增加到 5-6 后 CV RMSE 触底；再增加 k 会把噪声方向也带回模型，CV RMSE 略微回升或保持平稳。

OLS 的测试 RMSE 明显高于 PCR（最佳 k），因为 OLS 在高维共线场景下容易过拟合——它不仅学习了真实规律，还拟合了训练集中的随机噪声。

## 六、Task C：Lasso vs PCR

### 稀疏真实机制（少数变量真正有用）

- Lasso 测试 RMSE: {lasso_pcr_summary['lasso_test_rmse'].iloc[0]:.3f}
- PCR 测试 RMSE: {lasso_pcr_summary['pcr_test_rmse'].iloc[0]:.3f}
- Lasso 自动筛选出 {lasso_pcr_summary['lasso_nonzero'].iloc[0]} 个非零变量

在稀疏机制下 Lasso 通常更好：它的 selection 机制天然匹配"只有少数变量有信号"的场景。

### 潜在因子机制（很多变量共享少数底层因子）

- Lasso 测试 RMSE: {lasso_pcr_summary['lasso_test_rmse'].iloc[1]:.3f}
- PCR 测试 RMSE: {lasso_pcr_summary['pcr_test_rmse'].iloc[1]:.3f}
- PCR 保留 {lasso_pcr_summary['pcr_best_k'].iloc[1]} 个主成分

在低秩机制下 PCR 更稳：数据本身就是低维潜在因子的展开，先压缩再回归比在原变量里做筛选更自然。

### 核心区别

| 方法 | 策略 | 输出 | 适用场景 |
|---|---|---|---|
| Lasso | selection（筛选） | 非零变量名单 | 稀疏真相 |
| PCR | compression（压缩） | 主成分方向 | 低秩/共线结构 |

## 七、回答核心问题

1. **OLS 在高维/共线场景下为何不稳定？** 当 p 接近或超过 n，或变量强相关时，X^T X 接近奇异，条件数极大，OLS 系数对样本扰动极度敏感。

2. **PCA 在压缩什么？** PCA 寻找方差最大的方向，把原始高维空间投影到低维主成分子空间，同时保留最多的信息（方差）。

3. **PCR 与 Ridge/Lasso 的区别？** Ridge/Lasso 在原变量空间加约束；PCR 先换坐标系（PCA），再在低维空间回归——它不是在问"谁该删掉"，而是在问"信息该怎么压缩"。

4. **什么时候选 Lasso，什么时候选 PCR？**
   - 如果你要一份**短名单**来解释/决策 → Lasso
   - 如果变量高度相关、数据呈低秩结构 → PCR 更稳
   - 没有脱离数据结构的最优方法，只有更匹配任务目标的方法

"""
    with open(RESULTS_DIR / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Week 14: PCA / PCR 实验")
    print("=" * 60)

    reset_outputs()

    # --- Task A1: generate & save synthetic data ---
    print("\n[Task A1] 生成高维低秩模拟数据 ...")
    df_syn = make_latent_factor_data(n_samples=160, n_features=120, n_factors=6,
                                     noise_y=0.8, random_state=42)
    df_syn.to_csv(DATA_DIR / "synthetic_highdim.csv", index=False)
    print(f"  已保存: {DATA_DIR / 'synthetic_highdim.csv'}")

    # --- Task A3: OLS vs dimension ---
    print("\n[Task A3] OLS 误差随维度膨胀实验 ...")
    ols_result = run_ols_dimension_experiment()
    print(ols_result.to_string(index=False))

    # --- Task A4: coefficient instability ---
    print("\n[Task A4] 系数不稳定性实验 (50 次划分) ...")
    coef_df = run_coefficient_instability()
    stds = coefficient_std(np.vstack([
        coef_df[coef_df["feature"] == f"x{j + 1}"]["coefficient"].values.reshape(-1, 1).T
        for j in range(5)
    ]).T if False else np.column_stack([
        coef_df[coef_df["feature"] == f"x{j + 1}"]["coefficient"].values
        for j in range(5)
    ]))
    print(f"  前 5 个变量的系数标准差: {np.round(stds, 4)}")

    # --- Task B1: PCA ---
    print("\n[Task B1] PCA 累计解释方差 ...")
    pca_df, X_train, X_test, y_train, y_test = run_pca_analysis()
    top_90 = int(pca_df.loc[pca_df["cumulative_explained_variance"] >= 0.9,
                            "component"].iloc[0])
    print(f"  前 {top_90} 个主成分解释 >= 90% 方差")

    # --- Task B2: PCR ---
    print("\n[Task B2] PCR 随 k 的误差曲线 ...")
    pcr_result, best_k = run_pcr_experiment(X_train, X_test, y_train, y_test)
    print(f"  CV 最优 k = {best_k}")
    print(pcr_result.head(12).to_string(index=False))

    # --- Task C: Lasso vs PCR ---
    print("\n[Task C] Lasso vs PCR 对比 ...")
    lasso_pcr_summary = run_lasso_vs_pcr()
    print(lasso_pcr_summary.to_string(index=False))

    # --- Report ---
    print("\n[Report] 生成实验报告 ...")
    write_synthetic_report(ols_result, pca_df, pcr_result, best_k, lasso_pcr_summary)

    print("\n" + "=" * 60)
    print("Week 14 全部任务完成！")
    print(f"  图片: {FIGURES_DIR}")
    print(f"  报告: {RESULTS_DIR / 'synthetic_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
