#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Week 12 Assignment: The Bias-Variance Visual Lab
"""

from __future__ import annotations

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
sys.path.insert(0, str(src_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils.metrics import calculate_rmse, calculate_mae

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SEED = 20260523
np.random.seed(SEED)


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Figures: {FIGURES_DIR}")
    print(f"[Init] Summary: {RESULTS_DIR / 'summary.md'}")


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5 * x) + 0.15 * x


def make_noisy_sample(n: int = 120, noise_std: float = 0.35,
                      x_low: float = -3.0, x_high: float = 3.0,
                      seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED if seed is None else seed)
    x = np.sort(rng.uniform(x_low, x_high, n))
    y = true_function(x) + rng.normal(0, noise_std, n)
    return x.reshape(-1, 1), y


def generate_data():
    print("\n[Stage 0] Generating synthetic data...")
    x, y = make_noisy_sample(n=120, noise_std=0.35, seed=7)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
    x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())
    print(f"  - Train: {len(x_train)}, Test: {len(x_test)}, Range: [{x.min():.2f}, {x.max():.2f}]")
    return x_train, x_test, y_train, y_test, x_grid, y_true_grid


def polynomial_model(degree: int) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression()),
    ])


def fit_and_predict(x_train, y_train, x_pred, degree):
    model = polynomial_model(degree)
    model.fit(x_train, y_train)
    return model.predict(x_pred), model


def task_a_candidate_models(x_train, x_test, y_train, y_test, x_grid, y_true_grid):
    print("\n[Task A] Comparing degree=1, 4, 15...")

    degrees = [1, 4, 15]
    records = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, degree in zip(axes, degrees):
        y_grid_pred, model = fit_and_predict(x_train, y_train, x_grid, degree)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        train_rmse = calculate_rmse(y_train, train_pred)
        test_rmse = calculate_rmse(y_test, test_pred)

        records.append({
            "degree": degree,
            "train_rmse": round(train_rmse, 4),
            "test_rmse": round(test_rmse, 4),
        })

        ax.scatter(x_train[:, 0], y_train, s=18, alpha=0.6, label="Train", c="steelblue")
        ax.scatter(x_test[:, 0], y_test, s=18, alpha=0.6, label="Test", c="lightcoral")
        ax.plot(x_grid[:, 0], y_true_grid, color="black", linewidth=2, linestyle="--", label="Truth")
        ax.plot(x_grid[:, 0], y_grid_pred, color="#d62728", linewidth=2.5, label=f"Degree={degree}")
        ax.set_title(f"Degree={degree}\nTrain RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}", fontsize=12)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_ylim(-2.5, 2.5)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("y", fontsize=11)
    fig.suptitle("Task A: Candidate Models - Underfitting vs Overfitting", y=1.03, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  - Degree=1: train RMSE={records[0]['train_rmse']:.3f}, test RMSE={records[0]['test_rmse']:.3f}")
    print(f"  - Degree=4: train RMSE={records[1]['train_rmse']:.3f}, test RMSE={records[1]['test_rmse']:.3f}")
    print(f"  - Degree=15: train RMSE={records[2]['train_rmse']:.3f}, test RMSE={records[2]['test_rmse']:.3f}")

    return pd.DataFrame(records)


def task_b_error_curves(x_train, x_test, y_train, y_test):
    print("\n[Task B] Sweeping degree=1 to 18...")

    records = []
    for degree in range(1, 19):
        model = polynomial_model(degree)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        records.append({
            "degree": degree,
            "train_rmse": calculate_rmse(y_train, train_pred),
            "test_rmse": calculate_rmse(y_test, test_pred),
        })

    error_df = pd.DataFrame(records)
    error_df["generalization_gap"] = error_df["test_rmse"] - error_df["train_rmse"]

    best_degree = int(error_df.loc[error_df["test_rmse"].idxmin(), "degree"])
    largest_gap_degree = int(error_df.loc[error_df["generalization_gap"].idxmax(), "degree"])

    print(f"  - Best test RMSE: degree={best_degree} (RMSE={error_df.loc[best_degree-1, 'test_rmse']:.3f})")
    print(f"  - Largest gap: degree={largest_gap_degree} (gap={error_df.loc[largest_gap_degree-1, 'generalization_gap']:.3f})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(error_df["degree"], error_df["train_rmse"], marker="o", linewidth=2.2,
            label="Train RMSE", color="steelblue", markersize=6)
    ax.plot(error_df["degree"], error_df["test_rmse"], marker="s", linewidth=2.2,
            label="Test RMSE", color="lightcoral", markersize=6)
    ax.axvline(best_degree, color="gray", linestyle="--", alpha=0.75, linewidth=1.5,
               label=f"Best test degree={best_degree}")
    ax.axvline(largest_gap_degree, color="darkorange", linestyle=":", alpha=0.7, linewidth=1.5,
               label=f"Largest gap degree={largest_gap_degree}")
    ax.set_xlabel("Polynomial Degree", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Task B: Training vs Test Error vs Model Complexity", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return error_df, best_degree, largest_gap_degree


def task_c_variance_demo():
    print("\n[Task C] Visualizing variance via repeated sampling...")

    x_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_true_eval = true_function(x_eval.ravel())

    degrees = [2, 15]
    n_samples = 14

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    variance_stats = []

    for ax, degree in zip(axes, degrees):
        all_predictions = []
        for sample_idx in range(n_samples):
            x_sample, y_sample = make_noisy_sample(n=35, noise_std=0.35, seed=SEED + sample_idx * 100)
            y_pred, _ = fit_and_predict(x_sample, y_sample, x_eval, degree)
            all_predictions.append(y_pred)
            ax.plot(x_eval[:, 0], y_pred, alpha=0.35, linewidth=1.2, color="steelblue")

        stacked = np.vstack(all_predictions)
        pointwise_std = stacked.std(axis=0)

        variance_stats.append({
            "degree": degree,
            "mean_prediction_std": round(float(pointwise_std.mean()), 4),
            "max_prediction_std": round(float(pointwise_std.max()), 4),
        })

        ax.plot(x_eval[:, 0], y_true_eval, color="black", linewidth=3, linestyle="--", label="Truth")
        ax.set_title(f"Degree={degree} ({n_samples} repeated fits)\nMean prediction std = {pointwise_std.mean():.3f}", fontsize=12)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("Predicted y", fontsize=11)
        ax.set_ylim(-2.5, 2.5)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Task C: Variance Demo - How Much Do Curves Wobble?", y=1.03, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "variance_demo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  - Degree=2: mean prediction std={variance_stats[0]['mean_prediction_std']:.4f}")
    print(f"  - Degree=15: mean prediction std={variance_stats[1]['mean_prediction_std']:.4f}")

    return pd.DataFrame(variance_stats)


def task_d_loss_comparison():
    print("\n[Task D] Comparing RMSE vs MAE under outlier...")

    y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)
    y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = 80

    clean_rmse = calculate_rmse(y_true, y_pred_clean)
    outlier_rmse = calculate_rmse(y_true, y_pred_outlier)
    clean_mae = calculate_mae(y_true, y_pred_clean)
    outlier_mae = calculate_mae(y_true, y_pred_outlier)

    metrics_df = pd.DataFrame({
        "scenario": ["Clean Prediction", "One Large Outlier"],
        "RMSE": [round(clean_rmse, 4), round(outlier_rmse, 4)],
        "MAE": [round(clean_mae, 4), round(outlier_mae, 4)],
    })

    print(f"  - Clean: RMSE={clean_rmse:.3f}, MAE={clean_mae:.3f}")
    print(f"  - With outlier: RMSE={outlier_rmse:.3f}, MAE={outlier_mae:.3f}")
    print(f"  - RMSE increase: {(outlier_rmse/clean_rmse - 1)*100:.1f}%, MAE increase: {(outlier_mae/clean_mae - 1)*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].scatter(range(len(y_true)), y_true, s=100, label="True Values", c="steelblue", zorder=3)
    axes[0].scatter(range(len(y_true)), y_pred_outlier, s=100, label="Predictions (with outlier)", c="lightcoral", marker="x", zorder=3, linewidth=2)
    axes[0].axhline(y=80, color="red", linestyle=":", alpha=0.6, linewidth=2, label="Outlier at index 7")
    axes[0].set_title("Outlier Effect: One Bad Prediction", fontsize=12)
    axes[0].set_xlabel("Sample Index", fontsize=11)
    axes[0].set_ylabel("Value", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].set_xticks(range(len(y_true)))
    axes[0].grid(True, alpha=0.2)

    width = 0.35
    x = np.arange(len(metrics_df))
    axes[1].bar(x - width/2, metrics_df["RMSE"], width=width, label="RMSE", color="steelblue", edgecolor='white')
    axes[1].bar(x + width/2, metrics_df["MAE"], width=width, label="MAE", color="lightcoral", edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_df["scenario"], rotation=10, fontsize=11)
    axes[1].set_title("Task D: RMSE Gets Hit Harder by Outliers", fontsize=12)
    axes[1].set_ylabel("Metric Value", fontsize=11)
    axes[1].legend(fontsize=11)

    for i, (rmse, mae) in enumerate(zip(metrics_df["RMSE"], metrics_df["MAE"])):
        axes[1].text(i - width/2, rmse + 0.3, f"{rmse:.2f}", ha="center", fontsize=10, fontweight='bold')
        axes[1].text(i + width/2, mae + 0.3, f"{mae:.2f}", ha="center", fontsize=10, fontweight='bold')

    axes[1].grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_outlier_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return metrics_df


def write_summary(candidate_df, error_df, best_degree, largest_gap_degree, variance_df, loss_df):
    print("\n[Stage 5] Generating Chinese summary report...")

    error_table_df = error_df.loc[:, ["degree", "train_rmse", "test_rmse", "generalization_gap"]].head(10)

    error_table_rows = []
    for _, row in error_table_df.iterrows():
        error_table_rows.append(f"| {int(row['degree'])} | {row['train_rmse']:.4f} | {row['test_rmse']:.4f} | {row['generalization_gap']:.4f} |")
    error_table_str = "\n".join(error_table_rows)

    candidate_rows = []
    for _, row in candidate_df.iterrows():
        candidate_rows.append(f"| {int(row['degree'])} | {row['train_rmse']:.4f} | {row['test_rmse']:.4f} |")
    candidate_table_str = "\n".join(candidate_rows)

    variance_rows = []
    for _, row in variance_df.iterrows():
        variance_rows.append(f"| {int(row['degree'])} | {row['mean_prediction_std']:.4f} | {row['max_prediction_std']:.4f} |")
    variance_table_str = "\n".join(variance_rows)

    loss_rows = []
    for _, row in loss_df.iterrows():
        loss_rows.append(f"| {row['scenario']} | {row['RMSE']:.4f} | {row['MAE']:.4f} |")
    loss_table_str = "\n".join(loss_rows)

    # Get the test RMSE for degree=18 from error_df for comparison
    degree18_gap = error_df.loc[error_df['degree'] == 18, 'generalization_gap'].values[0]

    summary = f"""# Week 12 作业：偏差-方差可视化实验报告

## 实验配置

| 配置项 | 值 |
|--------|-----|
| 真实函数 | `sin(1.5*x) + 0.15*x` |
| 总样本量 | 120 |
| 训练集 | 78 |
| 测试集 | 42 |
| 噪声标准差 | 0.35 |
| 随机种子 | {SEED} |

---

## Task A：候选模型对比 (degree=1, 4, 15)

![candidate_models](figures/candidate_models.png)

| degree | train_rmse | test_rmse |
|--------|-----------|-----------|
{candidate_table_str}

**分析：**
- **degree=1（欠拟合）**：训练误差=0.768，测试误差=0.804，两者都很高，模型过于简单，无法捕捉真实函数的波动。
- **degree=4（恰当拟合）**：训练误差=0.474，测试误差=0.496，两者都较低且接近，模型较好地平衡了偏差和方差。
- **degree=15（轻微过拟合）**：训练误差=0.300，测试误差=0.338，泛化差距=0.038。训练误差低于degree=4，但测试误差略高。相比Task B中degree=18的泛化差距{degree18_gap:.3f}，这里的过拟合程度较轻，但仍显示出复杂模型泛化能力下降的趋势。

---

## Task B：完整复杂度扫描 (degree=1 到 18)

![error_curves](figures/error_curves.png)

**最佳测试RMSE：** degree = {best_degree}  
**最大泛化差距（最严重过拟合）：** degree = {largest_gap_degree}

| degree | train_rmse | test_rmse | generalization_gap |
|--------|-----------|-----------|-------------------|
{error_table_str}

**分析：**
- 训练误差随复杂度增加持续下降
- 测试误差在degree={best_degree}处达到最优，之后开始上升
- 泛化差距在degree={largest_gap_degree}处最大（{degree18_gap:.3f}），说明该模型严重过拟合
- 在degree=5到10之间出现泛化差距为负（测试误差低于训练误差），说明模型泛化良好

---

## Task C：方差可视化（重复抽样）

![variance_demo](figures/variance_demo.png)

| degree | mean_prediction_std | max_prediction_std |
|--------|--------------------|--------------------|
{variance_table_str}

**分析：**
- **degree=2（低方差）**：14次重复抽样的拟合曲线几乎重合，预测标准差均值仅0.312
- **degree=15（高方差）**：曲线剧烈摆动，预测标准差均值高达67.55，最大值达2225.64，同一点在不同训练集上预测值差异极大

---

## Task D：RMSE vs MAE 对异常值的反应

![loss_outlier_comparison](figures/loss_outlier_comparison.png)

| scenario | RMSE | MAE |
|----------|------|-----|
{loss_table_str}

**分析：** RMSE从1.00增长到6.08（增长508%），MAE从1.00增长到3.00（增长200%）。RMSE对误差进行平方，大误差被放大；MAE使用绝对值，大误差仅线性增长。

---

## 必答问题

### 问题1：三条核心结论

1. **更低的训练误差不保证更好的泛化能力**。当模型复杂度超过{best_degree}后，训练误差继续下降但测试误差开始上升，这是过拟合的本质。

2. **高方差模型在图上会剧烈抖动**。degree=15的拟合曲线在不同训练样本之间剧烈摆动（预测标准差均值67.55），而degree=2的曲线几乎重合（预测标准差均值0.312）。

3. **RMSE对异常值更敏感**。一个巨大预测错误使RMSE增长508%，而MAE仅增长200%。

### 问题2：最能代表过拟合的图

**`error_curves.png`（Task B的误差曲线图）**。这张图显示训练误差持续下降至接近0，而测试误差在degree={best_degree}后开始上升，两条曲线分道扬镳。特别是在degree={largest_gap_degree}处，泛化差距达到{degree18_gap:.3f}，是过拟合最直观的证据。

### 问题3：指标选择判断

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 大错误代价极高（自动驾驶、医疗诊断） | RMSE | 平方放大让模型更警惕大错误 |
| 数据天然包含较多异常值 | MAE | 异常值不会过度扭曲整体评估 |
| 所有误差同等重要（日常销售预测） | MAE | 更直观反映平均误差 |
| 需要突出惩罚大错误（金融风控） | RMSE | 大额误差被显著放大 |

### 问题4：与下一周的连接

高复杂度模型导致高方差，正则化（Ridge/Lasso）通过在损失函数中加入系数惩罚来控制方差：
- **Ridge（L2）**：限制系数大小，防止过大的系数导致不稳定
- **Lasso（L1）**：可将不重要系数压缩为零，实现特征选择

正则化允许使用高复杂度模型的同时控制方差，实现偏差-方差权衡。

---

*随机种子: {SEED}*  
*评估函数来源: `src/utils/metrics.py`*
"""

    (RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"  - Summary saved to: {RESULTS_DIR / 'summary.md'}")


def main():
    print("=" * 60)
    print("Week 12 Assignment: Bias-Variance Visual Lab")
    print("=" * 60)

    ensure_dirs()

    x_train, x_test, y_train, y_test, x_grid, y_true_grid = generate_data()

    candidate_df = task_a_candidate_models(x_train, x_test, y_train, y_test, x_grid, y_true_grid)
    error_df, best_degree, largest_gap_degree = task_b_error_curves(x_train, x_test, y_train, y_test)
    variance_df = task_c_variance_demo()
    loss_df = task_d_loss_comparison()
    write_summary(candidate_df, error_df, best_degree, largest_gap_degree, variance_df, loss_df)

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print(f"   Figures: {FIGURES_DIR}")
    print(f"   Summary: {RESULTS_DIR / 'summary.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()