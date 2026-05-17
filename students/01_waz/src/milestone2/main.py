"""
milestone2/main.py — 第二阶段里程碑大作业
===========================================
"工业流水线与无泄漏的泛化评估"

单一执行入口:
    uv run src/milestone2/main.py

功能:
    Task 3: 危险诱惑 — 全量预处理 + 5-Fold CV（存在数据泄露）
    Task 4: 无泄漏流水线 — 逐 Fold Pipeline（杜绝数据泄露）
    Task 5: 自动生成对比报告 & 可视化图表
"""

import shutil
import sys
from pathlib import Path

# ── 路径设置：确保可以 import utils ──────────────────────────────
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头模式，无需 GUI
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomImputer, CustomStandardScaler
from utils.models import GradientDescentOLS


# ═══════════════════════════════════════════════════════════════════
# 0. 数据加载
# ═══════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    从 homework/week09/data/ 加载脏数据。
    优先使用 dirty_q4_marketing.csv（若存在），否则使用 dirty_marketing.csv。
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent

    # 尝试 Q4 专用数据文件
    q4_path = project_root / "data" / "dirty_q4_marketing.csv"
    if q4_path.exists():
        return pd.read_csv(q4_path)

    # 回退到 week09 脏数据
    fallback_path = project_root / "homework" / "week09" / "data" / "dirty_marketing.csv"
    if not fallback_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件。已尝试:\n  {q4_path}\n  {fallback_path}"
        )
    return pd.read_csv(fallback_path)


# ═══════════════════════════════════════════════════════════════════
# Task 3: 危险的诱惑 — 全局预处理（数据泄露）
# ═══════════════════════════════════════════════════════════════════

def global_preprocess(df: pd.DataFrame, target_col: str):
    """
    全量数据预处理（包含 One-Hot 编码 + 缺失值填补 + 标准化）。
    
    ⚠️ 这是"错误的做法"：在切分之前就对全量数据 fit_transform，
    导致验证集信息泄露到训练过程中。
    """
    X_df = df.drop(columns=[target_col])
    y = df[target_col].values.astype(np.float64)

    # One-Hot 编码 Region
    X_encoded = pd.get_dummies(X_df, columns=["Region"], drop_first=True)
    X = X_encoded.values.astype(np.float64)

    # 用全局均值填补缺失值
    col_means = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        if np.any(mask):
            X[mask, i] = col_means[i]

    # 全量标准化（泄露！）
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GradientDescentOLS 无内置截距项，手动添加 bias 列
    X_scaled = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

    return X_scaled, y


def bad_cross_validation(df: pd.DataFrame, target_col: str, n_folds: int = 5):
    """
    Task 3: 危险的诱惑 — 存在数据泄露的交叉验证。

    先对全量数据做 fit_transform，再进行 5 折 CV。
    验证集的统计信息（均值、标准差）在训练前已被"看到"，
    导致评估结果偏乐观。
    """
    print("\n" + "=" * 60)
    print("Task 3: 危险的诱惑 — 全局预处理（数据泄露）")
    print("=" * 60)

    X, y = global_preprocess(df, target_col)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    rmse_list, mae_list, mape_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientDescentOLS(
            learning_rate=0.01, max_iter=1000, gd_type="full_batch"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = calculate_rmse(y_val, y_pred)
        mae = calculate_mae(y_val, y_pred)
        mape = calculate_mape(y_val, y_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

        print(f"  Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)

    print(f"\n  ▶ 平均 RMSE: {avg_rmse:.4f}")
    print(f"  ▶ 平均 MAE:  {avg_mae:.4f}")
    print(f"  ▶ 平均 MAPE: {avg_mape:.2f}%")

    return avg_rmse, avg_mae, avg_mape


# ═══════════════════════════════════════════════════════════════════
# Task 4: 坚不可摧的护城河 — 无泄露流水线
# ═══════════════════════════════════════════════════════════════════

def preprocess_fold_no_leak(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
):
    """
    在单个 Fold 内部执行绝对无泄漏的预处理流水线：

    1. One-Hot 编码（以训练集列模板对齐验证集）
    2. 缺失值填补（用训练集均值 fit，transform 两个集合）
    3. 标准化（用训练集 mean/std fit，transform 两个集合）

    关键原则：
        - 只对 X_train 调用 .fit() / .fit_transform()
        - 对 X_val 只调用 .transform()，绝不复用 .fit()
    """
    # ── 第 1 步: One-Hot 编码 ──
    X_train_enc = pd.get_dummies(X_train_df, columns=["Region"], drop_first=True)
    X_val_enc = pd.get_dummies(X_val_df, columns=["Region"], drop_first=True)

    # 对齐列：确保验证集与训练集列一致
    missing_cols = set(X_train_enc.columns) - set(X_val_enc.columns)
    for col in missing_cols:
        X_val_enc[col] = 0
    X_val_enc = X_val_enc[X_train_enc.columns]

    X_train = X_train_enc.values.astype(np.float64)
    X_val = X_val_enc.values.astype(np.float64)

    # ── 第 2 步: 缺失值填补（仅用训练集统计量）──
    imputer = CustomImputer()
    X_train_filled = imputer.fit_transform(X_train)     # fit on train
    X_val_filled = imputer.transform(X_val)             # transform val

    # ── 第 3 步: 标准化（仅用训练集统计量）──
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)  # fit on train
    X_val_scaled = scaler.transform(X_val_filled)           # transform val

    # GradientDescentOLS 无内置截距项，手动添加 bias 列
    X_train_scaled = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(X_val_scaled.shape[0]), X_val_scaled])

    return X_train_scaled, X_val_scaled, y_train, y_val


def good_cross_validation(df: pd.DataFrame, target_col: str, n_folds: int = 5):
    """
    Task 4: 坚不可摧的护城河 — 绝对无泄漏的交叉验证流水线。

    在每一 Fold 内部：
        1. 切分 X_train / X_val
        2. 用 X_train 去 fit Scaler & Imputer
        3. 用训练集学到的参数 transform 两者
        4. 训练模型 → 预测 → 评估

    验证集信息从未进入训练管道，保证评估的无偏性。
    """
    print("\n" + "=" * 60)
    print("Task 4: 无泄漏流水线 — 逐 Fold Pipeline（数据隔离）")
    print("=" * 60)

    X_df = df.drop(columns=[target_col])
    y_raw = df[target_col].values.astype(np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    rmse_list, mae_list, mape_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
        X_train_df = X_df.iloc[train_idx].copy()
        X_val_df = X_df.iloc[val_idx].copy()
        y_train = y_raw[train_idx]
        y_val = y_raw[val_idx]

        # ★ 核心：在 fold 内部执行无泄漏预处理 ★
        X_train_scaled, X_val_scaled, y_train, y_val = preprocess_fold_no_leak(
            X_train_df, X_val_df, y_train, y_val
        )

        model = GradientDescentOLS(
            learning_rate=0.01, max_iter=1000, gd_type="full_batch"
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rmse = calculate_rmse(y_val, y_pred)
        mae = calculate_mae(y_val, y_pred)
        mape = calculate_mape(y_val, y_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

        print(f"  Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)

    print(f"\n  ▶ 平均 RMSE: {avg_rmse:.4f}")
    print(f"  ▶ 平均 MAE:  {avg_mae:.4f}")
    print(f"  ▶ 平均 MAPE: {avg_mape:.2f}%")

    return avg_rmse, avg_mae, avg_mape


# ═══════════════════════════════════════════════════════════════════
# Task 5: 自动化 I/O
# ═══════════════════════════════════════════════════════════════════

def save_comparison_report(
    bad_res: tuple, good_res: tuple, output_dir: Path
) -> None:
    """
    生成 evaluation_comparison.md，对比有/无数据泄露的评估指标。
    """
    report_path = output_dir / "evaluation_comparison.md"
    metrics = ["RMSE", "MAE", "MAPE"]
    bad_rmse, bad_mae, bad_mape = bad_res
    good_rmse, good_mae, good_mape = good_res

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 数据泄露对比分析报告\n\n")
        f.write("> 对比 Task 3（有泄露）与 Task 4（无泄露）的 5-Fold CV 评估结果。\n\n")

        f.write("## 指标汇总\n\n")
        f.write("| 指标 | 有泄露 (Bad CV) | 无泄露 (Good CV) | 差异 (%) |\n")
        f.write("|------|:---------------:|:-----------------:|:--------:|\n")

        # RMSE
        diff_rmse = (bad_rmse - good_rmse) / good_rmse * 100 if good_rmse != 0 else 0
        f.write(f"| RMSE | {bad_rmse:.4f} | {good_rmse:.4f} | {diff_rmse:+.2f}% |\n")

        # MAE
        diff_mae = (bad_mae - good_mae) / good_mae * 100 if good_mae != 0 else 0
        f.write(f"| MAE  | {bad_mae:.4f} | {good_mae:.4f} | {diff_mae:+.2f}% |\n")

        # MAPE
        diff_mape = (bad_mape - good_mape) / good_mape * 100 if good_mape != 0 else 0
        f.write(f"| MAPE | {bad_mape:.2f}% | {good_mape:.2f}% | {diff_mape:+.2f}% |\n")

        f.write("\n## 结论\n\n")
        f.write(
            "### 为什么 Task 3 的「好成绩」是致命的？\n\n"
            "存在**数据泄露**（Data Leakage）的评估结果明显优于无泄露结果，"
            "这是因为在全量预处理阶段，验证集的信息（均值、标准差、缺失填补统计量）"
            "已经被模型\"看到\"。\n\n"
            "具体而言：\n"
            "- **全局标准化**使得 Scaler 的 `mean_` 和 `std_` 包含了验证集的分布信息。\n"
            "- **全局缺失值填补**使得 NaN 的填充值混合了训练集和验证集的中心趋势。\n\n"
            "这种「好看」的分数**不能代表模型在真实未知数据上的表现**。"
            "当模型部署上线后，面对全新的、从未见过的数据时，"
            "其真实误差将接近 Task 4 的结果而非 Task 3 的乐观估计。\n\n"
            "**工业级最佳实践**：Task 4 的无泄漏流水线（Per-Fold Pipeline）"
            "才是可信的泛化能力评估方式。给老板和业务团队看的，必须是 Task 4 的"
            "「差成绩」，因为这才是上线后的真实预期。"
        )

    print(f"\n✅ 报告已保存至 {report_path}")


def plot_comparison(
    bad_res: tuple, good_res: tuple, output_dir: Path
) -> None:
    """
    绘制柱状图，直观对比有/无数据泄露时的误差差异。
    """
    bad_rmse, bad_mae, bad_mape = bad_res
    good_rmse, good_mae, good_mape = good_res

    metrics = ["RMSE", "MAE", "MAPE (%)"]
    bad_vals = [bad_rmse, bad_mae, bad_mape]
    good_vals = [good_rmse, good_mae, good_mape]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_bad = ax.bar(
        x - width / 2, bad_vals, width,
        label="With Leakage (Bad CV)", color="#e74c3c", alpha=0.8,
    )
    bars_good = ax.bar(
        x + width / 2, good_vals, width,
        label="Leakage-Free (Good CV)", color="#2ecc71", alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Error Value", fontsize=12)
    ax.set_title("Data Leakage Impact on Evaluation Metrics", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 在柱子上标注数值
    for bar in bars_bad:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + height * 0.01,
            f"{height:.2f}", ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_good:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + height * 0.01,
            f"{height:.2f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    save_path = output_dir / "leakage_analysis.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 柱状图已保存至 {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── 动态清理 results/ 目录 ──
    results_dir = Path(__file__).resolve().parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Results directory ready: {results_dir}")

    # ── 加载数据 ──
    try:
        df = load_data()
        print(f"📊 数据加载成功 | shape={df.shape}")
        print(f"   列名: {df.columns.tolist()}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # ── 确认目标列 ──
    target_col = "Sales"
    if target_col not in df.columns:
        print(f"❌ 数据中不存在目标列 '{target_col}'，现有列: {df.columns.tolist()}")
        return

    # ── 数据概览 ──
    print(f"\n📋 缺失值统计:\n{df.isnull().sum()}")
    print(f"\n📋 数据描述:\n{df.describe()}")

    # ── 执行 Task 3 & Task 4 ──
    bad_res = bad_cross_validation(df, target_col, n_folds=5)
    good_res = good_cross_validation(df, target_col, n_folds=5)

    # ── 输出对比总结 ──
    print("\n" + "=" * 60)
    print("📊 最终对比")
    print("=" * 60)
    print(f"  有泄露 (Bad CV):  RMSE={bad_res[0]:.4f}, MAE={bad_res[1]:.4f}, MAPE={bad_res[2]:.2f}%")
    print(f"  无泄露 (Good CV): RMSE={good_res[0]:.4f}, MAE={good_res[1]:.4f}, MAPE={good_res[2]:.2f}%")

    # ── 生成报告 & 图表 ──
    save_comparison_report(bad_res, good_res, results_dir)
    plot_comparison(bad_res, good_res, results_dir)

    print("\n✅ 全部任务完成！查看 results/ 目录下的报告和图表。")


if __name__ == "__main__":
    main()
