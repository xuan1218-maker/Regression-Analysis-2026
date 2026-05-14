"""
Milestone Project 2: The Pipeline & The Leakage-Free Generalization

演示数据泄露的危害与防泄露流水线的正确实现
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import KFold

# ========== 中文字体配置 ==========
# 设置中文字体，解决图表中文显示问题
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# ==================================

from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler


def setup_results_dir():
    """创建结果目录"""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_data():
    """加载脏数据"""
    data_path = Path(__file__).parent.parent / "data" / "dirty_marketing.csv"

    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        return None

    df = pd.read_csv(data_path)
    print(f"[OK] 加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def preprocess_dataframe(df):
    """
    数据预处理：One-Hot 编码 + 提取特征和目标
    返回特征矩阵 X 和目标向量 y
    """
    # One-Hot 编码分类变量（drop_first=True 避免虚拟变量陷阱）
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"  One-Hot 编码后形状: {df.shape}")

    # 目标变量是 Sales
    target_col = "Sales"

    if target_col not in df.columns:
        raise ValueError(f"找不到目标列: {target_col}")

    # 特征矩阵（排除目标列）
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    return X, y, feature_cols


def fill_missing_with_mean(X):
    """用列均值填补缺失值"""
    X_filled = X.copy()
    col_means = np.nanmean(X_filled, axis=0)

    for i in range(X_filled.shape[1]):
        mask = np.isnan(X_filled[:, i])
        if mask.any():
            X_filled[mask, i] = col_means[i]

    return X_filled, col_means


def bad_cross_validation(X, y):
    """
    Task 3: 危险的诱惑 —— 制造数据泄露

    错误做法：在 CV 之前对全量数据进行全局预处理
    - 全局标准化（用全量数据 fit）
    - 全局均值填补缺失值
    """
    print("\n" + "=" * 60)
    print("[实验一] 错误的交叉验证（有数据泄露）")
    print("=" * 60)

    # 错误：在 CV 之前对全量数据做预处理
    # 1. 填补缺失值（用全量数据的均值）
    X_filled, _ = fill_missing_with_mean(X)

    # 2. 标准化（用全量数据 fit）
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        model = GradientDescentOLS(
            learning_rate=0.01,
            tol=1e-5,
            max_iter=500,
            gd_type="full_batch",
            fit_intercept=True,
        )
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        rmse_scores.append(calculate_rmse(y_val, y_pred))
        mae_scores.append(calculate_mae(y_val, y_pred))
        mape_scores.append(calculate_mape(y_val, y_pred))

        print(
            f"  第 {fold} 折: RMSE={rmse_scores[-1]:.4f}, MAE={mae_scores[-1]:.4f}, MAPE={mape_scores[-1]:.2f}%"
        )

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    print(
        f"\n[结果] 泄露版交叉验证平均: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, MAPE={avg_mape:.2f}%"
    )
    print("[警告] 这个结果虚高！因为验证集信息已泄露到训练过程中")

    return {"rmse": avg_rmse, "mae": avg_mae, "mape": avg_mape}


def good_cross_validation(X, y):
    """
    Task 4: 坚不可摧的护城河 —— 防泄露流水线

    正确做法：在 CV 循环内部进行预处理
    - 只用训练集 fit 标准化器和填补缺失值
    - 用训练集的参数转换验证集
    """
    print("\n" + "=" * 60)
    print("[实验二] 正确的交叉验证（无数据泄露）")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # 获取原始数据（注意：X 还包含缺失值）
        X_train_raw = X[train_idx]
        X_val_raw = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # 正确：用训练集拟合缺失值填补参数
        X_train_filled, col_means = fill_missing_with_mean(X_train_raw)

        # 正确：用训练集拟合标准化器
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)

        # 正确：用训练集的参数转换验证集
        # 1. 用训练集的均值填补验证集缺失值
        X_val_filled = X_val_raw.copy()
        for i in range(X_val_filled.shape[1]):
            mask = np.isnan(X_val_filled[:, i])
            if mask.any():
                X_val_filled[mask, i] = col_means[i]

        # 2. 用训练集的标准化器转换验证集
        X_val_scaled = scaler.transform(X_val_filled)

        # 训练模型
        model = GradientDescentOLS(
            learning_rate=0.01,
            tol=1e-5,
            max_iter=500,
            gd_type="full_batch",
            fit_intercept=True,
        )
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)

        rmse_scores.append(calculate_rmse(y_val, y_pred))
        mae_scores.append(calculate_mae(y_val, y_pred))
        mape_scores.append(calculate_mape(y_val, y_pred))

        print(
            f"  第 {fold} 折: RMSE={rmse_scores[-1]:.4f}, MAE={mae_scores[-1]:.4f}, MAPE={mape_scores[-1]:.2f}%"
        )

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    print(
        f"\n[结果] 无泄露版交叉验证平均: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, MAPE={avg_mape:.2f}%"
    )
    print("[成功] 这才是模型真实的泛化误差")

    return {"rmse": avg_rmse, "mae": avg_mae, "mape": avg_mape}


def plot_comparison(bad_results, good_results, results_dir):
    """绘制对比柱状图"""
    metrics = ["RMSE", "MAE", "MAPE"]
    bad_values = [bad_results["rmse"], bad_results["mae"], bad_results["mape"]]
    good_values = [good_results["rmse"], good_results["mae"], good_results["mape"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2,
        bad_values,
        width,
        label="有数据泄露 (错误交叉验证)",
        color="red",
        alpha=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        good_values,
        width,
        label="无数据泄露 (正确交叉验证)",
        color="green",
        alpha=0.7,
    )

    # 设置中文标题和标签
    ax.set_xlabel("评估指标", fontsize=12)
    ax.set_ylabel("误差值", fontsize=12)
    ax.set_title("数据泄露对模型评估的影响对比", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=10)

    # 添加数值标签
    for bar, val in zip(bars1, bad_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar, val in zip(bars2, good_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plot_path = results_dir / "leakage_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[成功] 对比图已保存: {plot_path}")


def generate_report(bad_results, good_results, results_dir):
    """生成 Markdown 报告"""
    report_path = results_dir / "evaluation_comparison.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Milestone 2: 数据泄露分析与防泄露流水线\n\n")

        f.write("## 实验目的\n\n")
        f.write("演示数据泄露的危害，以及如何构建无泄露的交叉验证流水线。\n\n")

        f.write("## 实验设计\n\n")
        f.write("| 实验 | 预处理方式 | 说明 |\n")
        f.write("|------|-----------|------|\n")
        f.write(
            "| 错误交叉验证 | 全局预处理（标准化 + 缺失值填补） | 验证集信息泄露 |\n"
        )
        f.write("| 正确交叉验证 | 循环内预处理（只用训练集拟合） | 无数据泄露 |\n\n")

        f.write("## 结果对比\n\n")
        f.write(
            "| 指标 | 有数据泄露 (错误交叉验证) | 无数据泄露 (正确交叉验证) | 差异 |\n"
        )
        f.write("|------|-------------------------|-------------------------|------|\n")
        f.write(
            f"| RMSE | {bad_results['rmse']:.4f} | {good_results['rmse']:.4f} | {bad_results['rmse'] - good_results['rmse']:.4f} |\n"
        )
        f.write(
            f"| MAE | {bad_results['mae']:.4f} | {good_results['mae']:.4f} | {bad_results['mae'] - good_results['mae']:.4f} |\n"
        )
        f.write(
            f"| MAPE | {bad_results['mape']:.2f}% | {good_results['mape']:.2f}% | {bad_results['mape'] - good_results['mape']:.2f}% |\n\n"
        )

        f.write("## 结论\n\n")
        f.write("### 为什么泄露版的指标更'好看'？\n\n")
        f.write("泄露版在预处理阶段使用了全量数据（包括验证集）的统计量：\n")
        f.write("1. 标准化时用了验证集的均值和标准差\n")
        f.write("2. 缺失值填补用了验证集的均值\n\n")
        f.write(
            "这导致验证集信息被偷看，模型评估结果虚高，无法反映真实的泛化能力。\n\n"
        )

        f.write("### 为什么老板应该看真实的泛化误差？\n\n")
        f.write("泄露版的好成绩是假的，模型上线后在新数据上表现会大幅下降。\n")
        f.write("无泄露版的成绩才是模型真实的泛化误差，能准确预测未来的业务表现。\n\n")

        f.write("### 业务解读\n\n")
        f.write(
            f"模型上线后，每天的销售额预测平均绝对百分比误差约为 **{good_results['mape']:.2f}%**。\n"
        )
        f.write(
            f"这意味着如果某天实际销售额为 100 万元，我们的预测误差大约在 ±{good_results['mape']:.2f} 万元左右。\n\n"
        )

        f.write("## 可视化\n\n")
        f.write("![对比图](leakage_analysis.png)\n")

    print(f"[成功] 报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Milestone Project 2: 工业流水线与无泄漏的泛化评估")
    print("=" * 60)

    # 设置结果目录
    results_dir = setup_results_dir()
    print(f"\n结果目录: {results_dir}")

    # 加载数据
    df = load_data()
    if df is None:
        return

    # 预处理：One-Hot 编码，提取 X, y
    X, y, feature_cols = preprocess_dataframe(df)
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")

    # 引入缺失值（随机将 5% 的值设为 NaN，模拟真实场景）
    np.random.seed(42)
    mask = np.random.random(X.shape) < 0.05
    X_with_nan = X.copy()
    X_with_nan[mask] = np.nan
    print(f"引入缺失值: {np.isnan(X_with_nan).sum()} 个 NaN")

    # 实验一：错误交叉验证（有数据泄露）
    bad_results = bad_cross_validation(X_with_nan, y)

    # 实验二：正确交叉验证（无数据泄露）
    good_results = good_cross_validation(X_with_nan, y)

    # 绘制对比图
    plot_comparison(bad_results, good_results, results_dir)

    # 生成报告
    generate_report(bad_results, good_results, results_dir)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print("\n思考题答案:")
    print("1. 泄露版指标更'好看'是因为验证集信息被偷看")
    print("2. 老板应该看真实的泛化误差")
    print(
        f"3. 业务解读: MAPE = {good_results['mape']:.2f}%，表示预测误差约为实际值的 {good_results['mape']:.2f}%"
    )


if __name__ == "__main__":
    main()
