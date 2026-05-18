import sys
import os
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 将项目根目录加入 sys.path，以便导入 src.utils
current_dir = Path(__file__).resolve().parent               # .../students/22_wjq/src/week10/milestone2
src_parent = current_dir.parent.parent.parent              # .../students/22_wjq
if str(src_parent) not in sys.path:
    sys.path.insert(0, str(src_parent))

from src.utils.models import GradientDescentOLS
from src.utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from src.utils.transformers import CustomStandardScaler

# ------------------------- 辅助函数 -------------------------
def fill_nan_with_col_mean(X: np.ndarray, col_means: np.ndarray = None):
    """用列均值填补 NaN，返回填补后的数组和使用的均值"""
    X = X.astype(np.float64)
    if col_means is None:
        col_means = np.nanmean(X, axis=0)
    # 若某列全为 NaN，则均值设为 0
    col_means = np.nan_to_num(col_means, nan=0.0)
    mask = np.isnan(X)
    if np.any(mask):
        X[mask] = np.take(col_means, np.where(mask)[1])
    return X, col_means

def load_and_prep_raw_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # 只保留数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    print(f"加载数据: {file_path}, 形状: {df.shape}")
    print(f"数值列: {numeric_cols}")
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing} 个缺失值")
    return df

# ------------------------- Task 3: 有数据泄露的 CV -------------------------
def bad_cross_validation(X_full: np.ndarray, y_full: np.ndarray, n_splits=5):
    print("\n" + "="*60)
    print("⚠️  BAD CV (存在数据泄露)")
    print("="*60)

    # 全局填补和标准化
    X_filled, _ = fill_nan_with_col_mean(X_full.copy())
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        if np.any(np.isnan(y_train)) or np.any(np.isnan(y_val)):
            print(f"Fold {fold}: 目标变量包含 NaN，跳过")
            continue

        model = GradientDescentOLS(learning_rate=0.01, max_iter=500, tol=1e-5, gd_type='full_batch')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if np.any(np.isnan(y_pred)):
            print(f"Fold {fold}: 预测值包含 NaN，跳过")
            continue

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))
        print(f"Fold {fold}: RMSE={rmse_list[-1]:.4f}, MAE={mae_list[-1]:.4f}, MAPE={mape_list[-1]:.2f}%")

    if len(rmse_list) == 0:
        print("所有折均失败，返回 NaN")
        return np.nan, np.nan, np.nan
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    print(f"\n平均: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, MAPE={avg_mape:.2f}%")
    return avg_rmse, avg_mae, avg_mape

# ------------------------- Task 4: 无数据泄露的 CV -------------------------
def good_cross_validation(X_full: np.ndarray, y_full: np.ndarray, n_splits=5):
    print("\n" + "="*60)
    print("✅ GOOD CV (无数据泄露)")
    print("="*60)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        X_train_raw = X_full[train_idx].copy()
        X_val_raw   = X_full[val_idx].copy()
        y_train = y_full[train_idx]
        y_val   = y_full[val_idx]

        if np.any(np.isnan(y_train)) or np.any(np.isnan(y_val)):
            print(f"Fold {fold}: 目标变量包含 NaN，跳过")
            continue

        # 1. 用训练集均值填补缺失值
        X_train_filled, col_means = fill_nan_with_col_mean(X_train_raw)
        X_val_filled, _ = fill_nan_with_col_mean(X_val_raw, col_means)

        # 2. 标准化（训练集拟合，验证集转换）
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)

        # 3. 训练模型
        model = GradientDescentOLS(learning_rate=0.01, max_iter=500, tol=1e-5, gd_type='full_batch')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        if np.any(np.isnan(y_pred)):
            print(f"Fold {fold}: 预测值包含 NaN，跳过")
            continue

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))
        print(f"Fold {fold}: RMSE={rmse_list[-1]:.4f}, MAE={mae_list[-1]:.4f}, MAPE={mape_list[-1]:.2f}%")

    if len(rmse_list) == 0:
        print("所有折均失败，返回 NaN")
        return np.nan, np.nan, np.nan
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    print(f"\n平均: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, MAPE={avg_mape:.2f}%")
    return avg_rmse, avg_mae, avg_mape

# ------------------------- Task 5: 输出报告与图表 -------------------------
def generate_report(bad_metrics, good_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_path = Path(output_dir) / "evaluation_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 交叉验证结果对比：数据泄露 vs 无泄漏\n\n")
        f.write("| 指标 | 有数据泄露 (Bad CV) | 无数据泄露 (Good CV) | 差异 |\n")
        f.write("|------|---------------------|----------------------|------|\n")
        f.write(f"| RMSE | {bad_metrics[0]:.4f} | {good_metrics[0]:.4f} | {bad_metrics[0]-good_metrics[0]:+.4f} |\n")
        f.write(f"| MAE  | {bad_metrics[1]:.4f} | {good_metrics[1]:.4f} | {bad_metrics[1]-good_metrics[1]:+.4f} |\n")
        f.write(f"| MAPE(%) | {bad_metrics[2]:.2f} | {good_metrics[2]:.2f} | {bad_metrics[2]-good_metrics[2]:+.2f} |\n\n")
        f.write("## 结论\n\n")
        f.write("存在数据泄露的 Bad CV 给出的误差指标通常更低（更好看），原因是预处理阶段使用了全量数据（包括验证集）的统计信息。\n\n")
        f.write("但在真实生产环境中，模型上线后只能看到训练时的统计量，因此 Bad CV 的“好成绩”是一种致命的乐观估计。\n\n")
        f.write("Good CV 模拟了真实的未来预测过程：只使用训练集的均值填补缺失值，只使用训练集的均值和标准差做标准化。\n\n")
        f.write(f"**业务解读**：以 Good CV 的 MAE 值为准，模型预测的误差大约为 {good_metrics[1]:.2f} 万元，MAPE 约为 {good_metrics[2]:.1f}%。")
    print(f"\n📄 报告已生成: {report_path}")

def plot_comparison(bad_metrics, good_metrics, output_dir):
    try:
        import matplotlib.pyplot as plt
        # 避免中文字体警告（不影响保存）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        metrics = ['RMSE', 'MAE', 'MAPE (%)']
        bad_vals = [bad_metrics[0], bad_metrics[1], bad_metrics[2]]
        good_vals = [good_metrics[0], good_metrics[1], good_metrics[2]]

        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, bad_vals, width, label='Bad CV (泄露)', color='salmon')
        ax.bar(x + width/2, good_vals, width, label='Good CV (无泄漏)', color='steelblue')
        ax.set_ylabel('误差值')
        ax.set_title('数据泄露对交叉验证误差的影响')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = Path(output_dir) / "leakage_analysis.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 柱状图已保存: {plot_path}")
        plt.close()
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图。")

# ------------------------- 主流程 -------------------------
def main():
    parser = argparse.ArgumentParser(description="无泄漏交叉验证评估")
    parser.add_argument("--data", type=str, default="homework/week09/data/dirty_marketing.csv",
                        help="数据文件路径")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="结果输出目录")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)

    results_dir = Path(args.output_dir)
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("✅ results 文件夹已清理并重建")

    df = load_and_prep_raw_data(data_path)
    if df.empty:
        print("数据为空，退出")
        sys.exit(1)

    # 假设目标列是最后一列
    target_col = df.columns[-1]
    print(f"目标列: {target_col}")
    X = df.drop(columns=[target_col]).values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    if np.any(np.isnan(y)):
        print("❌ 目标变量存在 NaN，请检查数据。")
        sys.exit(1)

    # 运行两种交叉验证
    bad_metrics = bad_cross_validation(X, y, n_splits=5)
    good_metrics = good_cross_validation(X, y, n_splits=5)

    generate_report(bad_metrics, good_metrics, results_dir)
    plot_comparison(bad_metrics, good_metrics, results_dir)

    print("\n🎉 里程碑大作业执行完毕！")

if __name__ == "__main__":
    main()