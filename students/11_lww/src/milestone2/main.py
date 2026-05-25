from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """加载并预处理第九周 dirty_marketing.csv 数据，自动处理分类变量"""
    data_path = Path(__file__).parent.parent.parent / "data" / "dirty_marketing.csv"
    df = pd.read_csv(data_path)
    target_col = "Sales"
    
    # ✅ 自动处理分类变量：One-Hot编码 + drop_first避免虚拟变量陷阱
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()
    return X, y

def bad_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    ❌ 有数据泄露的交叉验证：先全局预处理，再做CV
    """
    # 全局均值填充缺失值（泄露：用了验证集的均值）
    X_filled = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    # 全局标准化（泄露：用了验证集的均值和标准差）
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    # 添加截距项
    X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    mae_list = []
    mape_list = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientDescentOLS(
            learning_rate=0.01,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2
        ).fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

    return {
        "rmse": np.mean(rmse_list),
        "mae": np.mean(mae_list),
        "mape": np.mean(mape_list)
    }

def good_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    ✅ 无数据泄露的交叉验证：在每折内部独立做预处理
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    mae_list = []
    mape_list = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 仅用训练集的均值填充训练集和验证集
        train_mean = np.nanmean(X_train, axis=0)
        X_train_filled = np.nan_to_num(X_train, nan=train_mean)
        X_val_filled = np.nan_to_num(X_val, nan=train_mean)

        # 仅在训练集上拟合scaler，然后转换验证集
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)

        # 添加截距项
        X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])

        # 训练模型
        model = GradientDescentOLS(
            learning_rate=0.01,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2
        ).fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

    return {
        "rmse": np.mean(rmse_list),
        "mae": np.mean(mae_list),
        "mape": np.mean(mape_list)
    }

def generate_report(bad_results: dict, good_results: dict, results_dir: Path):
    """生成评估对比报告"""
    report_content = """# 数据泄露对比评估报告

## 1. 评估指标对比
| 方法 | RMSE | MAE | MAPE (%) |
|------|------|-----|----------|
| 有数据泄露（虚假好成绩） | {bad_rmse:.4f} | {bad_mae:.4f} | {bad_mape:.2f} |
| 无数据泄露（真实成绩） | {good_rmse:.4f} | {good_mae:.4f} | {good_mape:.2f} |

## 2. 结果分析
- 有数据泄露的模型误差更小，但这是**虚假的、不可信的**
- 因为预处理时提前"偷看"了验证集的统计信息
- 无数据泄露的结果才是模型上线后的**真实泛化能力**

## 3. 业务解读
- 模型上线后，广告预算预测的平均绝对误差约为 {good_mae:.2f} 万元
- 平均百分比误差约为 {good_mape:.2f}%
- 必须给老板看"差成绩"，因为这才是真实的业务风险
""".format(
        bad_rmse=bad_results["rmse"],
        bad_mae=bad_results["mae"],
        bad_mape=bad_results["mape"],
        good_rmse=good_results["rmse"],
        good_mae=good_results["mae"],
        good_mape=good_results["mape"]
    )

    with open(results_dir / "evaluation_comparison.md", "w", encoding="utf-8") as f:
        f.write(report_content)

def plot_comparison(bad_results: dict, good_results: dict, results_dir: Path):
    """绘制误差对比柱状图（加分项）"""
    metrics = ["RMSE", "MAE", "MAPE (%)"]
    bad_values = [bad_results["rmse"], bad_results["mae"], bad_results["mape"]]
    good_values = [good_results["rmse"], good_results["mae"], good_results["mape"]]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, bad_values, width, label="有数据泄露（虚假好成绩）", color="#ff6b6b")
    plt.bar(x + width/2, good_values, width, label="无数据泄露（真实成绩）", color="#4ecdc4")

    plt.xlabel("评估指标")
    plt.ylabel("误差值")
    plt.title("数据泄露对模型评估的致命影响")
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "leakage_analysis.png", dpi=150)
    plt.close()

def main():
    # 自动清理并新建results文件夹
    results_dir = Path(__file__).parent.parent.parent / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True)
    print(f"📁 结果目录: {results_dir}")

    # 加载数据（自动处理分类变量）
    X, y = load_data()
    print(f"📊 数据加载完成，样本数: {len(X)}, 特征数: {X.shape[1]}")

    # 运行有数据泄露的交叉验证
    print("\n=== 运行有数据泄露的交叉验证 ===")
    bad_results = bad_cross_validation(X, y)
    print(f"平均 RMSE: {bad_results['rmse']:.4f}")
    print(f"平均 MAE: {bad_results['mae']:.4f}")
    print(f"平均 MAPE: {bad_results['mape']:.2f}%")

    # 运行无数据泄露的交叉验证
    print("\n=== 运行无数据泄露的交叉验证 ===")
    good_results = good_cross_validation(X, y)
    print(f"平均 RMSE: {good_results['rmse']:.4f}")
    print(f"平均 MAE: {good_results['mae']:.4f}")
    print(f"平均 MAPE: {good_results['mape']:.2f}%")

    # 生成报告和图表
    generate_report(bad_results, good_results, results_dir)
    plot_comparison(bad_results, good_results, results_dir)
    print("\n✅ 所有任务完成！结果已保存到 results/ 文件夹")

if __name__ == "__main__":
    main()