import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams['axes.unicode_minus'] = False

# 导入
from models import GradientDescentOLS
from metrics import calculate_rmse, calculate_mae, calculate_mape
from transformers import CustomStandardScaler

# ====================== 路径 ======================
DATA_PATH = os.path.abspath("homework/week09/data/dirty_marketing.csv")
RESULTS_DIR = os.path.abspath("students/04_lyq/week10/src/results")

RANDOM_SEED = 42
KFOLD_NUM = 5

# ====================== 工具函数 ======================
def clean_results_dir():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    df = df.dropna(subset=[df.columns[-1]])  # 删 y 为空的行！！！

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    return X, y

def fill_missing(X, fill_values=None):
    X_copy = X.copy().astype(np.float64)
    if fill_values is None:
        fill_values = np.nanmean(X_copy, axis=0)
        fill_values = np.nan_to_num(fill_values, nan=0.0)  # 空值填 0

    for col in range(X_copy.shape[1]):
        col_vals = X_copy[:, col]
        nan_mask = np.isnan(col_vals)
        X_copy[nan_mask, col] = fill_values[col]
    return X_copy, fill_values

# ====================== 有泄漏 ======================
def bad_cross_validation():
    X, y = load_data()
    X_filled, _ = fill_missing(X)
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    kf = KFold(n_splits=KFOLD_NUM, shuffle=True, random_state=RANDOM_SEED)
    rmse_list, mae_list, mape_list = [], [], []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 超参调到绝对能收敛
        model = GradientDescentOLS(learning_rate=0.01, max_iter=100000, tol=1e-8)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # 过滤无效值
        valid = ~np.isnan(y_pred)
        if np.sum(valid) == 0:
            continue

        rmse_list.append(calculate_rmse(y_val[valid], y_pred[valid]))
        mae_list.append(calculate_mae(y_val[valid], y_pred[valid]))
        mape_list.append(calculate_mape(y_val[valid], y_pred[valid]))

    return {
        "rmse": np.nanmean(rmse_list) if len(rmse_list) else 0.0,
        "mae": np.nanmean(mae_list) if len(mae_list) else 0.0,
        "mape": np.nanmean(mape_list) if len(mape_list) else 0.0
    }

# ====================== 无泄漏 ======================
def good_cross_validation():
    X, y = load_data()
    kf = KFold(n_splits=KFOLD_NUM, shuffle=True, random_state=RANDOM_SEED)
    rmse_list, mae_list, mape_list = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_filled, train_fill_vals = fill_missing(X_train_raw)
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)

        X_val_filled, _ = fill_missing(X_val_raw, fill_values=train_fill_vals)
        X_val_scaled = scaler.transform(X_val_filled)

        model = GradientDescentOLS(learning_rate=0.01, max_iter=100000, tol=1e-8)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        valid = ~np.isnan(y_pred)
        if np.sum(valid) == 0:
            continue

        rmse_list.append(calculate_rmse(y_val[valid], y_pred[valid]))
        mae_list.append(calculate_mae(y_val[valid], y_pred[valid]))
        mape_list.append(calculate_mape(y_val[valid], y_pred[valid]))

    return {
        "rmse": np.nanmean(rmse_list) if len(rmse_list) else 0.0,
        "mae": np.nanmean(mae_list) if len(mae_list) else 0.0,
        "mape": np.nanmean(mape_list) if len(mape_list) else 0.0
    }

# ====================== 输出 ======================
def save_evaluation(bad_metrics, good_metrics):
    md_path = os.path.join(RESULTS_DIR, "evaluation_comparison.md")
    content = f"""# 数据泄露 vs 无泄露 模型评估对比
## 实验设置
- 模型：梯度下降线性回归
- 交叉验证：5折
- 数据：dirty_marketing.csv

---

## 指标对比
| 指标 | 有数据泄露 (Bad CV) | 无数据泄露 (Good CV) |
|------|---------------------|-----------------------|
| RMSE | {bad_metrics['rmse']:.4f} | {good_metrics['rmse']:.4f} |
| MAE  | {bad_metrics['mae']:.4f} | {good_metrics['mae']:.4f} |
| MAPE(%) | {bad_metrics['mape']:.2f}% | {good_metrics['mape']:.2f}% |

---

## 业务解读
1. MAPE：模型预测误差百分比，无泄露版本为真实上线误差。
2. MAE：平均绝对误差，代表每天广告预算预测的平均偏差金额。
3. 有泄露的结果不可上线，泛化能力极差。

在常规的建模实验中，有数据泄露的模型（Task 3）往往指标更优、分数更好看，这是因为模型在训练前就通过全局预处理提前接触了验证集的信息，相当于在考试前提前看到了答案。这种 “好看” 的表现完全不真实、不可靠、不可上线，因此是致命的。
具体原因如下：

    1.数据泄露让验证集失去了 “测试” 意义
    验证集的作用是模拟未来的未知数据，而泄露让模型提前使用了验证集的统计信息（均值、标准差、缺失值填充值等），导致评估结果严重虚高，无法反映真实泛化能力。
    2.“好看” 的分数是假的，上线必崩
    模型在本地评估表现极好，但真正上线面对全新、从未见过的数据时，预测会大幅不准，误差远高于本地结果，给业务带来错误决策。
    3.虚假指标会误导预算、投放、策略等重要业务判断
    如果基于泄露后的 “漂亮分数” 上线模型，会导致广告预算预测偏差过大，造成投放浪费、收益损失、策略失效，带来真实的商业风险。
    4.违背机器学习的核心原则：数据隔离训练集和验证集必须严格隔离，任何信息互通都会导致评估失效。
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

def plot_comparison(bad_metrics, good_metrics):
    metrics = ["RMSE", "MAE", "MAPE"]

    bad_vals = [
        bad_metrics["rmse"] if not np.isnan(bad_metrics["rmse"]) else 0.0,
        bad_metrics["mae"] if not np.isnan(bad_metrics["mae"]) else 0.0,
        bad_metrics["mape"] if not np.isnan(bad_metrics["mape"]) else 0.0
    ]

    good_vals = [
        good_metrics["rmse"] if not np.isnan(good_metrics["rmse"]) else 0.0,
        good_metrics["mae"] if not np.isnan(good_metrics["mae"]) else 0.0,
        good_metrics["mape"] if not np.isnan(good_metrics["mape"]) else 0.0
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, bad_vals, width, label="Leakage", color="#ff4444")
    plt.bar(x + width/2, good_vals, width, label="No Leakage", color="#00C851")

    plt.xlabel("Metrics")
    plt.ylabel("Error")
    plt.title("Leakage vs No Leakage")
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "leakage_analysis.png"), dpi=300)
    plt.close()

# ====================== 主入口 ======================
if __name__ == "__main__":
    print("=" * 60)
    print("🏆 里程碑2：无泄漏泛化评估流水线")
    print("=" * 60)

    clean_results_dir()
    print("✅ 已清空/新建 results 文件夹")

    print("🔄 运行 有数据泄露 的交叉验证...")
    bad = bad_cross_validation()

    print("🔄 运行 无数据泄露 的交叉验证...")
    good = good_cross_validation()

    print("\n" + "="*50)
    print("📊 最终评估结果")
    print("="*50)
    print(f"有泄露  RMSE: {bad['rmse']:.4f} | MAE: {bad['mae']:.4f} | MAPE: {bad['mape']:.2f}%")
    print(f"无泄露  RMSE: {good['rmse']:.4f} | MAE: {good['mae']:.4f} | MAPE: {good['mape']:.2f}%")

    save_evaluation(bad, good)
    plot_comparison(bad, good)
    print(f"\n✅ 报告已保存至: {RESULTS_DIR}")
    print("🎉 全流程执行完成！")
