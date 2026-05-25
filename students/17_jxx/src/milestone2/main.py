import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler

# ==========================
# 路径 100% 适配你！
# ==========================
BASE = Path(__file__).parent.parent.parent
DATA_FILE = BASE / "data" / "dirty_marketing.csv"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

for f in RESULT_DIR.glob("*"):
    f.unlink()

# ==========================
# Task 3 危险版：数据泄露
# ==========================
def bad_cross_validation(X_num, y):
    print("\n===== 危险模式：全局预处理 = 数据泄露 =====")

    # 只处理数值型特征！
    X_clean = np.nan_to_num(X_num, nan=np.nanmean(X_num, axis=0))
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_final = np.c_[np.ones(len(X_scaled)), X_scaled]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, val_idx in kf.split(X_final):
        X_tr, X_val = X_final[train_idx], X_final[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = GradientDescentOLS(learning_rate=0.01, max_iter=1000)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmse_list.append(calculate_rmse(y_val, y_pred))

    mean_rmse = np.mean(rmse_list)
    print(f"泄露版平均 RMSE: {mean_rmse:.2f}")
    return mean_rmse

# ==========================
# Task 4 安全版：无泄漏流水线
# ==========================
def good_cross_validation(X_num, y):
    print("\n===== 安全模式：无泄漏工业级流水线 =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    mae_list = []
    mape_list = []

    for train_idx, val_idx in kf.split(X_num):
        X_tr, X_val = X_num[train_idx], X_num[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        tr_mean = np.nanmean(X_tr, axis=0)
        X_tr_filled = np.nan_to_num(X_tr, nan=tr_mean)

        scaler = CustomStandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_filled)
        X_tr_final = np.c_[np.ones(len(X_tr_scaled)), X_tr_scaled]

        model = GradientDescentOLS(learning_rate=0.01, max_iter=1000)
        model.fit(X_tr_final, y_tr)

        X_val_filled = np.nan_to_num(X_val, nan=tr_mean)
        X_val_scaled = scaler.transform(X_val_filled)
        X_val_final = np.c_[np.ones(len(X_val_scaled)), X_val_scaled]

        y_pred = model.predict(X_val_final)
        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

    mean_rmse = np.mean(rmse_list)
    mean_mae = np.mean(mae_list)
    mean_mape = np.mean(mape_list)

    print(f"无泄漏 RMSE: {mean_rmse:.2f}")
    print(f"无泄漏 MAE:  {mean_mae:.2f}")
    print(f"无泄漏 MAPE: {mean_mape:.2f}%")

    return mean_rmse, mean_mae, mean_mape

# ==========================
# 自动生成报告
# ==========================
def generate_report(bad_rmse, good_rmse, good_mae, good_mape):
    content = f"""# Milestone 2 无泄漏泛化评估报告

## 实验对比结果
| 模式 | RMSE |
|------|------|
| 数据泄露（危险） | {bad_rmse:.2f} |
| 无泄漏（安全） | {good_rmse:.2f} |

## 真实业务误差
- MAE = {good_mae:.2f} 元
- MAPE = {good_mape:.2f} %

## 核心结论
数据泄露的分数更好看，但**致命错误**，因为：
1. 验证集信息提前被全局使用
2. 分数虚高，上线必崩
3. 无泄漏版本才是真实泛化能力

## 业务解读
模型上线后，每天广告销量预测真实误差约 {good_mae:.0f} 元，
相对误差 {good_mape:.1f}%。
必须给老板看无泄漏版本！
"""
    with open(RESULT_DIR / "evaluation_comparison.md", "w", encoding="utf-8") as f:
        f.write(content)

    plt.figure(figsize=(7, 4))
    plt.bar(["Leakage", "Leakage-Free"], [bad_rmse, good_rmse], color=["red", "green"])
    plt.title("Leakage vs Leakage-Free RMSE")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "leakage_analysis.png")
    plt.close()

    print(f"\n✅ 报告已保存到: {RESULT_DIR}/evaluation_comparison.md")

# ==========================
# 主程序
# ==========================
def main():
    df = pd.read_csv(DATA_FILE)

    # 🔥 🔥 🔥 只提取数值型特征！修复字符串报错！
    numeric_cols = ["TV_Budget", "Online_Video_Budget", "Radio_Budget"]
    X_num = df[numeric_cols].values
    y = df["Sales"].values

    bad_rmse = bad_cross_validation(X_num, y)
    good_rmse, good_mae, good_mape = good_cross_validation(X_num, y)
    generate_report(bad_rmse, good_rmse, good_mae, good_mape)

if __name__ == "__main__":
    main()