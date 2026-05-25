#!/usr/bin/env python3
"""
模型诊断与交叉验证评估脚本。

读取 data_prep.py 生成的清洗后 CSV 数据，执行：
  1. 多重共线性体检：调用 calculate_vif 计算 VIF，VIF > 10 红色警告
  2. 基线交叉验证：使用 CustomOLS 进行 5 折 CV，输出平均 R²

用法示例：
    python students/10_xzn/src/week9/evaluate.py
    python students/10_xzn/src/week9/evaluate.py --data students/10_xzn/data/clean_marketing.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 将 src 目录加入路径，确保能导入 utils 模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.diagnostics import calculate_vif
from utils.models import CustomOLS


def _is_onehot_column(series: pd.Series) -> bool:
    """
    判断一个列是否可能为 One-Hot 编码产生的哑变量列。
    哑变量列的值通常仅限于 {0, 1} 或 {True, False}。
    """
    unique_vals = set(series.dropna().unique())
    return unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0} or unique_vals <= {True, False}


def main():
    parser = argparse.ArgumentParser(
        description="模型诊断：VIF 共线性检测 + 5 折交叉验证"
    )
    parser.add_argument(
        "--data",
        default="students/10_xzn/data/clean_marketing.csv",
        help="清洗后 CSV 数据的路径（默认: students/10_xzn/data/clean_marketing.csv）"
    )
    args = parser.parse_args()

    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"[ERROR] 找不到数据文件: {args.data}")
        print("[INFO] 请先运行 data_prep.py 生成清洗数据，或通过 --data 参数指定路径：")
        print("  python students/10_xzn/src/week9/data_prep.py \\")
        print("      --input homework/week09/data/dirty_marketing.csv \\")
        print("      --output students/10_xzn/data/clean_marketing.csv")
        sys.exit(1)

    # 读取数据
    print(f"[INFO] 读取清洗后数据: {args.data}")
    df = pd.read_csv(args.data)
    print(f"[INFO] 数据形状: {df.shape}")
    print(f"[INFO] 列名: {list(df.columns)}")

    # 目标变量固定为 'Sales'（如果不存在则使用最后一列）
    if "Sales" in df.columns:
        target_col = "Sales"
    else:
        target_col = df.columns[-1]
    print(f"[INFO] 目标变量: '{target_col}'")

    # 分离特征和目标
    feature_cols = [c for c in df.columns if c != target_col]
    X_all = df[feature_cols].values.astype(float)
    y_all = df[target_col].values.astype(float)

    # ================================================================
    # 阶段一：多重共线性体检 (VIF)
    # ================================================================
    print("\n" + "=" * 60)
    print("阶段一：多重共线性体检 (VIF)")
    print("=" * 60)

    # 排除 One-Hot 编码产生的列，仅保留原始连续数值列
    non_onehot_cols = [c for c in feature_cols if not _is_onehot_column(df[c])]
    if not non_onehot_cols:
        print("[WARNING] 没有非 One-Hot 的数值列，跳过 VIF 计算")
    else:
        print(f"[INFO] 对以下非 One-Hot 数值列计算 VIF: {non_onehot_cols}")
        X_vif = df[non_onehot_cols].values.astype(float)

        if X_vif.shape[1] < 2:
            print("[INFO] 可用数值特征少于 2 个，无法计算 VIF（需要至少 2 个特征）")
        else:
            vif_values = calculate_vif(X_vif)

            print("\n特征 VIF 值:")
            print("-" * 40)
            high_vif_features = []
            for col, vif in zip(non_onehot_cols, vif_values):
                if np.isinf(vif):
                    print(f"  {col:<30s} VIF = inf (完美共线性)")
                    high_vif_features.append(col)
                else:
                    print(f"  {col:<30s} VIF = {vif:.4f}")
                    if vif > 10:
                        high_vif_features.append(col)

            if high_vif_features:
                red = "\033[91m"
                reset = "\033[0m"
                print(f"\n{red}⚠  [WARNING] 以下特征 VIF > 10，存在严重多重共线性：{reset}")
                for col in high_vif_features:
                    print(f"{red}  - {col}{reset}")
                print(f"{red}请业务方关注这些特征之间的高度相关性，考虑进行特征选择或正则化。{reset}")
            else:
                print("\n[OK] 所有特征 VIF <= 10，未检测到严重多重共线性")

    # ================================================================
    # 阶段二：5 折交叉验证
    # ================================================================
    print("\n" + "=" * 60)
    print("阶段二：5 折交叉验证 (CustomOLS)")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        model = CustomOLS(fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # 计算 R²
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2_scores.append(r2)
        print(f"  Fold {fold_idx}: R² = {r2:.6f}")

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)  # 样本标准差
    print(f"\n  平均 R² = {mean_r2:.6f}")
    print(f"  R² 标准差 = {std_r2:.6f}")

    # ================================================================
    # 讨论问题
    # ================================================================
    print("\n" + "=" * 60)
    print("讨论问题")
    print("=" * 60)
    print(
        "在 data_prep.py 中，我们使用了全量数据的均值来填补缺失值。\n"
        "然后在 5 折交叉验证时，训练集和验证集都是从这份'全量填补后'的数据中划分的。\n"
        "这引发了一个数据泄露（Data Leakage）问题：\n\n"
        "  验证集中的缺失值，实际上是用包含了验证集自身信息的全量均值填补的，\n"
        "  因此验证集并非真正意义上的'完全未见过的陌生数据'。\n"
        "  这会导致交叉验证给出的 R² 分数乐观偏高（overly optimistic）。\n\n"
        "正确的做法是：在交叉验证的每一折中，仅使用训练集的均值来填补\n"
        "训练集和验证集的缺失值，这样才能得到无偏的评估结果。"
    )


if __name__ == "__main__":
    main()