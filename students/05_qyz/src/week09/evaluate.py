"""
Week 9 模型诊断与交叉验证评估
"""

import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif_dataframe, print_vif_warning


def main():
    """主函数：诊断与评估流水线"""

    # 加载清洁数据（在 src/week09/data/ 目录下）
    data_path = Path(__file__).parent / "data" / "clean_marketing.csv"

    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        print(
            "请先运行: uv run src/week09/data_prep.py --input <原始数据> --output src/week09/data/clean_marketing.csv"
        )
        return

    df = pd.read_csv(data_path)
    df = df.replace({True: 1, False: 0})
    df = df.astype(float)
    print(f"[OK] 已加载清洁数据，形状: {df.shape}")

    # 分离特征和目标变量
    feature_cols = df.columns[:-1].tolist()
    X = df[feature_cols].values
    y = df.iloc[:, -1].values

    # 多重共线性诊断
    print("\n" + "=" * 60)
    print("第一阶段：多重共线性诊断（VIF）")
    print("=" * 60)

    vif_df = calculate_vif_dataframe(df, feature_cols)
    print_vif_warning(vif_df)

    print("\n[INFO] VIF 解释:")
    print("   VIF = 1      : 完全不相关")
    print("   1 < VIF < 5  : 中等相关，可接受")
    print("   VIF >= 10    : 严重多重共线性，需要处理")

    # 5折交叉验证
    print("\n" + "=" * 60)
    print("第二阶段：基线模型交叉验证（5-Fold CV）")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS(fit_intercept=True)
        model.fit(X_train, y_train)
        r2 = model.score(X_val, y_val)
        r2_scores.append(r2)

        print(f"  Fold {fold}: R² = {r2:.4f}")

    avg_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"\n[RESULT] 平均 R²: {avg_r2:.4f} (+-{std_r2:.4f})")

    # 思考题
    print("\n" + "=" * 60)
    print("[DISCUSSION] 思考题")
    print("=" * 60)
    print("""问题：用全量数据均值填补缺失值，验证集还算"未见过的数据"吗？

答案：不算！验证集信息已泄露到训练过程中，R² 会被虚高。
正确做法：只用训练集统计量处理验证集。""")


if __name__ == "__main__":
    main()
