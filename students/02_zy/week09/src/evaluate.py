import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 让 week9/evaluate.py 能找到 utils 文件夹
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.models import CustomOLS
from utils.diagnostics import calculate_vif


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def k_fold_cv(X, y, k=5):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    scores = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = CustomOLS()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)
        scores.append(score)

    return scores


def evaluate(input_path: str, target: str | None = None) -> None:
    df = pd.read_csv(input_path)

    if target is None:
        if "sales" in df.columns:
            target = "sales"
        elif "Sales" in df.columns:
            target = "Sales"
        else:
            target = df.columns[-1]

    print(f"Target column: {target}")

    y = df[target].values.astype(float)
    X_df = df.drop(columns=[target])
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    # 1. VIF 多重共线性诊断
    vif_values = calculate_vif(X)

    print("\n===== VIF Diagnostics =====")
    for name, vif in zip(feature_names, vif_values):
        print(f"{name}: VIF = {vif:.4f}")

        if vif > 10:
            print(f"\033[91mWarning: {name} has serious multicollinearity! VIF = {vif:.4f}\033[0m")

    # 2. 5 折交叉验证
    scores = k_fold_cv(X, y, k=5)

    print("\n===== 5-Fold Cross Validation =====")
    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: R^2 = {score:.4f}")

    print(f"\nAverage R^2: {np.mean(scores):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 9 Model Diagnostics and Baseline CV")

    parser.add_argument(
        "--input",
        required=True,
        help="清洗后的数据路径，例如 data/clean_marketing.csv",
    )

    parser.add_argument(
        "--target",
        required=False,
        default=None,
        help="目标变量列名，例如 sales。如果不写，默认找 sales，否则使用最后一列。",
    )

    args = parser.parse_args()

    evaluate(args.input, args.target)


if __name__ == "__main__":
    main()