import os
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from src.utils.models import GradientDescentOLS
from src.utils.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
)
from src.utils.transformers import (
    CustomStandardScaler
)

# =========================
# 使用 Week09 数据
# =========================
DATA_PATH = "src/week09/data/dirty_marketing.csv"

RESULTS_DIR = "results"


# =========================
# 自动清空 results/
# =========================
def prepare_results_dir():

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(RESULTS_DIR)


# =========================
# 读取数据
# =========================
def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Sales"])
    y = df["Sales"]

    # One-Hot编码
    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y


# =========================
# 缺失值填补
# =========================
def fill_nan_with_values(
    X,
    fill_values
):

    X_filled = X.copy()

    return X_filled.fillna(
        fill_values
    )


# =========================
# 计算指标
# =========================
def evaluate_metrics(
    y_true,
    y_pred
):

    return {
        "RMSE": calculate_rmse(
            y_true,
            y_pred
        ),
        "MAE": calculate_mae(
            y_true,
            y_pred
        ),
        "MAPE": calculate_mape(
            y_true,
            y_pred
        ),
    }


# =========================
# 训练GD模型
# =========================
def train_gradient_descent_model(
    X_train,
    y_train
):

    model = GradientDescentOLS(
        learning_rate=0.01,
        max_iter=1000,
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ==================================================
# Bad CV（存在数据泄漏）
# ==================================================
def bad_cross_validation():

    print(
        "\n========== Bad Cross Validation =========="
    )
    print(
        "存在数据泄漏\n"
    )

    X, y = load_data()

    # 全局均值填补（错误）
    global_mean = X.mean()

    X = fill_nan_with_values(
        X,
        global_mean
    )

    # 全局标准化（错误）
    scaler = CustomStandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    y = y.values

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        kf.split(X_scaled)
    ):

        X_train = X_scaled[
            train_idx
        ]

        X_val = X_scaled[
            val_idx
        ]

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        model = train_gradient_descent_model(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_val
        )

        metrics = evaluate_metrics(
            y_val,
            y_pred
        )

        rmse_scores.append(
            metrics["RMSE"]
        )

        mae_scores.append(
            metrics["MAE"]
        )

        mape_scores.append(
            metrics["MAPE"]
        )

        print(
            f"Fold {fold+1} RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

    result = {
        "RMSE": np.mean(
            rmse_scores
        ),
        "MAE": np.mean(
            mae_scores
        ),
        "MAPE": np.mean(
            mape_scores
        ),
    }

    print(
        f"\nBad CV 平均RMSE: "
        f"{result['RMSE']:.4f}"
    )

    return result


# ==================================================
# Good CV（无数据泄漏）
# ==================================================
def good_cross_validation():

    print(
        "\n========== Good Cross Validation =========="
    )
    print(
        "无数据泄漏\n"
    )

    X, y = load_data()

    y = y.values

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        kf.split(X)
    ):

        X_train = X.iloc[
            train_idx
        ].copy()

        X_val = X.iloc[
            val_idx
        ].copy()

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        # 只用训练集均值
        train_mean = X_train.mean()

        X_train = fill_nan_with_values(
            X_train,
            train_mean
        )

        X_val = fill_nan_with_values(
            X_val,
            train_mean
        )

        # 只fit训练集
        scaler = CustomStandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        # 验证集只能transform
        X_val_scaled = scaler.transform(
            X_val
        )

        model = train_gradient_descent_model(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_val_scaled
        )

        metrics = evaluate_metrics(
            y_val,
            y_pred
        )

        rmse_scores.append(
            metrics["RMSE"]
        )

        mae_scores.append(
            metrics["MAE"]
        )

        mape_scores.append(
            metrics["MAPE"]
        )

        print(
            f"Fold {fold+1} RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

    result = {
        "RMSE": np.mean(
            rmse_scores
        ),
        "MAE": np.mean(
            mae_scores
        ),
        "MAPE": np.mean(
            mape_scores
        ),
    }

    print(
        f"\nGood CV 平均RMSE: "
        f"{result['RMSE']:.4f}"
    )

    return result


# =========================
# 保存报告
# =========================
def save_comparison_report(
    bad_result,
    good_result
):

    output_path = os.path.join(
        RESULTS_DIR,
        "evaluation_comparison.md"
    )

    content = f"""# Evaluation Comparison

| Method | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Bad CV | {bad_result["RMSE"]:.4f} | {bad_result["MAE"]:.4f} | {bad_result["MAPE"]:.4f}% |
| Good CV | {good_result["RMSE"]:.4f} | {good_result["MAE"]:.4f} | {good_result["MAPE"]:.4f}% |

## Explanation

Bad Cross Validation：

先对全量数据进行了缺失值填补和标准化，
导致验证集信息提前泄漏。

Good Cross Validation：

严格在每一折内部使用训练集参数进行
fillna 和 scaler fit。

因此更真实反映模型泛化能力。
"""

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(content)

    print(
        f"\n报告已保存到: "
        f"{output_path}"
    )


# =========================
# 主函数
# =========================
def main():

    prepare_results_dir()

    bad_result = bad_cross_validation()

    good_result = good_cross_validation()

    save_comparison_report(
        bad_result,
        good_result
    )


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    main()