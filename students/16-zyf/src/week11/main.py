import os
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

from src.utils.models import GradientDescentOLS
from src.utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from src.utils.transformers import CustomStandardScaler
from src.utils.diagnostics import calculate_vif


DATA_DIR = "src/week11/data"
RESULTS_DIR = "src/week11/results"

SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_regression.csv")
KAGGLE_PATH = os.path.join(DATA_DIR, "train.csv")
KAGGLE_TARGET = "TARGET(PRICE_IN_LACS)"


def prepare_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(RESULTS_DIR, exist_ok=True)


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def generate_synthetic_data(n_samples=400, random_state=42):
    np.random.seed(random_state)

    tv_budget = np.random.uniform(50, 300, n_samples)
    online_video_budget = 0.85 * tv_budget + np.random.normal(0, 10, n_samples)
    radio_budget = np.random.uniform(10, 100, n_samples)
    competitor_price = np.random.uniform(80, 200, n_samples)

    region = np.random.choice(
        ["East", "North", "South", "West"],
        size=n_samples,
    )

    region_effect = {
        "East": 0,
        "North": 60,
        "South": 30,
        "West": -20,
    }

    noise = np.random.normal(0, 35, n_samples)

    sales = (
        100
        + 2.6 * tv_budget
        + 0.4 * online_video_budget
        + 0.8 * radio_budget
        - 1.5 * competitor_price
        + np.array([region_effect[r] for r in region])
        + noise
    )

    df = pd.DataFrame(
        {
            "TV_Budget": tv_budget,
            "Online_Video_Budget": online_video_budget,
            "Radio_Budget": radio_budget,
            "Competitor_Price": competitor_price,
            "Region": region,
            "Sales": sales,
        }
    )

    for col in ["TV_Budget", "Radio_Budget"]:
        missing_idx = np.random.choice(
            df.index,
            size=int(0.05 * n_samples),
            replace=False,
        )
        df.loc[missing_idx, col] = np.nan

    outlier_idx = np.random.choice(
        df.index,
        size=int(0.03 * n_samples),
        replace=False,
    )
    df.loc[outlier_idx, "Radio_Budget"] *= 15

    df.to_csv(SYNTHETIC_PATH, index=False)
    print(f"模拟数据已保存到: {SYNTHETIC_PATH}")


def fill_missing_by_train(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        train_median = X_train[numeric_cols].median()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(train_median)
        X_val[numeric_cols] = X_val[numeric_cols].fillna(train_median)

    for col in categorical_cols:
        mode_value = X_train[col].mode()
        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
        X_train[col] = X_train[col].fillna(fill_value)
        X_val[col] = X_val[col].fillna(fill_value)

    return X_train, X_val


def winsorize_by_train(X_train, X_val, q_low=0.01, q_high=0.99):
    X_train = X_train.copy()
    X_val = X_val.copy()

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        lower = X_train[col].quantile(q_low)
        upper = X_train[col].quantile(q_high)

        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)

    return X_train, X_val


def one_hot_align(X_train, X_val):
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    return X_train, X_val


def leakage_free_cv(df, target_col, drop_cols=None, use_gd=True):
    if drop_cols is None:
        drop_cols = []

    df = df.copy()
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train, X_val = fill_missing_by_train(X_train, X_val)
        X_train, X_val = winsorize_by_train(X_train, X_val)
        X_train, X_val = one_hot_align(X_train, X_val)

        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_gd:
            X_train_scaled = add_intercept(X_train_scaled)
            X_val_scaled = add_intercept(X_val_scaled)

            model = GradientDescentOLS(
                learning_rate=0.001,
                max_iter=3000,
            )
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rmse = calculate_rmse(y_val, y_pred)
        mae = calculate_mae(y_val, y_pred)
        mape = calculate_mape(y_val, y_pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)

        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    return {
        "RMSE": float(np.mean(rmse_scores)),
        "MAE": float(np.mean(mae_scores)),
        "MAPE": float(np.mean(mape_scores)),
    }


def prepare_full_data_for_vif(df, target_col, drop_cols=None, max_features=30):
    if drop_cols is None:
        drop_cols = []

    df = df.copy()
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        for col in numeric_cols:
            lower = X[col].quantile(0.01)
            upper = X[col].quantile(0.99)
            X[col] = X[col].clip(lower, upper)

    for col in categorical_cols:
        mode_value = X[col].mode()
        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
        X[col] = X[col].fillna(fill_value)

    X = pd.get_dummies(X, drop_first=True).astype(float)

    if X.shape[1] > max_features:
        X = X.iloc[:, :max_features]

    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X.columns.tolist()


def run_vif(df, target_col, drop_cols=None):
    X_scaled, feature_names = prepare_full_data_for_vif(
        df=df,
        target_col=target_col,
        drop_cols=drop_cols,
    )

    return calculate_vif(X_scaled, feature_names)


def write_synthetic_report(result, vif_result):
    output_path = os.path.join(RESULTS_DIR, "synthetic_report.md")

    vif_text = "\n".join(
        [f"- {name}: {value:.4f}" for name, value in vif_result]
    )

    content = f"""# Week 11 Task A Synthetic Report

## 1. Data Generating Process

本部分构造了一份广告预算与销售额的模拟回归数据。每一行代表一次广告投放记录，目标变量为 Sales。

生成公式为：

Sales = 100
      + 2.6 * TV_Budget
      + 0.4 * Online_Video_Budget
      + 0.8 * Radio_Budget
      - 1.5 * Competitor_Price
      + Region_Effect
      + noise

其中：

Online_Video_Budget = 0.85 * TV_Budget + random_noise

因此 TV_Budget 和 Online_Video_Budget 被故意设计为高度相关变量。

## 2. Expected Direction

- TV_Budget：正向影响 Sales
- Online_Video_Budget：正向影响 Sales
- Radio_Budget：正向影响 Sales
- Competitor_Price：负向影响 Sales
- Region_North：相对 East 正向影响 Sales
- Region_South：相对 East 正向影响 Sales
- Region_West：相对 East 负向影响 Sales

## 3. Added Data Problems

本模拟数据主动加入了缺失值、异常值、量纲差异和多重共线性。

## 4. Leakage-Free CV Results

| Metric | Value |
|---|---:|
| RMSE | {result["RMSE"]:.4f} |
| MAE | {result["MAE"]:.4f} |
| MAPE | {result["MAPE"]:.4f}% |

## 5. VIF Diagnostics

{vif_text}

## 6. Inference Discussion

由于模拟数据的 DGP 是已知的，所以可以检查模型估计结果是否与预设方向一致。广告预算类变量应当与销售额正相关，竞争对手价格应当与销售额负相关。

但是，TV_Budget 和 Online_Video_Budget 被人为构造成高度相关，因此单独解释这两个变量的系数时需要谨慎。
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"模拟数据报告已保存到: {output_path}")


def run_synthetic_task():
    print("\n========== Task A: Synthetic Data ==========")

    generate_synthetic_data()

    df = pd.read_csv(SYNTHETIC_PATH)

    result = leakage_free_cv(
        df=df,
        target_col="Sales",
        drop_cols=None,
        use_gd=True,
    )

    vif_result = run_vif(
        df=df,
        target_col="Sales",
        drop_cols=None,
    )

    write_synthetic_report(result, vif_result)

    return result


def write_kaggle_report(result, baseline_result, vif_result, df):
    output_path = os.path.join(RESULTS_DIR, "kaggle_report.md")

    vif_text = "\n".join(
        [f"- {name}: {value:.4f}" for name, value in vif_result]
    )

    content = f"""# Week 11 Task B Kaggle Report

## 1. Dataset Information

本部分使用 Kaggle 房产价格回归数据，原始文件名为 train.csv。

- 样本量：{df.shape[0]}
- 字段数：{df.shape[1]}
- 目标变量：TARGET(PRICE_IN_LACS)
- 每一行样本代表一套房产记录
- 任务目标：根据房屋面积、房型、是否在建、是否可入住、经纬度等信息预测房产价格

## 2. Why This Dataset Is Suitable

该数据适合本周回归作业，原因包括：

- 目标变量是连续变量
- 样本量大于 200
- 包含数值变量和类别变量
- 房价数据通常存在异常值和偏态分布
- 存在业务上可能相关的变量，例如 UNDER_CONSTRUCTION 和 READY_TO_MOVE

## 3. Cleaning Strategy

本流程中删除 ADDRESS 字段，因为该字段取值过多，直接 One-Hot 编码会造成维度过高。

主要处理包括：

- 数值变量使用训练集 median 填补缺失值
- 类别变量使用训练集 mode 填补缺失值
- 数值变量使用训练集 1% 和 99% 分位数进行缩尾
- 类别变量进行 One-Hot 编码
- 标准化只在训练集 fit，再 transform 验证集
- 使用 5 折无泄露交叉验证

## 4. Custom Utils Workflow Results

| Metric | Value |
|---|---:|
| RMSE | {result["RMSE"]:.4f} |
| MAE | {result["MAE"]:.4f} |
| MAPE | {result["MAPE"]:.4f}% |

## 5. Sklearn Ridge Baseline

| Metric | Value |
|---|---:|
| RMSE | {baseline_result["RMSE"]:.4f} |
| MAE | {baseline_result["MAE"]:.4f} |
| MAPE | {baseline_result["MAPE"]:.4f}% |

## 6. VIF Diagnostics

{vif_text}

## 7. Inference Discussion

真实房价数据的解释比模拟数据更困难。虽然 SQUARE_FT、BHK_NO.、房屋状态、经纬度等变量都可能与房价有关，但我们并不知道真实的数据生成机制。

此外，房价往往受到地段、时间、房屋质量等未观测因素影响。因此即使模型误差较低，也不能简单把某个变量解释为因果作用。

如果模型用于真实业务，我最担心的是高价房、特殊地段房产和信息缺失样本上的预测误差。
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Kaggle 报告已保存到: {output_path}")


def run_kaggle_task():
    print("\n========== Task B: Kaggle Real Data ==========")

    if not os.path.exists(KAGGLE_PATH):
        raise FileNotFoundError(
            f"没有找到 Kaggle 数据文件，请把 train.csv 放到: {KAGGLE_PATH}"
        )

    df = pd.read_csv(KAGGLE_PATH)

    drop_cols = ["ADDRESS"]

    print("Kaggle 数据形状:", df.shape)
    print("目标变量:", KAGGLE_TARGET)

    result = leakage_free_cv(
        df=df,
        target_col=KAGGLE_TARGET,
        drop_cols=drop_cols,
        use_gd=True,
    )

    baseline_result = leakage_free_cv(
        df=df,
        target_col=KAGGLE_TARGET,
        drop_cols=drop_cols,
        use_gd=False,
    )

    vif_result = run_vif(
        df=df,
        target_col=KAGGLE_TARGET,
        drop_cols=drop_cols,
    )

    write_kaggle_report(
        result=result,
        baseline_result=baseline_result,
        vif_result=vif_result,
        df=df,
    )

    return result


def write_summary_report(synthetic_result, kaggle_result):
    output_path = os.path.join(RESULTS_DIR, "summary_comparison.md")

    content = f"""# Week 11 Summary Comparison Report

## 1. Metric Comparison

| Task | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Synthetic Data | {synthetic_result["RMSE"]:.4f} | {synthetic_result["MAE"]:.4f} | {synthetic_result["MAPE"]:.4f}% |
| Kaggle Real Data | {kaggle_result["RMSE"]:.4f} | {kaggle_result["MAE"]:.4f} | {kaggle_result["MAPE"]:.4f}% |

## 2. Synthetic Data vs Real Data

模拟数据的优势是数据生成机制已知，因此可以直接判断模型识别出的变量方向是否符合预期。

真实 Kaggle 数据更接近业务场景，但解释难度更高。即使模型分数较好，也不能说明某个变量一定具有因果影响。

## 3. Influence of Data Problems

在模拟数据中，共线性、缺失值和异常值是人为加入的，因此来源明确、影响较容易解释。

在真实数据中，这些问题可能来自业务采集、录入错误、城市差异或隐藏变量，解释起来更复杂。

## 4. Why Leakage-Free CV Matters

无泄露交叉验证在真实数据中尤其重要。如果在全量数据上提前计算均值、标准差或异常值阈值，验证集信息就会提前进入训练流程，导致结果虚高。

本周流程中，所有会学习参数的步骤都放在每一折训练集内部完成，再应用到验证集，因此更接近真实上线场景。

## 5. Reuse of Utils

本周复用了以下自定义组件：

- GradientDescentOLS：主要回归模型
- CustomStandardScaler：标准化
- calculate_rmse / calculate_mae / calculate_mape：误差指标
- calculate_vif：共线性诊断

这些组件让模拟数据和真实数据可以使用统一的分析流程，也减少了重复代码。
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"总结报告已保存到: {output_path}")


def main():
    prepare_dirs()

    synthetic_result = run_synthetic_task()

    kaggle_result = run_kaggle_task()

    write_summary_report(
        synthetic_result=synthetic_result,
        kaggle_result=kaggle_result,
    )


if __name__ == "__main__":
    main()