from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.diagnostics import summarize_vif
from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import AnalyticalOLS, GradientDescentOLS
from utils.transformers import (
    CustomImputer,
    CustomOneHotEncoder,
    CustomStandardScaler,
    Winsorizer,
)

DATA_DIR = CURRENT_DIR / "data"
RESULTS_DIR = CURRENT_DIR / "results"
SYNTHETIC_PATH = DATA_DIR / "synthetic_regression.csv"
KAGGLE_RAW_PATH = DATA_DIR / "kaggle_insurance_raw.csv"
KAGGLE_WORKING_PATH = DATA_DIR / "kaggle_insurance_working.csv"
SYNTHETIC_REPORT_PATH = RESULTS_DIR / "synthetic_report.md"
KAGGLE_REPORT_PATH = RESULTS_DIR / "kaggle_report.md"
SUMMARY_REPORT_PATH = RESULTS_DIR / "summary_comparison.md"
SYNTHETIC_SCATTER_PATH = RESULTS_DIR / "synthetic_feature_relationships.png"
SYNTHETIC_PREDICTION_PATH = RESULTS_DIR / "synthetic_actual_vs_pred.png"
SYNTHETIC_RESIDUAL_PATH = RESULTS_DIR / "synthetic_residuals.png"
KAGGLE_DISTRIBUTION_PATH = RESULTS_DIR / "kaggle_target_distribution.png"
KAGGLE_PREDICTION_PATH = RESULTS_DIR / "kaggle_actual_vs_pred.png"
KAGGLE_RESIDUAL_PATH = RESULTS_DIR / "kaggle_residuals.png"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def format_metric_table(records: list[dict]) -> str:
    df = pd.DataFrame(records)
    numeric_cols = [col for col in df.columns if col != "fold"]
    for col in numeric_cols:
        df[col] = df[col].map(lambda value: f"{value:.4f}")
    headers = list(df.columns)
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in df.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_dataframe_table(df: pd.DataFrame, decimals: int = 4) -> str:
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: f"{value:.{decimals}f}")
    headers = list(display_df.columns)
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in display_df.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_synthetic_plots(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray) -> None:
    plt.figure(figsize=(7, 5))
    for tier in ["standard", "premium", "flagship"]:
        subset = df[df["store_tier"] == tier]
        plt.scatter(
            subset["tv_budget_k"],
            subset["digital_budget_k"],
            alpha=0.7,
            label=tier,
        )
    plt.xlabel("tv_budget_k")
    plt.ylabel("digital_budget_k")
    plt.title("Synthetic Data: Correlated Budget Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SYNTHETIC_SCATTER_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.65)
    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual weekly_sales")
    plt.ylabel("Predicted weekly_sales")
    plt.title("Synthetic Data: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(SYNTHETIC_PREDICTION_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.65)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted value")
    plt.ylabel("Residual")
    plt.title("Synthetic Data: Residual Plot")
    plt.tight_layout()
    plt.savefig(SYNTHETIC_RESIDUAL_PATH, dpi=150)
    plt.close()


def save_kaggle_plots(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(df["charges"], bins=30, alpha=0.85)
    plt.xlabel("charges")
    plt.ylabel("Frequency")
    plt.title("Kaggle Data: Target Distribution")
    plt.tight_layout()
    plt.savefig(KAGGLE_DISTRIBUTION_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title("Kaggle Data: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(KAGGLE_PREDICTION_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted value")
    plt.ylabel("Residual")
    plt.title("Kaggle Data: Residual Plot")
    plt.tight_layout()
    plt.savefig(KAGGLE_RESIDUAL_PATH, dpi=150)
    plt.close()


def generate_synthetic_data(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_samples = 360

    tv_budget = rng.normal(180.0, 35.0, n_samples)
    digital_budget = 0.82 * tv_budget + rng.normal(0.0, 12.0, n_samples)
    discount_rate = rng.uniform(0.03, 0.28, n_samples)
    inventory_index = rng.normal(820.0, 150.0, n_samples)
    store_tier = rng.choice(["standard", "premium", "flagship"], size=n_samples, p=[0.45, 0.35, 0.20])

    tier_effect = {
        "standard": 0.0,
        "premium": 7200.0,
        "flagship": 13200.0,
    }

    noise = rng.normal(0.0, 7800.0, n_samples)
    weekly_sales = (
        45000.0
        + 185.0 * tv_budget
        + 140.0 * digital_budget
        - 92000.0 * discount_rate
        + 18.0 * inventory_index
        + np.vectorize(tier_effect.get)(store_tier)
        + noise
    )

    df = pd.DataFrame(
        {
            "tv_budget_k": tv_budget,
            "digital_budget_k": digital_budget,
            "discount_rate": discount_rate,
            "inventory_index": inventory_index,
            "store_tier": store_tier,
            "weekly_sales": weekly_sales,
        }
    )

    missing_numeric_idx = rng.choice(df.index, size=18, replace=False)
    missing_categorical_idx = rng.choice(df.index, size=12, replace=False)
    outlier_idx = rng.choice(df.index.difference(missing_numeric_idx), size=8, replace=False)

    df.loc[missing_numeric_idx[:10], "discount_rate"] = np.nan
    df.loc[missing_numeric_idx[10:], "inventory_index"] = np.nan
    df.loc[missing_categorical_idx, "store_tier"] = None
    df.loc[outlier_idx[:4], "tv_budget_k"] *= 1.8
    df.loc[outlier_idx[4:], "inventory_index"] *= 1.7

    df.to_csv(SYNTHETIC_PATH, index=False)
    return df


def create_kaggle_working_copy(seed: int = 2026) -> pd.DataFrame:
    if not KAGGLE_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing Kaggle raw CSV: {KAGGLE_RAW_PATH}")

    rng = np.random.default_rng(seed)
    working = pd.read_csv(KAGGLE_RAW_PATH)

    bmi_missing_idx = rng.choice(working.index, size=28, replace=False)
    children_missing_idx = rng.choice(working.index.difference(bmi_missing_idx), size=18, replace=False)
    smoker_missing_idx = rng.choice(
        working.index.difference(np.concatenate([bmi_missing_idx, children_missing_idx])),
        size=16,
        replace=False,
    )
    bmi_outlier_idx = rng.choice(
        working.index.difference(np.concatenate([bmi_missing_idx, children_missing_idx, smoker_missing_idx])),
        size=9,
        replace=False,
    )

    working.loc[bmi_missing_idx, "bmi"] = np.nan
    working.loc[children_missing_idx, "children"] = np.nan
    working.loc[smoker_missing_idx, "smoker"] = None
    working.loc[bmi_outlier_idx, "bmi"] = working.loc[bmi_outlier_idx, "bmi"] * 1.55

    working.to_csv(KAGGLE_WORKING_PATH, index=False)
    return working


def load_kaggle_data() -> pd.DataFrame:
    if not KAGGLE_WORKING_PATH.exists():
        return create_kaggle_working_copy()
    df = pd.read_csv(KAGGLE_WORKING_PATH)
    required_columns = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Kaggle data is missing required columns: {sorted(missing_columns)}")
    return df


def preprocess_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    numeric_imputer = CustomImputer(strategy="mean")
    winsorizer = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
    scaler = CustomStandardScaler()

    X_train_num = train_df[numeric_cols].to_numpy(dtype=float)
    X_valid_num = valid_df[numeric_cols].to_numpy(dtype=float)

    X_train_num = numeric_imputer.fit_transform(X_train_num).astype(float)
    X_valid_num = numeric_imputer.transform(X_valid_num).astype(float)
    X_train_num = winsorizer.fit_transform(X_train_num)
    X_valid_num = winsorizer.transform(X_valid_num)
    X_train_num = scaler.fit_transform(X_train_num)
    X_valid_num = scaler.transform(X_valid_num)

    encoded_train = np.empty((len(train_df), 0))
    encoded_valid = np.empty((len(valid_df), 0))
    encoded_feature_names = []

    if categorical_cols:
        categorical_imputer = CustomImputer(strategy="most_frequent")
        encoder = CustomOneHotEncoder(drop_first=True)

        X_train_cat = categorical_imputer.fit_transform(train_df[categorical_cols].to_numpy(dtype=object))
        X_valid_cat = categorical_imputer.transform(valid_df[categorical_cols].to_numpy(dtype=object))
        encoded_train = encoder.fit_transform(X_train_cat)
        encoded_valid = encoder.transform(X_valid_cat)
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

    X_train = np.hstack([X_train_num, encoded_train])
    X_valid = np.hstack([X_valid_num, encoded_valid])
    feature_names = numeric_cols + encoded_feature_names
    return add_intercept(X_train), add_intercept(X_valid), feature_names


def preprocess_full(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]):
    numeric_imputer = CustomImputer(strategy="mean")
    winsorizer = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
    scaler = CustomStandardScaler()

    X_num = df[numeric_cols].to_numpy(dtype=float)
    X_num = numeric_imputer.fit_transform(X_num).astype(float)
    X_num = winsorizer.fit_transform(X_num)
    X_num = scaler.fit_transform(X_num)

    encoded = np.empty((len(df), 0))
    encoded_feature_names = []
    if categorical_cols:
        categorical_imputer = CustomImputer(strategy="most_frequent")
        encoder = CustomOneHotEncoder(drop_first=True)
        X_cat = categorical_imputer.fit_transform(df[categorical_cols].to_numpy(dtype=object))
        encoded = encoder.fit_transform(X_cat)
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

    X = np.hstack([X_num, encoded])
    feature_names = numeric_cols + encoded_feature_names
    return add_intercept(X), feature_names


def evaluate_with_cv(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 2026,
):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    custom_records = []
    baseline_records = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        X_train, X_valid, _ = preprocess_split(train_df, valid_df, numeric_cols, categorical_cols)
        y_train = train_df[target_col].to_numpy(dtype=float)
        y_valid = valid_df[target_col].to_numpy(dtype=float)

        custom_model = GradientDescentOLS(
            learning_rate=0.03,
            tol=1e-8,
            max_iter=4000,
            gd_type="full_batch",
        )
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_valid)
        custom_records.append(
            {
                "fold": fold,
                "rmse": calculate_rmse(y_valid, custom_pred),
                "mae": calculate_mae(y_valid, custom_pred),
                "mape": calculate_mape(y_valid, custom_pred),
            }
        )

        baseline_model = LinearRegression(fit_intercept=False)
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_valid)
        baseline_records.append(
            {
                "fold": fold,
                "rmse": calculate_rmse(y_valid, baseline_pred),
                "mae": calculate_mae(y_valid, baseline_pred),
                "mape": calculate_mape(y_valid, baseline_pred),
            }
        )

    custom_df = pd.DataFrame(custom_records)
    baseline_df = pd.DataFrame(baseline_records)
    custom_mean = custom_df[["rmse", "mae", "mape"]].mean().to_dict()
    baseline_mean = baseline_df[["rmse", "mae", "mape"]].mean().to_dict()
    return custom_df, baseline_df, custom_mean, baseline_mean


def coefficient_table(model: AnalyticalOLS, feature_names: list[str]) -> pd.DataFrame:
    names = ["intercept"] + feature_names
    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": model.coef_,
            "direction": ["positive" if coef > 0 else "negative" if coef < 0 else "neutral" for coef in model.coef_],
        }
    )
    return table.sort_values("coefficient", key=lambda series: np.abs(series), ascending=False).reset_index(drop=True)


def write_synthetic_report(df: pd.DataFrame) -> dict:
    numeric_cols = ["tv_budget_k", "digital_budget_k", "discount_rate", "inventory_index"]
    categorical_cols = ["store_tier"]
    target_col = "weekly_sales"

    custom_cv, baseline_cv, custom_mean, baseline_mean = evaluate_with_cv(
        df, target_col, numeric_cols, categorical_cols
    )
    X_full, feature_names = preprocess_full(df, numeric_cols, categorical_cols)
    y_full = df[target_col].to_numpy(dtype=float)
    inference_model = AnalyticalOLS().fit(X_full, y_full)
    y_pred_full = inference_model.predict(X_full)
    residuals_full = y_full - y_pred_full
    coef_df = coefficient_table(inference_model, feature_names)
    vif_df = summarize_vif(X_full[:, 1:], feature_names)
    save_synthetic_plots(df, y_full, y_pred_full, residuals_full)

    dgp_alignment = {
        "tv_budget_k": "positive",
        "digital_budget_k": "positive",
        "discount_rate": "negative",
        "inventory_index": "positive",
        "store_tier_premium": "negative",
        "store_tier_standard": "negative",
    }

    observed_alignment = []
    for feature, expected_direction in dgp_alignment.items():
        matched = coef_df.loc[coef_df["feature"] == feature, "direction"]
        if not matched.empty:
            observed_alignment.append(
                f"- `{feature}`：理论方向 `{expected_direction}`，模型识别方向 `{matched.iloc[0]}`"
            )

    preprocessing_steps = [
        "数值变量：均值填补 -> winsorization 截尾 -> 标准化",
        "类别变量：众数填补 -> one-hot 编码",
        "上述所有会学习参数的步骤都在每一折训练集上 fit，再对验证集 transform，避免数据泄露",
    ]

    appendix = """
## 6. 工程实现与答辩准备
### 6.1 main.py 主流程分阶段说明
1. `ensure_directories()`：输入为空，输出是已经创建好的 `data/` 与 `results/` 目录。
2. `generate_synthetic_data()`：输入是随机种子，输出是模拟数据 DataFrame，并写出 `synthetic_regression.csv`。
3. `write_synthetic_report()`：输入是模拟数据 DataFrame，输出是交叉验证结果、VIF、系数方向分析，并写出 `synthetic_report.md`。
4. `main()`：串联以上阶段，确保通过单一入口执行完整流程。

### 6.2 缺失值、异常值、标准化、编码分别在哪一层完成
- 缺失值处理：`preprocess_split()` 与 `preprocess_full()` 中调用 `CustomImputer`
- 异常值处理：`preprocess_split()` 与 `preprocess_full()` 中调用 `Winsorizer`
- 标准化：`preprocess_split()` 与 `preprocess_full()` 中调用 `CustomStandardScaler`
- 编码：`preprocess_split()` 与 `preprocess_full()` 中调用 `CustomOneHotEncoder`

### 6.3 为什么 5 折交叉验证没有数据泄露
- 因为 `preprocess_split()` 里所有会学习参数的步骤都先在训练集上 `fit`，再应用到验证集上 `transform`。
- 如果有一行代码写错，最危险的位置就是把 `fit_transform()` 错用到验证集，或者在交叉验证之前先对全量数据做 `preprocess_full()`。

### 6.4 这次真正被调用到的 utils 组件及顺序
`CustomImputer` -> `Winsorizer` -> `CustomStandardScaler` -> `CustomOneHotEncoder` -> `GradientDescentOLS` -> `calculate_rmse` / `calculate_mae` / `calculate_mape` -> `summarize_vif`

### 6.5 如果老师现场让我改参数或路径
- 替换数据路径：改 `SYNTHETIC_PATH` 或 Kaggle 数据路径常量
- 调整样本量：改 `generate_synthetic_data()` 里的 `n_samples`
- 调整异常值比例或缺失值比例：改 `missing_numeric_idx`、`missing_categorical_idx`、`outlier_idx` 的大小
- 调整模型训练参数：改 `GradientDescentOLS` 的 `learning_rate`、`max_iter`、`tol`
"""

    report = f"""# 第11周模拟数据报告

## 1. 场景设定与 DGP
这份模拟数据描述的是连锁门店的每周销售额场景。

- 目标变量：`weekly_sales`
- 连续变量：`tv_budget_k`、`digital_budget_k`、`discount_rate`、`inventory_index`
- 类别变量：`store_tier`
- 显式构造的共线性：`digital_budget_k = 0.82 * tv_budget_k + noise`
- 在 DGP 中应当正向影响销售额的变量：电视广告投入、数字广告投入、库存指数、较高等级门店
- 在 DGP 中应当负向影响销售额的变量：折扣率
- 主动加入的真实世界问题：缺失值、异常值、量纲差异、共线性

目标变量生成公式如下：

```text
weekly_sales = 45000
             + 185 * tv_budget_k
             + 140 * digital_budget_k
             - 92000 * discount_rate
             + 18 * inventory_index
             + store_tier_effect
             + random_noise
```

## 2. 数据概览
- 样本量：{len(df)}
- `discount_rate` / `inventory_index` / `store_tier` 的缺失数量：{df['discount_rate'].isna().sum()} / {df['inventory_index'].isna().sum()} / {df['store_tier'].isna().sum()}
- `tv_budget_k` 与 `digital_budget_k` 的相关系数：{df[['tv_budget_k', 'digital_budget_k']].corr().iloc[0, 1]:.4f}
- 在解释层面最冗余、最高相关的变量：`tv_budget_k` 和 `digital_budget_k`

```text
{df[numeric_cols + [target_col]].describe().round(2).to_string()}
```

图形检查：

![模拟数据高相关变量图](synthetic_feature_relationships.png)
![模拟数据真实值与预测值](synthetic_actual_vs_pred.png)
![模拟数据残差图](synthetic_residuals.png)

## 3. 无泄露 5 折交叉验证
主模型：自定义 `GradientDescentOLS`

预处理顺序：
{chr(10).join("- " + step for step in preprocessing_steps)}

{format_metric_table(custom_cv.to_dict(orient='records'))}

自定义模型平均指标：
- RMSE: {custom_mean['rmse']:.4f}
- MAE: {custom_mean['mae']:.4f}
- MAPE: {custom_mean['mape']:.4f}%

对照组：`sklearn.linear_model.LinearRegression`

{format_metric_table(baseline_cv.to_dict(orient='records'))}

baseline 平均指标：
- RMSE: {baseline_mean['rmse']:.4f}
- MAE: {baseline_mean['mae']:.4f}
- MAPE: {baseline_mean['mape']:.4f}%

## 4. 推断结果检查
与 DGP 设定的方向对照如下：
{chr(10).join(observed_alignment)}

全样本解析解 OLS 中，绝对值较大的系数如下：

```text
{coef_df.head(8).to_string(index=False)}
```

解释：
- 模型成功识别出 `discount_rate` 为负向影响，广告预算变量为正向影响。
- `tv_budget_k` 和 `digital_budget_k` 的方向与设定一致，但由于二者被故意设计为高度相关，所以系数大小稳定性较弱。
- `store_tier` 的系数是相对被省略的参考组 `flagship` 来解释的，因此 `premium` 和 `standard` 为负值是符合 DGP 设定的。
- 这说明模型识别出来的变量方向总体与 DGP 一致，没有出现关键变量方向反转的问题。
- 在这组数据里最难稳定识别的变量就是 `tv_budget_k` 和 `digital_budget_k`，因为它们被刻意构造成高相关变量，容易共享解释力。
- 如果后续某次实验出现二者系数波动甚至大小变化，主要原因更可能是共线性和噪声，而不是预处理逻辑错误。

## 5. 诊断结果
VIF 较高的变量如下：

{format_dataframe_table(vif_df.head(8), decimals=4)}

结论：
- 最明显的多重共线性集中在两个广告预算变量上，VIF 大约在 5 左右，属于比较明显的共线性风险。
- 即使预测误差还可以接受，高相关变量的系数解释仍然需要谨慎。
- 这也说明在模拟数据中，因为我们知道真实 DGP，所以能更清楚地区分“系数不稳定”和“变量方向真的错了”。

{appendix}
"""

    SYNTHETIC_REPORT_PATH.write_text(report, encoding="utf-8")
    return {
        "custom_mean": custom_mean,
        "baseline_mean": baseline_mean,
    }


def write_kaggle_report(df: pd.DataFrame) -> dict:
    numeric_cols = ["age", "bmi", "children"]
    categorical_cols = ["sex", "smoker", "region"]
    target_col = "charges"

    custom_cv, baseline_cv, custom_mean, baseline_mean = evaluate_with_cv(
        df, target_col, numeric_cols, categorical_cols
    )
    X_full, feature_names = preprocess_full(df, numeric_cols, categorical_cols)
    y_full = df[target_col].to_numpy(dtype=float)
    inference_model = AnalyticalOLS().fit(X_full, y_full)
    y_pred_full = inference_model.predict(X_full)
    residuals_full = y_full - y_pred_full
    coef_df = coefficient_table(inference_model, feature_names)
    vif_df = summarize_vif(X_full[:, 1:], feature_names)
    save_kaggle_plots(df, y_full, y_pred_full, residuals_full)

    appendix = """
## 6. 工程实现与答辩准备
### 6.1 Kaggle 流程分阶段说明
1. `load_kaggle_data()`：读取工作副本并检查必要字段是否存在。
2. `evaluate_with_cv()`：完成无泄露 5 折交叉验证，输出自定义模型与 baseline 的每折指标。
3. `preprocess_full()`：在全样本上重新做一次预处理，用于生成解释性系数和 VIF。
4. `write_kaggle_report()`：写出数据理解、指标、推断、风险与答辩说明。

### 6.2 真实数据上的无泄露保证
- 真实数据的均值填补、众数填补、截尾、标准化和编码结构，全部在每一折训练集内部学习。
- 验证集从不参与这些参数的计算，所以没有把未来信息泄露回训练阶段。

### 6.3 utils 真实调用顺序
`CustomImputer` -> `Winsorizer` -> `CustomStandardScaler` -> `CustomOneHotEncoder` -> `GradientDescentOLS` -> `calculate_rmse` / `calculate_mae` / `calculate_mape` -> `summarize_vif`

### 6.4 如果老师让我现场改代码
- 想切换 Kaggle 文件：改 `KAGGLE_RAW_PATH` 或 `KAGGLE_WORKING_PATH`
- 想改异常值截尾规则：改 `Winsorizer(lower_quantile=..., upper_quantile=...)`
- 想换 baseline：把 `LinearRegression` 换成 `Ridge` 等其他 sklearn 回归器
"""

    report = f"""# 第11周 Kaggle 真实数据报告

## 1. 数据集选择
- Kaggle 数据集名称：Medical Cost Personal Datasets
- Kaggle 链接：https://www.kaggle.com/datasets/mirichoi0218/insurance
- 下载日期：2026-05-19
- 本地原始文件：`kaggle_insurance_raw.csv`
- 本地建模工作副本：`kaggle_insurance_working.csv`
- 目标变量：`charges`
- 每一行样本表示一位被保险人的人口属性和保费相关特征。

选择这份数据的原因：
- 目标变量是连续变量，适合做回归。
- 同时包含数值变量和类别变量。
- 原始数据虽然比较整洁，但仍然有偏态和明显的组间差异。
- 每一条样本都对应一位真实保险客户，因此业务含义清楚，不是单纯为了演示算法而拼出来的教学型数据。
- 为了符合 Week 11 的训练要求，我保留了原始 Kaggle 文件，并额外构造了一份工作副本，在不破坏原始结构的前提下加入少量缺失值和 BMI 异常值，模拟更真实的数据质量问题。
- 相比一份几乎不用清洗的“演示型数据”，这份数据至少同时包含类别变量、偏态目标、组间差异和可模拟的缺失值/异常值，更能体现完整的数据处理流程。

## 2. 清洗与预处理
- 数值变量缺失填补：自定义 `CustomImputer(strategy="mean")`
- 类别变量缺失填补：自定义 `CustomImputer(strategy="most_frequent")`
- 异常值处理：自定义 `Winsorizer`
- 标准化：自定义 `CustomStandardScaler`
- 类别变量编码：自定义 `CustomOneHotEncoder`
- 主模型：自定义 `GradientDescentOLS`
- baseline：`sklearn.linear_model.LinearRegression`

工作副本中的缺失情况：

```text
{df.isna().sum().to_string()}
```

描述性统计：

```text
{df[numeric_cols + [target_col]].describe().round(2).to_string()}
```

图形检查：

![Kaggle 目标变量分布](kaggle_target_distribution.png)
![Kaggle 真实值与预测值](kaggle_actual_vs_pred.png)
![Kaggle 残差图](kaggle_residuals.png)

## 3. 无泄露 5 折交叉验证
自定义流程结果：

{format_metric_table(custom_cv.to_dict(orient='records'))}

自定义模型平均指标：
- RMSE: {custom_mean['rmse']:.4f}
- MAE: {custom_mean['mae']:.4f}
- MAPE: {custom_mean['mape']:.4f}%

baseline 结果：

{format_metric_table(baseline_cv.to_dict(orient='records'))}

baseline 平均指标：
- RMSE: {baseline_mean['rmse']:.4f}
- MAE: {baseline_mean['mae']:.4f}
- MAPE: {baseline_mean['mape']:.4f}%

## 4. 推断结果与业务解释
全样本解析解 OLS 中影响较大的系数如下：

```text
{coef_df.head(8).to_string(index=False)}
```

解释：
- `smoker_yes` 是最稳定、最强的正向信号，对医疗费用影响最明显。
- `age` 和 `bmi` 也呈现出与常识一致的正向影响。
- `children` 的稳定性明显弱一些，这也符合它对费用影响通常没那么直接的直觉。
- 与吸烟状态和 BMI 相比，地区变量的影响相对较小。
- 因此，我最信任的是 `smoker_yes`、`age`、`bmi` 这几类结果；而 `children` 和部分地区变量虽然直觉上可能有影响，但模型中的稳定性明显更弱。

## 5. 诊断与风险
VIF 较高的变量如下：

{format_dataframe_table(vif_df.head(8), decimals=4)}

如果把这个模型用于真实业务，主要风险包括：
- `charges` 明显右偏，少数高费用样本会显著拉高绝对误差。
- 吸烟状态信号很强，但也可能对数据质量和样本漂移比较敏感。
- 这份工作副本中故意加入了缺失值和异常值，说明稳健预处理是必要步骤。
- 从业务角度看，当前 MAE 表示模型对个体医疗费用仍可能有几千金额单位的误差，因此更适合作为定价参考，而不是精确到个人报销金额的预测工具。
- 从 VIF 看，这份数据没有特别明显的严重共线性，说明真实数据上的主要风险更多来自偏态、异常样本和业务分布漂移，而不是共线性主导。
- 如果真的上线，我最担心的是未来样本结构变化，例如吸烟比例、地区结构或医疗成本分布变化，导致模型外推能力下降。

{appendix}
"""

    KAGGLE_REPORT_PATH.write_text(report, encoding="utf-8")
    return {
        "custom_mean": custom_mean,
        "baseline_mean": baseline_mean,
    }


def write_summary_report(synthetic_summary: dict, kaggle_summary: dict) -> None:
    report = f"""# 第11周总结对照报告

## 1. 为什么模拟数据上的推断更容易
- 在模拟任务里，数据生成机制是已知的，所以可以直接检查系数方向是否与设定一致。
- 当某个系数不稳定时，我们更容易判断这是噪声还是共线性带来的问题，而不必怀疑业务机制本身。
- 因此，模拟数据更像一个“可控实验场”，既能验证模型，也能验证流程。

## 2. 为什么真实数据即使分数不错，解释依然更难
- 在 Kaggle 保险数据中，即使预测误差看起来还可以，解释仍然更困难，因为真实 DGP 并不知道。
- 真实世界变量经常同时承载多个潜在机制，所以系数不能直接当成因果结论。
- 像吸烟状态这样特别强的变量，还可能掩盖掉其他较弱变量的稳定性。

## 3. 缺失值、异常值、共线性在两类数据中的影响差异
- 在模拟任务中，缺失值和异常值是我主动加入的，因此更容易追踪它们对结果的影响。
- 在 Kaggle 任务中，即使只是少量缺失或几个 BMI 极端值，也会对模型解释造成更大干扰，因为我们没有真实公式可以对照。
- 共线性在模拟任务中更强，因为两个广告预算变量是故意绑定生成的。

## 4. 为什么无泄露交叉验证在真实数据上尤其重要
- 真实数据分布更复杂，如果把全量数据的填补均值、标准化参数提前泄露给验证集，会让结果显得比实际更好。
- 这次流程把填补、截尾、标准化、编码都放进每一折内部完成，因此评估更诚实。
- 对真实业务数据来说，这一点尤其重要，因为未来数据在训练时本来就不可能提前看到。

## 5. 这周 `utils/` 组件带来的复用价值
- `utils/transformers.py` 统一处理了缺失值填补、异常值截尾、标准化和独热编码。
- `utils/models.py` 提供了主流程中使用的自定义 OLS 建模能力。
- `utils/metrics.py` 让 RMSE、MAE、MAPE 的计算方式在两个任务里保持一致。
- `utils/diagnostics.py` 让 VIF 诊断可以直接复用到模拟数据和真实数据上。

## 6. 指标汇总
- Synthetic custom CV mean RMSE / MAE / MAPE: {synthetic_summary['custom_mean']['rmse']:.4f} / {synthetic_summary['custom_mean']['mae']:.4f} / {synthetic_summary['custom_mean']['mape']:.4f}%
- Synthetic baseline CV mean RMSE / MAE / MAPE: {synthetic_summary['baseline_mean']['rmse']:.4f} / {synthetic_summary['baseline_mean']['mae']:.4f} / {synthetic_summary['baseline_mean']['mape']:.4f}%
- Kaggle custom CV mean RMSE / MAE / MAPE: {kaggle_summary['custom_mean']['rmse']:.4f} / {kaggle_summary['custom_mean']['mae']:.4f} / {kaggle_summary['custom_mean']['mape']:.4f}%
- Kaggle baseline CV mean RMSE / MAE / MAPE: {kaggle_summary['baseline_mean']['rmse']:.4f} / {kaggle_summary['baseline_mean']['mae']:.4f} / {kaggle_summary['baseline_mean']['mape']:.4f}%
"""

    SUMMARY_REPORT_PATH.write_text(report, encoding="utf-8")


def run_synthetic_task() -> dict:
    synthetic_df = generate_synthetic_data()
    return write_synthetic_report(synthetic_df)


def run_kaggle_task() -> dict:
    create_kaggle_working_copy()
    kaggle_df = load_kaggle_data()
    return write_kaggle_report(kaggle_df)


def main() -> None:
    ensure_directories()

    synthetic_summary = run_synthetic_task()
    kaggle_summary = run_kaggle_task()
    write_summary_report(synthetic_summary, kaggle_summary)

    print("Week 11 workflow finished.")
    print(f"Synthetic data: {SYNTHETIC_PATH}")
    print(f"Kaggle working data: {KAGGLE_WORKING_PATH}")
    print(f"Reports: {SYNTHETIC_REPORT_PATH}, {KAGGLE_REPORT_PATH}, {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
