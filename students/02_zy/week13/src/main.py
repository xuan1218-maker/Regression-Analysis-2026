import re
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

WEEK13_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WEEK13_DIR / "src"
DATA_DIR = WEEK13_DIR / "data"
RESULTS_DIR = WEEK13_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(SRC_DIR))

from utils.metrics import calculate_mae, calculate_rmse
from utils.transformers import CustomStandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_SEED = 42


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(digits)

    columns = df.columns.tolist()

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def generate_correlated_data(n_samples: int = 400):
    rng = np.random.default_rng(RANDOM_SEED)

    z = rng.normal(0, 1, n_samples)

    x1 = z + rng.normal(0, 0.08, n_samples)
    x2 = 0.95 * z + rng.normal(0, 0.08, n_samples)
    x3 = 0.90 * z + rng.normal(0, 0.10, n_samples)

    x4 = rng.normal(0, 1, n_samples)
    x5 = rng.normal(0, 1, n_samples)
    x6 = rng.normal(0, 1, n_samples)

    x7 = rng.normal(0, 1, n_samples)
    x8 = rng.normal(0, 1, n_samples)
    x9 = rng.normal(0, 1, n_samples)
    x10 = rng.normal(0, 1, n_samples)

    noise = rng.normal(0, 1.2, n_samples)

    y = 4.0 * x1 - 3.0 * x4 + 2.5 * x5 + 1.5 * x6 + noise

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "x6": x6,
            "x7": x7,
            "x8": x8,
            "x9": x9,
            "x10": x10,
            "y": y,
        }
    )

    true_coef = {
        "x1": 4.0,
        "x2": 0.0,
        "x3": 0.0,
        "x4": -3.0,
        "x5": 2.5,
        "x6": 1.5,
        "x7": 0.0,
        "x8": 0.0,
        "x9": 0.0,
        "x10": 0.0,
    }

    df.to_csv(DATA_DIR / "synthetic_correlated.csv", index=False)

    return df, true_coef


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("scaler", CustomStandardScaler()),
            ("model", model),
        ]
    )


def evaluate_on_test(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "RMSE": calculate_rmse(y_test, y_pred),
        "MAE": calculate_mae(y_test, y_pred),
    }


def get_model_coefficients(model, feature_names: list[str]) -> dict:
    coef = model.named_steps["model"].coef_
    return {name: float(value) for name, value in zip(feature_names, coef)}


def build_cv_curve_dataframe(ridge_grid, lasso_grid, elastic_grid) -> pd.DataFrame:
    records = []

    for model_name, grid in [("Ridge", ridge_grid), ("Lasso", lasso_grid)]:
        params = grid.cv_results_["params"]
        scores = -grid.cv_results_["mean_test_score"]

        for param, score in zip(params, scores):
            records.append(
                {
                    "model": model_name,
                    "alpha": param["model__alpha"],
                    "cv_RMSE": score,
                }
            )

    elastic_df = pd.DataFrame(
        {
            "alpha": [
                param["model__alpha"] for param in elastic_grid.cv_results_["params"]
            ],
            "l1_ratio": [
                param["model__l1_ratio"] for param in elastic_grid.cv_results_["params"]
            ],
            "cv_RMSE": -elastic_grid.cv_results_["mean_test_score"],
        }
    )

    elastic_best_by_alpha = elastic_df.groupby("alpha", as_index=False)["cv_RMSE"].min()

    for _, row in elastic_best_by_alpha.iterrows():
        records.append(
            {
                "model": "ElasticNet",
                "alpha": row["alpha"],
                "cv_RMSE": row["cv_RMSE"],
            }
        )

    return pd.DataFrame(records)


def run_regularized_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names: list[str],
    true_coef: dict | None = None,
    alpha_grid=None,
):
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 3, 40)

    models = {}

    ols = build_pipeline(LinearRegression())
    ols_metrics = evaluate_on_test(ols, X_train, X_test, y_train, y_test)
    models["OLS"] = ols

    ridge_grid = GridSearchCV(
        estimator=build_pipeline(Ridge()),
        param_grid={"model__alpha": alpha_grid},
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    ridge_grid.fit(X_train, y_train)
    ridge = ridge_grid.best_estimator_
    ridge_metrics = evaluate_on_test(ridge, X_train, X_test, y_train, y_test)
    models["Ridge"] = ridge

    lasso_grid = GridSearchCV(
        estimator=build_pipeline(Lasso(max_iter=50000)),
        param_grid={"model__alpha": alpha_grid},
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    lasso_grid.fit(X_train, y_train)
    lasso = lasso_grid.best_estimator_
    lasso_metrics = evaluate_on_test(lasso, X_train, X_test, y_train, y_test)
    models["Lasso"] = lasso

    elastic_grid = GridSearchCV(
        estimator=build_pipeline(ElasticNet(max_iter=50000)),
        param_grid={
            "model__alpha": alpha_grid,
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    elastic_grid.fit(X_train, y_train)
    elastic = elastic_grid.best_estimator_
    elastic_metrics = evaluate_on_test(elastic, X_train, X_test, y_train, y_test)
    models["ElasticNet"] = elastic

    performance_df = pd.DataFrame(
        [
            {
                "model": "OLS",
                "best_alpha": "",
                "best_l1_ratio": "",
                "test_RMSE": ols_metrics["RMSE"],
                "test_MAE": ols_metrics["MAE"],
                "nonzero_features": int(
                    np.sum(np.abs(ols.named_steps["model"].coef_) > 1e-6)
                ),
            },
            {
                "model": "Ridge",
                "best_alpha": ridge_grid.best_params_["model__alpha"],
                "best_l1_ratio": "",
                "test_RMSE": ridge_metrics["RMSE"],
                "test_MAE": ridge_metrics["MAE"],
                "nonzero_features": int(
                    np.sum(np.abs(ridge.named_steps["model"].coef_) > 1e-6)
                ),
            },
            {
                "model": "Lasso",
                "best_alpha": lasso_grid.best_params_["model__alpha"],
                "best_l1_ratio": "",
                "test_RMSE": lasso_metrics["RMSE"],
                "test_MAE": lasso_metrics["MAE"],
                "nonzero_features": int(
                    np.sum(np.abs(lasso.named_steps["model"].coef_) > 1e-6)
                ),
            },
            {
                "model": "ElasticNet",
                "best_alpha": elastic_grid.best_params_["model__alpha"],
                "best_l1_ratio": elastic_grid.best_params_["model__l1_ratio"],
                "test_RMSE": elastic_metrics["RMSE"],
                "test_MAE": elastic_metrics["MAE"],
                "nonzero_features": int(
                    np.sum(np.abs(elastic.named_steps["model"].coef_) > 1e-6)
                ),
            },
        ]
    )

    coef_rows = []

    for feature in feature_names:
        row = {"feature": feature}

        if true_coef is not None:
            row["true_coefficient"] = true_coef.get(feature, "")

        for model_name, model in models.items():
            coef_map = get_model_coefficients(model, feature_names)
            row[model_name] = coef_map[feature]

        coef_rows.append(row)

    coef_df = pd.DataFrame(coef_rows)

    cv_curve_df = build_cv_curve_dataframe(
        ridge_grid=ridge_grid,
        lasso_grid=lasso_grid,
        elastic_grid=elastic_grid,
    )

    best_params = {
        "ridge_alpha": ridge_grid.best_params_["model__alpha"],
        "lasso_alpha": lasso_grid.best_params_["model__alpha"],
        "elastic_alpha": elastic_grid.best_params_["model__alpha"],
        "elastic_l1_ratio": elastic_grid.best_params_["model__l1_ratio"],
    }

    return performance_df, coef_df, cv_curve_df, best_params, models


def plot_model_performance(performance_df: pd.DataFrame, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(performance_df))
    width = 0.35

    ax.bar(x - width / 2, performance_df["test_RMSE"], width, label="Test RMSE")
    ax.bar(x + width / 2, performance_df["test_MAE"], width, label="Test MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(performance_df["model"])
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def plot_cv_curves(cv_curve_df: pd.DataFrame, best_params: dict, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in ["Ridge", "Lasso", "ElasticNet"]:
        sub = cv_curve_df[cv_curve_df["model"] == model_name]
        ax.plot(sub["alpha"], sub["cv_RMSE"], marker="o", label=model_name)

    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("CV RMSE")
    ax.set_title("GridSearchCV: CV Error vs Alpha")
    ax.legend()
    ax.grid(alpha=0.3)

    for alpha in [
        best_params["ridge_alpha"],
        best_params["lasso_alpha"],
        best_params["elastic_alpha"],
    ]:
        ax.axvline(alpha, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def plot_coefficients(
    coef_df: pd.DataFrame,
    filename: str,
    title: str,
    top_n: int | None = None,
) -> None:
    coef_df = coef_df.copy()

    model_cols = [col for col in ["OLS", "Ridge", "Lasso", "ElasticNet"] if col in coef_df]

    if top_n is not None and len(coef_df) > top_n:
        coef_df["max_abs_coef"] = coef_df[model_cols].abs().max(axis=1)
        coef_df = coef_df.sort_values("max_abs_coef", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))

    feature_names = coef_df["feature"].tolist()
    x = np.arange(len(feature_names))
    width = 0.18

    for i, model_name in enumerate(model_cols):
        ax.bar(
            x + (i - 1.5) * width,
            coef_df[model_name],
            width,
            label=model_name,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def run_repeated_split_stability(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    ridge_alpha: float,
    n_splits: int = 50,
) -> pd.DataFrame:
    records = []

    X = df[feature_cols]
    y = df[target_col]

    for seed in range(n_splits):
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=seed,
        )

        ols = build_pipeline(LinearRegression())
        ridge = build_pipeline(Ridge(alpha=ridge_alpha))

        ols.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        for model_name, model in [("OLS", ols), ("Ridge", ridge)]:
            coefs = get_model_coefficients(model, feature_cols)

            for feature in ["x1", "x2", "x3"]:
                records.append(
                    {
                        "split": seed,
                        "model": model_name,
                        "feature": feature,
                        "coefficient": coefs[feature],
                    }
                )

    stability_df = pd.DataFrame(records)
    plot_stability_boxplot(stability_df)

    return stability_df


def plot_stability_boxplot(stability_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    data = []

    for model_name in ["OLS", "Ridge"]:
        for feature in ["x1", "x2", "x3"]:
            sub = stability_df[
                (stability_df["model"] == model_name)
                & (stability_df["feature"] == feature)
            ]

            labels.append(f"{model_name}\n{feature}")
            data.append(sub["coefficient"].to_numpy())

    ax.boxplot(data, labels=labels)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Coefficient")
    ax.set_title("Coefficient Stability Under Repeated Splits")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coefficient_stability_boxplot.png", dpi=200)
    plt.close(fig)


def summarize_stability(stability_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stability_df.groupby(["model", "feature"])["coefficient"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "coef_mean", "std": "coef_std"})
    )


def cross_validated_rmse(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> float:
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = build_pipeline(LinearRegression())
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)
        fold_scores.append(calculate_rmse(y_val_fold, y_pred))

    return float(np.mean(fold_scores))


def forward_selection_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 5,
    cv_splits: int = 5,
    min_delta: float = 1e-4,
) -> tuple[list[str], pd.DataFrame]:
    remaining = list(X_train.columns)
    selected = []
    current_best = np.inf
    history = []

    for step in range(1, max_features + 1):
        candidate_scores = []

        for candidate in remaining:
            trial_features = selected + [candidate]
            score = cross_validated_rmse(
                X_train[trial_features],
                y_train,
                cv_splits=cv_splits,
            )

            candidate_scores.append((candidate, score))

        candidate_scores = sorted(candidate_scores, key=lambda item: item[1])
        best_candidate, best_score = candidate_scores[0]

        if best_score < current_best - min_delta:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_best = best_score

            history.append(
                {
                    "step": step,
                    "selected_feature": best_candidate,
                    "cv_RMSE": best_score,
                    "selected_set": ", ".join(selected),
                }
            )
        else:
            break

    return selected, pd.DataFrame(history)


def evaluate_selected_features(
    selected_features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    model = build_pipeline(LinearRegression())

    return evaluate_on_test(
        model,
        X_train[selected_features],
        X_test[selected_features],
        y_train,
        y_test,
    )


def get_lasso_selected_features(lasso_model, feature_names: list[str]) -> list[str]:
    coefs = get_model_coefficients(lasso_model, feature_names)

    return [name for name, coef in coefs.items() if abs(coef) > 1e-6]


def parse_money(value) -> float:
    cleaned = re.sub(r"[^0-9.]", "", str(value))

    if cleaned == "":
        return np.nan

    return float(cleaned)


def parse_mileage(value) -> float:
    cleaned = re.sub(r"[^0-9.]", "", str(value))

    if cleaned == "":
        return np.nan

    return float(cleaned)


def simplify_transmission(value) -> str:
    text = str(value).lower()

    if "manual" in text or "m/t" in text:
        return "Manual"

    if "automatic" in text or "a/t" in text or "dual" in text:
        return "Automatic"

    return "Other"

def locate_kaggle_file() -> Path | None:
    """
    Locate Kaggle used car data file.

    The function first looks for the file in week13/data.
    If not found, it tries to use the week11 data file.
    """
    candidates = [
        DATA_DIR / "kaggle_used_cars.csv",
        DATA_DIR / "used_cars.csv",
        WEEK13_DIR.parent / "week11" / "data" / "kaggle_used_cars.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None

def clean_kaggle_used_cars(input_path: Path) -> tuple[pd.DataFrame, dict]:
    raw_df = pd.read_csv(input_path)
    df = raw_df.copy()

    # 1. 统一列名，避免 Brand / brand / Brand & Model 这种差异
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("&", "and", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # 2. 兼容不同版本二手车数据的列名
    rename_map = {
        "brand_and_model": "brand_model",
        "brand_model": "brand_model",
        "model_year": "model_year",
        "milage": "milage",
        "mileage": "milage",
        "fuel_type": "fuel_type",
        "transmission": "transmission",
        "ext_col": "ext_col",
        "ext_color": "ext_col",
        "int_col": "int_col",
        "int_color": "int_col",
        "accident": "accident",
        "clean_title": "clean_title",
        "price": "price",
    }

    df = df.rename(columns=rename_map)

    # 3. 如果没有 brand，但有 brand_model 或 model，就从里面拆一个 brand
    if "brand" not in df.columns:
        if "brand_model" in df.columns:
            df["brand"] = (
                df["brand_model"]
                .astype(str)
                .str.split()
                .str[0]
                .fillna("Unknown")
            )
        elif "model" in df.columns:
            df["brand"] = (
                df["model"]
                .astype(str)
                .str.split()
                .str[0]
                .fillna("Unknown")
            )
        else:
            df["brand"] = "Unknown"

    # 4. price 转数值
    if "price" not in df.columns:
        raise ValueError(f"没有找到 price 列，当前数据列名为：{df.columns.tolist()}")

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 5. milage 转数值
    if "milage" in df.columns:
        df["milage"] = (
            df["milage"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" mi.", "", regex=False)
            .str.replace("mi.", "", regex=False)
            .str.replace(" miles", "", regex=False)
            .str.strip()
        )
        df["milage"] = pd.to_numeric(df["milage"], errors="coerce")
    else:
        df["milage"] = np.nan

    # 6. model_year 转数值，并构造 vehicle_age
    if "model_year" in df.columns:
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
        df["vehicle_age"] = 2026 - df["model_year"]
        df["vehicle_age"] = df["vehicle_age"].clip(lower=0)
    else:
        df["vehicle_age"] = np.nan

    # 7. 删除 price 缺失和明显异常价格
    df = df.dropna(subset=["price"]).copy()
    df = df[(df["price"] >= 1000) & (df["price"] <= 500000)].copy()

    # 8. 数值缺失填补
    for col in ["milage", "vehicle_age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # 9. 类别列：如果不存在就创建为 Unknown
    categorical_cols = [
        "brand",
        "fuel_type",
        "transmission",
        "ext_col",
        "int_col",
        "accident",
        "clean_title",
    ]

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = df[col].fillna("Unknown").astype(str)

        top_categories = df[col].value_counts().head(12).index
        df[col] = np.where(df[col].isin(top_categories), df[col], "Other")

    # 10. 保留建模字段
    keep_cols = [
        "price",
        "vehicle_age",
        "milage",
        "brand",
        "fuel_type",
        "transmission",
        "ext_col",
        "int_col",
        "accident",
        "clean_title",
    ]

    df = df[keep_cols].copy()

    info = {
        "source_path": str(input_path),
        "original_rows": len(raw_df),
        "cleaned_rows": len(df),
        "columns_after_cleaning": df.columns.tolist(),
    }

    return df, info


def build_kaggle_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["price"]
    X = df.drop(columns=["price"])
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    return X, y


def top_abs_coefficients(coef_df: pd.DataFrame, model_name: str, n: int = 10) -> pd.DataFrame:
    top_df = coef_df[["feature", model_name]].copy()
    top_df["abs_coefficient"] = top_df[model_name].abs()

    return top_df.sort_values("abs_coefficient", ascending=False).head(n)


def write_synthetic_report(
    performance_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    stability_summary_df: pd.DataFrame,
    forward_history_df: pd.DataFrame,
    selected_features: list[str],
    forward_metrics: dict,
    lasso_selected_features: list[str],
    best_params: dict,
) -> None:
    report_path = RESULTS_DIR / "synthetic_report.md"

    content = "\n".join(
        [
            "# Week 13：正则化回归与变量筛选模拟数据报告",
            "",
            "## 1. 数据生成机制 DGP",
            "",
            "本周我生成了一份带有明显多重共线性的模拟回归数据。样本量为 400，特征数为 10。",
            "",
            "`x1`、`x2`、`x3` 是一组高度相关变量，它们来自同一个潜在变量 `z`。真实目标变量只依赖部分变量：",
            "",
            "```text",
            "y = 4.0 * x1 - 3.0 * x4 + 2.5 * x5 + 1.5 * x6 + noise",
            "```",
            "",
            "真正有用的变量是 `x1`、`x4`、`x5`、`x6`。`x2` 和 `x3` 虽然与 `x1` 高度相关，但真实系数为 0；`x7`、`x8`、`x9`、`x10` 是纯噪声变量。",
            "",
            "## 2. 模型性能对比",
            "",
            dataframe_to_markdown(performance_df),
            "",
            "## 3. GridSearchCV 最优参数",
            "",
            f"- Ridge 最优 alpha：`{best_params['ridge_alpha']}`",
            f"- Lasso 最优 alpha：`{best_params['lasso_alpha']}`",
            f"- Elastic Net 最优 alpha：`{best_params['elastic_alpha']}`",
            f"- Elastic Net 最优 l1_ratio：`{best_params['elastic_l1_ratio']}`",
            "",
            "这些参数是通过 5 折交叉验证选择出来的，不是人为指定的。",
            "",
            "## 4. 系数对比",
            "",
            dataframe_to_markdown(coef_df),
            "",
            "Ridge 会整体缩小系数，但一般不会直接把系数变成 0；Lasso 会把部分变量系数压缩为 0，因此具有变量筛选效果；Elastic Net 介于 Ridge 和 Lasso 之间，既能收缩系数，也能缓解 Lasso 在高度相关变量组中只随机保留一个变量的问题。",
            "",
            "## 5. OLS 与 Ridge 的系数稳定性",
            "",
            dataframe_to_markdown(stability_summary_df),
            "",
            "通过 50 次不同随机切分可以看到，在高度相关变量组 `x1`、`x2`、`x3` 中，OLS 的系数更容易波动；Ridge 加入 L2 penalty 后，相关变量的系数更加平滑和稳定。",
            "",
            "## 6. 前向选择结果",
            "",
            dataframe_to_markdown(forward_history_df),
            "",
            f"前向选择最终选出的变量为：`{', '.join(selected_features)}`。",
            "",
            f"使用前向选择变量后，测试集 RMSE 为 `{forward_metrics['RMSE']:.4f}`，MAE 为 `{forward_metrics['MAE']:.4f}`。",
            "",
            f"Lasso 自动筛选出的非零变量为：`{', '.join(lasso_selected_features)}`。",
            "",
            "前向选择和 Lasso 的结果可能不完全一致。前向选择是逐步加入变量，每一步根据交叉验证误差判断是否值得加入；Lasso 是通过 L1 penalty 在一次模型拟合中自动压缩系数。",
            "",
            "## 7. 为什么正则化前必须标准化？",
            "",
            "Ridge 和 Lasso 的惩罚项直接作用在系数大小上。如果不同特征的量纲不同，系数大小就不能直接比较，惩罚项会不公平地影响某些变量。因此，在使用 Ridge、Lasso 和 Elastic Net 前，需要先对特征进行标准化。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def write_kaggle_report(
    info: dict,
    performance_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    best_model_name: str,
    best_params: dict,
    lasso_selected_features: list[str],
    total_features: int,
) -> None:
    report_path = RESULTS_DIR / "kaggle_report.md"

    ols_rmse = float(
        performance_df.loc[performance_df["model"] == "OLS", "test_RMSE"].iloc[0]
    )
    best_rmse = float(
        performance_df.loc[performance_df["model"] == best_model_name, "test_RMSE"].iloc[0]
    )

    if best_rmse < ols_rmse:
        improvement_text = "正则化模型相比 OLS 在测试集 RMSE 上有一定改善。"
    else:
        improvement_text = "正则化模型相比 OLS 没有明显降低测试集 RMSE，可能是因为 OLS 在当前特征下已经能够较好拟合。"

    content = "\n".join(
        [
            "# Week 13：Kaggle 二手车价格正则化回归报告",
            "",
            "## 1. 数据来源与业务背景",
            "",
            "- 数据集：Used Car Price Prediction Dataset",
            "- 业务问题：根据车辆年份、里程、品牌、燃料类型、变速箱、事故记录等信息预测二手车价格。",
            f"- 使用文件：`{info['source_path']}`",
            f"- 原始样本量：`{info['original_rows']}`",
            f"- 清洗后样本量：`{info['cleaned_rows']}`",
            f"- 建模特征数：`{total_features}`",
            "",
            "这份数据经过 One-Hot 编码后特征数量较多，同时品牌、燃料类型、事故记录等变量之间可能存在潜在相关性，因此适合练习正则化和变量筛选。",
            "",
            "## 2. 数据处理流程",
            "",
            "主要处理包括：价格和里程转为数值；构造 `vehicle_age`；简化 `transmission`；将缺失类别填为 `Unknown`；低频类别归为 `Other`；最后进行 One-Hot 编码，并在 Pipeline 中使用自定义 `CustomStandardScaler` 进行标准化。",
            "",
            "## 3. 模型性能对比",
            "",
            dataframe_to_markdown(performance_df),
            "",
            f"测试集 RMSE 最低的模型是 `{best_model_name}`。{improvement_text}",
            "",
            "## 4. GridSearchCV 最优参数",
            "",
            f"- Ridge 最优 alpha：`{best_params['ridge_alpha']}`",
            f"- Lasso 最优 alpha：`{best_params['lasso_alpha']}`",
            f"- Elastic Net 最优 alpha：`{best_params['elastic_alpha']}`",
            f"- Elastic Net 最优 l1_ratio：`{best_params['elastic_l1_ratio']}`",
            "",
            "这些参数来自 5 折交叉验证，目标是寻找验证误差较低的模型，而不是单纯追求变量越少越好。",
            "",
            "## 5. Lasso 变量筛选结果",
            "",
            f"Lasso 最终保留的非零变量数量为：`{len(lasso_selected_features)}`。",
            "",
            "部分 Lasso 保留变量如下：",
            "",
            "`" + ", ".join(lasso_selected_features[:20]) + "`",
            "",
            "Lasso 删除的变量不一定完全没有业务意义。特别是在类别变量或高度相关变量中，Lasso 可能只保留其中一部分，因此解释时需要结合业务背景。",
            "",
            "## 6. 最关键的影响因素",
            "",
            f"如果业务方要求给出最关键的影响因素，我会优先参考测试集表现最好的 `{best_model_name}` 模型的系数绝对值排序。",
            "",
            dataframe_to_markdown(top_features_df),
            "",
            "这些变量可以理解为模型中相对影响更明显的因素，但这只是相关关系，不应直接解释为严格因果关系。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def write_missing_kaggle_report() -> None:
    report_path = RESULTS_DIR / "kaggle_report.md"

    content = "\n".join(
        [
            "# Week 13：Kaggle 真实数据报告",
            "",
            "本次运行没有找到 `kaggle_used_cars.csv`，因此没有执行可选 Task B。",
            "",
            "如果需要运行 Task B，请将 Kaggle 二手车数据放到：",
            "",
            "```text",
            "week13/data/kaggle_used_cars.csv",
            "```",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def write_summary_comparison(
    synthetic_result: dict,
    kaggle_result: dict | None,
) -> None:
    report_path = RESULTS_DIR / "summary_comparison.md"

    synthetic_performance_df = synthetic_result["performance_df"]
    selected_features = synthetic_result["selected_features"]
    lasso_selected_features = synthetic_result["lasso_selected_features"]

    best_model_row = synthetic_performance_df.loc[
        synthetic_performance_df["test_RMSE"].idxmin()
    ]

    if kaggle_result is None:
        kaggle_text = "本次没有运行可选 Task B，因此总结主要基于模拟数据实验。"
    else:
        kaggle_text = (
            f"可选 Task B 中，我使用 Kaggle 二手车数据完成真实数据正则化回归。"
            f"经过 One-Hot 编码后共有 `{kaggle_result['total_features']}` 个特征，"
            f"测试集表现最好的模型是 `{kaggle_result['best_model_name']}`，"
            f"Lasso 保留了 `{len(kaggle_result['lasso_selected_features'])}` 个非零变量。"
        )

    content = "\n".join(
        [
            "# Week 13：理论与实践总结",
            "",
            "## 1. 本周核心结论",
            "",
            f"从模拟数据实验看，测试集 RMSE 最低的模型是 `{best_model_row['model']}`，对应 RMSE 为 `{best_model_row['test_RMSE']:.4f}`。",
            "",
            "OLS 在高度相关特征下容易出现系数不稳定；Ridge 通过 L2 正则化让系数整体收缩，提升稳定性；Lasso 通过 L1 正则化把部分系数压缩为 0，从而实现变量筛选；Elastic Net 同时结合 L1 和 L2，在共线性较强时通常比单纯 Lasso 更稳健。",
            "",
            "## 2. Task B 真实数据总结",
            "",
            kaggle_text,
            "",
            "## 3. Lasso 面对高度相关变量组的风险",
            "",
            "当多个变量高度相关时，Lasso 可能只保留其中一个变量，而把其他相关变量压缩为 0。这样虽然模型更稀疏，但也可能带来业务风险：如果这些变量在业务含义上都重要，Lasso 的结果可能让人误以为被删除的变量完全没有价值。",
            "",
            "## 4. Elastic Net 如何缓解这个问题？",
            "",
            "Elastic Net 同时包含 L1 和 L2 penalty。L1 部分帮助变量筛选，L2 部分让相关变量的系数更平滑，因此它不像 Lasso 那样容易在高度相关变量中只保留一个变量。它更适合处理存在共线性的真实业务数据。",
            "",
            "## 5. GridSearchCV 与主观追求稀疏性的区别",
            "",
            "GridSearchCV 是根据交叉验证误差选择超参数，它关注的是模型在未知数据上的预测表现；而主观追求“越稀疏越好”只关注变量少不少，可能会牺牲预测能力。因此，最优 alpha 不一定对应最稀疏的模型，而是对应验证误差较低、泛化能力较好的模型。",
            "",
            "## 6. 前向选择与 Lasso 的比较",
            "",
            f"前向选择最终选择的变量是：`{', '.join(selected_features)}`。",
            "",
            f"Lasso 在模拟数据中最终保留的变量是：`{', '.join(lasso_selected_features)}`。",
            "",
            "前向选择的优点是逻辑直观，能够看到变量一步步进入模型的过程；缺点是计算量较大。Lasso 的优点是可以在一次正则化建模过程中完成变量筛选，效率更高；缺点是在高度相关变量组中可能随机偏向其中一个变量。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def run_synthetic_task() -> dict:
    df, true_coef = generate_correlated_data()

    feature_cols = [col for col in df.columns if col != "y"]

    X = df[feature_cols]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_SEED,
    )

    performance_df, coef_df, cv_curve_df, best_params, models = run_regularized_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        true_coef=true_coef,
        alpha_grid=np.logspace(-4, 3, 40),
    )

    plot_model_performance(
        performance_df,
        filename="synthetic_model_performance.png",
        title="Synthetic Data: Model Performance",
    )

    plot_cv_curves(
        cv_curve_df,
        best_params,
        filename="synthetic_cv_alpha_curves.png",
    )

    plot_coefficients(
        coef_df,
        filename="synthetic_coefficient_comparison.png",
        title="Synthetic Data: Coefficient Comparison",
    )

    stability_df = run_repeated_split_stability(
        df=df,
        feature_cols=feature_cols,
        target_col="y",
        ridge_alpha=best_params["ridge_alpha"],
    )

    stability_summary_df = summarize_stability(stability_df)

    selected_features, forward_history_df = forward_selection_cv(
        X_train=X_train,
        y_train=y_train,
        max_features=5,
    )

    forward_metrics = evaluate_selected_features(
        selected_features=selected_features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    lasso_selected_features = get_lasso_selected_features(
        models["Lasso"],
        feature_cols,
    )

    write_synthetic_report(
        performance_df=performance_df,
        coef_df=coef_df,
        stability_summary_df=stability_summary_df,
        forward_history_df=forward_history_df,
        selected_features=selected_features,
        forward_metrics=forward_metrics,
        lasso_selected_features=lasso_selected_features,
        best_params=best_params,
    )

    return {
        "performance_df": performance_df,
        "selected_features": selected_features,
        "lasso_selected_features": lasso_selected_features,
    }


def run_kaggle_task() -> dict | None:
    kaggle_path = locate_kaggle_file()

    if kaggle_path is None:
        write_missing_kaggle_report()
        return None

    kaggle_df, info = clean_kaggle_used_cars(kaggle_path)
    X, y = build_kaggle_design_matrix(kaggle_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_SEED,
    )

    performance_df, coef_df, cv_curve_df, best_params, models = run_regularized_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X.columns.tolist(),
        true_coef=None,
        alpha_grid=np.logspace(0, 4, 18),
    )

    plot_model_performance(
        performance_df,
        filename="kaggle_model_performance.png",
        title="Kaggle Used Cars: Model Performance",
    )

    plot_cv_curves(
        cv_curve_df,
        best_params,
        filename="kaggle_cv_alpha_curves.png",
    )

    plot_coefficients(
        coef_df,
        filename="kaggle_top_coefficients.png",
        title="Kaggle Used Cars: Top Coefficients",
        top_n=20,
    )

    lasso_selected_features = get_lasso_selected_features(
        models["Lasso"],
        X.columns.tolist(),
    )

    best_model_name = performance_df.loc[performance_df["test_RMSE"].idxmin(), "model"]
    top_features_df = top_abs_coefficients(coef_df, best_model_name, n=10)

    write_kaggle_report(
        info=info,
        performance_df=performance_df,
        top_features_df=top_features_df,
        best_model_name=best_model_name,
        best_params=best_params,
        lasso_selected_features=lasso_selected_features,
        total_features=X.shape[1],
    )

    return {
        "performance_df": performance_df,
        "best_model_name": best_model_name,
        "lasso_selected_features": lasso_selected_features,
        "total_features": X.shape[1],
    }


def main() -> None:
    ensure_directories()

    print("===== Week 13 Regularized Regression Started =====")

    print("[Stage 1] Running Task A: synthetic correlated data...")
    synthetic_result = run_synthetic_task()

    print("[Stage 2] Running Task B: Kaggle used car data...")
    kaggle_result = run_kaggle_task()

    print("[Stage 3] Running Task C: writing summary comparison...")
    write_summary_comparison(
        synthetic_result=synthetic_result,
        kaggle_result=kaggle_result,
    )

    print("===== Week 13 Finished =====")
    print("Data saved to:", DATA_DIR / "synthetic_correlated.csv")
    print("Reports saved to:", RESULTS_DIR)
    print("Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()