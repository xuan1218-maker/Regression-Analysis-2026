"""Week 13 homework: regularized regression and variable selection.

Run from students/07_nc with:

    uv run week13/main.py

The script is self-contained: it generates the synthetic collinear data,
loads the included Kaggle House Prices CSV, runs Ridge/Lasso/Elastic Net
experiments, performs custom forward selection from src/utils/models.py, and
writes all reports/figures under week13/results/.
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.diagnostics import correlation_pairs  # noqa: E402
from utils.metrics import calculate_mae, calculate_rmse, summarize_regression_metrics  # noqa: E402
from utils.models import ForwardSelectorCV  # noqa: E402
from utils.transformers import CustomNumericImputer, CustomStandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_SEED = 42
WEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK_DIR / "data"
RESULTS_DIR = WEEK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def reset_outputs() -> None:
    """Create required folders and clear old reports/figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def clean_display_value(value: Any, float_digits: int = 4) -> str:
    """Convert numpy scalars and dicts into readable report strings."""
    if isinstance(value, dict):
        clean_items = {key: clean_display_value(val, float_digits) for key, val in value.items()}
        return str(clean_items)
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.{float_digits}f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str], float_digits: int = 4) -> str:
    """Small markdown table helper that avoids optional tabulate dependency."""
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in rows:
        cells = [clean_display_value(row.get(col, ""), float_digits) for col in columns]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def rmse_scorer():
    """GridSearchCV scorer using the student's own RMSE implementation."""
    return make_scorer(calculate_rmse, greater_is_better=False)


def get_model_coefficients(best_estimator: Pipeline, feature_names: list[str], model_name: str) -> pd.DataFrame:
    model = best_estimator.named_steps["model"]
    coefs = np.asarray(model.coef_, dtype=float).ravel()
    return pd.DataFrame({"model": model_name, "feature": feature_names, "coefficient": coefs})


def evaluate_pipeline(name: str, pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    pred = pipeline.predict(X_test)
    metrics = summarize_regression_metrics(y_test, pred)
    return {"model": name, **metrics}


def build_regularized_pipeline(model) -> Pipeline:
    """Pipeline required by Week 13: custom scaler + sklearn regularized model."""
    return Pipeline([("scaler", CustomStandardScaler()), ("model", model)])


def grid_search_model(
    model_name: str,
    model,
    param_grid: dict[str, list[float]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: KFold,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_regularized_pipeline(model),
        param_grid=param_grid,
        scoring=rmse_scorer(),
        cv=cv,
        n_jobs=None,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    print(f"[{model_name}] best params = {search.best_params_}, CV RMSE = {-search.best_score_:.4f}")
    return search


# ---------------------------------------------------------------------------
# Synthetic Task A
# ---------------------------------------------------------------------------

def make_correlated_regression_data(n_samples: int = 520, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a synthetic dataset with explicit multicollinearity.

    DGP:
        y = 20 + 4.0*x_signal_1 - 2.5*x_independent_1 + 1.8*x_independent_2 + noise

    The feature family x_signal_1/x_signal_2/x_signal_3 is intentionally
    generated from the same latent factor, so OLS should have unstable
    individual coefficients even though predictions remain good.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n_samples)
    macro = rng.normal(0, 1, n_samples)
    campaign = rng.normal(0, 1, n_samples)

    x_signal_1 = latent + rng.normal(0, 0.04, n_samples)
    x_signal_2 = 0.96 * latent + rng.normal(0, 0.05, n_samples)
    x_signal_3 = 1.08 * latent + rng.normal(0, 0.06, n_samples)
    x_independent_1 = macro + rng.normal(0, 0.15, n_samples)
    x_independent_2 = campaign + rng.normal(0, 0.15, n_samples)
    x_mixed_collinear = 0.75 * x_independent_1 + 0.25 * x_independent_2 + rng.normal(0, 0.08, n_samples)

    noise_1 = rng.normal(0, 1, n_samples)
    noise_2 = rng.normal(0, 1, n_samples)
    noise_3 = rng.normal(0, 1, n_samples)
    noise_4 = rng.normal(0, 1, n_samples)
    noise_5 = rng.normal(0, 1, n_samples)
    weak_signal = 0.30 * latent + rng.normal(0, 1.0, n_samples)

    epsilon = rng.normal(0, 1.6, n_samples)
    y = 20.0 + 4.0 * x_signal_1 - 2.5 * x_independent_1 + 1.8 * x_independent_2 + epsilon

    return pd.DataFrame(
        {
            "x_signal_1": x_signal_1,
            "x_signal_2": x_signal_2,
            "x_signal_3": x_signal_3,
            "x_independent_1": x_independent_1,
            "x_independent_2": x_independent_2,
            "x_mixed_collinear": x_mixed_collinear,
            "noise_1": noise_1,
            "noise_2": noise_2,
            "noise_3": noise_3,
            "noise_4": noise_4,
            "noise_5": noise_5,
            "weak_signal": weak_signal,
            "y": y,
        }
    )


def coefficient_stability_experiment(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_features: list[str],
    n_splits: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated train/test splits comparing OLS vs Ridge coefficient stability."""
    rows: list[dict[str, float | str | int]] = []
    target_indices = [feature_names.index(f) for f in target_features]

    for seed in range(n_splits):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=seed)
        models = {
            "OLS": build_regularized_pipeline(LinearRegression()),
            "Ridge_alpha_10": build_regularized_pipeline(Ridge(alpha=10.0)),
        }
        for model_name, pipe in models.items():
            pipe.fit(X_train, y_train)
            coefs = np.asarray(pipe.named_steps["model"].coef_, dtype=float).ravel()
            for idx in target_indices:
                rows.append(
                    {
                        "split_seed": seed,
                        "model": model_name,
                        "feature": feature_names[idx],
                        "coefficient": float(coefs[idx]),
                    }
                )

    coef_df = pd.DataFrame(rows)
    stability_df = (
        coef_df.groupby(["model", "feature"])["coefficient"]
        .agg(coef_mean="mean", coef_std="std", coef_min="min", coef_max="max")
        .reset_index()
    )
    return coef_df, stability_df


def plot_coefficient_stability(coef_df: pd.DataFrame, path: Path) -> None:
    target_features = list(coef_df["feature"].unique())
    models = list(coef_df["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    positions: list[float] = []
    data: list[np.ndarray] = []
    labels: list[str] = []
    for i, feature in enumerate(target_features):
        base = i * 3
        for j, model in enumerate(models):
            positions.append(base + j)
            values = coef_df[(coef_df["feature"] == feature) & (coef_df["model"] == model)]["coefficient"].to_numpy()
            data.append(values)
            labels.append(f"{feature}\n{model}")
    ax.boxplot(data, positions=positions, widths=0.65, showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Coefficient stability under repeated train/test splits")
    ax.set_ylabel("Coefficient on standardized feature")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cv_alpha_curves(searches: dict[str, GridSearchCV], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, search in searches.items():
        results = pd.DataFrame(search.cv_results_)
        if name == "ElasticNet":
            results["alpha"] = results["param_model__alpha"].astype(float)
            grouped = results.assign(rmse=-results["mean_test_score"]).groupby("alpha")["rmse"].min().reset_index()
            ax.plot(grouped["alpha"], grouped["rmse"], marker="o", label="ElasticNet best l1_ratio per alpha")
            best_alpha = search.best_params_["model__alpha"]
            best_rmse = -search.best_score_
        else:
            alpha_col = "param_model__alpha"
            results["alpha"] = results[alpha_col].astype(float)
            results["rmse"] = -results["mean_test_score"]
            results = results.sort_values("alpha")
            ax.plot(results["alpha"], results["rmse"], marker="o", label=name)
            best_alpha = search.best_params_["model__alpha"]
            best_rmse = -search.best_score_
        ax.scatter([best_alpha], [best_rmse], s=80)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("5-fold CV RMSE")
    ax.set_title("Validation error curve across alpha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_model_coefficients(coef_table: pd.DataFrame, top_features: list[str], path: Path) -> None:
    pivot = coef_table[coef_table["feature"].isin(top_features)].pivot(
        index="feature", columns="model", values="coefficient"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    width = 0.24
    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * width, pivot[col].values, width=width, label=col)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Coefficient on standardized feature")
    ax.set_title("How regularized models treat correlated and noisy features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_synthetic_task() -> dict[str, Any]:
    print("Running Task A: synthetic correlated data")
    df = make_correlated_regression_data()
    synthetic_path = DATA_DIR / "synthetic_correlated.csv"
    df.to_csv(synthetic_path, index=False)

    feature_names = [c for c in df.columns if c != "y"]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    corr_df = correlation_pairs(df[feature_names], threshold=0.75)
    coef_df, stability_df = coefficient_stability_experiment(
        X, y, feature_names, target_features=["x_signal_1", "x_signal_2", "x_signal_3"]
    )
    plot_coefficient_stability(coef_df, FIGURES_DIR / "synthetic_coefficient_stability.png")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    alpha_grid = np.logspace(-4, 3, 36)

    ridge_search = grid_search_model(
        "Ridge",
        Ridge(),
        {"model__alpha": list(alpha_grid)},
        X_train,
        y_train,
        cv,
    )
    lasso_search = grid_search_model(
        "Lasso",
        Lasso(max_iter=20000, random_state=RANDOM_SEED),
        {"model__alpha": list(alpha_grid)},
        X_train,
        y_train,
        cv,
    )
    enet_search = grid_search_model(
        "ElasticNet",
        ElasticNet(max_iter=30000, random_state=RANDOM_SEED),
        {"model__alpha": list(np.logspace(-4, 2, 28)), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        X_train,
        y_train,
        cv,
    )
    searches = {"Ridge": ridge_search, "Lasso": lasso_search, "ElasticNet": enet_search}
    plot_cv_alpha_curves(searches, FIGURES_DIR / "synthetic_cv_alpha_curves.png")

    ols_pipe = build_regularized_pipeline(LinearRegression()).fit(X_train, y_train)
    model_metrics = [evaluate_pipeline("OLS", ols_pipe, X_test, y_test)]
    for name, search in searches.items():
        model_metrics.append(evaluate_pipeline(name, search.best_estimator_, X_test, y_test))

    coef_tables = [get_model_coefficients(search.best_estimator_, feature_names, name) for name, search in searches.items()]
    coef_table = pd.concat(coef_tables, ignore_index=True)
    top_features = ["x_signal_1", "x_signal_2", "x_signal_3", "x_independent_1", "x_independent_2", "x_mixed_collinear", "noise_1", "noise_2"]
    plot_model_coefficients(coef_table, top_features, FIGURES_DIR / "synthetic_model_coefficients.png")

    scaler_for_selection = CustomStandardScaler().fit(X_train)
    X_train_scaled = scaler_for_selection.transform(X_train)
    X_test_scaled = scaler_for_selection.transform(X_test)
    selector = ForwardSelectorCV(max_features=5, cv=5, random_state=RANDOM_SEED)
    selector.fit(X_train_scaled, y_train, feature_names=feature_names)
    selector_pred = selector.predict(X_test_scaled)
    selector_metrics = {"model": "ForwardSelectionTop5", **summarize_regression_metrics(y_test, selector_pred)}

    lasso_coefs = get_model_coefficients(lasso_search.best_estimator_, feature_names, "Lasso")
    lasso_nonzero = lasso_coefs.loc[lasso_coefs["coefficient"].abs() > 1e-6, "feature"].tolist()

    write_synthetic_report(
        df=df,
        corr_df=corr_df,
        stability_df=stability_df,
        searches=searches,
        model_metrics=model_metrics,
        coef_table=coef_table,
        selector=selector,
        selector_metrics=selector_metrics,
        lasso_nonzero=lasso_nonzero,
    )

    return {
        "df": df,
        "feature_names": feature_names,
        "corr_df": corr_df,
        "stability_df": stability_df,
        "searches": searches,
        "model_metrics": model_metrics,
        "coef_table": coef_table,
        "selector": selector,
        "selector_metrics": selector_metrics,
        "lasso_nonzero": lasso_nonzero,
    }


# ---------------------------------------------------------------------------
# Optional Kaggle Task B
# ---------------------------------------------------------------------------

def load_kaggle_house_prices() -> pd.DataFrame:
    path = DATA_DIR / "kaggle_house_prices.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Missing week13/data/kaggle_house_prices.csv. The submission zip should include this file."
        )
    return pd.read_csv(path)


def prepare_kaggle_numeric_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Use numeric housing attributes to keep the comparison transparent."""
    target = np.log1p(df["SalePrice"].astype(float).to_numpy())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"SalePrice", "Id"}]

    # Drop numeric columns with excessive missingness, then keep enough features
    # for a high-dimensional regularization exercise.
    missing_rate = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missing_rate[c] < 0.40]
    X = df[feature_cols].copy()
    return X, target, feature_cols


def run_kaggle_task() -> dict[str, Any]:
    print("Running optional Task B: Kaggle House Prices")
    kaggle_df = load_kaggle_house_prices()
    X_df, y, feature_names = prepare_kaggle_numeric_data(kaggle_df)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=RANDOM_SEED
    )

    # Student's own imputer is fitted on training data only. The scaler is kept
    # inside every sklearn Pipeline for GridSearchCV.
    imputer = CustomNumericImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df).to_numpy(dtype=float)
    X_test = imputer.transform(X_test_df).to_numpy(dtype=float)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    alpha_grid = np.logspace(-4, 2, 30)

    ols_pipe = build_regularized_pipeline(LinearRegression()).fit(X_train, y_train)
    ridge_search = grid_search_model("Kaggle_Ridge", Ridge(), {"model__alpha": list(alpha_grid)}, X_train, y_train, cv)
    lasso_search = grid_search_model(
        "Kaggle_Lasso",
        Lasso(max_iter=30000, random_state=RANDOM_SEED),
        {"model__alpha": list(alpha_grid)},
        X_train,
        y_train,
        cv,
    )
    enet_search = grid_search_model(
        "Kaggle_ElasticNet",
        ElasticNet(max_iter=30000, random_state=RANDOM_SEED),
        {"model__alpha": list(np.logspace(-4, 1.5, 24)), "model__l1_ratio": [0.2, 0.5, 0.8]},
        X_train,
        y_train,
        cv,
    )

    model_metrics = [evaluate_pipeline("OLS", ols_pipe, X_test, y_test)]
    for name, search in {"Ridge": ridge_search, "Lasso": lasso_search, "ElasticNet": enet_search}.items():
        model_metrics.append(evaluate_pipeline(name, search.best_estimator_, X_test, y_test))

    coef_tables = [
        get_model_coefficients(ridge_search.best_estimator_, feature_names, "Ridge"),
        get_model_coefficients(lasso_search.best_estimator_, feature_names, "Lasso"),
        get_model_coefficients(enet_search.best_estimator_, feature_names, "ElasticNet"),
    ]
    coef_table = pd.concat(coef_tables, ignore_index=True)
    lasso_coef = coef_table[coef_table["model"] == "Lasso"].copy()
    lasso_nonzero = lasso_coef.loc[lasso_coef["coefficient"].abs() > 1e-6, "feature"].tolist()
    lasso_removed = lasso_coef.loc[lasso_coef["coefficient"].abs() <= 1e-6, "feature"].tolist()

    # Forward selection on already training-set-imputed + scaled data.
    scaler = CustomStandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    selector = ForwardSelectorCV(max_features=8, cv=5, random_state=RANDOM_SEED)
    selector.fit(X_train_scaled, y_train, feature_names=feature_names)
    selector_pred = selector.predict(X_test_scaled)
    selector_metrics = {"model": "ForwardSelectionTop8", **summarize_regression_metrics(y_test, selector_pred)}

    plot_kaggle_top_coefficients(coef_table, FIGURES_DIR / "kaggle_top_coefficients.png")
    plot_kaggle_cv_curves(
        {"Ridge": ridge_search, "Lasso": lasso_search, "ElasticNet": enet_search},
        FIGURES_DIR / "kaggle_cv_alpha_curves.png",
    )

    write_kaggle_report(
        kaggle_df=kaggle_df,
        feature_names=feature_names,
        model_metrics=model_metrics,
        selector=selector,
        selector_metrics=selector_metrics,
        coef_table=coef_table,
        searches={"Ridge": ridge_search, "Lasso": lasso_search, "ElasticNet": enet_search},
        lasso_nonzero=lasso_nonzero,
        lasso_removed=lasso_removed,
    )

    return {
        "kaggle_df": kaggle_df,
        "feature_names": feature_names,
        "model_metrics": model_metrics,
        "selector": selector,
        "selector_metrics": selector_metrics,
        "coef_table": coef_table,
        "searches": {"Ridge": ridge_search, "Lasso": lasso_search, "ElasticNet": enet_search},
        "lasso_nonzero": lasso_nonzero,
        "lasso_removed": lasso_removed,
    }


def plot_kaggle_top_coefficients(coef_table: pd.DataFrame, path: Path) -> None:
    # Select the union of the top 12 variables by absolute ElasticNet coefficient.
    enet = coef_table[coef_table["model"] == "ElasticNet"].copy()
    top_features = enet.reindex(enet["coefficient"].abs().sort_values(ascending=False).index).head(12)["feature"].tolist()
    pivot = coef_table[coef_table["feature"].isin(top_features)].pivot(index="feature", columns="model", values="coefficient")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pivot.index))
    width = 0.24
    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * width, pivot[col].values, width=width, label=col)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_title("Kaggle House Prices: largest standardized coefficients")
    ax.set_ylabel("Coefficient on standardized feature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_kaggle_cv_curves(searches: dict[str, GridSearchCV], path: Path) -> None:
    plot_cv_alpha_curves(searches, path)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_synthetic_report(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    searches: dict[str, GridSearchCV],
    model_metrics: list[dict[str, Any]],
    coef_table: pd.DataFrame,
    selector: ForwardSelectorCV,
    selector_metrics: dict[str, Any],
    lasso_nonzero: list[str],
) -> None:
    metric_rows = [{k: v for k, v in row.items()} for row in model_metrics + [selector_metrics]]
    metric_table = md_table(metric_rows, ["model", "RMSE", "MAE", "MAPE"])
    stability_rows = stability_df.to_dict(orient="records")
    stability_table = md_table(stability_rows, ["model", "feature", "coef_mean", "coef_std", "coef_min", "coef_max"])
    corr_rows = corr_df.head(12).to_dict(orient="records")
    corr_table = md_table(corr_rows, ["feature_1", "feature_2", "abs_corr"])

    best_param_rows = []
    for name, search in searches.items():
        best_param_rows.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "best_cv_rmse": -float(search.best_score_),
            }
        )
    best_param_table = md_table(best_param_rows, ["model", "best_params", "best_cv_rmse"])

    key_features = [
        "x_signal_1",
        "x_signal_2",
        "x_signal_3",
        "x_independent_1",
        "x_independent_2",
        "x_mixed_collinear",
        "noise_1",
        "noise_2",
    ]
    coef_rows = coef_table[coef_table["feature"].isin(key_features)].to_dict(orient="records")
    coef_table_md = md_table(coef_rows, ["model", "feature", "coefficient"])
    selector_history = selector.history_frame().to_dict(orient="records")
    selector_table = md_table(selector_history, ["step", "added_feature", "cv_rmse", "selected_features"])

    text = f"""# Week 13 Synthetic Report: Regularized Regression on Correlated Data

## 1. 数据生成设计与 DGP

本任务自己生成了 `week13/data/synthetic_correlated.csv`，样本量为 **{len(df)}**，特征数为 **{df.shape[1] - 1}**，满足 Week13 对“样本量不少于 300、至少 8 个特征、显式构造共线性特征族”的要求。

真实数据生成过程（DGP）为：

```text
 y = 20 + 4.0 * x_signal_1 - 2.5 * x_independent_1 + 1.8 * x_independent_2 + noise
```

其中：

- 高度相关特征族：`x_signal_1`, `x_signal_2`, `x_signal_3`，三者来自同一个 latent factor；
- 第二组潜在相关变量：`x_independent_1`, `x_independent_2`, `x_mixed_collinear`，其中 `x_mixed_collinear` 是前两个变量的线性混合；
- 纯噪声变量：`noise_1`, `noise_2`, `noise_3`, `noise_4`, `noise_5`；
- 弱信号变量：`weak_signal`，它与 latent factor 有轻微关系，但不在真实 DGP 中。

## 2. 高相关变量检查

下面是绝对相关系数超过 0.75 的变量对。可以看到 `x_signal_1/x_signal_2/x_signal_3` 这组变量高度相关，因此非常适合展示 OLS 系数不稳定和 Ridge/Lasso/Elastic Net 的差异。

{corr_table}

## 3. OLS 与 Ridge 的系数稳定性对比

作业要求至少做 50 次不同随机切分。我这里做了 **60 次 train/test split**，每次分别拟合 OLS 和 Ridge(alpha=10)，并收集高度相关特征族的系数。结果表明，OLS 在高度相关变量之间会“抢解释权”，单个变量的系数标准差较大；Ridge 通过 L2 penalty 把系数整体收缩，因此跨样本切分更稳定。

{stability_table}

对应图像：`week13/results/figures/synthetic_coefficient_stability.png`。

## 4. 为什么正则化前必须标准化？

Ridge、Lasso、Elastic Net 的 penalty 都直接作用在系数大小上。如果某个变量的量纲很大，模型可以用较小系数表达相同变化；如果某个变量量纲很小，则需要较大系数。若不标准化，penalty 会把“量纲差异”误当成“变量重要性差异”，导致正则化不公平。因此本实验用 `Pipeline([CustomStandardScaler(), model])`，其中 `CustomStandardScaler` 来自自己的 `src/utils/transformers.py`。

## 5. GridSearchCV 调参与最优参数

对 Ridge 和 Lasso 使用对数空间 `alpha`；对 Elastic Net 同时搜索 `alpha` 与 `l1_ratio`。5 折交叉验证的 RMSE 曲线保存为：`week13/results/figures/synthetic_cv_alpha_curves.png`。

{best_param_table}

## 6. 测试集模型表现

{metric_table}

RMSE 和 MAE 均由自己的 `src/utils/metrics.py` 计算，而不是直接调用 sklearn 指标。OLS 在预测上并不一定非常差，但它的系数解释不稳定；正则化的价值主要体现在稳定性和变量筛选解释上。

## 7. 模型性格：Ridge / Lasso / Elastic Net 如何处理相关变量？

下面列出关键变量在三个最优正则化模型中的标准化系数。

{coef_table_md}

解释：

- **Ridge**：倾向于把高度相关的一组变量一起保留，并较均匀地缩小系数；
- **Lasso**：倾向于在高度相关变量中挑选少数变量，把其他变量压到 0，因此有自动变量筛选效果，但也可能随机保留其中一个而丢掉同组变量；
- **Elastic Net**：同时有 L1 和 L2 penalty，通常比 Lasso 温和，会在稀疏性和组保留之间折中。

对应图像：`week13/results/figures/synthetic_model_coefficients.png`。

## 8. 自定义前向选择 vs Lasso 自动筛选

我在 `src/utils/models.py` 中实现了 `ForwardSelectorCV`，它每一步用 K 折 CV 比较所有候选变量，选择能使验证 RMSE 最低的变量。

前向选择过程：

{selector_table}

前向选择最终变量：

```text
{selector.selected_features_}
```

前向选择测试集指标：

{md_table([{k: v for k, v in selector_metrics.items()}], ["model", "RMSE", "MAE", "MAPE"])}

Lasso 非零变量名单：

```text
{lasso_nonzero}
```

两者不完全一致是正常的：前向选择是贪心搜索，每一步只看“当前新增一个变量后 CV RMSE 是否下降”；Lasso 是在同一个优化目标中同时平衡 loss 和 L1 penalty。面对高度相关变量时，Lasso 更容易保留其中一个代表变量，而前向选择可能根据当时已经入选的变量组合继续补充其他变量。
"""
    (RESULTS_DIR / "synthetic_report.md").write_text(text, encoding="utf-8")


def write_kaggle_report(
    kaggle_df: pd.DataFrame,
    feature_names: list[str],
    model_metrics: list[dict[str, Any]],
    selector: ForwardSelectorCV,
    selector_metrics: dict[str, Any],
    coef_table: pd.DataFrame,
    searches: dict[str, GridSearchCV],
    lasso_nonzero: list[str],
    lasso_removed: list[str],
) -> None:
    metric_table = md_table(model_metrics + [selector_metrics], ["model", "RMSE", "MAE", "MAPE"])
    best_param_rows = [
        {"model": name, "best_params": search.best_params_, "best_cv_rmse": -float(search.best_score_)}
        for name, search in searches.items()
    ]
    best_param_table = md_table(best_param_rows, ["model", "best_params", "best_cv_rmse"])

    enet_top = (
        coef_table[coef_table["model"] == "ElasticNet"]
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(10)
    )
    top_rows = enet_top[["model", "feature", "coefficient", "abs_coef"]].to_dict(orient="records")
    top_table = md_table(top_rows, ["model", "feature", "coefficient", "abs_coef"])
    selector_table = md_table(selector.history_frame().to_dict(orient="records"), ["step", "added_feature", "cv_rmse", "selected_features"])

    removed_preview = lasso_removed[:20]
    nonzero_preview = lasso_nonzero[:30]

    text = f"""# Week 13 Kaggle Report: House Prices Regularization Study

## 1. 数据来源与业务背景

本部分完成 Week13 的选做加分任务。数据来自 Kaggle 竞赛 **House Prices - Advanced Regression Techniques**，目标是根据 Ames, Iowa 房屋属性预测 `SalePrice`。本提交将原始训练集保存为：

```text
week13/data/kaggle_house_prices.csv
```

该数据集适合练习正则化和变量筛选，因为它有 **{kaggle_df.shape[0]}** 行、**{kaggle_df.shape[1]}** 列，特征数量较多；同时房屋面积、楼层面积、地下室面积、车库面积、质量评分等变量之间存在明显业务相关性，容易产生共线性。

本实验为了保持模型解释清楚，使用数值型变量建模，共使用 **{len(feature_names)}** 个特征。目标变量使用 `log1p(SalePrice)`，因此 RMSE/MAE 是对数价格误差，不是美元误差。

## 2. 数据清洗与自定义预处理

- 删除 `Id`，因为它只是样本编号；
- 仅保留数值型特征，保证线性模型系数含义清晰；
- 删除缺失率过高的数值列；
- 用自己的 `CustomNumericImputer` 在训练集上拟合中位数并填补缺失；
- 在 Ridge/Lasso/Elastic Net 的 `Pipeline` 中使用自己的 `CustomStandardScaler`；
- 训练/测试划分后，测试集只使用训练集学到的填补值和标准化参数，避免数据泄露。

## 3. GridSearchCV 最优参数

{best_param_table}

CV 曲线图：`week13/results/figures/kaggle_cv_alpha_curves.png`。

## 4. 测试集表现

{metric_table}

解释：如果正则化方法没有显著优于 OLS，可能原因有三点：第一，当前只使用数值变量，维度虽高但样本量也有 1460 行；第二，OLS 在预测层面未必崩溃，但系数解释可能不稳定；第三，真正能提升 Kaggle 排名的特征工程通常还包括大量类别变量编码、异常值处理和非线性模型，本作业重点是正则化回归的推测比较。

## 5. Lasso 删除了哪些特征？业务上是否合理？

Lasso 非零变量预览：

```text
{nonzero_preview}
```

Lasso 压缩为 0 的变量预览：

```text
{removed_preview}
```

从业务上看，被压为 0 并不意味着这些房屋属性“没有用”。很多面积类变量高度相关，例如 `TotalBsmtSF`, `1stFlrSF`, `GrLivArea`, `GarageArea`；质量类变量也可能互相替代。Lasso 面对相关变量时倾向于保留一个代表变量，因此被剔除的变量更准确地说是“在当前 penalty 和其他变量共同存在的条件下，边际贡献不足”。

## 6. 如果业务方要最关键的 5 个影响因素，我会以什么方法为准？

我不会直接只看 Lasso。Lasso 的优点是稀疏，但它在高度相关变量组内可能随机保留其中一个。业务解释时，我更倾向于综合：

1. Elastic Net 的绝对系数排名；
2. Lasso 的非零变量名单；
3. 前向选择的 Top-K 结果；
4. 房地产业务常识。

Elastic Net 的前 10 个标准化系数如下：

{top_table}

前向选择过程如下：

{selector_table}

最终前向选择变量：

```text
{selector.selected_features_}
```

图像：`week13/results/figures/kaggle_top_coefficients.png`。

## 7. 真实决策风险

该模型可以帮助理解哪些房屋属性与价格相关，但不能直接当作因果结论。例如 `OverallQual` 重要，并不等价于“只要人为提高评分就能同比例提高售价”。此外，房价受地段、宏观周期、供需、学校和社区等影响，单纯线性模型只能提供一个可解释基准。
"""
    (RESULTS_DIR / "kaggle_report.md").write_text(text, encoding="utf-8")


def write_summary_comparison(synthetic: dict[str, Any], kaggle: dict[str, Any]) -> None:
    synth_metrics = md_table(
        synthetic["model_metrics"] + [synthetic["selector_metrics"]], ["model", "RMSE", "MAE", "MAPE"]
    )
    kaggle_metrics = md_table(kaggle["model_metrics"] + [kaggle["selector_metrics"]], ["model", "RMSE", "MAE", "MAPE"])

    text = f"""# Week 13 Summary Comparison

## 1. Lasso 面对高度相关变量组的业务风险，Elastic Net 如何缓解？

在模拟数据中，`x_signal_1`, `x_signal_2`, `x_signal_3` 几乎表达同一类信息。Lasso 的 L1 penalty 会产生稀疏解，因此它可能只保留其中一个变量，把其他同组变量压到 0。从预测角度这未必是坏事，但从业务解释角度有风险：业务方可能误以为被压为 0 的变量完全不重要。

Elastic Net 同时包含 L1 与 L2 penalty。L1 带来变量筛选，L2 让相关变量可以成组保留并共同收缩。因此 Elastic Net 通常比 Lasso 更适合“变量高度相关但都代表同一业务维度”的场景。

## 2. GridSearchCV 的最低验证误差，与“越稀疏越好/越稳越好”有什么异同？

GridSearchCV 的目标是选择验证 RMSE 最低的超参数，它直接服务于预测泛化表现。但“越稀疏越好”是解释性偏好，“越稳越好”是系数稳定性偏好。三者相关但不完全相同：

- 最低验证误差：关注预测准不准；
- 稀疏：关注变量名单短不短；
- 稳定：关注样本变化时结论是否可靠。

因此，如果任务是纯预测，可以优先看 GridSearchCV；如果任务是向业务方解释关键因素，则还要看系数稳定性、变量共线性和 Lasso/Elastic Net 的筛选差异。

## 3. 前向选择/后向剔除与 Lasso 的效率和结果差异

本作业实现的是前向选择 Top-K。它的优点是过程直观，每一步都能解释“为什么加入这个变量”；缺点是计算成本较高，因为每一步要遍历剩余候选变量并做交叉验证，而且它是贪心算法，早期选择可能影响后续结果。

Lasso 则是在一个优化目标中同时完成拟合和筛选，计算上通常更统一、更适合高维数据。但在高度相关变量组中，Lasso 的选择可能不稳定，可能只保留其中一个代表变量。

## 4. 模拟数据结果概览

{synth_metrics}

模拟数据的主要结论是：OLS 预测不一定很差，但高度相关变量的单个系数不稳定；Ridge 更稳定，Lasso 更稀疏，Elastic Net 介于二者之间。

## 5. Kaggle 真实数据结果概览

{kaggle_metrics}

Kaggle House Prices 数据的主要结论是：正则化方法提供了更稳健的系数解释框架。即使测试误差提升不巨大，Ridge/Lasso/Elastic Net 仍能帮助我们理解高维、共线性特征下的变量收缩与筛选。

## 6. 本周文件与代码位置

- 入口：`week13/main.py`
- 自定义指标：`src/utils/metrics.py`
- 自定义标准化、填补、预处理：`src/utils/transformers.py`
- 自定义前向选择：`src/utils/models.py` 中的 `ForwardSelectorCV`
- 模拟数据：`week13/data/synthetic_correlated.csv`
- Kaggle 数据：`week13/data/kaggle_house_prices.csv`
- 图像目录：`week13/results/figures/`
"""
    (RESULTS_DIR / "summary_comparison.md").write_text(text, encoding="utf-8")


def main() -> None:
    reset_outputs()
    synthetic_results = run_synthetic_task()
    kaggle_results = run_kaggle_task()
    write_summary_comparison(synthetic_results, kaggle_results)
    print("Week 13 finished. Reports are in week13/results/.")


if __name__ == "__main__":
    # Keep BLAS from over-using resources on small homework data.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
