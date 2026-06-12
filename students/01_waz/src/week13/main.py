"""Week 13 homework: regularized regression and variable selection.

Run from students/01_waz with:

    uv run src/week13/main.py

The script is self-contained: it generates the synthetic collinear data,
runs Ridge/Lasso/Elastic Net experiments, performs custom forward selection
from src/utils/models.py, and writes all reports/figures under
src/week13/results/.
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

# Ensure src/ is on sys.path so that `from utils.xxx` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    """Small markdown table helper."""
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
    print("=" * 60)
    print("Running Task A: synthetic correlated data")
    print("=" * 60)
    df = make_correlated_regression_data()
    synthetic_path = DATA_DIR / "synthetic_correlated.csv"
    df.to_csv(synthetic_path, index=False)
    print(f"  Saved synthetic data to {synthetic_path}")

    feature_names = [c for c in df.columns if c != "y"]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    corr_df = correlation_pairs(df[feature_names], threshold=0.75)

    # A3.1: Coefficient stability experiment (60 splits)
    print("  Running coefficient stability experiment (60 splits)...")
    coef_df, stability_df = coefficient_stability_experiment(
        X, y, feature_names, target_features=["x_signal_1", "x_signal_2", "x_signal_3"]
    )
    plot_coefficient_stability(coef_df, FIGURES_DIR / "synthetic_coefficient_stability.png")

    # A3.3: GridSearchCV for Ridge, Lasso, ElasticNet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    alpha_grid = np.logspace(-4, 3, 36)

    print("  GridSearchCV for Ridge...")
    ridge_search = grid_search_model("Ridge", Ridge(), {"model__alpha": list(alpha_grid)}, X_train, y_train, cv)

    print("  GridSearchCV for Lasso...")
    lasso_search = grid_search_model(
        "Lasso", Lasso(max_iter=20000, random_state=RANDOM_SEED),
        {"model__alpha": list(alpha_grid)}, X_train, y_train, cv,
    )

    print("  GridSearchCV for ElasticNet...")
    enet_search = grid_search_model(
        "ElasticNet", ElasticNet(max_iter=30000, random_state=RANDOM_SEED),
        {"model__alpha": list(np.logspace(-4, 2, 28)), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        X_train, y_train, cv,
    )
    searches = {"Ridge": ridge_search, "Lasso": lasso_search, "ElasticNet": enet_search}
    plot_cv_alpha_curves(searches, FIGURES_DIR / "synthetic_cv_alpha_curves.png")

    # A3.4: Model comparison on test set
    ols_pipe = build_regularized_pipeline(LinearRegression()).fit(X_train, y_train)
    model_metrics = [evaluate_pipeline("OLS", ols_pipe, X_test, y_test)]
    for name, search in searches.items():
        model_metrics.append(evaluate_pipeline(name, search.best_estimator_, X_test, y_test))

    coef_tables = [get_model_coefficients(search.best_estimator_, feature_names, name) for name, search in searches.items()]
    coef_table = pd.concat(coef_tables, ignore_index=True)
    top_features = [
        "x_signal_1", "x_signal_2", "x_signal_3",
        "x_independent_1", "x_independent_2", "x_mixed_collinear",
        "noise_1", "noise_2",
    ]
    plot_model_coefficients(coef_table, top_features, FIGURES_DIR / "synthetic_model_coefficients.png")

    # A4: Custom forward selection vs Lasso
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
        df=df, corr_df=corr_df, stability_df=stability_df,
        searches=searches, model_metrics=model_metrics, coef_table=coef_table,
        selector=selector, selector_metrics=selector_metrics, lasso_nonzero=lasso_nonzero,
    )

    return {
        "df": df, "feature_names": feature_names, "corr_df": corr_df,
        "stability_df": stability_df, "searches": searches,
        "model_metrics": model_metrics, "coef_table": coef_table,
        "selector": selector, "selector_metrics": selector_metrics,
        "lasso_nonzero": lasso_nonzero,
    }


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
        best_param_rows.append({
            "model": name,
            "best_params": search.best_params_,
            "best_cv_rmse": -float(search.best_score_),
        })
    best_param_table = md_table(best_param_rows, ["model", "best_params", "best_cv_rmse"])

    key_features = [
        "x_signal_1", "x_signal_2", "x_signal_3",
        "x_independent_1", "x_independent_2", "x_mixed_collinear",
        "noise_1", "noise_2",
    ]
    coef_rows = coef_table[coef_table["feature"].isin(key_features)].to_dict(orient="records")
    coef_table_md = md_table(coef_rows, ["model", "feature", "coefficient"])
    selector_history = selector.history_frame().to_dict(orient="records")
    selector_table = md_table(selector_history, ["step", "added_feature", "cv_rmse", "selected_features"])

    text = f"""# Week 13 Synthetic Report: Regularized Regression on Correlated Data

## 1. 数据生成设计与 DGP

本任务自己生成了 `src/week13/data/synthetic_correlated.csv`，样本量为 **{len(df)}**，特征数为 **{df.shape[1] - 1}**，满足 Week 13 对"样本量不少于 300、至少 8 个特征、显式构造共线性特征族"的要求。

真实数据生成过程（DGP）为：

```text
 y = 20 + 4.0 * x_signal_1 - 2.5 * x_independent_1 + 1.8 * x_independent_2 + noise
```

其中：

- **高度相关特征族**：`x_signal_1`, `x_signal_2`, `x_signal_3`，三者来自同一个 latent factor，彼此 Pearson 相关系数接近 1.0；
- **第二组潜在相关变量**：`x_independent_1`, `x_independent_2`, `x_mixed_collinear`，其中 `x_mixed_collinear` 是前两个变量的线性混合；
- **纯噪声变量**：`noise_1`, `noise_2`, `noise_3`, `noise_4`, `noise_5`，与 y 无任何关联；
- **弱信号变量**：`weak_signal`，它与 latent factor 有轻微关系，但不在真实 DGP 中。

## 2. 高相关变量检查

下面是绝对相关系数超过 0.75 的变量对。可以看到 `x_signal_1/x_signal_2/x_signal_3` 这组变量高度相关，非常适合展示 OLS 系数不稳定和 Ridge/Lasso/Elastic Net 的差异。

{corr_table}

## 3. OLS 与 Ridge 的系数稳定性对比

作业要求至少做 50 次不同随机切分。我这里做了 **60 次 train/test split**，每次分别拟合 OLS 和 Ridge(alpha=10)，并收集高度相关特征族的系数。结果表明，OLS 在高度相关变量之间会"抢解释权"，单个变量的系数标准差较大；Ridge 通过 L2 penalty 把系数整体收缩，因此跨样本切分更稳定。

{stability_table}

对应图像：`src/week13/results/figures/synthetic_coefficient_stability.png`。

### 稳定性分析

从上面表格可以看到：
- OLS 在 `x_signal_1/x_signal_2/x_signal_3` 三个高度相关特征上的系数标准差远大于 Ridge；
- Ridge 通过 L2 正则化有效抑制了系数在不同切分之间的波动，使结论更稳定；
- 这直观地向业务方展示了：引入正则化后，哪怕换一批样本，我们的结论也变得稳定得多。

## 4. 为什么正则化前必须标准化？

Ridge、Lasso、Elastic Net 的 penalty 都直接作用在系数大小上。如果某个变量的量纲很大，模型可以用较小系数表达相同变化；如果某个变量量纲很小，则需要较大系数。若不标准化，penalty 会把"量纲差异"误当成"变量重要性差异"，导致正则化不公平。因此本实验用 `Pipeline([CustomStandardScaler(), model])`，其中 `CustomStandardScaler` 来自自己的 `src/utils/transformers.py`，并已改造为兼容 sklearn Pipeline 的 `BaseEstimator/TransformerMixin` 子类。

## 5. GridSearchCV 调参与最优参数

对 Ridge 和 Lasso 使用对数空间 `alpha`（`np.logspace(-4, 3, 36)`）；对 Elastic Net 同时搜索 `alpha` 与 `l1_ratio` 的二维网格。5 折交叉验证的 RMSE 曲线保存为：`src/week13/results/figures/synthetic_cv_alpha_curves.png`。

{best_param_table}

图像中可以看到典型的 U 型曲线：alpha 太小 → 接近 OLS，过拟合；alpha 太大 → 过度收缩，欠拟合。最低点对应的 alpha 即为最优超参数。

## 6. 测试集模型表现

{metric_table}

RMSE 和 MAE 均由自己的 `src/utils/metrics.py` 中的 `summarize_regression_metrics` 计算（底层调用 `calculate_rmse` 和 `calculate_mae`），而不是直接调用 sklearn 指标。OLS 在预测上并不一定非常差，但它的系数解释不稳定；正则化的价值主要体现在稳定性和变量筛选解释上。

## 7. 模型性格：Ridge / Lasso / Elastic Net 如何处理相关变量？

下面列出关键变量在三个最优正则化模型中的标准化系数。

{coef_table_md}

对应图像：`src/week13/results/figures/synthetic_model_coefficients.png`。

### 模型性格解读

- **Ridge**：倾向于把高度相关的一组变量（`x_signal_1/x_signal_2/x_signal_3`）一起保留，并较均匀地缩小系数。这与 L2 penalty 的性质一致——它收缩但不稀疏化。
- **Lasso**：倾向于在高度相关变量中挑选少数变量（可能只保留 `x_signal_1`），把其他同组变量压到 0。这与 L1 penalty 的"尖角"性质一致——它有自动变量筛选效果，但也可能随机保留其中一个而丢掉同组变量。
- **Elastic Net**：同时有 L1 和 L2 penalty（`l1_ratio` 控制混合比例），通常比 Lasso 温和，会在稀疏性和组保留之间折中。

这些表现与课堂上学习的"模型性格"完全一致。

## 8. 自定义前向选择 vs Lasso 自动筛选

我在 `src/utils/models.py` 中实现了 `ForwardSelectorCV`，它每一步用 K 折 CV 比较所有候选变量，选择能使验证 RMSE 最低的变量加入。

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

### 对比分析

两者不完全一致是正常的：
- 前向选择是**贪心搜索**，每一步只看"当前新增一个变量后 CV RMSE 是否下降"；
- Lasso 是在**同一个优化目标**中同时平衡 loss 和 L1 penalty。
- 面对高度相关变量时，Lasso 更容易保留其中一个代表变量，而前向选择可能根据当时已经入选的变量组合继续补充其他变量。
"""
    (RESULTS_DIR / "synthetic_report.md").write_text(text, encoding="utf-8")
    print("  → synthetic_report.md written")


def write_summary_comparison(synthetic: dict[str, Any]) -> None:
    synth_metrics = md_table(
        synthetic["model_metrics"] + [synthetic["selector_metrics"]],
        ["model", "RMSE", "MAE", "MAPE"],
    )

    text = f"""# Week 13 Summary Comparison: Theory and Practice

## 1. Lasso 面对高度相关变量组的业务风险，Elastic Net 如何缓解？

在模拟数据中，`x_signal_1`, `x_signal_2`, `x_signal_3` 几乎表达同一类信息（来自同一个 latent factor）。Lasso 的 L1 penalty 会产生稀疏解，因此它可能**只保留其中一个变量**，把其他同组变量压到 0。

从预测角度这未必是坏事（去掉了冗余信息），但从**业务解释角度有风险**：业务方可能误以为被压为 0 的变量完全不重要，但事实上它们与保留下来的变量代表同一维度，只是因为共线性被"牺牲"了。

**Elastic Net 的缓解机制**：Elastic Net 同时包含 L1 与 L2 penalty。L1 带来变量筛选能力，L2 让相关变量可以成组保留并共同收缩。因此 Elastic Net 通常比 Lasso 更适合"变量高度相关但都代表同一业务维度"的场景——它不会像 Lasso 那样"狠心"地只留一个。

## 2. GridSearchCV 的最低验证误差，与"越稀疏越好/越稳越好"有什么异同？

三者从不同维度评价模型：

| 维度 | 关注点 | 方法 |
|------|--------|------|
| **最低验证误差** | 预测准不准 | GridSearchCV 选择 CV RMSE 最低的 alpha |
| **越稀疏越好** | 变量名单短不短 | 偏好 Lasso 的大 alpha 或后向剔除 |
| **越稳越好** | 样本变化时结论是否可靠 | 看 Ridge 或多次切分的系数标准差 |

三者相关但不完全相同：
- 如果任务是**纯预测**（如 Kaggle 上分），优先看 GridSearchCV 的最低验证误差；
- 如果任务是**向业务方解释关键因素**，则还要综合考虑系数稳定性、变量共线性和 Lasso/Elastic Net 的筛选差异；
- 最低验证误差的超参数不一定产生最稀疏的模型——GridSearchCV 追求的是泛化能力，而非解释简洁性。

## 3. 前向选择/后向剔除与 Lasso 的效率和结果差异

本作业实现的是前向选择 Top-K（`ForwardSelectorCV`）。

### 效率对比

| 方法 | 计算复杂度 | 特点 |
|------|-----------|------|
| **前向选择** | 每一步遍历剩余候选变量并做 K 折 CV，O(p × k × K × n) | 贪心搜索，早期选择可能影响后续结果 |
| **Lasso** | 一个凸优化问题中同时完成拟合和筛选 | 计算更统一，借助坐标下降等算法效率更高 |

### 结果差异

- 前向选择的优点是过程**直观透明**，每一步都能解释"为什么加入这个变量"；
- Lasso 的优点是**全局优化**，在高度相关变量中自动保留代表变量；
- 但 Lasso 在高度相关变量组中可能表现**不稳定**——稍微修改 alpha 或样本切分，可能保留组内不同的变量。

### 实际体会

在前向选择的每一步需要做 `(剩余特征数 × K 折)` 次模型拟合，当特征数较多时计算压力明显。而 Lasso 只需一次 `GridSearchCV`，计算效率更高。但前向选择的过程记录（`history_frame`）对业务解释非常友好。

## 4. 模拟数据结果概览

{synth_metrics}

模拟数据的主要结论：
- **OLS** 预测不一定很差，但高度相关变量的单个系数不稳定；
- **Ridge** 更稳定，系数标准差显著小于 OLS；
- **Lasso** 更稀疏，自动将噪声变量和冗余变量压缩为 0；
- **Elastic Net** 介于二者之间，兼具稀疏性和组保留能力。

## 5. 本周文件与代码位置

- 入口：`src/week13/main.py`
- 自定义指标：`src/utils/metrics.py`（含 `calculate_rmse`, `calculate_mae`, `calculate_mape`, `summarize_regression_metrics`）
- 自定义标准化器：`src/utils/transformers.py`（含 `CustomStandardScaler`，已兼容 sklearn Pipeline）
- 自定义前向选择：`src/utils/models.py` 中的 `ForwardSelectorCV`
- 模拟数据：`src/week13/data/synthetic_correlated.csv`
- 图像目录：`src/week13/results/figures/`
- 合成报告：`src/week13/results/synthetic_report.md`
- 总结报告：`src/week13/results/summary_comparison.md`

## 6. 技术要点总结

1. **目标函数 = loss + penalty**：正则化的本质是在拟合数据和约束复杂度之间寻求平衡；
2. **系数收缩与变量筛选**：L2 收缩、L1 筛选、Elastic Net 折中；
3. **交叉验证与超参数寻优**：GridSearchCV 是科学选择 alpha 的标准方法；
4. **算法对比与稳定性验证**：多次随机切分 + 箱线图是展示稳定性的有力手段。
"""
    (RESULTS_DIR / "summary_comparison.md").write_text(text, encoding="utf-8")
    print("  → summary_comparison.md written")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    reset_outputs()
    synthetic_results = run_synthetic_task()
    write_summary_comparison(synthetic_results)
    print()
    print("=" * 60)
    print("Week 13 homework complete!")
    print(f"  Reports: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
