#!/usr/bin/env python3
"""Week 13: Regularization Methods — Complete Pipeline (Task A + Task B + Task C).

Run:
    uv run src/week13/main.py
    python src/week13/main.py

This single-entry script generates synthetic correlated data, compares OLS / Ridge /
Lasso / ElasticNet with GridSearchCV, implements forward selection, downloads and
processes the Boston Housing dataset, and writes all reports and figures.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # .../students/10_xzn/src/week13/
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Fallback metrics (if utils/metrics.py is unavailable)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # students/10_xzn/src
    from utils.metrics import calculate_rmse, calculate_mae
except ImportError:
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def config_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "figure.figsize": (10, 5),
        }
    )
    sns.set_style("whitegrid")


def rmse_scorer() -> callable:
    """Return a callable scorer (lower is better) for use in grid-search."""
    return make_scorer(
        lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        greater_is_better=False,
    )


# ---------------------------------------------------------------------------
# Task A — Synthetic Data Generation
# ---------------------------------------------------------------------------
def generate_synthetic_data(
    n: int = 300,
    noise_sigma: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """A1 & A2: Generate synthetic data with strong collinearity.

    Features:
      x1 ~ N(0, 1)
      x2 = x1 + N(0, 0.1)
      x3 = 2*x1 + N(0, 0.1)
      x4, x5 ~ N(0, 1)  (pure noise)
      x6 ~ N(0, 1)  (additional useful, independent feature)
      x7, x8 ~ N(0, 1)  (extra noise)

    True DGP:
      y = 3*x1 + 1.5*x2 - 2*x3 + 0*x4 + 0*x5 + 2*x6 + 0*x7 + 0*x8 + eps
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.1, n)
    x3 = 2 * x1 + rng.normal(0, 0.1, n)
    x4 = rng.normal(0, 1, n)
    x5 = rng.normal(0, 1, n)
    x6 = rng.normal(0, 1, n)
    x7 = rng.normal(0, 1, n)
    x8 = rng.normal(0, 1, n)

    eps = rng.normal(0, noise_sigma, n)
    y = 3 * x1 + 1.5 * x2 - 2 * x3 + 2 * x6 + eps

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
            "y": y,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Task A3 — Coefficient Stability Boxplot (OLS vs Ridge over 50 splits)
# ---------------------------------------------------------------------------
def run_stability_experiment(
    df: pd.DataFrame,
    n_splits: int = 50,
    alpha: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Repeated random splits → fit OLS & Ridge → collect coefficients of x1,x2,x3."""
    X_all = df[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]].values
    y_all = df["y"].values

    records = []
    for i in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.3, random_state=seed + i
        )
        # OLS
        pipe_ols = Pipeline(
            [("scaler", StandardScaler()), ("lr", LinearRegression())]
        )
        pipe_ols.fit(X_tr, y_tr)
        coef_ols = pipe_ols.named_steps["lr"].coef_

        # Ridge
        pipe_ridge = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )
        pipe_ridge.fit(X_tr, y_tr)
        coef_ridge = pipe_ridge.named_steps["ridge"].coef_

        for j, fname in enumerate(["x1", "x2", "x3"]):
            records.append(
                {
                    "split": i,
                    "model": "OLS",
                    "feature": fname,
                    "coefficient": coef_ols[j],
                }
            )
            records.append(
                {
                    "split": i,
                    "model": "Ridge",
                    "feature": fname,
                    "coefficient": coef_ridge[j],
                }
            )

    return pd.DataFrame(records)


def plot_coefficient_stability(df_coef: pd.DataFrame) -> Path:
    """Boxplot of coefficients for x1, x2, x3 under OLS and Ridge."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, feat in zip(axes, ["x1", "x2", "x3"]):
        sub = df_coef[df_coef["feature"] == feat]
        sns.boxplot(
            data=sub,
            x="model",
            y="coefficient",
            palette="Set2",
            ax=ax,
            width=0.5,
        )
        ax.set_title(f"Coefficient stability — {feat}")
        ax.set_xlabel("")
        ax.set_ylabel("Estimated coefficient")
    fig.tight_layout()
    path = FIGURES_DIR / "coefficient_stability_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Task A3 — GridSearch & CV curves
# ---------------------------------------------------------------------------
def grid_search_task_a(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Grid-search for Ridge, Lasso, ElasticNet; return best estimators + CV tables."""
    alpha_grid_ridge_lasso = np.logspace(-4, 3, 50)
    alpha_grid_en = np.logspace(-4, 3, 20)
    l1_ratio_grid = np.linspace(0.1, 1.0, 5)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    base_pipe = Pipeline([("scaler", StandardScaler())])

    # Ridge
    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gs_ridge = GridSearchCV(
        pipe_ridge,
        param_grid={"ridge__alpha": alpha_grid_ridge_lasso},
        cv=cv,
        scoring=rmse_scorer(),
        n_jobs=-1,
    )
    gs_ridge.fit(X_train, y_train)

    # Lasso
    pipe_lasso = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))])
    gs_lasso = GridSearchCV(
        pipe_lasso,
        param_grid={"lasso__alpha": alpha_grid_ridge_lasso},
        cv=cv,
        scoring=rmse_scorer(),
        n_jobs=-1,
    )
    gs_lasso.fit(X_train, y_train)

    # ElasticNet
    pipe_en = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("en", ElasticNet(max_iter=10000)),
        ]
    )
    gs_en = GridSearchCV(
        pipe_en,
        param_grid={"en__alpha": alpha_grid_en, "en__l1_ratio": l1_ratio_grid},
        cv=cv,
        scoring=rmse_scorer(),
        n_jobs=-1,
    )
    gs_en.fit(X_train, y_train)

    best = {
        "Ridge": gs_ridge.best_estimator_,
        "Lasso": gs_lasso.best_estimator_,
        "ElasticNet": gs_en.best_estimator_,
    }

    # CV curve dataframes
    ridge_cv_df = pd.DataFrame(
        {
            "alpha": alpha_grid_ridge_lasso,
            "mean_cv_rmse": -gs_ridge.cv_results_["mean_test_score"],
            "std_cv_rmse": gs_ridge.cv_results_["std_test_score"],
        }
    )
    lasso_cv_df = pd.DataFrame(
        {
            "alpha": alpha_grid_ridge_lasso,
            "mean_cv_rmse": -gs_lasso.cv_results_["mean_test_score"],
            "std_cv_rmse": gs_lasso.cv_results_["std_test_score"],
        }
    )
    return best, ridge_cv_df, lasso_cv_df


def plot_cv_error_vs_alpha(
    ridge_df: pd.DataFrame,
    lasso_df: pd.DataFrame,
) -> Path:
    """Ridge & Lasso CV RMSE vs alpha log-scale line plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        ridge_df["alpha"],
        ridge_df["mean_cv_rmse"],
        yerr=ridge_df["std_cv_rmse"],
        label="Ridge",
        marker="o",
        markersize=3,
        capsize=2,
    )
    ax.errorbar(
        lasso_df["alpha"],
        lasso_df["mean_cv_rmse"],
        yerr=lasso_df["std_cv_rmse"],
        label="Lasso",
        marker="s",
        markersize=3,
        capsize=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("5-fold CV RMSE")
    ax.set_title("CV error vs regularisation strength (alpha)")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "cv_error_vs_alpha.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Task A4 — Forward Selection (custom implementation)
# ---------------------------------------------------------------------------
def forward_selection_cv(
    X: np.ndarray,
    y: np.ndarray,
    max_features: int = 5,
    cv: int = 5,
    seed: int = 42,
) -> Tuple[List[int], List[float]]:
    """Forward selection using 5-fold CV RMSE as criterion.

    Returns:
        selected_indices: list of feature indices in selection order.
        cv_rmse_history: CV RMSE after each addition.
    """
    n, p = X.shape
    remaining = set(range(p))
    selected: List[int] = []
    cv_rmse_history: List[float] = []

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    # baseline (intercept-only)
    baseline = np.mean(
        [np.sqrt(np.mean((y[tst] - np.mean(y[trn])) ** 2)) for trn, tst in kf.split(X)]
    )
    best_global_rmse = baseline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max_features):
            best_candidate = None
            best_candidate_score = np.inf
            for cand in sorted(remaining):
                cols = selected + [cand]
                X_sub = X[:, cols]
                scores = []
                for trn, tst in kf.split(X_sub):
                    # fit OLS
                    X_trn = X_sub[trn]
                    y_trn = y[trn]
                    X_tst = X_sub[tst]
                    y_tst = y[tst]
                    # Standardize within fold
                    scaler = StandardScaler()
                    X_trn_s = scaler.fit_transform(X_trn)
                    X_tst_s = scaler.transform(X_tst)
                    lr = LinearRegression()
                    lr.fit(X_trn_s, y_trn)
                    preds = lr.predict(X_tst_s)
                    scores.append(np.sqrt(mean_squared_error(y_tst, preds)))
                mean_score = float(np.mean(scores))
                if mean_score < best_candidate_score:
                    best_candidate_score = mean_score
                    best_candidate = cand
            if best_candidate is None:
                break
            # Only add if it improves CV RMSE
            if best_candidate_score < best_global_rmse:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                best_global_rmse = best_candidate_score
                cv_rmse_history.append(best_candidate_score)
            else:
                break

    return selected, cv_rmse_history


# ---------------------------------------------------------------------------
# Task A — evaluate models on test set
# ---------------------------------------------------------------------------
def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute test-set RMSE/MAE and collect coefficients for all models."""
    rows = []
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)
        # extract coefficients (last step is the estimator)
        if name == "OLS":
            coef = pipe.named_steps["lr"].coef_
        elif name == "Ridge":
            coef = pipe.named_steps["ridge"].coef_
        elif name == "Lasso":
            coef = pipe.named_steps["lasso"].coef_
        elif name == "ElasticNet":
            coef = pipe.named_steps["en"].coef_
        else:
            coef = np.zeros(len(feature_names))
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mae,
                **{f"coef_{fn}": c for fn, c in zip(feature_names, coef)},
                "nonzero_features": sum(
                    1 for c in coef if abs(c) > 1e-6
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task B — Boston Housing (real-world data)
# ---------------------------------------------------------------------------
BOSTON_URL = (
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)
BOSTON_PATH = DATA_DIR / "real_estate.csv"


def download_boston_housing() -> pd.DataFrame:
    """Download Boston Housing CSV from public URL, save locally, return DataFrame."""
    if BOSTON_PATH.exists():
        return pd.read_csv(BOSTON_PATH)

    print("[Task B] Downloading Boston Housing dataset from GitHub...")
    try:
        df = pd.read_csv(BOSTON_URL)
    except Exception:
        print("[Task B] URL failed; trying sklearn California Housing as fallback.")
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=True)
        df = data.frame
        df.rename(columns={"MedHouseVal": "medv"}, inplace=True)
        print("[Task B] Note: California Housing has 8 features (<15).")
    df.to_csv(BOSTON_PATH, index=False)
    return df


def task_b_pipeline(df: pd.DataFrame) -> Dict:
    """Run full modelling pipeline on the Boston Housing dataset.

    Returns dict with model results, best models, etc.
    """
    # Separate target
    target_col = "medv"
    if target_col not in df.columns:
        # Try California Housing fallback naming
        target_col = "MedHouseVal"
        if target_col not in df.columns:
            raise KeyError(f"Target column not found. Columns: {list(df.columns)}")
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    feature_names = list(X.columns)

    # Basic cleaning
    n_missing = X.isnull().sum().sum()
    if n_missing > 0:
        print(f"[Task B] {n_missing} missing values found; filling with column means.")
        X = X.fillna(X.mean())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, random_state=42
    )

    # OLS
    pipe_ols = Pipeline(
        [("scaler", StandardScaler()), ("lr", LinearRegression())]
    )
    pipe_ols.fit(X_train, y_train)

    # GridSearch for regularised models
    alpha_grid = np.logspace(-4, 3, 50)
    alpha_grid_en = np.logspace(-4, 3, 20)
    l1_grid = np.linspace(0.1, 1.0, 5)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Ridge
    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gs_ridge = GridSearchCV(
        pipe_ridge, {"ridge__alpha": alpha_grid}, cv=cv, scoring=rmse_scorer(), n_jobs=-1
    )
    gs_ridge.fit(X_train, y_train)

    # Lasso
    pipe_lasso = Pipeline(
        [("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))]
    )
    gs_lasso = GridSearchCV(
        pipe_lasso, {"lasso__alpha": alpha_grid}, cv=cv, scoring=rmse_scorer(), n_jobs=-1
    )
    gs_lasso.fit(X_train, y_train)

    # ElasticNet
    pipe_en = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("en", ElasticNet(max_iter=10000)),
        ]
    )
    gs_en = GridSearchCV(
        pipe_en,
        {"en__alpha": alpha_grid_en, "en__l1_ratio": l1_grid},
        cv=cv,
        scoring=rmse_scorer(),
        n_jobs=-1,
    )
    gs_en.fit(X_train, y_train)

    best_models = {
        "OLS": pipe_ols,
        "Ridge": gs_ridge.best_estimator_,
        "Lasso": gs_lasso.best_estimator_,
        "ElasticNet": gs_en.best_estimator_,
    }

    # Evaluate
    eval_rows = []
    for name, pipe in best_models.items():
        y_pred = pipe.predict(X_test)
        rmse_v = calculate_rmse(y_test, y_pred)
        mae_v = calculate_mae(y_test, y_pred)
        # Extract coefficients
        if name == "OLS":
            coef = pipe.named_steps["lr"].coef_
        elif name == "Ridge":
            coef = pipe.named_steps["ridge"].coef_
        elif name == "Lasso":
            coef = pipe.named_steps["lasso"].coef_
        elif name == "ElasticNet":
            coef = pipe.named_steps["en"].coef_
        else:
            coef = np.zeros(len(feature_names))
        nonzero = [feature_names[i] for i, c in enumerate(coef) if abs(c) > 1e-6]
        zeroed = [feature_names[i] for i, c in enumerate(coef) if abs(c) <= 1e-6]
        eval_rows.append(
            {
                "model": name,
                "rmse": float(rmse_v),
                "mae": float(mae_v),
                "best_params": (
                    {}
                    if name == "OLS"
                    else {
                        k.replace("ridge__", "").replace("lasso__", "").replace("en__", ""): float(v)
                        for k, v in (
                            gs_ridge.best_params_
                            if name == "Ridge"
                            else gs_lasso.best_params_
                            if name == "Lasso"
                            else gs_en.best_params_
                        ).items()
                    }
                ),
                "nonzero_features": nonzero,
                "zeroed_features": zeroed,
                **{f"coef_{fn}": float(c) for fn, c in zip(feature_names, coef)},
            }
        )

    eval_df = pd.DataFrame(eval_rows)

    return {
        "eval_df": eval_df,
        "feature_names": feature_names,
        "best_lasso": best_models["Lasso"],
        "best_ridge": best_models["Ridge"],
        "best_en": best_models["ElasticNet"],
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_synthetic_report(
    df: pd.DataFrame,
    eval_df: pd.DataFrame,
    selected_fs: List[int],
    feature_names: List[str],
    lasso_nonzero: List[str],
) -> None:
    """Write synthetic_report.md."""
    # Determine Lasso non-zero feature names
    lasso_coef = {}
    for _, row in eval_df.iterrows():
        if row["model"] == "Lasso":
            for fn in feature_names:
                lasso_coef[fn] = row.get(f"coef_{fn}", 0)
    lasso_nz_names = [fn for fn, v in lasso_coef.items() if abs(v) > 1e-6]
    fs_names = [feature_names[i] for i in selected_fs]

    report = f"""# Task A — Synthetic Correlated Data Report

## A1 & A2: Data Generating Process

**Sample size:** n = {len(df)}
**Features:** {feature_names}

**True DGP:**
```
y = 3*x1 + 1.5*x2 - 2*x3 + 2*x6 + ε,  ε ~ N(0, 0.5)
```

**Highly correlated group:** `x1, x2, x3`
- `x2 = x1 + N(0, 0.1)`
- `x3 = 2*x1 + N(0, 0.1)`

**Pure noise features:** `x4, x5, x7, x8` — coefficients = 0.

**Additional useful independent feature:** `x6` — coefficient = 2.

---

## A3: Model Comparison & Regularisation

### Why standardisation is essential
Regularised models penalise coefficient magnitude. If features are on different
scales, those with larger raw values are penalised more heavily for a given
coefficient, distorting the optimisation. `StandardScaler` centres each feature
to zero mean and unit variance so that the penalty is applied fairly.

### Test-set performance (best models from GridSearchCV)

| Model | RMSE | MAE | Non-zero coefs |
|-------|------|-----|----------------|
{chr(10).join(f"| {row['model']} | {row['rmse']:.4f} | {row['mae']:.4f} | {int(row['nonzero_features'])} |" for _, row in eval_df.iterrows())}

### Coefficient behaviour on correlated features
Because `x1, x2, x3` are nearly collinear, OLS assigns extreme and unstable
coefficients across splits (see boxplot).  Ridge shrinks all coefficients and
stabilises them; Lasso tends to select one or two from the group and zero the
rest.  Elastic Net balances the two behaviours via the `l1_ratio` mixing
parameter.

---

## A4: Forward Selection vs Lasso

**Forward Selection selected features**: {fs_names if fs_names else 'none (stopped early)'}

**Lasso non-zero features**: {lasso_nz_names}

**Comparison**: Forward selection adds features greedily based on OLS CV RMSE;
it tends to include one representative from the correlated group plus the
independent signal (`x6`).  Lasso, driven by the L1 penalty, also selects a
sparse subset but may choose slightly different features because the L1 path
trades off all coefficients simultaneously rather than greedily.

"""
    (RESULTS_DIR / "synthetic_report.md").write_text(report, encoding="utf-8")
    print("[Task A] synthetic_report.md written.")


def write_kaggle_report(eval_df: pd.DataFrame, feature_names: List[str]) -> None:
    """Write kaggle_report.md."""
    ols_row = eval_df[eval_df["model"] == "OLS"].iloc[0]
    ridge_row = eval_df[eval_df["model"] == "Ridge"].iloc[0]
    lasso_row = eval_df[eval_df["model"] == "Lasso"].iloc[0]
    en_row = eval_df[eval_df["model"] == "ElasticNet"].iloc[0]

    report = f"""# Task B — Real-World Dataset Report (Boston Housing)

## Data Source & Business Context

**Dataset**: Boston Housing (originally from the 1978 UCI ML Repository, also
available on Kaggle).  Downloaded from a public GitHub mirror:
`{BOSTON_URL}`.

**Target variable**: `medv` — median value of owner-occupied homes in $1000s.
**Features**: 13 (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO,
B, LSTAT).

**Business goal**: Predict home prices to support valuation, investment, or
policy decisions.

## B1 & B2: Modelling Results

| Model | Test RMSE | Test MAE | Best Params | Zeroed Features |
|-------|-----------|----------|-------------|-----------------|
| OLS   | {ols_row['rmse']:.4f} | {ols_row['mae']:.4f} | N/A | None |
| Ridge | {ridge_row['rmse']:.4f} | {ridge_row['mae']:.4f} | {ridge_row['best_params']} | {ridge_row.get('zeroed_features', [])} |
| Lasso | {lasso_row['rmse']:.4f} | {lasso_row['mae']:.4f} | {lasso_row['best_params']} | {lasso_row['zeroed_features']} |
| ElasticNet | {en_row['rmse']:.4f} | {en_row['mae']:.4f} | {en_row['best_params']} | {en_row.get('zeroed_features', [])} |

## B3: Interpretation

### 1. Did regularisation significantly improve over OLS?
{_interpret_improvement(ols_row, ridge_row, lasso_row)}

### 2. What did Lasso remove, and is it reasonable?
Lasso zeroed: **{lasso_row['zeroed_features']}**.
{_interpret_lasso_zero(lasso_row['zeroed_features'])}

### 3. Top 5 key factors
If a business stakeholder asked for the five most important predictors, I would
use the **Lasso** results (or Elastic Net, if Lasso is too aggressive) because:
- Lasso performs automatic feature selection with a principled CV-tuned
  threshold.
- The non-zero Lasso coefficients offer a parsimonious, interpretable list.
- OLS includes every variable regardless of relevance; forward selection can
  be greedy and unstable.  Lasso's simultaneous shrinkage provides a more
  robust variable-importance ranking.

---
"""
    (RESULTS_DIR / "kaggle_report.md").write_text(report, encoding="utf-8")
    print("[Task B] kaggle_report.md written.")


def _interpret_improvement(
    ols_row: pd.Series,
    ridge_row: pd.Series,
    lasso_row: pd.Series,
) -> str:
    ols_rmse = ols_row["rmse"]
    ridge_rmse = ridge_row["rmse"]
    lasso_rmse = lasso_row["rmse"]
    diff_r = ols_rmse - ridge_rmse
    diff_l = ols_rmse - lasso_rmse

    if diff_r > 0.05:
        msg = (
            f"Ridge reduced RMSE by {diff_r:.3f} over OLS, suggesting that some "
            "over-fitting was present and the L2 penalty helped. "
        )
    elif diff_r < -0.05:
        msg = (
            f"Ridge RMSE was {abs(diff_r):.3f} worse than OLS, indicating that the "
            "bias introduced by the L2 penalty hurt more than the variance reduction "
            "helped. The data may be nearly orthogonal or scarce in true signal. "
        )
    else:
        msg = (
            f"Ridge RMSE differed by only {diff_r:.3f} from OLS, indicating the model "
            "is not severely over-fitting and/or the dataset is large enough for OLS "
            "to be stable. "
        )

    if diff_l > 0.05:
        msg += (
            f"Lasso reduced RMSE by {diff_l:.3f}, likely because it eliminated noisy "
            "predictors and reduced variance."
        )
    elif diff_l < -0.05:
        msg += (
            f"Lasso RMSE was {abs(diff_l):.3f} worse than OLS, suggesting the sparsity "
            "constraint eliminated useful predictors at the CV-optimal alpha."
        )
    else:
        msg += (
            f"Lasso RMSE differed by only {diff_l:.3f} from OLS, so the sparsity "
            "benefit came at little or no cost in predictive accuracy."
        )
    return msg


def _interpret_lasso_zero(zeroed: List[str]) -> str:
    if not zeroed:
        return "No features were zeroed by Lasso."
    # Boston Housing known interpretations
    notes = {
        "INDUS": "proportion of non-retail business acres — may be redundant with other area-level features",
        "AGE": "proportion of owner-occupied units built before 1940 — may correlate with other structural variables",
        "CHAS": "Charles River dummy — infrequently active, so its signal may be weak",
        "ZN": "proportion of large lots — may be collinear with other zoning information",
        "RAD": "index of accessibility to radial highways — may overlap with TAX",
    }
    lines = []
    for f in zeroed:
        note = notes.get(f, f"this feature may be redundant given others in the dataset")
        lines.append(f"  - **{f}**: {note}")
    return (
        "From a business perspective, these removals are plausible:\n"
        + "\n".join(lines)
        + "\n\nIn real-estate pricing, many location-based features are correlated; "
        + "Lasso retains the strongest signal from each cluster."
    )


def write_summary_comparison() -> None:
    """Write summary_comparison.md (Task C)."""
    report = """# Task C — Theory & Practice Summary

## 1. Lasso's coefficient shrinkage and correlated groups

When features are highly correlated, Lasso tends to arbitrarily pick one
feature from the group and zero out the others.  In a business setting, this
can be risky: a manager reviewing the model might conclude that the dropped
features are irrelevant, when in fact they carry near-identical predictive
information.  For example, if both "years of education" and "degree level" are
candidates, Lasso might keep only one, obscuring a deeper behavioural
relationship.

**Elastic Net** mitigates this by mixing an L2 penalty with the L1 penalty.
The L2 component encourages coefficients within a correlated group to be
similar and shared, so Elastic Net is less likely to drop all but one.  The
`l1_ratio` parameter controls the balance, giving practitioners a knob to
trade off sparsity and stability.

---

## 2. GridSearchCV “best” vs subjective goals

`GridSearchCV` selects the hyper-parameters that minimise cross-validated
error (e.g., lowest RMSE).  This is an objective, data-driven criterion.
However, "sparser is better" or "more stable coefficients are better" are
often subjective goals driven by interpretability or deployment constraints.

The grid-search optimum may yield a model with many small non-zero
coefficients, whereas a business requirement for a 3-factor model would prefer
a higher alpha (more regularisation) at a small cost in accuracy.  Similarly,
the most stable model in repeated splits may not coincide with the lowest CV
error.  These trade-offs must be discussed with stakeholders and validated in
the deployment context.

---

## 3. Forward selection / backward elimination vs Lasso

**Computational efficiency**: Forward selection fits OLS repeatedly (O(k·p)
models for k selected features), while Lasso solves a convex optimisation once
over a path of alpha values.  For large p, Lasso is substantially faster
because its coordinate-descent solver scales well and does not require
refitting from scratch for each candidate.

**Final results**: Forward selection produces a hard inclusion/exclusion
decision, while Lasso yields continuous shrinkage.  Because forward selection
uses greedy OLS fits, it can be unstable under collinearity — small data
perturbations may change the feature order.  Lasso's simultaneous
regularisation is more principled for correlated data, but its arbitrary
selection within groups can be misleading.  In practice, I prefer Lasso (or
Elastic Net) for screening, then validate the selected feature set with domain
experts.

---
"""
    (RESULTS_DIR / "summary_comparison.md").write_text(report, encoding="utf-8")
    print("[Task C] summary_comparison.md written.")


# ===================================================================
# Main entry
# ===================================================================
def main() -> None:
    """Orchestrate all tasks."""
    ensure_dirs()
    config_matplotlib()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ------------------------------------------------------------------
    # Task A
    # ------------------------------------------------------------------
    print("\n[Task A1] Generating synthetic correlated data ...")
    df_syn = generate_synthetic_data(noise_sigma=1.0)
    syn_path = DATA_DIR / "synthetic_correlated.csv"
    df_syn.to_csv(syn_path, index=False)
    print(f"        Saved to {syn_path}")

    feature_names_syn = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    X_syn = df_syn[feature_names_syn].values
    y_syn = df_syn["y"].values

    # A3 — stability experiment
    print("[Task A3] Running coefficient stability experiment (50 splits) ...")
    df_coef = run_stability_experiment(df_syn)
    plot_path = plot_coefficient_stability(df_coef)
    print(f"        Saved {plot_path}")

    # Train/test split for GridSearch
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_syn, y_syn, test_size=0.3, random_state=42
    )

    print("[Task A3] GridSearchCV for Ridge, Lasso, ElasticNet ...")
    best_syn, ridge_cv_df, lasso_cv_df = grid_search_task_a(X_tr, y_tr)
    print(
        f"        Ridge best alpha: {best_syn['Ridge'].named_steps['ridge'].alpha:.4f}"
    )
    print(
        f"        Lasso best alpha: {best_syn['Lasso'].named_steps['lasso'].alpha:.4f}"
    )

    # CV curve
    cv_plot_path = plot_cv_error_vs_alpha(ridge_cv_df, lasso_cv_df)
    print(f"        Saved {cv_plot_path}")

    # OLS via pipeline
    pipe_ols_syn = Pipeline(
        [("scaler", StandardScaler()), ("lr", LinearRegression())]
    )
    pipe_ols_syn.fit(X_tr, y_tr)

    all_models_syn = {
        "OLS": pipe_ols_syn,
        "Ridge": best_syn["Ridge"],
        "Lasso": best_syn["Lasso"],
        "ElasticNet": best_syn["ElasticNet"],
    }

    print("[Task A3] Evaluating on test set ...")
    eval_syn_df = evaluate_models(all_models_syn, X_te, y_te, feature_names_syn)
    print(eval_syn_df[["model", "rmse", "mae", "nonzero_features"]].to_string(index=False))

    # A4 — Forward selection
    print("[Task A4] Forward selection (max 5 features, 5-fold CV RMSE) ...")
    selected_fs, fs_rmse = forward_selection_cv(X_tr, y_tr, max_features=5)
    fs_names = [feature_names_syn[i] for i in selected_fs]
    print(f"        Selected: {fs_names}  |  CV RMSE history: {[round(x,4) for x in fs_rmse]}")

    # Lasso non-zero
    lasso_coefs = best_syn["Lasso"].named_steps["lasso"].coef_
    lasso_nz_syn = [feature_names_syn[i] for i, c in enumerate(lasso_coefs) if abs(c) > 1e-6]
    print(f"        Lasso non-zero: {lasso_nz_syn}")

    # Write synthetic report
    print("[Task A] Writing synthetic_report.md ...")
    write_synthetic_report(df_syn, eval_syn_df, selected_fs, feature_names_syn, lasso_nz_syn)

    # ------------------------------------------------------------------
    # Task B
    # ------------------------------------------------------------------
    print("\n[Task B1] Loading Boston Housing dataset ...")
    df_real = download_boston_housing()
    print(f"        Shape: {df_real.shape}")

    print("[Task B2] Running full modelling pipeline ...")
    result_b = task_b_pipeline(df_real)
    eval_b_df = result_b["eval_df"]
    ft_names_b = result_b["feature_names"]
    print(eval_b_df[["model", "rmse", "mae"]].to_string(index=False))

    print("[Task B] Writing kaggle_report.md ...")
    write_kaggle_report(eval_b_df, ft_names_b)

    # ------------------------------------------------------------------
    # Task C
    # ------------------------------------------------------------------
    print("\n[Task C] Writing summary_comparison.md ...")
    write_summary_comparison()

    print("\n" + "=" * 60)
    print("All tasks complete. Output files:")
    print(f"  {syn_path}")
    print(f"  {BOSTON_PATH}")
    print(f"  {RESULTS_DIR / 'synthetic_report.md'}")
    print(f"  {RESULTS_DIR / 'kaggle_report.md'}")
    print(f"  {RESULTS_DIR / 'summary_comparison.md'}")
    print(f"  {FIGURES_DIR / 'coefficient_stability_boxplot.png'}")
    print(f"  {FIGURES_DIR / 'cv_error_vs_alpha.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()