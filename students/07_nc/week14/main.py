#!/usr/bin/env python3
"""第 14 周作业：高维回归、PCA、PCR，以及变量选择与信息压缩。

在 students/07_nc 目录下运行：
    uv run week14/main.py

脚本设置为可复现：每次运行都会重新生成模拟数据、图像和 Markdown 报告。
"""
from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import KFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.metrics import calculate_mae, calculate_rmse  # noqa: E402
from utils.models import (  # noqa: E402
    PCRRegressor,
    design_rank_and_condition,
    pcr_cv_rmse,
    prediction_stability_score,
)
from utils.transformers import CustomStandardScaler  # noqa: E402

WEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK_DIR / "data"
RESULTS_DIR = WEEK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42


@dataclass
class ModelResult:
    model: str
    rmse: float
    mae: float
    complexity_name: str
    complexity: float
    stability: float | None = None


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------


def reset_outputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def feature_names(p: int) -> list[str]:
    return [f"x{j + 1:03d}" for j in range(p)]


def savefig(name: str) -> Path:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """把较小的 DataFrame 渲染成 GitHub 风格 Markdown 表格，不依赖 tabulate。"""
    if df.empty:
        return "(empty table)"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "nan" if pd.isna(x) else format(float(x), floatfmt))
        else:
            display[col] = display[col].astype(str)
    headers = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, CustomStandardScaler]:
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def pca_components_from_scaled(X_scaled: np.ndarray, max_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """基于已标准化的 X 快速计算 PCA 载荷和解释方差比例。"""
    X_centered = X_scaled - np.mean(X_scaled, axis=0)
    _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (singular_values**2) / max(1, X_scaled.shape[0] - 1)
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    if max_k is None:
        max_k = vt.shape[0]
    return vt[:max_k].T, explained[:max_k]


def fit_pcr_np_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """拟合“标准化 -> PCA(k) -> OLS”流程，并对 X_eval 做预测。

    返回预测值，以及主成分空间里的回归系数。
    """
    scaler = CustomStandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    components, _ = pca_components_from_scaled(X_train_scaled, max_k=k)
    Z_train = X_train_scaled @ components
    Z_eval = X_eval_scaled @ components
    Z_design = np.column_stack([np.ones(Z_train.shape[0]), Z_train])
    coef = np.linalg.lstsq(Z_design, y_train, rcond=1e-8)[0]
    pred = np.column_stack([np.ones(Z_eval.shape[0]), Z_eval]) @ coef
    return pred, coef


def fit_ols_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """使用 numpy lstsq 快速拟合带截距的 OLS。

    这样可以避免在 p >= n 的实验中反复调用 sklearn 的 SVD 导致速度变慢，
    同时仍然计算普通最小二乘的最小范数解。
    """
    X_train_scaled, X_test_scaled, _ = standardize_train_test(X_train, X_test)
    X_train_design = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
    X_test_design = np.column_stack([np.ones(X_test_scaled.shape[0]), X_test_scaled])
    coef_full = np.linalg.lstsq(X_train_design, y_train, rcond=1e-8)[0]
    train_pred = X_train_design @ coef_full
    test_pred = X_test_design @ coef_full
    sse = np.sum((y_test - test_pred) ** 2)
    sst = np.sum((y_test - np.mean(y_test)) ** 2)
    test_r2 = 0.0 if np.isclose(sst, 0.0) else 1.0 - sse / sst
    return (
        calculate_rmse(y_train, train_pred),
        calculate_rmse(y_test, test_pred),
        calculate_mae(y_test, test_pred),
        float(test_r2),
        coef_full[1:].copy(),
    )


# ---------------------------------------------------------------------------
# 任务 A/B：高维低秩潜因子模拟数据
# ---------------------------------------------------------------------------


def make_high_dimensional_data(
    n_samples: int = 160,
    n_features: int = 80,
    n_factors: int = 6,
    noise_x: float = 0.18,
    noise_y: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从低秩潜因子模型生成高维模拟数据。"""
    rng = np.random.default_rng(random_state)
    factors = rng.normal(size=(n_samples, n_factors))
    loadings = rng.normal(size=(n_factors, n_features))

    # 让前几个原始特征明显受第一个潜因子影响，便于解释系数不稳定图。
    loadings[:, 0] = np.array([2.2, 0.2, 0.0, 0.0, 0.0, 0.1])
    loadings[:, 1] = np.array([2.0, 0.3, 0.0, 0.0, 0.0, -0.1])
    loadings[:, 2] = np.array([1.8, 0.1, 0.2, 0.0, 0.0, 0.0])

    X = factors @ loadings + rng.normal(scale=noise_x, size=(n_samples, n_features))
    beta_factors = np.array([3.0, -2.1, 0.0, 1.6, 0.0, 0.9])
    y = 15.0 + factors @ beta_factors + rng.normal(scale=noise_y, size=n_samples)

    df = pd.DataFrame(X, columns=feature_names(n_features))
    df.insert(0, "y", y)
    factor_df = pd.DataFrame(factors, columns=[f"latent_{i + 1}" for i in range(n_factors)])
    return df, factor_df


def run_ols_dimension_experiment() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for p in [10, 30, 60, 120, 160]:
        df, _ = make_high_dimensional_data(n_samples=140, n_features=p, random_state=100 + p)
        X = df.drop(columns=["y"]).to_numpy()
        y = df["y"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE
        )
        train_rmse, test_rmse, test_mae, test_r2, _ = fit_ols_evaluate(X_train, X_test, y_train, y_test)
        X_train_scaled, _, _ = standardize_train_test(X_train, X_test)
        rank, cond = design_rank_and_condition(X_train_scaled)
        rows.append(
            {
                "p": p,
                "n_train": X_train.shape[0],
                "rank_X_train": rank,
                "condition_number": cond,
                "train_RMSE": train_rmse,
                "test_RMSE": test_rmse,
                "test_MAE": test_mae,
                "test_R2": test_r2,
            }
        )
    return pd.DataFrame(rows)


def plot_ols_dimension_results(dimension_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.6))
    plt.plot(dimension_df["p"], dimension_df["train_RMSE"], marker="o", label="OLS train RMSE")
    plt.plot(dimension_df["p"], dimension_df["test_RMSE"], marker="o", label="OLS test RMSE")
    plt.xlabel("Number of raw features p")
    plt.ylabel("RMSE")
    plt.title("OLS error as feature dimension increases")
    plt.legend()
    savefig("ols_error_vs_p.png")

    cond_plot = dimension_df.copy()
    finite = cond_plot["condition_number"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_max = finite.max() if not finite.empty else 1.0
    cond_plot["condition_for_plot"] = cond_plot["condition_number"].replace(
        [np.inf, -np.inf], finite_max * 10.0
    )

    plt.figure(figsize=(7, 4.6))
    plt.plot(cond_plot["p"], cond_plot["rank_X_train"], marker="o", label="rank(X_train)")
    plt.plot(cond_plot["p"], cond_plot["condition_for_plot"], marker="s", label="condition number (inf capped)")
    plt.yscale("log")
    plt.xlabel("Number of raw features p")
    plt.ylabel("Rank / condition number (log scale)")
    plt.title("Matrix structure becomes ill-conditioned as p grows")
    plt.legend()
    savefig("matrix_structure_vs_p.png")


def run_ols_stability_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()
    watched = ["x001", "x002", "x003"]
    watched_idx = [df.drop(columns=["y"]).columns.get_loc(col) for col in watched]
    rows: list[dict[str, float | str | int]] = []

    for seed in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
        train_rmse, test_rmse, _, _, coef = fit_ols_evaluate(X_train, X_test, y_train, y_test)
        for col, idx in zip(watched, watched_idx):
            rows.append(
                {
                    "split": seed,
                    "feature": col,
                    "coef": coef[idx],
                    "train_RMSE": train_rmse,
                    "test_RMSE": test_rmse,
                }
            )

    coef_df = pd.DataFrame(rows)
    summary = (
        coef_df.groupby("feature")
        .agg(coef_mean=("coef", "mean"), coef_std=("coef", "std"), coef_min=("coef", "min"), coef_max=("coef", "max"))
        .reset_index()
    )

    plt.figure(figsize=(7, 4.6))
    data = [coef_df.loc[coef_df["feature"] == col, "coef"].to_numpy() for col in watched]
    plt.boxplot(data, tick_labels=watched)
    plt.xlabel("Watched raw feature")
    plt.ylabel("OLS coefficient across 50 splits")
    plt.title("OLS coefficient instability for correlated high-dimensional features")
    savefig("ols_coefficient_instability.png")

    return coef_df, summary


def fast_pcr_cv_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_k: int,
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """每个交叉验证折只做一次 numpy SVD，计算 k=1..max_k 的 PCR CV RMSE。"""
    scores_by_k: list[list[float]] = [[] for _ in range(max_k)]
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        scaler = CustomStandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        components, _ = pca_components_from_scaled(X_tr_scaled, max_k=max_k)
        Z_tr = X_tr_scaled @ components
        Z_val = X_val_scaled @ components
        for k in range(1, max_k + 1):
            Z_design = np.column_stack([np.ones(Z_tr.shape[0]), Z_tr[:, :k]])
            coef = np.linalg.lstsq(Z_design, y_tr, rcond=1e-8)[0]
            pred = np.column_stack([np.ones(Z_val.shape[0]), Z_val[:, :k]]) @ coef
            scores_by_k[k - 1].append(calculate_rmse(y_val, pred))
    return np.array([np.mean(values) for values in scores_by_k])


def run_pca_pcr(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    _, explained_ratio = pca_components_from_scaled(X_scaled)
    cumulative = np.cumsum(explained_ratio)
    pca_df = pd.DataFrame(
        {
            "n_components": np.arange(1, len(cumulative) + 1),
            "cumulative_explained_variance": cumulative,
        }
    )

    plt.figure(figsize=(7, 4.6))
    plt.plot(pca_df["n_components"], pca_df["cumulative_explained_variance"], marker="o", markersize=3)
    plt.axhline(0.90, linestyle="--", label="90% variance")
    plt.axhline(0.95, linestyle="--", label="95% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA cumulative explained variance on synthetic high-dimensional data")
    plt.legend()
    savefig("pca_cumulative_variance.png")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    ols_train_rmse, ols_test_rmse, _, _, _ = fit_ols_evaluate(X_train, X_test, y_train, y_test)

    max_k = 20
    scaler_train = CustomStandardScaler().fit(X_train)
    X_train_scaled = scaler_train.transform(X_train)
    X_test_scaled = scaler_train.transform(X_test)
    components, _ = pca_components_from_scaled(X_train_scaled, max_k=max_k)
    Z_train = X_train_scaled @ components
    Z_test = X_test_scaled @ components
    cv_scores = fast_pcr_cv_scores(X_train, y_train, max_k=max_k, cv=5, random_state=RANDOM_STATE)

    rows: list[dict[str, float | int]] = []
    for k in range(1, max_k + 1):
        coef = np.linalg.lstsq(np.column_stack([np.ones(Z_train.shape[0]), Z_train[:, :k]]), y_train, rcond=1e-8)[0]
        train_pred = np.column_stack([np.ones(Z_train.shape[0]), Z_train[:, :k]]) @ coef
        test_pred = np.column_stack([np.ones(Z_test.shape[0]), Z_test[:, :k]]) @ coef
        rows.append(
            {
                "k": k,
                "train_RMSE": calculate_rmse(y_train, train_pred),
                "test_RMSE": calculate_rmse(y_test, test_pred),
                "cv_RMSE": float(cv_scores[k - 1]),
                "OLS_train_RMSE": ols_train_rmse,
                "OLS_test_RMSE": ols_test_rmse,
            }
        )
    pcr_df = pd.DataFrame(rows)
    best_k = int(pcr_df.loc[pcr_df["cv_RMSE"].idxmin(), "k"])

    plt.figure(figsize=(7, 4.6))
    plt.plot(pcr_df["k"], pcr_df["train_RMSE"], marker="o", label="PCR train RMSE")
    plt.plot(pcr_df["k"], pcr_df["test_RMSE"], marker="o", label="PCR test RMSE")
    plt.plot(pcr_df["k"], pcr_df["cv_RMSE"], marker="o", label="PCR 5-fold CV RMSE")
    plt.axhline(ols_test_rmse, linestyle="--", label="OLS test RMSE baseline")
    plt.axvline(best_k, linestyle=":", label=f"best CV k={best_k}")
    plt.xlabel("Number of retained principal components k")
    plt.ylabel("RMSE")
    plt.title("PCR error curves as k increases")
    plt.legend()
    savefig("pcr_rmse_by_k.png")

    return pca_df, best_k, pcr_df


# ---------------------------------------------------------------------------
# 任务 C：稀疏真实机制和潜因子真实机制下的 Lasso 与 PCR 对比
# ---------------------------------------------------------------------------


def make_sparse_truth_data(n_samples: int = 180, n_features: int = 40, random_state: int = 707) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[[0, 3, 20, 30]] = [3.2, -2.5, 1.8, 2.2]
    y = 10.0 + X @ beta + rng.normal(scale=1.4, size=n_samples)
    return X, y, feature_names(n_features)


def make_latent_truth_data(n_samples: int = 180, n_features: int = 40, n_factors: int = 5, random_state: int = 808) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(random_state)
    factors = rng.normal(size=(n_samples, n_factors))
    loadings = rng.normal(size=(n_factors, n_features))
    X = factors @ loadings + rng.normal(scale=0.2, size=(n_samples, n_features))
    y = 5.0 + factors @ np.array([2.8, -1.9, 1.4, 0.0, 1.1]) + rng.normal(scale=1.2, size=n_samples)
    return X, y, feature_names(n_features)


def fit_lasso_train_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> ModelResult:
    X_train_scaled, X_test_scaled, _ = standardize_train_test(X_train, X_test)
    lasso = LassoCV(cv=4, random_state=RANDOM_STATE, max_iter=12000, alphas=40)
    lasso.fit(X_train_scaled, y_train)
    pred = lasso.predict(X_test_scaled)
    nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-8))
    result = ModelResult(
        model="LassoCV",
        rmse=calculate_rmse(y_test, pred),
        mae=calculate_mae(y_test, pred),
        complexity_name="nonzero_coefficients",
        complexity=float(nonzero),
    )
    # 为第 14 周实验保留简单的动态属性：后续稳定性分析可以复用 CV 选出的 alpha，
    # 避免多次重复运行开销较大的 LassoCV。
    result.alpha_ = float(lasso.alpha_)  # type: ignore[attr-defined]
    result.coef_ = lasso.coef_.copy()  # type: ignore[attr-defined]
    return result


def fit_best_pcr_train_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, max_k: int = 20) -> tuple[ModelResult, pd.DataFrame]:
    max_k = min(max_k, X_train.shape[1], X_train.shape[0] - 1)
    cv_scores = fast_pcr_cv_scores(X_train, y_train, max_k=max_k, cv=5, random_state=RANDOM_STATE)
    cv_df = pd.DataFrame({"k": np.arange(1, max_k + 1), "cv_RMSE": cv_scores})
    best_k = int(cv_df.loc[cv_df["cv_RMSE"].idxmin(), "k"])
    pred, _ = fit_pcr_np_predict(X_train, y_train, X_test, best_k)
    return (
        ModelResult(
            model="PCR",
            rmse=calculate_rmse(y_test, pred),
            mae=calculate_mae(y_test, pred),
            complexity_name="retained_components",
            complexity=float(best_k),
        ),
        cv_df,
    )


def repeated_prediction_stability(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    lasso_alpha: float | None = None,
    pcr_k: int | None = None,
    repeats: int = 4,
    anchor_size: int = 30,
) -> float:
    """在固定模型复杂度下快速计算预测稳定性得分。

    先在普通 train/test 实验中选出 alpha 或 k，然后在不同训练样本上反复拟合同一类模型。
    这样可以衡量稳定性，同时避免非常慢的嵌套交叉验证。
    """
    rng = np.random.default_rng(999)
    anchor_idx = rng.choice(X.shape[0], size=anchor_size, replace=False)
    X_anchor = X[anchor_idx]
    predictions: list[np.ndarray] = []
    all_idx = np.arange(X.shape[0])

    for r in range(repeats):
        train_pool = np.setdiff1d(all_idx, anchor_idx)
        train_idx = rng.choice(train_pool, size=int(0.70 * len(train_pool)), replace=False)
        X_train, y_train = X[train_idx], y[train_idx]

        if method == "lasso":
            scaler = CustomStandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_anchor_scaled = scaler.transform(X_anchor)
            model = Lasso(alpha=0.01 if lasso_alpha is None else lasso_alpha, max_iter=8000)
            model.fit(X_train_scaled, y_train)
            predictions.append(model.predict(X_anchor_scaled))
        elif method == "pcr":
            k = int(pcr_k if pcr_k is not None else min(10, X.shape[1]))
            pred, _ = fit_pcr_np_predict(X_train, y_train, X_anchor, k)
            predictions.append(pred)
        else:
            raise ValueError("method must be 'lasso' or 'pcr'")

    return prediction_stability_score(np.vstack(predictions))

def selected_design_condition(X_train: np.ndarray, coef: np.ndarray) -> float:
    """计算 Lasso 非零系数所选特征子矩阵的条件数。"""
    scaler = CustomStandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    selected = np.where(np.abs(coef) > 1e-8)[0]
    if selected.size == 0:
        return float("inf")
    _, cond = design_rank_and_condition(X_scaled[:, selected])
    return cond


def pcr_component_condition(X_train: np.ndarray, k: int) -> float:
    """计算 PCR 保留主成分得分矩阵的条件数。"""
    scaler = CustomStandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    components, _ = pca_components_from_scaled(X_scaled, max_k=k)
    Z = X_scaled @ components
    _, cond = design_rank_and_condition(Z)
    return cond


def run_lasso_vs_pcr_scenarios() -> pd.DataFrame:
    scenario_rows: list[dict[str, float | str]] = []
    scenarios = [
        ("Sparse truth", make_sparse_truth_data, 8),
        ("Latent-factor truth", make_latent_truth_data, 5),
    ]
    for scenario_name, maker, pcr_k in scenarios:
        X, y, _ = maker()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE
        )

        lasso_result = fit_lasso_train_test(X_train, y_train, X_test, y_test)
        pcr_pred, _ = fit_pcr_np_predict(X_train, y_train, X_test, pcr_k)
        pcr_result = ModelResult(
            model="PCR",
            rmse=calculate_rmse(y_test, pcr_pred),
            mae=calculate_mae(y_test, pcr_pred),
            complexity_name="retained_components",
            complexity=float(pcr_k),
        )

        lasso_result.stability = selected_design_condition(X_train, getattr(lasso_result, "coef_"))
        pcr_result.stability = pcr_component_condition(X_train, pcr_k)

        for result in [lasso_result, pcr_result]:
            scenario_rows.append(
                {
                    "scenario": scenario_name,
                    "method": result.model,
                    "test_RMSE": result.rmse,
                    "test_MAE": result.mae,
                    "complexity_metric": result.complexity_name,
                    "complexity_value": result.complexity,
                    "stability_metric_condition_number": result.stability,
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, scenario in zip(axes, scenario_df["scenario"].unique()):
        subset = scenario_df[scenario_df["scenario"] == scenario]
        ax.bar(subset["method"], subset["test_RMSE"])
        ax.set_title(scenario)
        ax.set_ylabel("Test RMSE")
        ax.set_xlabel("Method")
    fig.suptitle("Lasso selection vs PCR compression under two data-generating worlds")
    savefig("lasso_vs_pcr_scenarios.png")

    return scenario_df


# ---------------------------------------------------------------------------
# 选做任务 D：Kaggle House Prices 高维回归实验
# ---------------------------------------------------------------------------


def load_kaggle_house_prices() -> pd.DataFrame | None:
    path = DATA_DIR / "kaggle_house_prices.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def prepare_house_prices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "SalePrice" not in df.columns:
        raise ValueError("kaggle_house_prices.csv must contain SalePrice")
    numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [col for col in ["Id", "SalePrice"] if col in numeric.columns]
    X_df = numeric.drop(columns=drop_cols)
    y = np.log1p(numeric["SalePrice"].to_numpy(dtype=float))
    names = list(X_df.columns)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df)
    return X, y, names


def run_kaggle_optional() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    df = load_kaggle_house_prices()
    if df is None:
        return None, None
    X, y, names = prepare_house_prices(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)

    ols_train_rmse, ols_test_rmse, ols_mae, _, ols_coef = fit_ols_evaluate(X_train, X_test, y_train, y_test)
    lasso_result = fit_lasso_train_test(X_train, y_train, X_test, y_test)
    pcr_result, pcr_cv_df = fit_best_pcr_train_test(X_train, y_train, X_test, y_test, max_k=min(30, X.shape[1]))

    results = pd.DataFrame(
        [
            {
                "model": "OLS",
                "test_RMSE": ols_test_rmse,
                "test_MAE": ols_mae,
                "complexity_metric": "numeric_coefficients",
                "complexity_value": X.shape[1],
            },
            {
                "model": "LassoCV",
                "test_RMSE": lasso_result.rmse,
                "test_MAE": lasso_result.mae,
                "complexity_metric": lasso_result.complexity_name,
                "complexity_value": lasso_result.complexity,
            },
            {
                "model": "PCR",
                "test_RMSE": pcr_result.rmse,
                "test_MAE": pcr_result.mae,
                "complexity_metric": pcr_result.complexity_name,
                "complexity_value": pcr_result.complexity,
            },
        ]
    )

    plt.figure(figsize=(7, 4.6))
    plt.bar(results["model"], results["test_RMSE"])
    plt.xlabel("Model")
    plt.ylabel("Test RMSE on log1p(SalePrice)")
    plt.title("Kaggle House Prices: OLS vs Lasso vs PCR")
    savefig("kaggle_model_rmse.png")

    plt.figure(figsize=(7, 4.6))
    plt.plot(pcr_cv_df["k"], pcr_cv_df["cv_RMSE"], marker="o")
    plt.xlabel("Number of principal components k")
    plt.ylabel("PCR 5-fold CV RMSE")
    plt.title("Kaggle House Prices: PCR component selection")
    savefig("kaggle_pcr_cv_curve.png")

    # 报告用相关性快照：数值特征中绝对相关系数最高的若干对。
    numeric_corr = pd.DataFrame(X, columns=names).corr().abs()
    rows = []
    for i, col_a in enumerate(names):
        for col_b in names[i + 1 :]:
            rows.append({"feature_1": col_a, "feature_2": col_b, "abs_corr": numeric_corr.loc[col_a, col_b]})
    corr_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).head(8)

    # 保留 OLS 系数表，便于在报告中讨论解释稳定性。
    top_ols = (
        pd.DataFrame({"feature": names, "OLS_standardized_coef": ols_coef})
        .assign(abs_coef=lambda d: d["OLS_standardized_coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(10)
        .drop(columns="abs_coef")
    )
    extra = pd.concat(
        {
            "top_numeric_correlations": corr_df.reset_index(drop=True),
            "top_ols_coefficients": top_ols.reset_index(drop=True),
        },
        names=["section", "row"],
    )
    return results, extra



# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def write_synthetic_report(
    highdim_df: pd.DataFrame,
    dimension_df: pd.DataFrame,
    coef_summary: pd.DataFrame,
    pca_df: pd.DataFrame,
    best_k: int,
    pcr_df: pd.DataFrame,
) -> None:
    """写出中文模拟数据报告：高维 OLS、PCA 和 PCR。"""
    first90 = int(pca_df.loc[pca_df["cumulative_explained_variance"] >= 0.90, "n_components"].iloc[0])
    first95 = int(pca_df.loc[pca_df["cumulative_explained_variance"] >= 0.95, "n_components"].iloc[0])
    best_row = pcr_df.loc[pcr_df["cv_RMSE"].idxmin()]

    condition_display = dimension_df.copy()
    condition_display["condition_number"] = condition_display["condition_number"].map(
        lambda x: "inf" if math.isinf(float(x)) else f"{float(x):.2e}"
    )

    text = f"""# 第 14 周模拟数据报告：高维回归、PCA 与 PCR

## 1. 数据生成机制

本次自己生成的高维模拟数据保存在：

```text
week14/data/synthetic_highdim.csv
```

保存后的数据共有 **{highdim_df.shape[0]} 行**，包含 **{highdim_df.shape[1] - 1} 个原始特征**和目标列 `y`。
数据来自一个低秩潜因子模型：

```text
F in R^(n x 6)
X = F L + E_x
Y = 15 + 3.0 F_1 - 2.1 F_2 + 1.6 F_4 + 0.9 F_6 + E_y
```

也就是说，表面上 `X` 有很多列，但主要信息实际上由 6 个潜在方向驱动。很多原始变量只是同一批潜在因子的不同噪声投影，因此数据同时具有两个特点：**维度高**、**信息冗余强**。这正好适合观察 OLS 在高维和共线性环境中的不稳定性，也适合展示 PCA/PCR 的压缩思想。

## 2. 任务 A3：OLS 与“训练误差很低”的风险

我固定样本量，然后逐步增加原始特征数量 `p`。对每个 `p`，都划分训练集和测试集，拟合 OLS，并记录训练 RMSE、测试 RMSE、`rank(X_train)` 和条件数。

{markdown_table(condition_display)}

图像：`week14/results/figures/ols_error_vs_p.png`

- **横轴**：原始特征数量 `p`。
- **纵轴**：RMSE。
- **曲线**：OLS 训练 RMSE 与 OLS 测试 RMSE。
- **结论**：当 `p` 接近甚至超过训练样本数时，OLS 可以把训练误差压得很低，但这并不代表模型真正学到了稳定规律。它可能只是利用了很多不稳定方向来贴合训练数据。

图像：`week14/results/figures/matrix_structure_vs_p.png`

- **横轴**：原始特征数量 `p`。
- **纵轴**：设计矩阵的秩和条件数，使用对数尺度。
- **曲线**：`rank(X_train)` 与 condition number。
- **结论**：当 `p` 变大时，设计矩阵会变得秩不足或病态。此时很多组系数都能差不多解释训练数据，所以 OLS 的系数会变得不唯一或非常不稳定。

## 3. 任务 A4：重复切分下的系数不稳定

我固定一份高维模拟数据，然后重复 50 次 train/test split。每次都拟合 OLS，并记录三个相关原始变量 `x001`、`x002`、`x003` 的标准化系数。

{markdown_table(coef_summary)}

图像：`week14/results/figures/ols_coefficient_instability.png`

- **横轴**：被观察的特征名。
- **纵轴**：50 次切分下的 OLS 系数。
- **箱线图**：同一个变量在不同训练集划分下的系数分布。
- **结论**：这些系数在不同切分下波动明显。风险不只是预测误差变化，更重要的是“哪个变量重要”的解释也会随训练样本变化。因此，系数不稳定本身就是高维回归中的建模风险。

## 4. 任务 B1：PCA 与低维信息子空间

图像：`week14/results/figures/pca_cumulative_variance.png`

- **横轴**：主成分数量。
- **纵轴**：累计解释方差比例。
- **曲线**：保留前 `k` 个主成分后解释了多少标准化特征方差。

结果显示，前 **{first90}** 个主成分已经能解释至少 90% 的方差，前 **{first95}** 个主成分能解释至少 95% 的方差。这说明原始高维特征空间其实接近一个更低维的潜在子空间。

## 5. 任务 B2/B3：PCR 流程和 k 的选择

我的 PCR 流程是：

```text
标准化 X -> PCA -> 保留前 k 个主成分 -> 在主成分得分 Z_k 上做线性回归
```

我比较了 `k = 1, 2, ..., 20`。最终选择 5 折交叉验证 RMSE 最低的 `k`。

交叉验证选出的最佳 PCR：

```text
best k = {best_k}
CV RMSE = {best_row['cv_RMSE']:.4f}
train RMSE = {best_row['train_RMSE']:.4f}
test RMSE = {best_row['test_RMSE']:.4f}
```

图像：`week14/results/figures/pcr_rmse_by_k.png`

- **横轴**：保留的主成分数量 `k`。
- **纵轴**：RMSE。
- **曲线**：PCR 训练 RMSE、PCR 测试 RMSE、PCR 五折 CV RMSE。
- **虚线基准**：原始高维特征空间里的 OLS 测试 RMSE。
- **竖线**：交叉验证选择的最佳 `k`。

这里的 `PCR CV RMSE` 是在每个交叉验证折内部完整拟合标准化、PCA 和线性回归流程后得到的平均验证误差。这样可以避免数据泄露，因为标准化和 PCA 都只在训练折上重新拟合。

OLS 在原始高维空间中可能获得很低的训练 RMSE，因为它可以贴合不稳定方向。PCR 则故意把原始特征压缩成较少的主成分后再回归，牺牲一部分训练拟合能力来换取更稳定的预测和解释。

## 6. 必要概念和公式

### OLS 估计量

如果设计矩阵满秩，OLS 估计量为：

```text
beta_hat_OLS = (X^T X)^(-1) X^T y
```

当 `X^T X` 奇异或接近奇异时，这个估计会不稳定或不唯一。这正是高维数据和强共线性数据中的核心风险。

### 第一主成分

第一主成分方向是让投影方差最大的单位向量：

```text
v_1 = argmax_{{||v||=1}} Var(X v)
```

它表示在特征空间里，能够捕捉最大方差信息的方向。

### PCR 的符号表达

设 `V_k` 是前 `k` 个 PCA loading 向量。PCR 先把标准化后的原始特征投影到主成分得分空间：

```text
Z_k = X V_k
```

然后在压缩后的空间中拟合线性回归：

```text
y = Z_k gamma + epsilon
```

因此，PCR 回答的问题和变量选择不同。它不是问“哪些原始列应该留下”，而是问“哪些低维信息方向应该留下”。
"""
    (RESULTS_DIR / "synthetic_report.md").write_text(text, encoding="utf-8")


def write_summary_report(scenario_df: pd.DataFrame) -> None:
    """写出中文总结报告：Lasso 变量选择 vs PCR 信息压缩。"""
    pivot = scenario_df.copy()
    text = f"""# 第 14 周总结对比：Lasso 变量选择 vs PCR 信息压缩

## 1. 实验对比

我在两种不同的数据生成机制下比较了 Lasso 和 PCR。

{markdown_table(pivot)}

图像：`week14/results/figures/lasso_vs_pcr_scenarios.png`

- **横轴**：每个场景下的方法，包含 `LassoCV` 和 `PCR`。
- **纵轴**：测试集 RMSE。
- **两个子图**：稀疏真实机制和潜因子真实机制。
- **主要结论**：当真实关系由少数几个原始变量决定时，Lasso 更自然；当很多原始变量都是少数潜在因子的噪声投影时，PCR 更自然。

这里的稳定性指标是我自己设计的一个代理指标：Lasso 选中特征子矩阵的条件数，或者 PCR 保留主成分得分矩阵的条件数。数值越低，说明拟合设计矩阵越不病态，系数估计越稳定。

## 2. 如果真实机制是稀疏的，为什么 Lasso 更自然？

在稀疏真实机制中，只有少数几个原始变量直接决定 `y`。Lasso 的 L1 惩罚正好适合这种场景，因为它可以把无关变量的系数压缩为 0。这样得到的模型变量列表更短，也更容易用原始变量向业务方解释。

## 3. 如果真实机制是潜因子驱动的，为什么 PCR 更自然？

在潜因子场景中，原始变量本身并不是彼此独立且清晰有意义的变量，而是少数隐藏因子的许多噪声测量。PCR 更适合这种情况，因为它把冗余变量压缩成主成分方向。它不强迫模型从高度相关的一组变量中挑一个，而是保留这一组变量共享的信息方向。

## 4. 两类方法分别回答什么问题？

- **Lasso** 回答的是变量选择问题：哪些原始变量应该保留？
- **PCR** 回答的是信息压缩问题：哪些低维信息方向应该保留？

这是 Week14 的核心区别。Lasso 输出的是更短的原始变量列表；PCR 输出的是更少的转换后主成分。

## 5. 如果业务方想要更短的变量列表，我会选什么？

我通常会优先考虑 Lasso 或其他变量选择方法，因为输出仍然是原始变量。解释时可以直接说：哪些列被保留、哪些列被剔除。

## 6. 如果业务方想要更稳定的预测器，我会选什么？

如果数据看起来像潜因子数据，或者存在很强多重共线性，我会考虑 PCR 或 Ridge 这类压缩/收缩方法。特别是当很多列都在测量同一个潜在概念，而单个原始变量系数并不可靠时，PCR 会更有吸引力。

## 7. 为什么这周不把前向/后向选择放在中心？

前向选择和后向剔除都属于 **selection** 家族，它们决定哪些原始变量进入模型。Week14 的重点是更广的概念对比：selection vs compression。Lasso 已经代表了现代正则化变量选择路线，而 PCR 代表压缩路线。如果把前向/后向选择也放到中心，它和 Lasso 更接近，并不能像 PCR 那样清楚展示“压缩”的思想。
"""
    (RESULTS_DIR / "summary_comparison.md").write_text(text, encoding="utf-8")


def write_kaggle_report(kaggle_results: pd.DataFrame | None, kaggle_extra: pd.DataFrame | None) -> None:
    """写出中文 Kaggle 选做报告。"""
    if kaggle_results is None:
        text = "# 第 14 周 Kaggle 报告\n\n没有找到可选 Kaggle 数据，因此本部分跳过。\n"
    else:
        # 将 MultiIndex 的补充结果拆回可读表格。
        corr_df = kaggle_extra.loc["top_numeric_correlations"].reset_index(drop=True) if kaggle_extra is not None else pd.DataFrame()
        top_ols = kaggle_extra.loc["top_ols_coefficients"].reset_index(drop=True) if kaggle_extra is not None else pd.DataFrame()
        if not corr_df.empty:
            corr_df = corr_df[["feature_1", "feature_2", "abs_corr"]]
        if not top_ols.empty:
            top_ols = top_ols[["feature", "OLS_standardized_coef"]]
        best = kaggle_results.loc[kaggle_results["test_RMSE"].idxmin()]
        text = f"""# 第 14 周 Kaggle 报告：House Prices 房价数据

## 1. 数据集和目标变量

选做真实数据实验使用的是：

```text
week14/data/kaggle_house_prices.csv
```

该数据来自 House Prices 回归任务，目标变量是 `SalePrice`。我对目标变量做了 `log1p(SalePrice)` 变换，用来减弱房价右偏分布的影响。

这一版作业只使用数值型特征。即使如此，这份数据仍然适合 Week14，因为房屋属性天然存在相关性：居住面积、地下室面积、车库容量、房间数、建造年份和质量评分往往一起变化。

## 2. 模型对比

{markdown_table(kaggle_results)}

本次运行中，测试集 RMSE 最低的模型是 **{best['model']}**，其在 `log1p(SalePrice)` 上的 RMSE 为 **{best['test_RMSE']:.4f}**。

图像：`week14/results/figures/kaggle_model_rmse.png`

- **横轴**：模型名称。
- **纵轴**：`log1p(SalePrice)` 上的测试集 RMSE。
- **柱子**：OLS、LassoCV 和 PCR。
- **结论**：真实数据中不一定有某一个方法压倒性胜出，因此解释时不能只看误差，还要考虑模型稳定性和复杂度。

图像：`week14/results/figures/kaggle_pcr_cv_curve.png`

- **横轴**：保留的主成分数量 `k`。
- **纵轴**：PCR 五折交叉验证 RMSE。
- **曲线**：随着压缩程度减弱，PCR 验证误差如何变化。
- **结论**：这条曲线帮助判断保留多少个主成分已经足够，避免盲目保留所有高维原始变量。

## 3. 高维与共线性的证据

数值特征中绝对相关系数最高的若干对是：

{markdown_table(corr_df)}

这些高相关在房价数据中是合理的，因为很多变量都在描述相近概念，例如总面积、房间数量、地下室面积和车库面积。

标准化后 OLS 系数绝对值较大的变量是：

{markdown_table(top_ols)}

OLS 在这里并没有完全失败，但解释时需要谨慎。原因是房地产特征之间经常相关，一个变量的系数可能取决于模型中是否同时包含了其他相近变量。

## 4. Lasso 和 PCR 哪个更合适？

对于这份只使用数值变量的房价数据，数据结构既有稀疏信号，也有潜因子结构。一些变量本身就很有业务含义，例如 `OverallQual` 和 `GrLivArea`，这支持 Lasso 式变量选择；同时，很多变量又共同反映房屋规模、质量和年代等潜在概念，这也支持 PCR 式信息压缩。

如果业务方希望得到一个短变量清单，我会先使用 Lasso；如果目标是构建在相关特征下更稳定的预测基线，我会同时比较 PCR 和 Ridge 类模型。
"""
    (RESULTS_DIR / "kaggle_report.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------


def main() -> None:
    """第 14 周唯一执行入口：生成数据、图像和中文报告。"""
    print("[Week14] 重置输出目录", flush=True)
    reset_outputs()

    # reset_outputs 只清理 results，不删除 data，因此选做 Kaggle 数据会被保留。
    # 下面重新生成并保存必做的高维模拟数据。
    print("[Week14] 生成高维模拟数据", flush=True)
    highdim_df, _ = make_high_dimensional_data()
    highdim_df.to_csv(DATA_DIR / "synthetic_highdim.csv", index=False)

    print("[Week14] 运行 OLS 维度增长实验", flush=True)
    dimension_df = run_ols_dimension_experiment()
    plot_ols_dimension_results(dimension_df)

    print("[Week14] 运行 OLS 系数稳定性实验", flush=True)
    _, coef_summary = run_ols_stability_experiment(highdim_df)
    print("[Week14] 运行 PCA 和 PCR 流程", flush=True)
    pca_df, best_k, pcr_df = run_pca_pcr(highdim_df)
    print("[Week14] 比较 Lasso 变量选择与 PCR 信息压缩", flush=True)
    scenario_df = run_lasso_vs_pcr_scenarios()
    print("[Week14] 运行可选 Kaggle 房价实验", flush=True)
    kaggle_results, kaggle_extra = run_kaggle_optional()

    print("[Week14] 写出中文报告", flush=True)
    write_synthetic_report(highdim_df, dimension_df, coef_summary, pca_df, best_k, pcr_df)
    write_summary_report(scenario_df)
    write_kaggle_report(kaggle_results, kaggle_extra)

    print("第 14 周作业已完成。")
    print(f"数据目录：{DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"结果目录：{RESULTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    # `uv run week14/main.py` 会把本文件作为脚本执行。
    # 通过 sys.executable 重新执行一次，可使其行为更接近
    # `uv run -- python week14/main.py`，在部分平台上对 numpy/matplotlib 更稳定。
    if os.environ.get("WEEK14_REEXECUTED") != "1":
        os.environ["WEEK14_REEXECUTED"] = "1"
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])
    main()
