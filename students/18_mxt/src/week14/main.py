#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
sys.path.append(str(SRC_DIR))

from utils.diagnostics import coefficient_stability, condition_number, matrix_rank
from utils.metrics import calculate_mae, calculate_rmse
from utils.models import PCRRegressor
from utils.transformers import CustomStandardScaler


DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
for path in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    path.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42


def configure_matplotlib_fonts():
    """Use a Chinese-capable font when figures are rendered from WSL."""
    candidates = [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


def make_latent_factor_data(
    n_samples: int = 180,
    n_features: int = 140,
    n_factors: int = 5,
    noise_scale: float = 0.25,
    target_noise: float = 0.8,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Generate high-dimensional X from a low-dimensional latent-factor model."""
    rng = np.random.default_rng(random_state)
    factors = rng.normal(size=(n_samples, n_factors))
    loadings = rng.normal(size=(n_factors, n_features))

    # Give each block of variables a strong shared factor, then add small cross-loadings.
    for j in range(n_features):
        dominant = j % n_factors
        loadings[:, j] *= 0.25
        loadings[dominant, j] += rng.uniform(0.9, 1.4)

    X = factors @ loadings + rng.normal(scale=noise_scale, size=(n_samples, n_features))
    beta_factors = np.array([3.2, -2.4, 1.6, 0.0, 1.1])
    y = factors @ beta_factors + rng.normal(scale=target_noise, size=n_samples)

    columns = [f"x{j:03d}" for j in range(1, n_features + 1)]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="y"), factors


def make_sparse_truth_data(
    n_samples: int = 220,
    n_features: int = 100,
    random_state: int = 2026,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Generate data where only a few raw variables directly drive y."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    true_idx = [0, 4, 12, 33, 57]
    beta = np.zeros(n_features)
    beta[true_idx] = [3.0, -2.5, 2.0, -1.7, 1.3]
    y = X @ beta + rng.normal(scale=1.0, size=n_samples)
    columns = [f"x{j:03d}" for j in range(1, n_features + 1)]
    true_features = [columns[i] for i in true_idx]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="y"), true_features


def standardize_train_test(X_train, X_test):
    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(np.asarray(X_train, dtype=float))
    X_test_s = scaler.transform(np.asarray(X_test, dtype=float))
    return X_train_s, X_test_s


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": round(float(calculate_rmse(np.asarray(y_true), np.asarray(y_pred))), 4),
        "MAE": round(float(calculate_mae(np.asarray(y_true), np.asarray(y_pred))), 4),
    }


def pcr_cv_rmse(X, y, k: int, cv_splits: int = 5, random_state: int = RANDOM_SEED) -> float:
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    errors = []
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    for train_idx, val_idx in kf.split(X_arr):
        model = PCRRegressor(n_components=k).fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[val_idx])
        errors.append(calculate_rmse(y_arr[val_idx], pred))
    return float(np.mean(errors))


def select_pcr_k(X, y, max_k: int = 20, cv_splits: int = 3) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(1, max_k + 1):
        rows.append({"k": k, "CV_RMSE": pcr_cv_rmse(X, y, k, cv_splits=cv_splits)})
    df = pd.DataFrame(rows)
    best_k = int(df.loc[df["CV_RMSE"].idxmin(), "k"])
    return best_k, df


def fit_lasso_1se(X_train_s, y_train, alphas, cv: int = 3, random_state: int = RANDOM_SEED):
    """Fit Lasso with the one-standard-error rule for a shorter variable list."""
    cv_model = LassoCV(
        alphas=alphas,
        cv=cv,
        random_state=random_state,
        max_iter=100000,
    ).fit(X_train_s, y_train)
    mean_mse = cv_model.mse_path_.mean(axis=1)
    se_mse = cv_model.mse_path_.std(axis=1, ddof=1) / np.sqrt(cv_model.mse_path_.shape[1])
    best_idx = int(np.argmin(mean_mse))
    threshold = mean_mse[best_idx] + se_mse[best_idx]
    eligible = np.where(mean_mse <= threshold)[0]
    chosen_idx = int(eligible[np.argmax(cv_model.alphas_[eligible])])
    alpha = float(cv_model.alphas_[chosen_idx])
    model = Lasso(alpha=alpha, max_iter=100000).fit(X_train_s, y_train)
    return model, alpha


def run_task_a(X_full: pd.DataFrame, y: pd.Series):
    p_values = [10, 30, 60, 120, 140]
    rows = []

    for p in p_values:
        X_p = X_full.iloc[:, :p]
        X_train, X_test, y_train, y_test = train_test_split(
            X_p, y, test_size=0.35, random_state=RANDOM_SEED
        )
        X_train_s, X_test_s = standardize_train_test(X_train, X_test)
        ols = LinearRegression().fit(X_train_s, y_train)
        train_pred = ols.predict(X_train_s)
        test_pred = ols.predict(X_test_s)
        rows.append(
            {
                "p": p,
                "train_n": len(X_train),
                "train_RMSE": calculate_rmse(y_train, train_pred),
                "test_RMSE": calculate_rmse(y_test, test_pred),
                "rank_X_train": matrix_rank(X_train_s),
                "condition_number": condition_number(X_train_s),
            }
        )

    ols_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ols_df["p"], ols_df["train_RMSE"], marker="o", label="训练集 RMSE")
    ax.plot(ols_df["p"], ols_df["test_RMSE"], marker="o", label="测试集 RMSE")
    ax.axvline(ols_df["train_n"].iloc[0], color="gray", linestyle="--", label="训练样本数 n_train")
    ax.set_xlabel("原始特征数量 p")
    ax.set_ylabel("RMSE")
    ax.set_title("OLS 误差随特征维度增加的变化")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ols_error_by_dimension.png", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(ols_df["p"], ols_df["rank_X_train"], width=8, alpha=0.75, label="rank(X_train)")
    ax1.set_xlabel("原始特征数量 p")
    ax1.set_ylabel("矩阵秩 rank(X_train)")
    ax2 = ax1.twinx()
    finite_cond = ols_df["condition_number"].replace([np.inf, -np.inf], np.nan)
    ax2.plot(ols_df["p"], finite_cond, color="tab:red", marker="o", label="条件数 condition number")
    ax2.set_yscale("log")
    ax2.set_ylabel("条件数 condition number（对数刻度）")
    ax1.set_title("设计矩阵结构随特征维度增加的变化")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matrix_structure_by_dimension.png", dpi=220)
    plt.close(fig)

    coef_rows = []
    key_features = ["x001", "x002", "x003"]
    X_key = X_full.iloc[:, :120]
    for seed in range(50):
        X_train, _, y_train, _ = train_test_split(X_key, y, test_size=0.35, random_state=seed)
        scaler = CustomStandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        ols = LinearRegression().fit(X_train_s, y_train)
        for feature in key_features:
            idx = X_key.columns.get_loc(feature)
            coef_rows.append({"split": seed, "feature": feature, "coef": ols.coef_[idx]})

    coef_df = pd.DataFrame(coef_rows)
    stability = (
        coef_df.pivot(index="split", columns="feature", values="coef")
        .pipe(lambda frame: coefficient_stability(frame.to_numpy()))
    )
    stability_df = pd.DataFrame({"feature": key_features, "coef_sd": stability})

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [coef_df.loc[coef_df["feature"] == feat, "coef"] for feat in key_features]
    ax.boxplot(box_data, tick_labels=key_features, patch_artist=True)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("选定原始特征")
    ax.set_ylabel("50 次随机划分下的 OLS 系数")
    ax.set_title("重复训练/测试划分下的系数不稳定性")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ols_coefficient_instability.png", dpi=220)
    plt.close(fig)

    return ols_df, stability_df


def run_task_b(X_full: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.35, random_state=RANDOM_SEED
    )
    X_train_s, X_test_s = standardize_train_test(X_train, X_test)

    pca = PCA().fit(X_train_s)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(
        {
            "component": np.arange(1, len(cum_var) + 1),
            "cumulative_explained_variance": cum_var,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pca_df["component"], pca_df["cumulative_explained_variance"], marker="o", markersize=3)
    ax.axhline(0.9, color="tab:red", linestyle="--", label="90% 方差")
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("主成分数量")
    ax.set_ylabel("累计解释方差比例")
    ax.set_title("PCA 累计解释方差")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_cumulative_variance.png", dpi=220)
    plt.close(fig)

    k_rows = []
    for k in range(1, 21):
        model = PCRRegressor(n_components=k).fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        k_rows.append(
            {
                "k": k,
                "train_RMSE": calculate_rmse(y_train, train_pred),
                "test_RMSE": calculate_rmse(y_test, test_pred),
                "CV_RMSE": pcr_cv_rmse(X_train, y_train, k),
            }
        )
    pcr_df = pd.DataFrame(k_rows)

    ols = LinearRegression().fit(X_train_s, y_train)
    ols_test_rmse = calculate_rmse(y_test, ols.predict(X_test_s))
    best_k = int(pcr_df.loc[pcr_df["CV_RMSE"].idxmin(), "k"])
    k90 = int(np.searchsorted(cum_var, 0.9) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcr_df["k"], pcr_df["train_RMSE"], marker="o", label="PCR 训练集 RMSE")
    ax.plot(pcr_df["k"], pcr_df["test_RMSE"], marker="o", label="PCR 测试集 RMSE")
    ax.plot(pcr_df["k"], pcr_df["CV_RMSE"], marker="o", label="PCR CV RMSE")
    ax.axhline(ols_test_rmse, color="gray", linestyle="--", label="OLS 测试集 RMSE 基准")
    ax.axvline(best_k, color="tab:red", linestyle=":", label=f"CV 最优 k={best_k}")
    ax.set_xlabel("保留主成分数量 k")
    ax.set_ylabel("RMSE")
    ax.set_title("PCR 误差曲线随保留主成分数量的变化")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pcr_error_by_k.png", dpi=220)
    plt.close(fig)

    return pca_df, pcr_df, best_k, k90, ols_test_rmse


def lasso_vs_pcr_once(X: pd.DataFrame, y: pd.Series, scenario: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=RANDOM_SEED
    )
    X_train_s, X_test_s = standardize_train_test(X_train, X_test)

    alphas = np.logspace(-3, 1.5, 30)
    lasso, lasso_alpha = fit_lasso_1se(X_train_s, y_train, alphas)
    lasso_pred = lasso.predict(X_test_s)
    lasso_complexity = int(np.sum(np.abs(lasso.coef_) > 1e-8))

    best_k, cv_df = select_pcr_k(X_train, y_train, max_k=20)
    pcr = PCRRegressor(n_components=best_k).fit(X_train, y_train)
    pcr_pred = pcr.predict(X_test)

    rows = [
        {
            "scenario": scenario,
            "method": "Lasso",
            **evaluate_predictions(y_test, lasso_pred),
            "complexity": lasso_complexity,
            "complexity_definition": "nonzero coefficients",
        },
        {
            "scenario": scenario,
            "method": "PCR",
            **evaluate_predictions(y_test, pcr_pred),
            "complexity": best_k,
            "complexity_definition": "retained principal components",
        },
    ]
    return pd.DataFrame(rows), cv_df


def stability_over_splits(X: pd.DataFrame, y: pd.Series, method: str, n_splits: int = 6) -> float:
    rmses = []
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed)
        if method == "Lasso":
            X_train_s, X_test_s = standardize_train_test(X_train, X_test)
            model = Lasso(alpha=0.05, max_iter=100000).fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
        else:
            model = PCRRegressor(n_components=min(8, X_train.shape[1])).fit(X_train, y_train)
            pred = model.predict(X_test)
        rmses.append(calculate_rmse(y_test, pred))
    return float(np.std(rmses, ddof=1))


def plot_scenario_comparison(df: pd.DataFrame, filename: str = "lasso_vs_pcr_scenarios.png"):
    scenario_labels = [
        ("Sparse truth", "稀疏真相（sparse truth）"),
        ("Latent-factor truth", "潜在因子真相（latent-factor truth）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    colors = ["tab:blue", "tab:orange"]
    for row, (scenario, label) in enumerate(scenario_labels):
        subset = df[df["scenario"] == scenario].copy()
        axes[row, 0].bar(subset["method"], subset["RMSE"], color=colors)
        axes[row, 0].set_ylabel("测试集 RMSE")
        axes[row, 0].set_title(f"{label}：预测误差")
        axes[row, 1].bar(subset["method"], subset["complexity"], color=colors)
        axes[row, 1].set_ylabel("模型复杂度")
        axes[row, 1].set_title(f"{label}：模型复杂度")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def run_task_c():
    sparse_X, sparse_y, true_features = make_sparse_truth_data()
    latent_X, latent_y, _ = make_latent_factor_data(
        n_samples=220,
        n_features=100,
        n_factors=5,
        noise_scale=0.25,
        target_noise=0.8,
        random_state=2027,
    )

    sparse_df, _ = lasso_vs_pcr_once(sparse_X, sparse_y, "Sparse truth")
    latent_df, _ = lasso_vs_pcr_once(latent_X, latent_y, "Latent-factor truth")
    comparison = pd.concat([sparse_df, latent_df], ignore_index=True)

    stability_rows = []
    for scenario, X, y in [
        ("Sparse truth", sparse_X, sparse_y),
        ("Latent-factor truth", latent_X, latent_y),
    ]:
        for method in ["Lasso", "PCR"]:
            stability_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "stability_RMSE_sd": stability_over_splits(X, y, method),
                }
            )
    stability_df = pd.DataFrame(stability_rows)
    comparison = comparison.merge(stability_df, on=["scenario", "method"])

    plot_scenario_comparison(comparison)

    return comparison, true_features




def load_diabetes_real_data():
    """Load a public real regression dataset and expand it into a high-dimensional design."""
    diabetes = load_diabetes(as_frame=True)
    raw_X = diabetes.data.copy()
    y = diabetes.target.copy()

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(raw_X)
    feature_names = poly.get_feature_names_out(raw_X.columns)
    X = pd.DataFrame(X_poly, columns=feature_names)

    cleaned = X.copy()
    cleaned["disease_progression"] = y.to_numpy()
    cleaned.to_csv(DATA_DIR / "diabetes_polynomial_features.csv", index=False)

    return raw_X, X, y


def run_task_d():
    raw_X, X, y = load_diabetes_real_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )
    X_train_s, X_test_s = standardize_train_test(X_train, X_test)

    pca = PCA().fit(X_train_s)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k90 = int(np.searchsorted(cum_var, 0.9) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o", markersize=3)
    ax.axhline(0.9, color="tab:red", linestyle="--", label="90% 方差")
    ax.axvline(k90, color="gray", linestyle=":", label=f"达到 90% 的 k={k90}")
    ax.set_xlim(1, min(45, len(cum_var)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("主成分数量")
    ax.set_ylabel("累计解释方差比例")
    ax.set_title("真实数据的 PCA 累计解释方差")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_pca_cumulative_variance.png", dpi=220)
    plt.close(fig)

    ols = LinearRegression().fit(X_train_s, y_train)
    ols_test_pred = ols.predict(X_test_s)

    lasso, _ = fit_lasso_1se(
        X_train_s,
        y_train,
        alphas=np.logspace(-2, 2.2, 35),
        cv=5,
    )
    lasso_test_pred = lasso.predict(X_test_s)

    best_k, _ = select_pcr_k(X_train, y_train, max_k=25, cv_splits=3)
    pcr = PCRRegressor(n_components=best_k).fit(X_train, y_train)
    pcr_test_pred = pcr.predict(X_test)

    comparison = pd.DataFrame(
        [
            {
                "method": "OLS",
                **evaluate_predictions(y_test, ols_test_pred),
                "complexity": X.shape[1],
            },
            {
                "method": "Lasso",
                **evaluate_predictions(y_test, lasso_test_pred),
                "complexity": int(np.sum(np.abs(lasso.coef_) > 1e-8)),
            },
            {
                "method": "PCR",
                **evaluate_predictions(y_test, pcr_test_pred),
                "complexity": best_k,
            },
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["tab:gray", "tab:blue", "tab:orange"]
    axes[0].bar(comparison["method"], comparison["RMSE"], color=colors)
    axes[0].set_ylabel("测试集 RMSE")
    axes[0].set_title("真实数据：预测误差")
    axes[1].bar(comparison["method"], comparison["complexity"], color=colors)
    axes[1].set_ylabel("模型复杂度")
    axes[1].set_title("真实数据：模型复杂度")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_ols_lasso_pcr_comparison.png", dpi=220)
    plt.close(fig)

def main():
    X_full, y, _ = make_latent_factor_data()
    synthetic_df = X_full.copy()
    synthetic_df["y"] = y
    synthetic_df.to_csv(DATA_DIR / "synthetic_highdim.csv", index=False)

    print("Running Task A: OLS dimension and coefficient stability...", flush=True)
    run_task_a(X_full, y)
    print("Running Task B: PCA and PCR...", flush=True)
    run_task_b(X_full, y)
    print("Running Task C: Lasso vs PCR scenarios...", flush=True)
    run_task_c()
    print("Running Task D: real public regression data...", flush=True)
    run_task_d()


    print("Week 14 completed.")
    print(f"Synthetic data: {DATA_DIR / 'synthetic_highdim.csv'}")


if __name__ == "__main__":
    main()
