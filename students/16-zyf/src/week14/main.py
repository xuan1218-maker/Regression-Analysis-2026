import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import calculate_rmse, calculate_mae


warnings.filterwarnings("ignore")


DATA_DIR = "src/week14/data"
RESULTS_DIR = "src/week14/results"
FIGURES_DIR = "src/week14/results/figures"
SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_highdim.csv")


def prepare_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(FIGURES_DIR, exist_ok=True)


def dataframe_to_markdown(df, digits=4):
    df = df.copy()
    df = df.round(digits)

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    return "\n".join(lines)


def make_latent_factor_data(n_samples=150, n_features=80, n_factors=5, random_state=42):
    rng = np.random.default_rng(random_state)

    Z = rng.normal(size=(n_samples, n_factors))

    loadings = rng.normal(size=(n_factors, n_features))

    X = Z @ loadings + rng.normal(scale=0.15, size=(n_samples, n_features))

    y = 3.0 * Z[:, 0] - 2.0 * Z[:, 1] + rng.normal(scale=1.0, size=n_samples)

    columns = [f"x{i+1}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=columns)
    df["y"] = y

    return df


def save_main_synthetic_data():
    df = make_latent_factor_data(
        n_samples=150,
        n_features=80,
        n_factors=5,
        random_state=42,
    )

    df.to_csv(SYNTHETIC_PATH, index=False)

    return df


def evaluate_ols_under_different_p():
    rows = []

    p_list = [10, 30, 60, 120]

    for p in p_list:
        df = make_latent_factor_data(
            n_samples=150,
            n_features=p,
            n_factors=5,
            random_state=100 + p,
        )

        X = df.drop(columns=["y"]).values
        y = df["y"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.4,
            random_state=42,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        rank = np.linalg.matrix_rank(X_train_scaled)
        cond_number = np.linalg.cond(X_train_scaled)

        rows.append(
            {
                "p": p,
                "train_rmse": calculate_rmse(y_train, train_pred),
                "test_rmse": calculate_rmse(y_test, test_pred),
                "rank_X_train": rank,
                "condition_number": cond_number,
            }
        )

    result = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(result["p"], result["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(result["p"], result["test_rmse"], marker="o", label="Test RMSE")
    plt.xlabel("Number of features p")
    plt.ylabel("RMSE")
    plt.title("OLS Error Changes When p Increases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ols_error_vs_p.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result["p"], result["rank_X_train"], marker="o", label="rank(X_train)")
    plt.xlabel("Number of features p")
    plt.ylabel("Matrix Rank")
    plt.title("Rank of X_train Changes When p Increases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rank_vs_p.png"), dpi=300)
    plt.close()

    return result


def coefficient_stability_experiment(df, n_repeats=50):
    feature_cols = [c for c in df.columns if c != "y"]

    key_features = ["x1", "x2", "x3"]

    X = df[feature_cols].values
    y = df["y"].values

    rows = []

    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.4,
            random_state=seed,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        for feature in key_features:
            idx = feature_cols.index(feature)

            rows.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "coef": model.coef_[idx],
                }
            )

    coef_df = pd.DataFrame(rows)

    stability_summary = (
        coef_df.groupby("feature")["coef"]
        .std()
        .reset_index()
        .rename(columns={"coef": "coef_std_across_splits"})
    )

    data = [
        coef_df[coef_df["feature"] == feature]["coef"].values
        for feature in key_features
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=key_features)
    plt.xlabel("Feature")
    plt.ylabel("OLS Coefficient")
    plt.title("OLS Coefficient Instability Across 50 Splits")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ols_coef_instability_boxplot.png"), dpi=300)
    plt.close()

    return coef_df, stability_summary


def run_pca_analysis(df):
    X = df.drop(columns=["y"]).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    pca_df = pd.DataFrame(
        {
            "n_components": np.arange(1, len(cumulative_variance) + 1),
            "cumulative_explained_variance": cumulative_variance,
        }
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        pca_df["n_components"],
        pca_df["cumulative_explained_variance"],
        marker="o",
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.axhline(0.9, linestyle="--", label="90% variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pca_cumulative_variance.png"), dpi=300)
    plt.close()

    k_90 = int(np.argmax(cumulative_variance >= 0.9) + 1)

    return pca_df, k_90


def pcr_fit_predict(X_train, X_test, y_train, k):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=k)
    Z_train = pca.fit_transform(X_train_scaled)
    Z_test = pca.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(Z_train, y_train)

    train_pred = model.predict(Z_train)
    test_pred = model.predict(Z_test)

    return train_pred, test_pred


def pcr_cv_rmse(X, y, k, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []

    for train_idx, val_idx in cv.split(X):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_scaled)
        Z_val = pca.transform(X_val_scaled)

        model = LinearRegression()
        model.fit(Z_train, y_train)

        pred = model.predict(Z_val)

        scores.append(calculate_rmse(y_val, pred))

    return float(np.mean(scores))


def run_pcr_k_experiment(df):
    X = df.drop(columns=["y"]).values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    rows = []

    for k in range(1, 21):
        train_pred, test_pred = pcr_fit_predict(X_train, X_test, y_train, k)

        cv_rmse = pcr_cv_rmse(X_train, y_train, k)

        rows.append(
            {
                "k_components": k,
                "pcr_train_rmse": calculate_rmse(y_train, train_pred),
                "pcr_test_rmse": calculate_rmse(y_test, test_pred),
                "pcr_cv_rmse": cv_rmse,
            }
        )

    result = pd.DataFrame(rows)

    best_k = int(result.loc[result["pcr_cv_rmse"].idxmin(), "k_components"])

    plt.figure(figsize=(8, 5))
    plt.plot(result["k_components"], result["pcr_train_rmse"], marker="o", label="Train RMSE")
    plt.plot(result["k_components"], result["pcr_test_rmse"], marker="o", label="Test RMSE")
    plt.plot(result["k_components"], result["pcr_cv_rmse"], marker="o", label="CV RMSE")
    plt.xlabel("Number of principal components k")
    plt.ylabel("RMSE")
    plt.title("PCR Error Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pcr_rmse_vs_k.png"), dpi=300)
    plt.close()

    return result, best_k


def make_sparse_truth_data(n_samples=150, n_features=80, random_state=123):
    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))

    beta = np.zeros(n_features)
    beta[0] = 3.0
    beta[4] = -2.0
    beta[9] = 1.5

    y = X @ beta + rng.normal(scale=1.0, size=n_samples)

    return X, y


def make_latent_truth_arrays(n_samples=150, n_features=80, n_factors=5, random_state=456):
    df = make_latent_factor_data(
        n_samples=n_samples,
        n_features=n_features,
        n_factors=n_factors,
        random_state=random_state,
    )

    X = df.drop(columns=["y"]).values
    y = df["y"].values

    return X, y


def evaluate_lasso_vs_pcr(X, y, scenario_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = LassoCV(cv=5, random_state=42, max_iter=20000)
    lasso.fit(X_train_scaled, y_train)

    lasso_pred = lasso.predict(X_test_scaled)

    lasso_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-6))

    best_pcr_rmse = np.inf
    best_pcr_k = None
    best_pcr_pred = None

    for k in range(1, 21):
        _, test_pred = pcr_fit_predict(X_train, X_test, y_train, k)
        rmse = calculate_rmse(y_test, test_pred)

        if rmse < best_pcr_rmse:
            best_pcr_rmse = rmse
            best_pcr_k = k
            best_pcr_pred = test_pred

    rows = [
        {
            "scenario": scenario_name,
            "model": "Lasso",
            "test_rmse": calculate_rmse(y_test, lasso_pred),
            "test_mae": calculate_mae(y_test, lasso_pred),
            "complexity": lasso_nonzero,
            "complexity_meaning": "nonzero coefficients",
        },
        {
            "scenario": scenario_name,
            "model": "PCR",
            "test_rmse": calculate_rmse(y_test, best_pcr_pred),
            "test_mae": calculate_mae(y_test, best_pcr_pred),
            "complexity": best_pcr_k,
            "complexity_meaning": "selected principal components",
        },
    ]

    return pd.DataFrame(rows)


def run_task_c_comparison():
    X_sparse, y_sparse = make_sparse_truth_data()

    X_latent, y_latent = make_latent_truth_arrays()

    sparse_result = evaluate_lasso_vs_pcr(
        X_sparse,
        y_sparse,
        "Sparse truth",
    )

    latent_result = evaluate_lasso_vs_pcr(
        X_latent,
        y_latent,
        "Latent-factor truth",
    )

    result = pd.concat([sparse_result, latent_result], ignore_index=True)

    plt.figure(figsize=(8, 5))

    x_labels = result["scenario"] + " - " + result["model"]

    plt.bar(x_labels, result["test_rmse"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Lasso vs PCR Test RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lasso_vs_pcr_rmse.png"), dpi=300)
    plt.close()

    return result


def write_synthetic_report(
    ols_p_result,
    stability_summary,
    pca_df,
    k_90,
    pcr_result,
    best_k,
):
    path = os.path.join(RESULTS_DIR, "synthetic_report.md")

    lines = []

    lines.append("# Week 14 Synthetic Report")
    lines.append("")
    lines.append("## 1. 数据生成机制")
    lines.append("")
    lines.append("本周自己生成了一份高维低秩结构数据。")
    lines.append("")
    lines.append("- 样本量 n = 150")
    lines.append("- 特征数 p = 80")
    lines.append("- 潜在因子数 latent factors = 5")
    lines.append("- 目标变量 y 主要由前两个 latent factors 决定")
    lines.append("")
    lines.append("这说明原始 80 个自变量并不是完全独立的，它们大部分来自少数几个潜在因子，因此存在明显的信息冗余。")
    lines.append("")
    lines.append("## 2. OLS 在不同 p 下的表现")
    lines.append("")
    lines.append(dataframe_to_markdown(ols_p_result))
    lines.append("")
    lines.append("图 ols_error_vs_p.png 展示了横轴为特征数 p，纵轴为 RMSE。")
    lines.append("图中 Train RMSE 和 Test RMSE 同时展示，用来说明 p 增大时训练误差可能变低，但测试误差可能变差。")
    lines.append("")
    lines.append("训练误差接近 0 并不一定是好事，因为它可能说明 OLS 已经开始记忆训练集中的噪声。")
    lines.append("")
    lines.append("## 3. OLS 系数不稳定")
    lines.append("")
    lines.append(dataframe_to_markdown(stability_summary))
    lines.append("")
    lines.append("图 ols_coef_instability_boxplot.png 展示了 x1、x2、x3 三个变量在 50 次随机切分中的系数波动。")
    lines.append("系数波动很大说明模型解释不稳定，即使预测误差看起来还可以，模型解释也可能不可信。")
    lines.append("")
    lines.append("## 4. PCA 累计解释方差")
    lines.append("")
    lines.append(f"前 {k_90} 个主成分可以解释至少 90% 的方差。")
    lines.append("")
    lines.append("图 pca_cumulative_variance.png 的横轴是主成分个数，纵轴是累计解释方差比例。")
    lines.append("如果前几个主成分已经解释大部分方差，就说明原始高维空间接近一个低维子空间。")
    lines.append("")
    lines.append("## 5. PCR 不同 k 的表现")
    lines.append("")
    lines.append(dataframe_to_markdown(pcr_result))
    lines.append("")
    lines.append(f"根据 CV RMSE，最佳主成分个数 k = {best_k}。")
    lines.append("")
    lines.append("图 pcr_rmse_vs_k.png 的横轴是主成分个数 k，纵轴是 RMSE。")
    lines.append("图中展示了 PCR train RMSE、test RMSE 和 CV RMSE。")
    lines.append("")
    lines.append("## 6. 公式解释")
    lines.append("")
    lines.append("OLS 估计式：")
    lines.append("")
    lines.append("beta_hat = (X^T X)^(-1) X^T y")
    lines.append("")
    lines.append("第一主成分定义：")
    lines.append("")
    lines.append("v1 = argmax Var(Xv), subject to ||v|| = 1")
    lines.append("")
    lines.append("PCR 流程：")
    lines.append("")
    lines.append("Z_k = X V_k，然后在 Z_k 上做线性回归。")
    lines.append("")
    lines.append("也就是说，PCR 不是直接在原始变量 X 上回归，而是先把 X 压缩成前 k 个主成分。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Synthetic report saved to: {path}")


def write_summary_report(task_c_result):
    path = os.path.join(RESULTS_DIR, "summary_comparison.md")

    lines = []

    lines.append("# Week 14 Summary Comparison")
    lines.append("")
    lines.append("## 1. Lasso vs PCR 结果")
    lines.append("")
    lines.append(dataframe_to_markdown(task_c_result))
    lines.append("")
    lines.append("图 lasso_vs_pcr_rmse.png 分别展示了 Sparse truth 和 Latent-factor truth 两种场景下 Lasso 与 PCR 的测试误差。")
    lines.append("")
    lines.append("## 2. Sparse truth 下为什么 Lasso 更自然")
    lines.append("")
    lines.append("Sparse truth 的意思是：很多变量中只有少数原始变量真正决定 y。")
    lines.append("这时 Lasso 更自然，因为它的作用就是把一部分变量系数压缩为 0，相当于自动筛选变量。")
    lines.append("")
    lines.append("## 3. Latent-factor truth 下为什么 PCR 更自然")
    lines.append("")
    lines.append("Latent-factor truth 的意思是：很多原始变量其实来自少数几个潜在因子。")
    lines.append("这时 PCR 更自然，因为 PCR 不是挑某几个原始变量，而是把整体信息压缩成少数主成分。")
    lines.append("")
    lines.append("## 4. Lasso 和 PCR 回答的问题不同")
    lines.append("")
    lines.append("Lasso 回答的问题更像是：谁留下？谁被删掉？")
    lines.append("PCR 回答的问题更像是：这些变量背后能不能压缩成更少的核心方向？")
    lines.append("")
    lines.append("## 5. 业务解释")
    lines.append("")
    lines.append("如果业务方想要一个更短的变量名单，我会优先考虑 Lasso。")
    lines.append("如果业务方想要一个更稳定的预测器，尤其是变量很多且高度相关时，我会优先考虑 PCR。")
    lines.append("")
    lines.append("## 6. 为什么这周不把前向/后向选择作为主线")
    lines.append("")
    lines.append("前向/后向选择本质上属于 selection 路线，也就是筛变量。")
    lines.append("但 Week14 的重点是比较 selection 和 compression。")
    lines.append("因此这周主线更适合比较 Lasso 和 PCR，而不是继续重点做前向/后向选择。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Summary report saved to: {path}")


def main():
    prepare_dirs()

    print("[Task A1] Generating synthetic high-dimensional data...")
    df = save_main_synthetic_data()

    print("[Task A3] Evaluating OLS under different feature dimensions...")
    ols_p_result = evaluate_ols_under_different_p()

    print("[Task A4] Running coefficient stability experiment...")
    coef_df, stability_summary = coefficient_stability_experiment(df)

    print("[Task B1] Running PCA analysis...")
    pca_df, k_90 = run_pca_analysis(df)

    print("[Task B2] Running PCR k experiment...")
    pcr_result, best_k = run_pcr_k_experiment(df)

    print("[Task C] Comparing Lasso and PCR...")
    task_c_result = run_task_c_comparison()

    print("[Report] Writing synthetic report...")
    write_synthetic_report(
        ols_p_result,
        stability_summary,
        pca_df,
        k_90,
        pcr_result,
        best_k,
    )

    print("[Report] Writing summary comparison report...")
    write_summary_report(task_c_result)

    print("\nWeek14 finished.")


if __name__ == "__main__":
    main()