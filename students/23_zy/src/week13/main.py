import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "synthetic_correlated.csv")
REPORT_PATH = os.path.join(RESULTS_DIR, "summary.md")

FIG_CORR = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
FIG_ALPHA = os.path.join(FIGURES_DIR, "alpha_cv_curves.png")
FIG_COEF = os.path.join(FIGURES_DIR, "model_coefficients.png")
FIG_STABILITY = os.path.join(FIGURES_DIR, "coefficient_stability.png")
FIG_FORWARD = os.path.join(FIGURES_DIR, "forward_selection_curve.png")


# ============================================================
# 2. 基础函数
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def df_to_markdown(df, float_digits=4):
    """
    自己写一个 DataFrame 转 Markdown 表格的函数，
    避免 pandas.to_markdown 因为缺少 tabulate 包而报错。
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.{float_digits}f}")

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    return "\n".join(lines)


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "模型": model_name,
        "RMSE": rmse(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }


# ============================================================
# 3. 生成模拟数据
# ============================================================

def generate_synthetic_data(n_samples=500, random_state=42):
    """
    生成一个带有多重共线性的模拟数据集。

    x1、x2、x3 之间高度相关；
    x4、x5 之间高度相关；
    x6、x7、x8 相对独立。

    y 主要由 x1、x4、x6、x8 决定。
    """
    rng = np.random.default_rng(random_state)

    x1 = rng.normal(0, 1, n_samples)
    x2 = x1 + rng.normal(0, 0.08, n_samples)
    x3 = 0.8 * x1 + rng.normal(0, 0.10, n_samples)

    z = rng.normal(0, 1, n_samples)
    x4 = z + rng.normal(0, 0.08, n_samples)
    x5 = 0.9 * z + rng.normal(0, 0.10, n_samples)

    x6 = rng.normal(0, 1, n_samples)
    x7 = rng.normal(0, 1, n_samples)
    x8 = rng.normal(0, 1, n_samples)

    noise = rng.normal(0, 1.5, n_samples)

    y = 5.0 * x1 + 3.0 * x4 - 2.5 * x6 + 1.8 * x8 + noise

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "x6": x6,
        "x7": x7,
        "x8": x8,
        "y": y,
    })

    return df


# ============================================================
# 4. 画相关系数热力图
# ============================================================

def plot_correlation_heatmap(df):
    corr = df.drop(columns=["y"]).corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Feature Correlation Heatmap")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_CORR, dpi=200)
    plt.close()

    return corr


# ============================================================
# 5. 正则化模型调参
# ============================================================

def tune_regularized_models(df):
    X = df.drop(columns=["y"])
    y = df["y"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 2, 20)

    configs = {
        "Ridge": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            "params": {
                "model__alpha": alphas
            }
        },
        "Lasso": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=20000))
            ]),
            "params": {
                "model__alpha": alphas
            }
        },
        "ElasticNet": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=20000))
            ]),
            "params": {
                "model__alpha": alphas,
                "model__l1_ratio": [0.2, 0.5, 0.8]
            }
        }
    }

    best_models = {}
    cv_records = []

    for name, cfg in configs.items():
        grid = GridSearchCV(
            estimator=cfg["pipeline"],
            param_grid=cfg["params"],
            scoring="neg_root_mean_squared_error",
            cv=cv
        )

        grid.fit(X, y)
        best_models[name] = grid.best_estimator_

        result_df = pd.DataFrame(grid.cv_results_)

        for _, row in result_df.iterrows():
            record = {
                "模型": name,
                "Alpha": float(row["param_model__alpha"]),
                "交叉验证RMSE": float(-row["mean_test_score"])
            }

            if "param_model__l1_ratio" in row:
                record["L1比例"] = row["param_model__l1_ratio"]
            else:
                record["L1比例"] = np.nan

            cv_records.append(record)

    cv_df = pd.DataFrame(cv_records)

    return best_models, cv_df


def plot_alpha_curves(cv_df):
    plt.figure(figsize=(9, 6))

    for model_name in ["Ridge", "Lasso", "ElasticNet"]:
        subset = cv_df[cv_df["模型"] == model_name].copy()

        if model_name == "ElasticNet":
            subset = subset.groupby("Alpha", as_index=False)["交叉验证RMSE"].min()

        subset = subset.sort_values("Alpha")

        plt.plot(
            subset["Alpha"],
            subset["交叉验证RMSE"],
            marker="o",
            label=model_name
        )

    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("CV RMSE")
    plt.title("CV RMSE under Different Alpha Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_ALPHA, dpi=200)
    plt.close()


# ============================================================
# 6. 模型效果比较
# ============================================================

def compare_models(df, best_models):
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=2026
    )

    models = {
        "OLS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": best_models["Ridge"],
        "Lasso": best_models["Lasso"],
        "ElasticNet": best_models["ElasticNet"],
    }

    metric_records = []

    for name, model in models.items():
        metric_records.append(
            evaluate_model(name, model, X_train, X_test, y_train, y_test)
        )

    metric_df = pd.DataFrame(metric_records)

    return metric_df, models


def extract_coefficients(models, feature_names):
    records = []

    for name, model in models.items():
        coefs = model.named_steps["model"].coef_

        for feature, coef in zip(feature_names, coefs):
            records.append({
                "模型": name,
                "变量": feature,
                "系数": coef
            })

    return pd.DataFrame(records)


def plot_model_coefficients(coef_df):
    feature_names = sorted(coef_df["变量"].unique())
    model_names = ["OLS", "Ridge", "Lasso", "ElasticNet"]

    x = np.arange(len(feature_names))
    width = 0.2

    plt.figure(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        subset = coef_df[coef_df["模型"] == model_name].set_index("变量")
        values = [subset.loc[f, "系数"] for f in feature_names]
        plt.bar(x + (i - 1.5) * width, values, width=width, label=model_name)

    plt.xticks(x, feature_names)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.title("Model Coefficient Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_COEF, dpi=200)
    plt.close()


# ============================================================
# 7. OLS 与 Ridge 系数稳定性比较
# ============================================================

def coefficient_stability_experiment(df, n_repeats=60):
    X = df.drop(columns=["y"])
    y = df["y"]
    feature_names = X.columns.tolist()

    records = []

    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=seed
        )

        ols = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ])

        ols.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        ols_coef = ols.named_steps["model"].coef_
        ridge_coef = ridge.named_steps["model"].coef_

        for feature, coef in zip(feature_names, ols_coef):
            records.append({
                "重复编号": seed,
                "模型": "OLS",
                "变量": feature,
                "系数": coef
            })

        for feature, coef in zip(feature_names, ridge_coef):
            records.append({
                "重复编号": seed,
                "模型": "Ridge",
                "变量": feature,
                "系数": coef
            })

    return pd.DataFrame(records)


def plot_coefficient_stability(stability_df):
    features = sorted(stability_df["变量"].unique())
    model_names = ["OLS", "Ridge"]

    data = []
    labels = []
    positions = []

    pos = 1

    for feature in features:
        for model_name in model_names:
            subset = stability_df[
                (stability_df["变量"] == feature)
                & (stability_df["模型"] == model_name)
            ]["系数"].values

            data.append(subset)
            labels.append(f"{feature}\n{model_name}")
            positions.append(pos)
            pos += 1

        pos += 0.8

    plt.figure(figsize=(14, 6))
    plt.boxplot(data, positions=positions, widths=0.6, showfliers=False)
    plt.xticks(positions, labels, rotation=45)
    plt.ylabel("Coefficient")
    plt.title("Coefficient Stability: OLS vs Ridge")
    plt.tight_layout()
    plt.savefig(FIG_STABILITY, dpi=200)
    plt.close()


def summarize_stability(stability_df):
    summary = (
        stability_df
        .groupby(["模型", "变量"])["系数"]
        .std()
        .reset_index()
        .rename(columns={"系数": "系数标准差"})
    )

    wide = summary.pivot(index="变量", columns="模型", values="系数标准差").reset_index()

    if "OLS" in wide.columns and "Ridge" in wide.columns:
        wide["OLS/Ridge标准差比值"] = wide["OLS"] / wide["Ridge"]

    return wide


# ============================================================
# 8. 前向变量选择
# ============================================================

def forward_selection(df):
    X = df.drop(columns=["y"])
    y = df["y"]

    selected = []
    remaining = list(X.columns)
    records = []

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    step = 1

    while len(remaining) > 0:
        candidate_records = []

        for feature in remaining:
            current_features = selected + [feature]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])

            scores = cross_val_score(
                model,
                X[current_features],
                y,
                scoring="neg_root_mean_squared_error",
                cv=cv
            )

            mean_rmse = -scores.mean()

            candidate_records.append({
                "候选变量": feature,
                "交叉验证RMSE": mean_rmse
            })

        candidate_df = pd.DataFrame(candidate_records)
        best_row = candidate_df.sort_values("交叉验证RMSE").iloc[0]

        best_feature = best_row["候选变量"]
        best_rmse = best_row["交叉验证RMSE"]

        selected.append(best_feature)
        remaining.remove(best_feature)

        records.append({
            "步骤": step,
            "本轮选入变量": best_feature,
            "当前已选变量": ", ".join(selected),
            "交叉验证RMSE": best_rmse
        })

        step += 1

    return pd.DataFrame(records)


def plot_forward_selection(forward_df):
    plt.figure(figsize=(8, 5))
    plt.plot(
        forward_df["步骤"],
        forward_df["交叉验证RMSE"],
        marker="o"
    )
    plt.xlabel("Number of Selected Features")
    plt.ylabel("CV RMSE")
    plt.title("Forward Selection Curve")
    plt.tight_layout()
    plt.savefig(FIG_FORWARD, dpi=200)
    plt.close()


# ============================================================
# 9. 写中文报告
# ============================================================

def write_report(
    df,
    corr_df,
    cv_df,
    metric_df,
    best_models,
    coef_df,
    stability_summary_df,
    forward_df
):
    feature_names = df.drop(columns=["y"]).columns.tolist()

    best_param_records = []

    for name, model in best_models.items():
        final_model = model.named_steps["model"]

        best_param_records.append({
            "模型": name,
            "最优Alpha": getattr(final_model, "alpha", np.nan),
            "L1比例": getattr(final_model, "l1_ratio", np.nan)
        })

    best_param_df = pd.DataFrame(best_param_records)

    lasso_coef = coef_df[coef_df["模型"] == "Lasso"].copy()
    lasso_selected = lasso_coef[np.abs(lasso_coef["系数"]) > 1e-6]["变量"].tolist()

    forward_best = forward_df.sort_values("交叉验证RMSE").iloc[0]

    corr_table = corr_df.reset_index().rename(columns={"index": "变量"})

    lines = []

    lines.append("# 第13周作业报告：正则化回归与变量选择")
    lines.append("")
    lines.append("## 1. 作业说明")
    lines.append("")
    lines.append("本次作业主要围绕正则化回归和变量选择展开。")
    lines.append("我使用自己生成的模拟数据进行实验，数据中故意加入了较强的多重共线性。")
    lines.append("这样可以更清楚地观察 OLS、Ridge、Lasso、ElasticNet 这些模型之间的区别。")
    lines.append("")
    lines.append("本次主要完成了以下内容：")
    lines.append("")
    lines.append("1. 生成带有共线性的模拟回归数据；")
    lines.append("2. 比较 OLS、Ridge、Lasso 和 ElasticNet 的预测效果；")
    lines.append("3. 使用交叉验证选择正则化参数；")
    lines.append("4. 比较 OLS 和 Ridge 的系数稳定性；")
    lines.append("5. 使用 Lasso 和前向选择方法进行变量选择；")
    lines.append("6. 生成图像和总结报告。")
    lines.append("")
    lines.append("## 2. 数据说明")
    lines.append("")
    lines.append(f"样本数量：{df.shape[0]}")
    lines.append("")
    lines.append(f"自变量数量：{len(feature_names)}")
    lines.append("")
    lines.append("自变量包括：")
    lines.append("")
    lines.append(", ".join(feature_names))
    lines.append("")
    lines.append("因变量为：y")
    lines.append("")
    lines.append("其中，x1、x2、x3 之间高度相关，x4、x5 之间高度相关。")
    lines.append("这种设置主要是为了模拟实际建模中常见的多重共线性问题。")
    lines.append("")
    lines.append("数据文件保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/data/synthetic_correlated.csv")
    lines.append("```")
    lines.append("")
    lines.append("## 3. 自变量相关系数矩阵")
    lines.append("")
    lines.append(df_to_markdown(corr_table, float_digits=3))
    lines.append("")
    lines.append("从相关系数矩阵可以看出，部分变量之间的相关性比较强。")
    lines.append("比如 x1、x2、x3 之间相关性很高，x4 和 x5 之间相关性也很高。")
    lines.append("这说明数据中确实存在多重共线性。")
    lines.append("")
    lines.append("相关系数热力图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/figures/correlation_heatmap.png")
    lines.append("```")
    lines.append("")
    lines.append("## 4. 正则化模型参数选择")
    lines.append("")
    lines.append("本次使用 5 折交叉验证对 Ridge、Lasso 和 ElasticNet 进行参数选择。")
    lines.append("下面是每个模型最后选出的较优参数：")
    lines.append("")
    lines.append(df_to_markdown(best_param_df, float_digits=4))
    lines.append("")
    lines.append("Alpha 可以理解为正则化强度。")
    lines.append("Alpha 越大，模型对回归系数的压缩越明显。")
    lines.append("")
    lines.append("交叉验证曲线保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/figures/alpha_cv_curves.png")
    lines.append("```")
    lines.append("")
    lines.append("## 5. 不同模型预测效果比较")
    lines.append("")
    lines.append("下面是在测试集上的模型表现：")
    lines.append("")
    lines.append(df_to_markdown(metric_df, float_digits=4))
    lines.append("")
    lines.append("从结果来看，几个模型的预测误差整体差距不算特别大。")
    lines.append("但是这类问题不能只看预测误差，还要关注模型系数是否稳定，以及变量选择是否合理。")
    lines.append("")
    lines.append("## 6. 不同模型系数比较")
    lines.append("")
    lines.append("模型系数对比图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/figures/model_coefficients.png")
    lines.append("```")
    lines.append("")
    lines.append("从系数对比可以看出，OLS 在面对共线性变量时，系数可能会出现不稳定的情况。")
    lines.append("Ridge 会保留所有变量，但是会把系数压缩得更平稳。")
    lines.append("Lasso 会把部分变量的系数压缩为 0，因此它可以起到变量选择的作用。")
    lines.append("ElasticNet 结合了 Ridge 和 Lasso 的特点，属于一种折中方法。")
    lines.append("")
    lines.append("## 7. 系数稳定性分析")
    lines.append("")
    lines.append("为了观察系数稳定性，我多次随机划分训练集和测试集，比较 OLS 和 Ridge 的系数标准差。")
    lines.append("")
    lines.append(df_to_markdown(stability_summary_df, float_digits=4))
    lines.append("")
    lines.append("一般来说，系数标准差越小，说明模型系数越稳定。")
    lines.append("从结果可以看出，Ridge 的系数波动通常比 OLS 更小。")
    lines.append("这说明在存在共线性的情况下，Ridge 对提高模型稳定性有帮助。")
    lines.append("")
    lines.append("系数稳定性图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/figures/coefficient_stability.png")
    lines.append("```")
    lines.append("")
    lines.append("## 8. Lasso 变量选择结果")
    lines.append("")
    lines.append("Lasso 最终保留下来的变量为：")
    lines.append("")
    lines.append("```text")
    if len(lasso_selected) > 0:
        lines.append(", ".join(lasso_selected))
    else:
        lines.append("没有变量被明显保留")
    lines.append("```")
    lines.append("")
    lines.append("Lasso 的特点是可以把一些变量的系数直接压缩到 0。")
    lines.append("所以它不仅能做预测，也可以用于变量筛选。")
    lines.append("不过在变量之间高度相关时，Lasso 可能只保留其中一个代表变量。")
    lines.append("")
    lines.append("## 9. 前向变量选择结果")
    lines.append("")
    lines.append("前向变量选择的过程如下：")
    lines.append("")
    lines.append(df_to_markdown(forward_df, float_digits=4))
    lines.append("")
    lines.append("交叉验证 RMSE 最小的一步为：")
    lines.append("")
    lines.append(f"- 步骤：{int(forward_best['步骤'])}")
    lines.append(f"- 选中的变量：{forward_best['当前已选变量']}")
    lines.append(f"- 交叉验证 RMSE：{forward_best['交叉验证RMSE']:.4f}")
    lines.append("")
    lines.append("前向变量选择曲线保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/figures/forward_selection_curve.png")
    lines.append("```")
    lines.append("")
    lines.append("## 10. 总结")
    lines.append("")
    lines.append("通过本次实验可以看出，在存在多重共线性的情况下，OLS 虽然简单，但是系数容易不稳定。")
    lines.append("Ridge 的优点是可以提高系数稳定性，适合处理共线性问题。")
    lines.append("Lasso 的优点是可以进行变量选择，它能把一些不重要变量的系数压缩为 0。")
    lines.append("ElasticNet 同时结合了 L1 和 L2 正则化，因此在变量选择和稳定性之间做了折中。")
    lines.append("")
    lines.append("前向变量选择和 Lasso 都可以做变量筛选，但它们的思路不同。")
    lines.append("前向变量选择是一种逐步加入变量的方法，而 Lasso 是通过正则化自动压缩系数。")
    lines.append("在实际分析中，可以把这些方法结合起来看，而不是只依赖单一结果。")
    lines.append("")
    lines.append("## 11. 本次生成的文件")
    lines.append("")
    lines.append("```text")
    lines.append("src/week13/main.py")
    lines.append("src/week13/data/synthetic_correlated.csv")
    lines.append("src/week13/results/summary.md")
    lines.append("src/week13/figures/correlation_heatmap.png")
    lines.append("src/week13/figures/alpha_cv_curves.png")
    lines.append("src/week13/figures/model_coefficients.png")
    lines.append("src/week13/figures/coefficient_stability.png")
    lines.append("src/week13/figures/forward_selection_curve.png")
    lines.append("```")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 10. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("第13周作业：正则化回归与变量选择")
    print("=" * 70)

    print("\n[阶段1] 正在生成模拟数据...")
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"数据已保存：{DATA_PATH}")
    print(f"数据规模：{df.shape}")

    print("\n[阶段2] 正在计算相关系数并绘图...")
    corr_df = plot_correlation_heatmap(df)
    print(f"相关系数热力图已保存：{FIG_CORR}")

    print("\n[阶段3] 正在进行 Ridge、Lasso、ElasticNet 交叉验证调参...")
    best_models, cv_df = tune_regularized_models(df)
    plot_alpha_curves(cv_df)
    print(f"交叉验证曲线已保存：{FIG_ALPHA}")

    print("\n[阶段4] 正在比较 OLS、Ridge、Lasso、ElasticNet 的预测效果...")
    metric_df, fitted_models = compare_models(df, best_models)
    print(metric_df)

    print("\n[阶段5] 正在提取并绘制模型系数...")
    feature_names = df.drop(columns=["y"]).columns.tolist()
    coef_df = extract_coefficients(fitted_models, feature_names)
    plot_model_coefficients(coef_df)
    print(f"模型系数图已保存：{FIG_COEF}")

    print("\n[阶段6] 正在进行 OLS 与 Ridge 系数稳定性实验...")
    stability_df = coefficient_stability_experiment(df)
    stability_summary_df = summarize_stability(stability_df)
    plot_coefficient_stability(stability_df)
    print(f"系数稳定性图已保存：{FIG_STABILITY}")

    print("\n[阶段7] 正在进行前向变量选择...")
    forward_df = forward_selection(df)
    plot_forward_selection(forward_df)
    print(f"前向变量选择曲线已保存：{FIG_FORWARD}")

    print("\n[阶段8] 正在生成中文报告...")
    write_report(
        df=df,
        corr_df=corr_df,
        cv_df=cv_df,
        metric_df=metric_df,
        best_models=best_models,
        coef_df=coef_df,
        stability_summary_df=stability_summary_df,
        forward_df=forward_df
    )
    print(f"中文报告已保存：{REPORT_PATH}")

    print("\n全部任务完成！")
    print("\n生成文件如下：")
    print(f"1. {DATA_PATH}")
    print(f"2. {REPORT_PATH}")
    print(f"3. {FIG_CORR}")
    print(f"4. {FIG_ALPHA}")
    print(f"5. {FIG_COEF}")
    print(f"6. {FIG_STABILITY}")
    print(f"7. {FIG_FORWARD}")


if __name__ == "__main__":
    main()