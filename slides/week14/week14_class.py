# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# # Week 14：高维问题、共线性与降维回归
#
# **主题**：`PCA` / `PCR`
#
# **本节课主线**：
# - 为什么 `p > n` 和共线性会一起把 `OLS` 逼到危险地带？
# - `PCA` 到底在压缩什么？
# - `PCR` 为什么能在高维相关数据里稳住回归？
# - `Lasso` 和 `PCR` 分别更适合什么世界？
#
# **课堂规则**：
# - 先猜，再运行；
# - 先看现象，再命名；
# - 先做判断，再背术语。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## Prologue：今天我们不再只问“谁该留下”，而要问“信息该怎么压缩”
#
# 上周我们在原变量空间里讨论了 `Ridge`、`Lasso`、`Elastic Net`。
# 它们都在回答一个问题：
#
# > 当 `OLS` 不够稳时，我们怎么在原变量空间里加约束？
#
# 但现实里还有另一种很常见的情况：
#
# - 变量很多；
# - 变量彼此相关；
# - 我们掌握的信息，不足以对每一列都做出稳定而准确的估计。
#
# 这背后的原因通常有两类：
# - 要么是 `p > n`，每一列平均分到的信息太少；
# - 要么是共线性太强，几列变量在争夺同一份解释权。
#
# 这时，问题不一定是“删掉谁”，也可能是：
#
# > 要不要先换一个更紧凑的坐标系，再做回归？

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 本周的主线不要讲成“PCA 定义 + PCR 定义”。
# 更自然的说法是：
# - 原变量空间已经很拥挤；
# - 我们试着先压缩信息，再回归；
# - 最后再和 Lasso 做一次世界观对比。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti"]

RANDOM_SEED = 14


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_high_dimensional_data(
    n_samples=90,
    n_features=120,
    n_latent=6,
    noise_x=0.55,
    noise_y=1.0,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, n_latent))
    loadings = rng.normal(size=(n_latent, n_features))
    X = latent @ loadings + rng.normal(scale=noise_x, size=(n_samples, n_features))
    latent_coef = np.array([3.0, -2.2, 1.6, 0.0, 0.0, 0.0])[:n_latent]
    y = latent @ latent_coef + rng.normal(scale=noise_y, size=n_samples)
    return X, y


def make_collinearity_demo(strength="strong", n_samples=150, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n_samples)
    if strength == "strong":
        eps = 0.12
    else:
        eps = 0.95

    x1 = base + rng.normal(scale=eps, size=n_samples)
    x2 = 0.96 * base + rng.normal(scale=eps, size=n_samples)
    x3 = 0.92 * base + rng.normal(scale=eps, size=n_samples)
    x4 = rng.normal(size=n_samples)
    x5 = rng.normal(size=n_samples)
    x6 = rng.normal(size=n_samples)
    X = np.column_stack([x1, x2, x3, x4, x5, x6])
    beta = np.array([2.6, 0.0, 0.0, 1.0, 0.0, 0.0])
    y = X @ beta + rng.normal(scale=1.2, size=n_samples)
    return X, y


def repeated_ols_coefficients(X, y, n_repeats=60):
    records = []
    cond_numbers = []
    for split_seed in range(n_repeats):
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=split_seed
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        singular_values = np.linalg.svd(X_train_scaled, full_matrices=False)[1]
        cond_numbers.append(singular_values[0] / singular_values[-1])
        for feature_idx, coef in enumerate(model.coef_):
            records.append(
                {
                    "split": split_seed,
                    "feature": f"x{feature_idx + 1}",
                    "coefficient": coef,
                }
            )
    return pd.DataFrame(records), float(np.mean(cond_numbers))


def make_two_dimensional_cloud(n_samples=180, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n_samples)
    z2 = 0.35 * z1 + rng.normal(scale=0.55, size=n_samples)
    x1 = 1.9 * z1 + 0.2 * z2
    x2 = 1.4 * z1 + 1.1 * z2
    return np.column_stack([x1, x2])


def draw_pc_arrow(ax, start, vector, color, label):
    arrow = FancyArrowPatch(
        posA=start,
        posB=start + vector,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=3,
        color=color,
    )
    ax.add_patch(arrow)
    text_pos = start + 1.08 * vector
    ax.text(text_pos[0], text_pos[1], label, color=color, fontsize=12, weight="bold")


class PCRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.pca_ = PCA(n_components=self.n_components)
        scores = self.pca_.fit_transform(X_scaled)
        self.reg_ = LinearRegression()
        self.reg_.fit(scores, y)
        self.coef_ = (
            self.pca_.components_.T @ self.reg_.coef_
        ) / self.scaler_.scale_
        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        scores = self.pca_.transform(X_scaled)
        return self.reg_.predict(scores)


def evaluate_dimension_blowup():
    rows = []
    n_samples = 84
    feature_grid = [12, 36, 72, 140]
    for n_features in feature_grid:
        X, y = make_high_dimensional_data(
            n_samples=n_samples,
            n_features=n_features,
            n_latent=6,
            noise_x=0.6,
            noise_y=1.1,
            seed=100 + n_features,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        singular_values = np.linalg.svd(X_train_scaled, full_matrices=False)[1]
        rank = int(np.linalg.matrix_rank(X_train_scaled))
        smallest_sv = singular_values[-1]
        if smallest_sv < 1e-10:
            cond_text = "very large"
        else:
            cond_text = f"{singular_values[0] / smallest_sv:,.0f}"

        rows.append(
            {
                "p": n_features,
                "n_train": X_train.shape[0],
                "p_over_n_train": round(n_features / X_train.shape[0], 2),
                "rank(X_train)": rank,
                "train_rmse": rmse(y_train, model.predict(X_train_scaled)),
                "test_rmse": rmse(y_test, model.predict(X_test_scaled)),
                "condition_number": cond_text,
            }
        )
    return pd.DataFrame(rows)


def pcr_curve_data():
    X, y = make_high_dimensional_data(
        n_samples=90,
        n_features=120,
        n_latent=6,
        noise_x=0.55,
        noise_y=1.0,
        seed=RANDOM_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )

    component_grid = list(range(1, 26))
    results = []
    for n_components in component_grid:
        model = PCRRegressor(n_components=n_components)
        model.fit(X_train, y_train)
        results.append(
            {
                "n_components": n_components,
                "train_rmse": rmse(y_train, model.predict(X_train)),
                "test_rmse": rmse(y_test, model.predict(X_test)),
            }
        )

    ols = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )
    ols.fit(X_train, y_train)
    ols_train_rmse = rmse(y_train, ols.predict(X_train))
    ols_test_rmse = rmse(y_test, ols.predict(X_test))

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        PCRRegressor(),
        param_grid={"n_components": component_grid},
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    search.fit(X_train, y_train)
    cv_df = pd.DataFrame(
        {
            "n_components": component_grid,
            "cv_rmse": -search.cv_results_["mean_test_score"],
        }
    )

    return (
        pd.DataFrame(results),
        cv_df,
        search.best_params_["n_components"],
        ols_train_rmse,
        ols_test_rmse,
    )


def make_sparse_signal_data(
    n_samples=120,
    n_features=80,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[[1, 7, 19, 42]] = [3.0, -2.6, 2.0, 1.5]
    y = X @ beta + rng.normal(scale=1.1, size=n_samples)
    return X, y, beta


def compare_lasso_and_pcr(make_data_fn, scenario_name, n_splits=10):
    rows = []
    coef_records = []
    for split_seed in range(n_splits):
        X, y, _ = make_data_fn(seed=400 + split_seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=split_seed
        )

        lasso = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LassoCV(
                        cv=5,
                        alphas=np.logspace(-3, 1, 25),
                        max_iter=20000,
                        random_state=split_seed,
                    ),
                ),
            ]
        )
        lasso.fit(X_train, y_train)
        lasso_coef = lasso.named_steps["model"].coef_ / lasso.named_steps["scaler"].scale_
        rows.append(
            {
                "scenario": scenario_name,
                "method": "Lasso",
                "split": split_seed,
                "test_rmse": rmse(y_test, lasso.predict(X_test)),
                "model_size": int(np.sum(np.abs(lasso.named_steps["model"].coef_) > 1e-6)),
            }
        )
        coef_records.append(
            {
                "scenario": scenario_name,
                "method": "Lasso",
                "split": split_seed,
                "coef_vector": lasso_coef,
            }
        )

        pcr_search = GridSearchCV(
            PCRRegressor(),
            param_grid={"n_components": list(range(1, 13))},
            scoring="neg_root_mean_squared_error",
            cv=5,
        )
        pcr_search.fit(X_train, y_train)
        pcr = pcr_search.best_estimator_
        rows.append(
            {
                "scenario": scenario_name,
                "method": "PCR",
                "split": split_seed,
                "test_rmse": rmse(y_test, pcr.predict(X_test)),
                "model_size": int(pcr_search.best_params_["n_components"]),
            }
        )
        coef_records.append(
            {
                "scenario": scenario_name,
                "method": "PCR",
                "split": split_seed,
                "coef_vector": pcr.coef_,
            }
        )

    metric_df = pd.DataFrame(rows)
    coef_rows = []
    for record in coef_records:
        coef_rows.append(
            {
                "scenario": record["scenario"],
                "method": record["method"],
                "split": record["split"],
                "coef_vector": np.asarray(record["coef_vector"]),
            }
        )
    coef_df = pd.DataFrame(coef_rows)
    return metric_df, coef_df


def summarize_comparison(metric_df, coef_df):
    summary_rows = []
    for (scenario, method), subset in metric_df.groupby(["scenario", "method"]):
        coef_matrix = np.vstack(
            coef_df.loc[
                (coef_df["scenario"] == scenario) & (coef_df["method"] == method),
                "coef_vector",
            ].to_list()
        )
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "mean_test_rmse": subset["test_rmse"].mean(),
                "sd_test_rmse": subset["test_rmse"].std(ddof=1),
                "avg_model_size": subset["model_size"].mean(),
                "coef_instability": np.mean(np.std(coef_matrix, axis=0)),
            }
        )
    return pd.DataFrame(summary_rows)


# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第一幕：维度一高，`OLS` 还能不能信？
#
# 这一幕先不急着谈 `PCA`。
# 先回答更基础的问题：
#
# > 当特征数不断上升，尤其开始逼近甚至超过训练样本数时，`OLS` 会发生什么？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜一个趋势：
#
# 如果我们固定样本量，只让特征数 `p` 越来越大，你觉得哪件事更可能发生？
#
# 1. 训练误差和测试误差都会稳定下降；
# 2. 训练误差会越来越漂亮，但泛化未必跟着更好。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先看一个极端例子：
#
# - 一元回归里，如果训练集只有 `n = 1` 个点；
# - 我们总能找到一条直线，把这个点“拟合得完美无缺”；
# - 于是训练误差一定是 0。
#
# 但这并不说明模型学到了规律。
# 它只说明：**约束太少时，模型可以轻易把训练集讲圆。**

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：随着 `p` 上升，`OLS` 的 train/test 误差与矩阵秩变化
#
# 这张图要回答：
# > `OLS` 的危险到底是“算不出来”，还是“太容易给出一个看起来很完美的答案”？

# %% slideshow={"slide_type": "skip"} tags=["skip"]
dimension_df = evaluate_dimension_blowup()

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].plot(dimension_df["p"], dimension_df["train_rmse"], marker="o", label="train RMSE")
axes[0].plot(dimension_df["p"], dimension_df["test_rmse"], marker="o", label="test RMSE")
axes[0].axvline(dimension_df["n_train"].iloc[0], color="#64748b", linestyle="--", linewidth=1.6)
axes[0].text(
    dimension_df["n_train"].iloc[0] + 2,
    axes[0].get_ylim()[1] * 0.92,
    "p = n_train",
    color="#475569",
)
axes[0].set_title("维度上升时，训练误差会越来越乐观")
axes[0].set_xlabel("number of features p")
axes[0].set_ylabel("RMSE")
axes[0].legend()

axes[1].plot(dimension_df["p"], dimension_df["rank(X_train)"], marker="o", color="#0f766e")
axes[1].axhline(dimension_df["n_train"].iloc[0], color="#64748b", linestyle="--", linewidth=1.6)
for _, row in dimension_df.iterrows():
    axes[1].text(
        row["p"],
        row["rank(X_train)"] + 1.5,
        f"cond={row.condition_number}",
        ha="center",
        fontsize=9,
    )
axes[1].set_title(f"rank 的上限是 min(n_train, p)；这里 n_train = {dimension_df['n_train'].iloc[0]}")
axes[1].set_xlabel("number of features p")
axes[1].set_ylabel("rank of standardized X_train")

fig.suptitle("`p > n` 时，OLS 往往更像是在记住训练样本，而不是稳稳学到规律", y=1.02)
plt.tight_layout()
plt.show()

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
dimension_df

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - `p > n` 时，`OLS` 不一定报错，但自由度会非常大；
# - 训练误差接近 0，不代表模型已经“学得很好”；
# - 问题的核心不是能不能拟合，而是拟合是否稳定、是否可泛化。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **这里顺手补两个词**
#
# - `condition number`：最大奇异值与最小奇异值之比
#   $$
#   \kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}
#   $$
#   它越大，说明设计矩阵越接近“某些方向上快要塌掉”。
# - `自由度`：这里可以先直观理解成“模型可以独立摆动的方向数”。
#
# 当 `p` 上升、而样本量没跟上时，可摆动的方向越来越多，但每个方向获得的证据越来越少。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 这里看到的风险，可以用 $OLS$ 的标准写法来表达：
#
# $$
# \hat{\beta}_{OLS} = (X^\top X)^{-1} X^\top y
# $$
#
# 当 $p$ 逼近甚至超过 $n$ 时，$X^\top X$ 更容易：
#
# - 不可逆；
# - 或者虽然可逆，但病态得非常严重。
#
# 同时还要记住一个简单关系：
# - $rank(X^\top X) = rank(X)$；
# - 所以一旦 $rank(X) < p$，$X^\top X$ 就不可能满秩。
#
# 这时图里看到的“训练误差很低、但模型不稳”，就对应着：
# 系数解不唯一，或者对样本扰动极度敏感。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 先让大家从图上形成危机感，再补公式。
# 这里不要展开 Moore-Penrose 伪逆细节，而是做桥接：
# - 训练集几乎被“讲圆了”；
# - 这恰恰可能是危险信号，而不是安全信号；
# - 接着指出：图上的危险，在公式里对应的是 $X^\top X$ 不可逆或病态。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第二幕：共线性为什么会破坏解释？
#
# 上一幕强调的是“变量太多”。
# 这一幕强调的是“信息太重复”。
#
# > 就算 `p` 还没有超过 `n`，只要变量彼此强相关，`OLS` 也会开始摇摆。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 如果三个变量几乎都在描述同一件事，你觉得 `OLS` 更可能：
#
# 1. 很稳定地把功劳分给它们；
# 2. 在不同样本切分下，反复改变谁来“扛主力”。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 回忆一下上周的 `Ridge`：
#
# - 它不是把某一列直接删掉；
# - 而是通过 `L2 penalty` 让几列相关变量更倾向于**平滑地共享功劳**；
# - 所以在“谁该留下”之外，`Ridge` 更关心“怎样让估计更稳定”。
#
# 这一幕可以把 `OLS` 和 `Ridge` 的气质先区分开：
# `OLS` 更容易在相关变量之间剧烈摇摆，`Ridge` 更愿意保守地平均分担。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：弱相关 vs 强相关 场景下，`OLS` 系数跨切分波动对比
#
# 这张图要回答：
# > 共线性破坏的，到底是误差、系数，还是解释本身？

# %% slideshow={"slide_type": "skip"} tags=["skip"]
X_weak, y_weak = make_collinearity_demo(strength="weak")
X_strong, y_strong = make_collinearity_demo(strength="strong")
coef_weak_df, weak_cond = repeated_ols_coefficients(X_weak, y_weak)
coef_strong_df, strong_cond = repeated_ols_coefficients(X_strong, y_strong)

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
focus_features = ["x1", "x2", "x3", "x4"]

for ax, coef_df, title, cond_value, facecolor in [
    (axes[0], coef_weak_df, "弱相关：谁重要相对更稳定", weak_cond, "#dbeafe"),
    (axes[1], coef_strong_df, "强相关：几个变量开始互相抢解释权", strong_cond, "#fee2e2"),
]:
    data = [
        coef_df.loc[coef_df["feature"] == feature, "coefficient"].to_numpy()
        for feature in focus_features
    ]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for box in bp["boxes"]:
        box.set(facecolor=facecolor, alpha=0.8)
    for median in bp["medians"]:
        median.set(color="#0f172a", linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(focus_features) + 1))
    ax.set_xticklabels(focus_features)
    ax.set_title(title)
    ax.text(0.04, 0.95, f"avg cond number ≈ {cond_value:,.1f}", transform=ax.transAxes, va="top")

axes[0].set_ylabel("OLS coefficient across repeated splits")
fig.suptitle("共线性最先破坏的，往往不是“能不能拟合”，而是“解释是否稳定”", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要理解的是：
#
# - 多重共线性会放大系数方差，让“谁更重要”这件事变得摇摆；
# - 高维和共线性不是同一个原因，但它们常常同时出现，并共同削弱原变量空间里的稳定估计。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 如果仍然写成
#
# $$
# \hat{\beta}_{OLS} = (X^\top X)^{-1} X^\top y
# $$
#
# 那么强共线性对应的是：
#
# - $X$ 的列彼此接近线性相关；
# - $X^\top X$ 虽然可能可逆，但条件数会很大；
# - 因而 $\hat{\beta}$ 对样本切分和噪声非常敏感。
#
# 所以图里看到的“几个变量互相抢解释权”，本质上就是矩阵病态带来的系数不稳定。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这一幕的桥接重点不是再说一遍“共线性不好”，而是帮大家接上：
# - 有时预测误差没坏得那么明显；
# - 但参数解释已经先碎掉了；
# - 在公式里，这不是玄学，而是 $X^\top X$ 条件数变大。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第三幕：`PCA` 在压缩什么？
#
# 既然原变量空间里充满重复信息，一个自然问题就是：
#
# > 我们能不能先换个坐标系，让主要信息沿着更少的方向表达？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 看二维图之前先猜：
#
# 如果一团点呈现出明显的“斜着拉长”的椭圆形，你觉得最值得保留的方向会是：
#
# 1. 沿着最长扩散方向；
# 2. 沿着最窄、最安静的方向。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：二维相关数据 + 主成分方向
#
# 这张图要回答：
# > 主成分不是在挑变量，而是在找“变化最大的信息方向”。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
cloud = make_two_dimensional_cloud()
cloud_scaled = StandardScaler().fit_transform(cloud)
pca_2d = PCA(n_components=2)
pca_2d.fit(cloud_scaled)
pc_vectors = pca_2d.components_
pc_scales = np.sqrt(pca_2d.explained_variance_) * 2.6

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, ax = plt.subplots(figsize=(8.5, 7))
ax.scatter(cloud_scaled[:, 0], cloud_scaled[:, 1], alpha=0.35, color="#2563eb", s=26)
origin = np.array([0.0, 0.0])
draw_pc_arrow(ax, origin, pc_vectors[0] * pc_scales[0], "#dc2626", "PC1")
draw_pc_arrow(ax, origin, pc_vectors[1] * pc_scales[1], "#16a34a", "PC2")
ax.axhline(0, color="#94a3b8", linewidth=1)
ax.axvline(0, color="#94a3b8", linewidth=1)
ax.set_title("PCA 先找的是数据变化最大的方向，而不是某一列变量")
ax.set_xlabel("standardized x1")
ax.set_ylabel("standardized x2")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# $PCA$ 的一个关键词是：
#
# $projection$
#
# 它做的不是“删掉某一列”，而是把原始变量投影到一组新的正交方向上。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **先把算法步骤说清楚**
#
# 一个最朴素的 `PCA` 过程可以写成：
#
# 1. 先标准化各列变量；
# 2. 计算协方差结构；
# 3. 找到投影方差最大的方向作为 `PC1`；
# 4. 在与 `PC1` 正交的约束下，继续找 `PC2`、`PC3`；
# 5. 用前几个主成分来近似原来的高维数据。
#
# 所以 `PCA` 不是“随便取平均”，而是在系统地寻找最能保留变化的方向。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕辅图**：高维低秩数据中的累计解释方差
#
# 这张图要把二维直觉迁移到高维：
# > 如果前几个主成分已经解释了大部分方差，就说明原数据虽然住在高维空间里，但主要变化其实贴近一个更低维的平面或子空间。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
X_latent, _ = make_high_dimensional_data(
    n_samples=140,
    n_features=60,
    n_latent=6,
    noise_x=0.5,
    noise_y=1.0,
    seed=222,
)
X_latent_scaled = StandardScaler().fit_transform(X_latent)
pca_full = PCA().fit(X_latent_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(np.arange(1, 21), cum_var[:20], marker="o", color="#7c3aed")
ax.axhline(0.8, color="#64748b", linestyle="--", linewidth=1.5)
ax.axhline(0.9, color="#94a3b8", linestyle="--", linewidth=1.5)
ax.set_title("前几个主成分就能吃下大部分方差，说明原变量中有大量重复信息")
ax.set_xlabel("number of principal components kept")
ax.set_ylabel("cumulative explained variance ratio")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - `PCA` 解决的是“如何用更少的方向保留主要信息”；
# - 它优先保留的是 `X` 中方差大的方向；
# - 所以 `PCA` 还不是回归，它只是为后面的 `PCR` 准备了一个低维表示。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 在严格定义里，第一主成分方向 $v_1$ 满足：
#
# $$
# v_1 = \arg\max_{\|v\|=1} \mathrm{Var}(Xv)
# $$
#
# 后续主成分继续满足：
#
# - 仍然最大化投影方差；
# - 彼此正交。
#
# 所以刚才图里的“最长扩散方向”，在数学上就是：
# 在所有单位方向里，投影方差最大的那个方向。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 先用图让大家说出“最长方向”，再补优化定义。
# 这里很适合明确区分两件事：
# - 方差大，不一定就对 y 最有用；
# - 但如果许多列在重复表达少数因子，PCA 至少能先把拥挤空间整理干净；
# - 严格说，这个“整理”就是寻找最大投影方差的正交方向组。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第四幕：`PCR` 到底做了什么？
#
# 有了 `PCA` 之后，一个自然流程就是：
#
# 1. 先标准化；
# 2. 做 `PCA`；
# 3. 保留前 `k` 个主成分；
# 4. 在这些主成分上做线性回归。
#
# 这就是 `PCR`。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜：
#
# 如果 $PCR$ 里的主成分个数 $k$ 从 1 一直加到很多，你觉得测试误差更可能：
#
# 1. 单调下降；
# 2. 先下降，后面可能又把噪声带回来。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：`PCR` 随主成分个数变化的 train/test/CV 误差曲线
#
# 这张图要回答：
# > `PCR` 的关键调参对象不是“留哪个变量”，而是“保留多少个方向”。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
pcr_results_df, pcr_cv_df, best_k, ols_train_rmse, ols_test_rmse = pcr_curve_data()

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, ax = plt.subplots(figsize=(10.5, 6))
ax.plot(
    pcr_results_df["n_components"],
    pcr_results_df["train_rmse"],
    marker="o",
    label="PCR train RMSE",
    color="#2563eb",
)
ax.plot(
    pcr_results_df["n_components"],
    pcr_results_df["test_rmse"],
    marker="o",
    label="PCR test RMSE",
    color="#dc2626",
)
ax.plot(
    pcr_cv_df["n_components"],
    pcr_cv_df["cv_rmse"],
    marker="o",
    linestyle="--",
    label="PCR CV RMSE",
    color="#16a34a",
)
ax.axhline(ols_test_rmse, color="#7c2d12", linestyle=":", linewidth=2, label=f"OLS test RMSE = {ols_test_rmse:.2f}")
ax.axhline(ols_train_rmse, color="#9a3412", linestyle="--", linewidth=1.8, label=f"OLS train RMSE = {ols_train_rmse:.2f}")
ax.axvline(best_k, color="#475569", linestyle="--", linewidth=1.6)
ax.text(best_k + 0.4, ax.get_ylim()[1] * 0.95, f"best CV k = {best_k}", color="#334155")
ax.set_title("PCR 不是主成分越多越好，而是在“保留结构”和“带回噪声”之间找平衡")
ax.set_xlabel("number of principal components")
ax.set_ylabel("RMSE")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 这张图最值得课堂上反复强调的，是这三点：
#
# - `OLS` 在原始高维空间里可以把训练误差压得很低；
# - `PCR` 用少量主成分就能抓住主要结构，测试误差反而更稳；
# - 成分太多时，后面的方向也会把噪声重新带回来。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **怎么读 `PCR CV RMSE` 这条线？**
#
# - 它是交叉验证下的平均测试误差，用来估计“如果换一批新样本，模型大概会表现怎样”；
# - 它不像 train RMSE 那样容易被“贴着训练集讲故事”迷惑；
# - 它和 test RMSE 的趋势接近，说明 `PCR` 的优劣主要确实取决于保留多少个主成分。
#
# 这张图里 `OLS` 的 train error 很低，甚至接近 0，并不是因为它更聪明，
# 而是因为原始高维空间给了它过多自由度。
# `CV` 在这里的主要作用，就是帮我们选择更合适的主成分个数 `k`。

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# $PCR$ 的本质不是变量筛选，而是：
#
# > 先做 `information compression`，再回归。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# $PCR$ 的流程可以写成：
#
# 1. 先把标准化后的 $X$ 投影到前 $k$ 个主成分上，得到
#    $$
#    Z_k = X V_k
#    $$
# 2. 再做线性回归：
#    $$
#    y = Z_k \gamma + \varepsilon
#    $$
#
# 也就是说，$PCR$ 并不是在原变量 $X_1, \ldots, X_p$ 上直接筛选，
# 而是在低维表示 $Z_k$ 上回归。
#
# 所以图里“成分个数 `k` 的选择”，在数学上对应的正是：
# 要保留多少个主成分方向进入回归。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这里要补足“图 -> 公式 -> 判断”的桥：
# - Lasso 给的是变量名单；
# - PCR 给的是低维表示；
# - 公式里出现的是 $Z_k = X V_k$，不是“原始第几列留下来”。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第五幕：`Lasso` vs `PCR`，你到底想解决哪一种问题？
#
# 到这里，终于可以把这周和上周接起来了。
#
# 真正要问的不是“谁更高级”，而是：
#
# > 数据的真实结构更像“少数原始变量直接有效”，还是“原始变量主要是由少数潜在因子的线性组合生成出来的”？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先做一个方法下注：
#
# 如果真实世界是“只有少数原始变量直接有用”，你更偏向 `Lasso` 还是 `PCR`？
#
# 如果真实世界是“很多变量一起表达几个潜在因子”，你又会偏向谁？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：两种数据世界里，`Lasso` 与 `PCR` 的表现对比
#
# 我们直接造两个世界：
#
# 1. `Sparse truth`：只有少数原始变量真正有用；
# 2. `Latent-factor truth`：原始变量主要由少数潜在因子线性组合生成。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
latent_metric_df, latent_coef_df = compare_lasso_and_pcr(
    lambda seed: (*make_high_dimensional_data(n_samples=120, n_features=80, n_latent=6, noise_x=0.55, noise_y=1.0, seed=seed), np.zeros(80)),
    scenario_name="Latent-factor truth",
)
sparse_metric_df, sparse_coef_df = compare_lasso_and_pcr(
    make_sparse_signal_data,
    scenario_name="Sparse truth",
)
comparison_metric_df = pd.concat([latent_metric_df, sparse_metric_df], ignore_index=True)
comparison_coef_df = pd.concat([latent_coef_df, sparse_coef_df], ignore_index=True)
comparison_summary_df = summarize_comparison(comparison_metric_df, comparison_coef_df)

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
scenario_order = ["Sparse truth", "Latent-factor truth"]
for ax, scenario in zip(axes, scenario_order):
    subset = comparison_summary_df.loc[comparison_summary_df["scenario"] == scenario].copy()
    subset = subset.set_index("method").loc[["Lasso", "PCR"]].reset_index()
    ax.bar(
        subset["method"],
        subset["mean_test_rmse"],
        yerr=subset["sd_test_rmse"],
        color=["#2563eb", "#dc2626"],
        alpha=0.85,
        capsize=6,
    )
    ax.set_title(scenario)
    ax.set_ylabel("mean test RMSE across repeated splits")
    for row in subset.itertuples():
        ax.text(
            row.method,
            row.mean_test_rmse + row.sd_test_rmse + 0.05,
            f"size={row.avg_model_size:.1f}",
            ha="center",
            fontsize=10,
        )

fig.suptitle("方法优劣会随“真实结构”改变：selection 和 compression 不是同一个问题", y=1.02)
plt.tight_layout()
plt.show()

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
comparison_summary_df.round(
    {
        "mean_test_rmse": 3,
        "sd_test_rmse": 3,
        "avg_model_size": 2,
        "coef_instability": 3,
    }
)

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 这张表里最值得读的，不只是预测误差，还有两个维度：
#
# - `avg_model_size`
#   - 对 `Lasso`，它表示平均留下了多少个非零系数；
#   - 对 `PCR`，它表示平均保留了多少个主成分。
# - `coef_instability`
#   - 它这里定义为：先对每个系数看“跨切分的标准差”，再对所有系数取平均；
#   - 数值越大，说明同一方法在不同切分下给出的系数更容易波动。

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - `Lasso` 更适合回答：**谁留下？**
# - `PCR` 更适合回答：**先把重复信息压缩掉，再预测会不会更稳？**

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 对应到真实场景：
#
# - 基因表达、文本词项、传感器阵列：往往更容易出现高维和强相关；
# - 如果你更需要短名单，`Lasso` 更自然；
# - 如果你更需要把成团的信息压缩成稳定表征，`PCR` 往往更顺手。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## Synthesis
#
# 今天的整节课，其实都在回答同一个问题：
#
# > 当原变量空间已经拥挤、重复、摇摆时，我们该在原空间里筛选，还是先换坐标、再回归？
#
# 到这里，大家需要记住 5 个判断：
#
# 1. `p > n` 会让 `OLS` 更容易把训练集讲圆，却更难保证泛化；
# 2. 共线性会让系数解释摇摆，哪怕预测误差还没立刻爆炸；
# 3. `PCA` 做的是坐标变换与信息压缩；
# 4. `PCR` 做的是先压缩、再回归，不是直接给变量名单；
# 5. `Lasso` 和 `PCR` 的差异，本质上是 `selection` vs `compression`。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## Transition
#
# 今天我们一直在默认：先把 `y` 当作旁观者，只根据 `X` 的结构来整理特征空间。
#
# 但一个自然问题是：
#
# > 如果某个方差很大的方向，其实和 `y` 没那么相关，怎么办？
#
# 这就把我们带向下一个更大的主题：
#
# > 当响应变量不再只是连续数值，而是类别标签时，我们该怎样直接建模 $P(y \mid X)$？
#
# 这会把我们带到下一周：`广义线性模型（GLM）核心：逻辑回归`。
