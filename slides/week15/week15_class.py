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
# # Week 15：广义线性模型（GLM）核心
#
# **主题**：`Logistic Regression`
#
# **本节课主线**：
# - 为什么二分类任务不能直接把 `OLS` 硬套上去？
# - `sigmoid`、`Bernoulli` 和 `MLE` 是怎样自然连起来的？
# - `log-odds` 为什么让逻辑回归的系数变得可解释？
# - 为什么分类问题不能只看 `accuracy`？
# - 为什么正则化逻辑回归在工业界一直不过时？
#

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## 目录
#
# 这节课一共有 6 幕：
#
# 1. 为什么分类不能直接拿 `OLS` 硬做？
# 2. 如果要预测概率，`sigmoid` 为什么自然？
# 3. 为什么这里自然会走到 `Bernoulli + MLE`？
# 4. `log-odds` 到底是什么，系数为什么能解释？
# 5. 分类指标为什么不能只看 `accuracy`？
# 6. 正则化逻辑回归为什么一直不过时？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 这 6 幕之间的关系可以压成一句话：
#
# > 我们先证明“普通回归不适合直接做分类”，
# > 再一步步回答：如果还想保留线性结构，
# > 输出该怎么改、分布该怎么改、目标函数该怎么改、解释该怎么改，
# > 最后再讨论工业界为什么仍然离不开逻辑回归。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## Prologue：这节课真正要讨论的，不是“会不会分类”，而是“怎么正确地建模概率”
#
# 上周我们还在做连续响应下的回归问题。
# 但现实里还有一大类任务，响应变量不是一个连续数值，而是：
#
# - 是否违约；
# - 是否流失；
# - 是否点击；
# - 是否患病。
#
# 这些任务的共同点是：
# - 响应变量常常只有 `0/1`；
# - 我们更关心“发生的概率”；
# - 模型不仅要给出分类结果，还要给出可解释的风险信号。
#
# 所以如果还想沿用回归框架，直觉上只有两条路：
# - 要么直接用线性模型去拟合 $y \in \{0,1\}$；
# - 要么先回归出一个连续值，再把这个连续值硬解释成概率。
#
# 这两条路听起来都很顺手，但其实都埋着问题。
#
# 所以这节课的核心不是“又学一个新算法”，而是：
#
# > 当响应变量变成 `0/1` 时，回归框架要怎样改造，才既保留线性结构，又符合概率建模？

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 开场不要急着说“logistic regression”定义。
# 更自然的开法是：
# - 先从业务里常见的 `0/1` 决策说起；
# - 再强调：我们不仅要分对，还要给出靠谱的概率；
# - 这样 sigmoid、Bernoulli、MLE 就会显得是“被逼出来的”，而不是凭空出现。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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

RANDOM_SEED = 15


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def safe_prob(p, eps=1e-6):
    return np.clip(p, eps, 1 - eps)


def logit(p):
    p_safe = safe_prob(p)
    return np.log(p_safe / (1 - p_safe))


def make_binary_curve_data(n_samples=240, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=n_samples)
    eta = -0.6 + 1.8 * x
    p = sigmoid(eta)
    y = rng.binomial(1, p)
    return x.reshape(-1, 1), y, p


def make_churn_like_data(n_samples=520, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    tenure = rng.uniform(0, 10, size=n_samples)
    service_calls = rng.poisson(1.8, size=n_samples)
    discount = rng.binomial(1, 0.4, size=n_samples)
    eta = -1.5 - 0.32 * tenure + 0.75 * service_calls + 0.95 * discount
    p = sigmoid(eta)
    y = rng.binomial(1, p)
    X = np.column_stack([tenure, service_calls, discount])
    columns = ["tenure", "service_calls", "discount_flag"]
    return pd.DataFrame(X, columns=columns), y


def make_high_dimensional_classification_data(
    n_samples=320,
    n_features=28,
    n_latent=5,
    noise_x=0.6,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, n_latent))
    loadings = rng.normal(size=(n_latent, n_features))
    X = latent @ loadings + rng.normal(scale=noise_x, size=(n_samples, n_features))
    gamma = np.array([1.7, -1.4, 1.0, 0.0, 0.0])[:n_latent]
    eta = latent @ gamma
    p = sigmoid(eta)
    y = rng.binomial(1, p)
    columns = [f"x{j + 1}" for j in range(n_features)]
    return pd.DataFrame(X, columns=columns), y


def threshold_metrics(y_true, probas, thresholds):
    rows = []
    for threshold in thresholds:
        pred = (probas >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, pred),
                "recall": recall_score(y_true, pred),
            }
        )
    return pd.DataFrame(rows)


def summarize_classifier(name, y_true, probas, threshold=0.5, model_size=None):
    pred = (probas >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, probas),
        "log_loss": log_loss(y_true, safe_prob(probas)),
        "model_size": model_size,
    }


# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第一幕：分类问题为什么不能直接拿 `OLS` 硬做？
#
# 这一幕先不急着定义逻辑回归。
# 先让大家对一件事感到不舒服：
#
# > 如果标签只有 `0/1`，普通线性回归到底哪里不自然？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜：
#
# 如果我们把一个二分类任务直接交给 `OLS`，你觉得最先出问题的会是：
#
# 1. 预测值可能跑出 `[0,1]`；
# 2. 模型完全不能分开两类；
# 3. 两者都会。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：同一份 `0/1` 数据上，`OLS` 预测值与逻辑回归概率曲线对比
#
# 这张图要回答：
# > 即使线性回归偶尔能“看起来分得开”，为什么它仍然不适合直接充当概率模型？

# %% slideshow={"slide_type": "skip"} tags=["skip"]
X_curve, y_curve, _ = make_binary_curve_data()
ols_curve = LinearRegression().fit(X_curve, y_curve)
logit_curve = LogisticRegression().fit(X_curve, y_curve)
x_grid = np.linspace(X_curve.min() - 0.2, X_curve.max() + 0.2, 300).reshape(-1, 1)
ols_pred = ols_curve.predict(x_grid)
logit_pred = logit_curve.predict_proba(x_grid)[:, 1]

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, ax = plt.subplots(figsize=(10.5, 6))
jitter = np.where(y_curve == 1, 0.04, -0.04)
ax.scatter(
    X_curve[:, 0],
    y_curve + jitter,
    alpha=0.35,
    color="#64748b",
    s=26,
    label="observed 0/1 labels",
)
ax.plot(x_grid[:, 0], ols_pred, color="#dc2626", label="OLS fitted value")
ax.plot(x_grid[:, 0], logit_pred, color="#2563eb", label="logistic predicted probability")
ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
ax.axhline(1, color="#94a3b8", linestyle="--", linewidth=1)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("single risk feature x")
ax.set_ylabel("predicted value / predicted probability")
ax.set_title("同样是线性结构，OLS 会自然跑出 `[0,1]` 之外，而逻辑回归不会")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 图里最别扭的地方有两个：
#
# - `OLS` 输出的是实数，所以可能给出大于 `1` 或小于 `0` 的预测；
# - 即使局部看起来能分开两类，它也不是在按“概率模型”的方式训练自己。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 在二分类问题里，我们面对的是：
#
# $$
# Y \in \{0,1\}
# $$
#
# 如果还直接写
#
# $$
# E[Y \mid X] = X\beta
# $$
#
# 那么右边作为线性预测量可以取任意实数，就天然会遇到边界问题：
# 它并不保证输出落在概率应该属于的区间 `[0,1]` 里。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **顺手回顾一下：分类和回归最本质的区别是什么？**
#
# - 回归问题里，响应变量通常是连续数值；
# - 分类问题里，响应变量是离散标签。
#
# 这听起来像一句废话，但其实我们早就见过它的后果：
# 在做 `one-hot encoding` 时，我们已经看到“离散变量”一旦进入建模流程，
# 就不能再被当作普通连续量随意处理。
#
# 只不过那时这个区别出现在 `X` 的范畴里；
# 这周，这个区别第一次直接落在 `Y` 的范畴里。

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - 分类任务不是不能保留“线性预测”；
# - 但不能继续把输出直接当作一个无边界实数；
# - 所以逻辑回归的第一步，不是换掉线性部分，而是换掉“输出如何被解释”。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这幕不要让大家误会成“OLS 一点都不能分类”。
# 更准确的说法是：
# - 它有时也能给出看起来像分类边界的东西；
# - 但概率解释不自然，训练目标也不对；
# - 所以我们要的不是“分得开”而已，而是“概率建模要站得住”。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第二幕：如果要预测概率，为什么 `sigmoid` 很自然？
#
# 既然问题出在“输出不该再是任意实数”，一个自然问题就是：
#
# > 如果我们还想保留线性部分 $\eta = X\beta$，怎样把它稳定地映射到 `[0,1]`？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜：
#
# 如果一个模型先算出线性预测量 $\eta$，然后再把它变成概率，
# 你觉得这个映射最重要的性质是什么？
#
# 1. 对输入没有额外限制；
# 2. 输出必须落在 `[0,1]`；
# 3. 两者都必须满足。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：`sigmoid` 如何把线性预测量压成概率
#
# 这张图要回答：
# > 为什么大家一讲逻辑回归就会提到 `sigmoid`？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先把定义放在前面：
#
# $$
# \sigma(\eta) = \frac{1}{1 + e^{-\eta}}
# $$
#
# 它有两个最关键的性质：
# - 对输入 $\eta$ 没有额外限制；
# - 输出稳定落在 $(0,1)$，因此可以被解释成概率。

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
eta_grid = np.linspace(-6, 6, 400)
p_grid = sigmoid(eta_grid)
highlight_eta = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
highlight_p = sigmoid(highlight_eta)

fig, ax = plt.subplots(figsize=(10.5, 6))
ax.plot(eta_grid, p_grid, color="#2563eb", label="sigmoid probability")
ax.scatter(highlight_eta, highlight_p, color="#dc2626", s=50, zorder=3)
for eta_value, p_value in zip(highlight_eta, highlight_p):
    ax.plot([eta_value, eta_value], [0, p_value], linestyle="--", linewidth=1, color="#94a3b8")
    ax.text(eta_value + 0.08, p_value + 0.03, f"({eta_value:.0f}, {p_value:.2f})", fontsize=10)
ax.axhline(0.5, color="#16a34a", linestyle="--", linewidth=1.4)
ax.axvline(0, color="#16a34a", linestyle="--", linewidth=1.4)
ax.set_xlabel("linear predictor eta")
ax.set_ylabel("predicted probability")
ax.set_title("`sigmoid` 允许线性预测量取任意实数，同时把输出稳定压到 `[0,1]`")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 从图上可以直接读出三件事：
#
# - $\eta$ 很大时，概率接近 `1`；
# - $\eta$ 很小时，概率接近 `0`；
# - $\eta = 0$ 正好对应 $p = 0.5$。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 逻辑回归保留了线性预测子：
#
# $$
# \eta = X\beta
# $$
#
# 但不直接把 $\eta$ 当作输出，而是通过 `sigmoid` 把它映射成概率：
#
# $$
# p(x) = \frac{1}{1 + e^{-\eta}}
# $$
#
# 所以更准确的说法是：
# **线性预测量先算出来，再被 link function 压成概率。**

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - 逻辑回归并没有抛弃线性结构；
# - 它是在“线性预测量”外面，加了一个把输出压成概率的映射；
# - 所以逻辑回归更准确的说法是：**线性预测 + 概率映射**。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这里很适合强调：
# - sigmoid 不是为了“让曲线更好看”；
# - 它解决的是概率区间约束问题；
# - 这样 Bernoulli 和 MLE 才能自然接上。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第三幕：为什么这里自然会走到 `Bernoulli + MLE`？
#
# 一旦输出被解释成概率，接下来的问题就是：
#
# > 那我们到底该用什么分布、什么目标函数来训练这个模型？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先问一个更基础的子问题：
#
# 如果我们现在已经把输出压成了概率，
# 那还能不能继续像普通回归那样，用 `RMSE / MSE` 作为损失函数？
#
# 很多人第一反应会说：
# “为什么不行？反正真值还是 `0/1`，预测值还是一个数。”
#
# 这一幕就是要回答：**这里为什么更自然地转向 `Bernoulli + MLE`。**

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **先把对象定义清楚**
#
# 这里我们不是在做“带阈值的回归”，而是在做概率建模。
#
# 如果记
#
# $$
# Y \sim Bernoulli(p)
# $$
#
# 那么单个样本的 likelihood 就是：
#
# $$
# L(p; y) = p^y (1-p)^{1-y}
# $$
#
# 也就是说：
# - 当 $y = 1$ 时，我们希望 $p$ 尽量大；
# - 当 $y = 0$ 时，我们希望 $p$ 尽量小。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：为什么这里不再优先使用平方误差？
#
# 这张图要回答：
# > 如果输出已经被解释成概率，那么平方误差和基于似然的损失，到底在惩罚什么？

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
p_values = np.linspace(0.01, 0.99, 300)
sqerr_y1 = (1 - p_values) ** 2
sqerr_y0 = p_values**2
logloss_y1 = -np.log(p_values)
logloss_y0 = -np.log(1 - p_values)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharex=True)
axes[0].plot(p_values, sqerr_y1, color="#2563eb", label="squared loss when y = 1")
axes[0].plot(p_values, sqerr_y0, color="#dc2626", label="squared loss when y = 0")
axes[0].set_title("平方误差：会惩罚，但对“错得很自信”不够敏感")
axes[0].set_xlabel("predicted probability for class 1")
axes[0].set_ylabel("squared loss")
axes[0].legend()

axes[1].plot(p_values, logloss_y1, color="#2563eb", label="y = 1")
axes[1].plot(p_values, logloss_y0, color="#dc2626", label="y = 0")
axes[1].set_title("负对数似然 / log loss：错得越自信，惩罚越重")
axes[1].set_xlabel("predicted probability for class 1")
axes[1].set_ylabel("log loss")
axes[1].legend()

fig.suptitle("一旦输出被解释成概率，我们就更关心“概率是否支持了正确标签”", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **这张图的纵轴和图像含义怎么读？**
#
# - 左图纵轴是“损失大小”：越高表示这个概率预测越糟；
# - 右图纵轴也是“损失大小”，但它来自负对数似然；
# - 两张图都在看：给定真值后，不同预测概率会受到怎样的惩罚。
#
# 真正关键的区别是：
# - 平方误差当然也会惩罚错误；
# - 但当你把错误概率压得特别极端时，它的惩罚仍然是有上界的；
# - 负对数似然会对“错得很自信”给出更重的惩罚。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 如果从 `Bernoulli likelihood` 出发，
# 把所有样本的 likelihood 乘起来，再取对数，
# 就得到逻辑回归训练时使用的对数似然；
# 再乘一个负号，得到的正是大家常见的 `log loss` / 交叉熵目标。
#
# 所以这里不是“先研究了半天 log loss，后来才顺手说它和 likelihood 有关”；
# 而是：
#
# $$
# Bernoulli \ probability \ model
# \Longrightarrow
# likelihood
# \Longrightarrow
# negative\ log\ likelihood = log\ loss
# $$

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - 严格说，平方误差并不是“绝对不能用”，但它不是这里最自然的统计目标；
# - 逻辑回归从一开始就在建模 `Bernoulli` 概率；
# - 所以它的训练目标自然来自 `MLE`，也就是负对数似然 / `log loss`。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这幕最重要的是把“概率输出”真正和“目标函数变化”连起来。
# 不要只说“逻辑回归用交叉熵”，而是要让大家看到：
# - 为什么会是 Bernoulli；
# - 为什么会是 likelihood；
# - 为什么错得越自信惩罚越重。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第四幕：`log-odds` 到底是什么，系数为什么能解释？
#
# 到这里，概率已经出来了。
# 但逻辑回归真正最强的一点，还不是“会给概率”，而是：
#
# > 它的系数仍然可以被解释，只不过解释落在 `log-odds` 上。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜：
#
# 很多人第一次见逻辑回归时，都会很自然地想：
#
# > 如果模型最后输出的是概率，那系数能不能直接解释成“概率加多少”？
#
# 这个直觉很自然，但恰恰不对。
# 所以下一页我们要把注意力集中到：
#
# > 为什么逻辑回归的系数解释，不能直接落在概率上？
#
# 也就是说，为什么它不会直接说成：
#
# > `x` 增加 1，概率就增加 `beta` 这么多？
#
# 你直觉上觉得，问题更可能出在：
# 1. 概率本身有边界；
# 2. 概率和自变量的关系不是线性的；
# 3. 两者都对。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **先把定义放在前面**
#
# 我们先定义：
#
# $$
# odds = \frac{p}{1-p}
# $$
#
# 再定义：
#
# $$
# logit(p) = \log\left(\frac{p}{1-p}\right)
# $$
#
# 逻辑回归真正写成线性形式时，是：
#
# $$
# logit(p) = X\beta
# $$
#
# 也就是说，这一幕接下来讨论的对象，从一开始就是：
# **odds / log-odds，而不是概率本身。**

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：概率、odds、log-odds 的对应关系
#
# 这张图要回答：
# > 为什么逻辑回归真正线性的是 `log-odds`，而不是概率本身？

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
p_demo = np.linspace(0.02, 0.98, 300)
log_odds_demo = logit(p_demo)
highlight_p = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
highlight_logodds = logit(highlight_p)

fig, ax = plt.subplots(figsize=(10.5, 6))
ax.plot(p_demo, log_odds_demo, color="#2563eb")
ax.scatter(highlight_p, highlight_logodds, color="#dc2626", s=48, zorder=3)
for p_val, l_val in zip(highlight_p, highlight_logodds):
    ax.text(p_val + 0.01, l_val + 0.2, f"p={p_val:.2f}", fontsize=10)
ax.axhline(0, color="#16a34a", linestyle="--", linewidth=1.4)
ax.axvline(0.5, color="#16a34a", linestyle="--", linewidth=1.4)
ax.set_xlabel("probability p")
ax.set_ylabel("log-odds = log(p / (1 - p))")
ax.set_title("概率不是线性的，但 `log-odds` 可以和线性预测量自然对接")
plt.tight_layout()
plt.show()

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
odds_table = pd.DataFrame(
    {
        "probability p": [0.10, 0.25, 0.50, 0.75, 0.90],
        "odds = p / (1-p)": [0.11, 0.33, 1.00, 3.00, 9.00],
        "log-odds": np.round(logit(np.array([0.10, 0.25, 0.50, 0.75, 0.90])), 2),
    }
)
odds_table

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **系数解释落在哪里？**
#
# 一旦写成
#
# $$
# logit(p) = X\beta
# $$
#
# 系数的解释就变成：
#
# - `x_j` 增加 1 单位；
# - 其他变量不变时；
# - `log-odds` 会改变 $\beta_j$。
#
# 如果想转成更业务的说法，可以进一步看：
#
# $$
# \exp(\beta_j)
# $$
#
# 它表示 odds 会被乘上多少倍。

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - 逻辑回归真正线性的是 `log-odds`，不是概率本身；
# - 因此系数的统计解释是：自变量变化如何影响 `log-odds`；
# - 对业务方来说，更自然的翻译通常是：某个变量会让“发生的相对可能性”上升还是下降。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这幕不要让大家陷在 odds 公式里出不来。
# 更自然的顺序是：
# - 先承认“概率本身不是线性的”；
# - 再说为什么要换成 odds / log-odds；
# - 最后才落到 exp(beta) 的业务解释。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第五幕：分类指标为什么不能只看 `accuracy`？
#
# 到这里，模型形式已经立住了。
# 但如果只会报一个 `accuracy`，分类任务仍然很容易被讲得过于粗糙。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜：
#
# 如果我们在做违约预警、欺诈检测、用户流失这样的任务，
# 你觉得下面哪件事更危险？
#
# 1. 总体准确率不是特别高；
# 2. 真正危险的样本经常没被抓出来。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：threshold 改变时，`accuracy` 与 `recall` 会怎么变？
#
# 这张图要回答：
# > 为什么分类器不是只给一个结果，还要讨论阈值和指标偏好？

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **先把定义放在前面**
#
# 在分类任务里，最基础的对象是混淆矩阵：
#
# - `TP`：把正类判成正类；
# - `TN`：把负类判成负类；
# - `FP`：把负类误判成正类；
# - `FN`：把正类漏判成负类。
#
# 从这里往上，至少有两层指标：
#
# - 一级：`TP / TN / FP / FN` 这四个计数；
# - 二级：`accuracy`、`precision`、`recall`、`F1`；
# - 再往上还有 `ROC-AUC` 和 `log loss` 这种更整体的概率 / 排序型指标。

# %% slideshow={"slide_type": "skip"} tags=["skip"]
X_churn, y_churn = make_churn_like_data()
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.35, random_state=RANDOM_SEED, stratify=y_churn
)
churn_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_SEED)),
    ]
)
churn_model.fit(X_train_churn, y_train_churn)
churn_proba = churn_model.predict_proba(X_test_churn)[:, 1]
threshold_df = threshold_metrics(y_test_churn, churn_proba, np.linspace(0.1, 0.9, 17))

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
fig, ax = plt.subplots(figsize=(10.5, 6))
ax.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="accuracy", color="#2563eb")
ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="recall", color="#dc2626")
ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=1.4)
ax.set_xlabel("classification threshold")
ax.set_ylabel("metric value")
ax.set_title("阈值一变，分类器在“抓多少”和“分对多少”之间的取舍也会变化")
ax.legend(loc="center right")
plt.tight_layout()
plt.show()

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
roc_x, roc_y, _ = roc_curve(y_test_churn, churn_proba)
dummy_proba = np.repeat(y_train_churn.mean(), len(y_test_churn))
metric_summary = pd.DataFrame(
    [
        summarize_classifier("Logistic @ 0.5", y_test_churn, churn_proba, threshold=0.5),
        summarize_classifier("Logistic @ 0.3", y_test_churn, churn_proba, threshold=0.3),
        summarize_classifier("Base-rate only", y_test_churn, dummy_proba, threshold=0.5),
    ]
).round(3)
metric_summary

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把定义和图重新接起来**
#
# 这节课至少要把 6 个指标分开：
#
# - `accuracy`：总共分对了多少；
# - `precision`：判成正类的样本里，真的有多少是正类；
# - `recall`：真正应该抓住的正类，抓住了多少；
# - `F1`：`precision` 和 `recall` 的调和平均；
# - `ROC-AUC`：模型把正类排在负类前面的整体能力；
# - `log loss`：模型给出的概率分布是否靠谱，尤其会重罚“错得很自信”的情况。

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - 分类指标不是彼此替代，而是在回答不同问题；
# - 逻辑回归的概率输出，使它非常适合和阈值、风险偏好、概率损失联系起来；
# - 所以评估分类器时，不能只盯着“分对了多少个”。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这幕很适合落到具体业务：
# - 违约预测：漏掉高风险客户通常代价更大，所以 recall 很关键；
# - 营销投放：可能会更在意 precision 或成本；
# - log loss 用来提醒大家：概率模型不只是输出类别，还要输出“靠谱程度”。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ---
# ## 第六幕：正则化逻辑回归为什么在工业界一直不过时？
#
# 到这里，终于可以把这周和 week13 真正接起来了：
#
# > 如果逻辑回归也面对高维、相关特征和过拟合风险，它当然也会需要正则化。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 先猜：
#
# 在高维分类里，
# 你觉得 `L1` 和 `L2` 的“性格”会不会和在线性回归里很像？
#
# 1. `L1` 更偏稀疏筛选；
# 2. `L2` 更偏平滑收缩；
# 3. 两者都会，只是强度不同。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **本幕主图**：高维分类里，`L1` 与 `L2` 逻辑回归的系数与指标对比
#
# 这张图要回答：
# > 为什么逻辑回归即使不复杂，仍然是一个非常强的工业 baseline？

# %% slideshow={"slide_type": "skip"} tags=["skip"]
X_hd, y_hd = make_high_dimensional_classification_data()
X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(
    X_hd, y_hd, test_size=0.3, random_state=RANDOM_SEED, stratify=y_hd
)
l2_logit = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegressionCV(
                Cs=np.logspace(-2, 1, 10),
                cv=5,
                penalty="l2",
                solver="lbfgs",
                random_state=RANDOM_SEED,
                max_iter=4000,
            ),
        ),
    ]
)
l1_logit = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegressionCV(
                Cs=np.logspace(-2, 1, 10),
                cv=5,
                penalty="l1",
                solver="saga",
                random_state=RANDOM_SEED,
                max_iter=5000,
            ),
        ),
    ]
)
base_logit = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1e6, random_state=RANDOM_SEED, max_iter=5000)),
    ]
)
for model in [base_logit, l1_logit, l2_logit]:
    model.fit(X_train_hd, y_train_hd)

base_coef = base_logit.named_steps["model"].coef_[0]
l1_coef = l1_logit.named_steps["model"].coef_[0]
l2_coef = l2_logit.named_steps["model"].coef_[0]
top_idx = np.argsort(np.abs(l2_coef))[-10:]
top_features = X_hd.columns[top_idx]

reg_metric_summary = pd.DataFrame(
    [
        summarize_classifier(
            "Approx. unregularized",
            y_test_hd,
            base_logit.predict_proba(X_test_hd)[:, 1],
            model_size=int(np.sum(np.abs(base_coef) > 1e-6)),
        ),
        summarize_classifier(
            "L1 logistic",
            y_test_hd,
            l1_logit.predict_proba(X_test_hd)[:, 1],
            model_size=int(np.sum(np.abs(l1_coef) > 1e-6)),
        ),
        summarize_classifier(
            "L2 logistic",
            y_test_hd,
            l2_logit.predict_proba(X_test_hd)[:, 1],
            model_size=int(np.sum(np.abs(l2_coef) > 1e-6)),
        ),
    ]
).round(3)

# %% slideshow={"slide_type": "subslide"} tags=["hide-input"]
plot_df = pd.DataFrame(
    {
        "feature": top_features,
        "L1 logistic": l1_coef[top_idx],
        "L2 logistic": l2_coef[top_idx],
    }
)
plot_df = plot_df.sort_values("L2 logistic")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
axes[0].barh(plot_df["feature"], plot_df["L1 logistic"], alpha=0.78, label="L1 logistic", color="#dc2626")
axes[0].barh(plot_df["feature"], plot_df["L2 logistic"], alpha=0.55, label="L2 logistic", color="#2563eb")
axes[0].axvline(0, color="#475569", linewidth=1)
axes[0].set_title("L1 更像筛选器，L2 更像平滑收缩器")
axes[0].set_xlabel("coefficient")
axes[0].legend(loc="lower right")

axes[1].axis("off")
table_text = reg_metric_summary.to_string(index=False)
axes[1].text(
    0.0,
    1.0,
    table_text,
    va="top",
    ha="left",
    family="monospace",
    fontsize=10,
)
axes[1].set_title("测试集指标与模型规模", loc="left")

fig.suptitle("高维分类里，正则化逻辑回归兼顾了概率输出、稳定性与可解释性", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# 这张图里至少要看两层对比：
#
# - 左边：`L1` 更容易把很多系数直接压到 `0`，`L2` 更倾向于整体收缩；
# - 右边：即使指标差距不夸张，模型规模和解释风格也会非常不同。

# %% [markdown] slideshow={"slide_type": "subslide"} tags=["sub-slide"]
# **把直观和严格定义对应起来**
#
# 逻辑回归一旦加入正则化，目标函数就变成：
#
# $$
# \text{negative log-likelihood} + \lambda \cdot penalty(\beta)
# $$
#
# 其中：
#
# - `L1` penalty 更容易产生稀疏解，适合做高维分类里的特征筛选；
# - `L2` penalty 更偏向平滑收缩，适合在相关特征存在时稳定估计。
#
# 所以在线性回归里学到的“`L1` vs `L2` 性格差异”，在逻辑回归里仍然保留。

# %% [markdown] slideshow={"slide_type": "fragment"} tags=["fragment"]
# 这里，大家需要记住的是：
#
# - 逻辑回归不是过时模型，而是非常强的 baseline 和解释工具；
# - `L1` 更适合稀疏筛选，`L2` 更适合稳定收缩；
# - 在很多工业场景里，“可解释 + 可校准 + 可快速部署”本身就是重要优势。

# %% [markdown] slideshow={"slide_type": "skip"} tags=["script", "skip"]
# 这幕收尾时可以明确回答“为什么工业界长期不过时”：
# - 概率输出自然；
# - 训练快，部署稳；
# - 系数有解释；
# - 正则化后还能处理中高维特征。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## Synthesis
#
# 今天整节课其实都在回答同一个问题：
#
# > 当响应变量从连续值变成 `0/1` 时，回归框架要怎样改造，才能既保留线性结构，又符合概率建模？
#
# 到这里，大家需要记住 6 个判断：
#
# 1. 分类任务不能继续把输出直接当作无边界实数；
# 2. `sigmoid` 让线性预测量自然映射到概率；
# 3. 一旦输出被解释成概率，`Bernoulli + MLE` 就会自然接上；
# 4. 逻辑回归真正线性的是 `log-odds`，所以系数仍然可解释；
# 5. 分类评估不能只看 `accuracy`，还要结合 recall、ROC-AUC、log loss；
# 6. 正则化逻辑回归在高维分类里仍然是很强的工业 baseline。

# %% [markdown] slideshow={"slide_type": "slide"} tags=["slide"]
# ## Transition
#
# 今天我们看到：当响应变量是 `0/1` 时，
# `Bernoulli + logit link` 会把线性预测框架改造成逻辑回归。
#
# 这就自然引出一个更大的问题：
#
# > 如果响应变量不是二分类，而是计数、比例，或者多类别，GLM 该怎样继续扩展？
#
# 下一个自然方向就是：
# - 更一般的 `GLM`
# - 不同分布与不同 link function 的对应关系
# - 以及更复杂的分类建模形式
