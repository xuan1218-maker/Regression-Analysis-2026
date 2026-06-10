from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

from utils.metrics import rmse, mae
from utils.diagnostics import matrix_rank, condition_number, coefficient_std
from utils.models import PCRModel, pcr_cv_rmse
from utils.transformers import standardize_train_test


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def setup_chinese_font():
    font_candidates = [
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/msyhbd.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for font_path in font_candidates:
        path = Path(font_path)
        if path.exists():
            fm.fontManager.addfont(str(path))
            font_name = fm.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return

    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()
def make_latent_factor_data(
    n_samples=160,
    n_features=120,
    n_factors=6,
    noise=0.8,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    latent_factors = rng.normal(size=(n_samples, n_factors))
    loadings = rng.normal(size=(n_factors, n_features))
    X = latent_factors @ loadings + 0.25 * rng.normal(size=(n_samples, n_features))

    beta_factor = np.array([3.0, -2.5, 1.5, 0.8, 0.0, 0.0])[:n_factors]
    y = latent_factors @ beta_factor + noise * rng.normal(size=n_samples)

    columns = [f"x{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y
    return df


def make_sparse_truth_data(
    n_samples=160,
    n_features=120,
    n_active=6,
    noise=1.0,
    random_state=123,
):
    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[:n_active] = np.array([4.0, -3.0, 2.5, -2.0, 1.5, 1.0])[:n_active]

    y = X @ beta + noise * rng.normal(size=n_samples)
    return X, y, beta


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def df_to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def run_ols_dimension_experiment():
    rows = []
    p_values = [10, 30, 60, 120]

    for p in p_values:
        df = make_latent_factor_data(
            n_samples=160,
            n_features=p,
            n_factors=6,
            noise=0.8,
            random_state=100 + p,
        )

        X = df.drop(columns=["y"]).to_numpy()
        y = df["y"].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.35,
            random_state=42,
        )

        X_train_scaled, X_test_scaled, _ = standardize_train_test(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        rows.append(
            {
                "p": p,
                "train_rmse": rmse(y_train, train_pred),
                "test_rmse": rmse(y_test, test_pred),
                "rank_X_train": matrix_rank(X_train_scaled),
                "condition_number": condition_number(X_train_scaled),
            }
        )

    result = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4.5))
    plt.plot(result["p"], result["train_rmse"], marker="o", label="训练集误差")
    plt.plot(result["p"], result["test_rmse"], marker="o", label="测试集误差")
    plt.xlabel("特征数量")
    plt.ylabel("均方根误差")
    plt.title("普通最小二乘法误差随特征维度增加的变化")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(FIGURES_DIR / "普通最小二乘法误差变化.png")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(result["p"], result["rank_X_train"], marker="o", color="tab:blue")
    ax1.set_xlabel("特征数量")
    ax1.set_ylabel("训练集矩阵秩", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(result["p"], result["condition_number"], marker="s", color="tab:red")
    ax2.set_ylabel("条件数", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("矩阵秩和条件数随特征维度的变化")
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "矩阵结构变化.png", dpi=180)
    plt.close()

    return result


def run_coefficient_instability():
    df = make_latent_factor_data(
        n_samples=160,
        n_features=120,
        n_factors=6,
        noise=0.8,
        random_state=42,
    )

    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    selected_features = [0, 1, 2, 3, 4]
    records = []
    coef_list = []

    for seed in range(50):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.35,
            random_state=seed,
        )

        X_train_scaled, X_test_scaled, _ = standardize_train_test(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        coef_list.append(model.coef_)

        for j in selected_features:
            records.append(
                {
                    "split": seed,
                    "feature": f"x{j + 1}",
                    "coefficient": model.coef_[j],
                    "train_rmse": rmse(y_train, train_pred),
                    "test_rmse": rmse(y_test, test_pred),
                }
            )

    coef_df = pd.DataFrame(records)
    coef_matrix = np.vstack(coef_list)
    std_values = coefficient_std(coef_matrix)

    plt.figure(figsize=(8, 4.8))
    data = [
        coef_df.loc[coef_df["feature"] == f"x{j + 1}", "coefficient"]
        for j in selected_features
    ]

    plt.boxplot(data, tick_labels=[f"x{j + 1}" for j in selected_features])
    plt.xlabel("选取的原始变量")
    plt.ylabel("五十次随机划分下的回归系数")
    plt.title("普通最小二乘法系数在不同随机划分下的波动")
    plt.grid(axis="y", alpha=0.3)
    save_figure(FIGURES_DIR / "系数不稳定性.png")

    return coef_df, std_values


def run_pca_and_pcr():
    df = pd.read_csv(DATA_DIR / "synthetic_highdim.csv")
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    pca_df = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "cumulative_explained_variance": cumulative,
        }
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        pca_df["component"],
        pca_df["cumulative_explained_variance"],
        marker="o",
    )
    plt.axhline(0.8, color="gray", linestyle="--", label="百分之八十方差")
    plt.axhline(0.9, color="red", linestyle="--", label="百分之九十方差")
    plt.xlabel("主成分个数")
    plt.ylabel("累计解释方差比例")
    plt.title("主成分分析累计解释方差曲线")
    plt.xlim(1, 30)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(FIGURES_DIR / "累计解释方差曲线.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=42,
    )

    rows = []

    for k in range(1, 21):
        model = PCRModel(n_components=k)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        rows.append(
            {
                "k": k,
                "train_rmse": rmse(y_train, train_pred),
                "test_rmse": rmse(y_test, test_pred),
                "cv_rmse": pcr_cv_rmse(X_train, y_train, k, n_splits=5),
            }
        )

    result = pd.DataFrame(rows)
    best_k = int(result.loc[result["cv_rmse"].idxmin(), "k"])

    plt.figure(figsize=(8, 4.8))
    plt.plot(result["k"], result["train_rmse"], marker="o", label="训练集误差")
    plt.plot(result["k"], result["test_rmse"], marker="o", label="测试集误差")
    plt.plot(result["k"], result["cv_rmse"], marker="o", label="交叉验证误差")
    plt.axvline(best_k, color="gray", linestyle="--", label=f"交叉验证最优数量={best_k}")
    plt.xlabel("保留的主成分个数")
    plt.ylabel("均方根误差")
    plt.title("不同主成分个数下的主成分回归误差曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(FIGURES_DIR / "主成分回归误差曲线.png")

    return pca_df, result, best_k


def evaluate_lasso_vs_pcr_for_scenario(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=42,
    )

    X_train_scaled, X_test_scaled, _ = standardize_train_test(X_train, X_test)

    lasso = LassoCV(cv=5, random_state=42, max_iter=20000)
    lasso.fit(X_train_scaled, y_train)

    lasso_pred = lasso.predict(X_test_scaled)
    lasso_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-8))

    pcr_rows = []

    for k in range(1, 21):
        model = PCRModel(n_components=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cv = pcr_cv_rmse(X_train, y_train, k, n_splits=5)

        pcr_rows.append(
            {
                "k": k,
                "test_rmse": rmse(y_test, pred),
                "test_mae": mae(y_test, pred),
                "cv_rmse": cv,
            }
        )

    pcr_df = pd.DataFrame(pcr_rows)
    best = pcr_df.loc[pcr_df["cv_rmse"].idxmin()]
    best_k = int(best["k"])

    best_model = PCRModel(n_components=best_k)
    best_model.fit(X_train, y_train)
    pcr_pred = best_model.predict(X_test)

    summary = pd.DataFrame(
        [
            {
                "scenario": name,
                "method": "套索回归",
                "test_rmse": rmse(y_test, lasso_pred),
                "test_mae": mae(y_test, lasso_pred),
                "complexity": lasso_nonzero,
                "complexity_definition": "非零系数个数",
                "stability_metric": "系数绝对值总和",
                "stability_value": float(np.sum(np.abs(lasso.coef_))),
            },
            {
                "scenario": name,
                "method": "主成分回归",
                "test_rmse": rmse(y_test, pcr_pred),
                "test_mae": mae(y_test, pcr_pred),
                "complexity": best_k,
                "complexity_definition": "保留的主成分个数",
                "stability_metric": "交叉验证选择的主成分个数",
                "stability_value": float(best_k),
            },
        ]
    )

    return summary


def run_lasso_vs_pcr():
    X_sparse, y_sparse, _ = make_sparse_truth_data(
        n_samples=160,
        n_features=120,
        n_active=6,
        noise=1.0,
        random_state=123,
    )

    df_latent = make_latent_factor_data(
        n_samples=160,
        n_features=120,
        n_factors=6,
        noise=0.8,
        random_state=42,
    )

    X_latent = df_latent.drop(columns=["y"]).to_numpy()
    y_latent = df_latent["y"].to_numpy()

    summary = pd.concat(
        [
            evaluate_lasso_vs_pcr_for_scenario("稀疏真实机制", X_sparse, y_sparse),
            evaluate_lasso_vs_pcr_for_scenario("潜在因子机制", X_latent, y_latent),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, scenario in zip(axes, ["稀疏真实机制", "潜在因子机制"]):
        part = summary[summary["scenario"] == scenario]
        ax.bar(part["method"], part["test_rmse"])
        ax.set_title(scenario)
        ax.set_ylabel("测试集均方根误差")
        ax.grid(axis="y", alpha=0.3)

        for i, row in enumerate(part.itertuples()):
            ax.text(
                i,
                row.test_rmse,
                f"复杂度={row.complexity}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("稀疏机制与潜在因子机制下的两类方法对比")
    save_figure(FIGURES_DIR / "两类方法对比.png")

    return summary


def write_synthetic_report(ols_result, pca_df, pcr_result, best_k):
    top_variance_90 = int(
        pca_df.loc[
            pca_df["cumulative_explained_variance"] >= 0.9,
            "component",
        ].iloc[0]
    )

    ols_show = ols_result.rename(
        columns={
            "p": "特征数量",
            "train_rmse": "训练集均方根误差",
            "test_rmse": "测试集均方根误差",
            "rank_X_train": "训练集矩阵秩",
            "condition_number": "条件数",
        }
    )

    pcr_show = pcr_result.rename(
        columns={
            "k": "主成分个数",
            "train_rmse": "训练集均方根误差",
            "test_rmse": "测试集均方根误差",
            "cv_rmse": "交叉验证均方根误差",
        }
    )

    report = f"""# 第十四周模拟数据实验报告

## 一、数据生成机制

本次实验生成了一份高维回归模拟数据，样本量为一百六十，原始特征数量为一百二十。数据文件保存在数据文件夹中。

这份数据不是让一百二十个变量各自独立决定因变量，而是先生成六个潜在因子，再由这些潜在因子线性组合生成大量原始特征。因变量也主要由这些潜在因子驱动，并加入少量随机噪声。

因此，这份数据具有两个特点：

一是特征维度较高，原始变量数量较多；二是信息存在明显冗余，很多变量本质上来自少数几个共同的潜在方向。

这说明原始高维空间虽然列数很多，但真正有效的信息可能集中在一个较低维的子空间中。

## 二、普通最小二乘法在高维场景下的问题

本实验固定样本量，分别设置特征数量为十、三十、六十、一百二十，并比较普通最小二乘法在线性回归中的表现。

误差变化图的含义如下：

- 横轴表示特征数量；
- 纵轴表示均方根误差；
- 一条曲线表示训练集误差；
- 另一条曲线表示测试集误差。

从图中可以观察到，随着特征数量增加，训练误差可能变得很低，但测试误差并不一定同步下降。这说明模型可能只是更好地记住了训练集，而不是真正学到了更稳定的规律。

矩阵结构图的含义如下：

- 横轴表示特征数量；
- 左侧纵轴表示训练集设计矩阵的秩；
- 右侧纵轴表示条件数；
- 条件数越大，说明矩阵越病态，系数估计越不稳定。

实验结果如下：

{df_to_markdown(ols_show)}

当训练误差接近零时，在高维问题中反而可能是危险信号。因为此时模型可能已经拥有足够多的自由度去拟合训练集中的噪声。训练集表现很好，并不代表模型在新数据上也稳定可靠。

## 三、系数不稳定性

系数波动图的含义如下：

- 横轴表示选取的几个原始变量；
- 纵轴表示这些变量在五十次随机划分中的回归系数；
- 每个箱线图表示同一个变量在不同训练集划分下的系数波动范围。

可以看到，同一个变量的系数在不同随机划分下会出现明显波动。这说明在高维和多重共线性场景中，普通最小二乘法对数据划分非常敏感。

这种风险不仅体现在预测误差上，也体现在解释性上。如果同一个变量在不同划分下系数大小甚至方向都不稳定，那么我们就很难可靠地说明这个变量到底是否重要。

## 四、主成分分析降维结果

累计解释方差图的含义如下：

- 横轴表示主成分个数；
- 纵轴表示累计解释方差比例；
- 曲线表示保留前若干个主成分后，能够解释原始数据总方差的比例。

在本次实验中，前 {top_variance_90} 个主成分已经能够解释至少百分之九十的总方差。这说明虽然原始数据有一百二十个特征，但主要信息集中在少数几个主成分方向上。

这也验证了前面的数据生成设定：原始高维变量并不是完全独立地携带信息，而是接近一个较低维的潜在因子空间。

## 五、主成分回归实验结果

主成分回归的基本流程如下：

一是先对原始特征进行标准化；二是使用主成分分析提取主成分；三是保留前若干个主成分；四是在这些主成分上建立线性回归模型。

误差曲线图的含义如下：

- 横轴表示保留的主成分个数；
- 纵轴表示均方根误差；
- 图中展示训练集误差、测试集误差和交叉验证误差；
- 竖线表示交叉验证误差最小时选择的主成分个数。

本次实验中，通过交叉验证选择的最佳主成分个数为：{best_k}。

实验结果如下：

{df_to_markdown(pcr_show)}

交叉验证误差表示模型在训练集内部进行多次划分后得到的平均验证误差。它比单纯的训练误差更适合用来选择主成分个数。

训练误差通常会随着主成分个数增大而降低，但测试误差和交叉验证误差可以帮助我们判断：继续增加主成分到底是在补充有效信息，还是在引入噪声。

普通最小二乘法可以在原始高维空间中取得很低的训练误差，但这不代表它更好。主成分回归通过先压缩信息，再回归，可以在一定程度上降低模型方差，提高预测稳定性。

## 六、公式与定义

普通最小二乘法的估计式为：

`beta_hat = (X^T X)^(-1) X^T y`

当矩阵不可逆或接近不可逆时，系数估计会变得非常不稳定。

第一主成分的定义可以写成：

`v1 = argmax_{{||v|| = 1}} Var(Xv)`

也就是说，第一主成分是在所有单位方向中，使投影后方差最大的方向。

主成分回归可以表示为：

`Z_k = X V_k`

其中，`V_k` 表示前若干个主成分方向，`Z_k` 表示原始数据投影后的主成分得分矩阵。

然后在主成分得分矩阵上建立线性回归：

`y = Z_k gamma + epsilon`

也就是说，主成分回归不是直接在原始变量上回归，而是在压缩后的主成分空间中回归。
"""

    (RESULTS_DIR / "synthetic_report.md").write_text(report, encoding="utf-8")


def write_summary_report(summary):
    summary_show = summary.rename(
        columns={
            "scenario": "数据机制",
            "method": "方法",
            "test_rmse": "测试集均方根误差",
            "test_mae": "测试集平均绝对误差",
            "complexity": "模型复杂度",
            "complexity_definition": "复杂度含义",
            "stability_metric": "稳定性指标",
            "stability_value": "稳定性指标值",
        }
    )

    report = f"""# 第十四周方法对比总结

## 一、比较目标

本部分主要比较两类方法：

一类是套索回归，代表变量筛选思路；另一类是主成分回归，代表信息压缩思路。

为了说明两类方法适合的场景不同，本实验构造了两种数据机制：

第一种是稀疏真实机制，只有少数几个原始变量真正直接决定因变量。

第二种是潜在因子机制，大量原始变量由少数潜在因子生成，因变量也主要由这些潜在因子驱动。

## 二、两类方法的实验对比

对比图的含义如下：

- 每个子图对应一种数据机制；
- 横轴表示方法；
- 纵轴表示测试集均方根误差；
- 柱子上方标注的是模型复杂度；
- 对套索回归来说，复杂度表示非零系数个数；
- 对主成分回归来说，复杂度表示保留的主成分个数。

实验结果如下：

{df_to_markdown(summary_show)}

## 三、核心问题讨论

当数据真实机制是稀疏的，也就是只有少数原始变量真正影响因变量时，套索回归往往更加自然。因为套索回归直接对原始变量进行筛选，它回答的是“哪些变量应该留下”的问题。

当数据更接近潜在因子机制时，主成分回归往往更加自然。因为此时有用信息并不集中在少数几个原始变量上，而是分散在许多高度相关的变量中。主成分回归不强行选择某几个原始变量，而是把这些变量压缩成少数几个主成分方向。

因此，两种方法回答的问题不同。

套索回归回答的是：谁留下？

主成分回归回答的是：用哪些低维信息来代表原始高维变量？

如果业务方需要的是一个更短的变量名单，那么套索回归更合适，因为它保留的是原始变量名称。

如果业务方需要的是一个更稳定的预测器，那么主成分回归可能更合适，特别是在变量之间存在强相关、信息高度冗余的情况下。主成分回归通过压缩信息，可以减少模型对单个原始变量系数的依赖。

## 四、关于前向选择和后向选择

本周主线更适合比较套索回归和主成分回归，而不是重新把前向选择和后向选择作为重点。

原因是，本周的核心问题是：变量筛选和信息压缩有什么区别？

前向选择和后向选择本质上仍然属于变量筛选方法。它们是在原始变量集合中一步步加入或删除变量，并没有像主成分回归那样改变变量空间，也没有把高维信息压缩到主成分空间中。

如果一定要加入前向选择或后向选择，那么它们更接近套索回归所代表的变量筛选路线，而不是主成分回归所代表的信息压缩路线。
"""

    (RESULTS_DIR / "summary_comparison.md").write_text(report, encoding="utf-8")


def write_optional_kaggle_report():
    report = """# 真实数据扩展报告

本次作业没有完成选做的真实数据部分。

当前已经完成的内容包括：

一、高维低秩模拟数据生成；

二、普通最小二乘法在高维和多重共线性场景下的不稳定性分析；

三、主成分分析累计解释方差分析；

四、主成分回归建模与主成分个数选择；

五、套索回归与主成分回归在稀疏真实机制和潜在因子机制下的对比。

真实数据部分属于选做内容，因此这里不再额外展开。
"""

    (RESULTS_DIR / "kaggle_report.md").write_text(report, encoding="utf-8")


def main():
    synthetic = make_latent_factor_data(
        n_samples=160,
        n_features=120,
        n_factors=6,
        noise=0.8,
        random_state=42,
    )
    synthetic.to_csv(DATA_DIR / "synthetic_highdim.csv", index=False)

    ols_result = run_ols_dimension_experiment()
    run_coefficient_instability()
    pca_df, pcr_result, best_k = run_pca_and_pcr()
    comparison = run_lasso_vs_pcr()

    write_synthetic_report(ols_result, pca_df, pcr_result, best_k)
    write_summary_report(comparison)
    write_optional_kaggle_report()

    print("第十四周作业运行完成。")
    print(f"数据保存位置：{DATA_DIR}")
    print(f"报告保存位置：{RESULTS_DIR}")
    print(f"图片保存位置：{FIGURES_DIR}")


if __name__ == "__main__":
    main()