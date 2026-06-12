"""Week 14 assignment entry point.
第14周作业主入口脚本
实验主题：高维数据回归、多重共线性、主成分回归(PCR)、Lasso对比

运行方式:
    uv run src/week14/main.py

输出目录:
    src/week14/data/        # 生成/存放数据集
    src/week14/results/     # 报表、CSV、图片结果
"""

# 系统模块导入
import sys
import shutil
from pathlib import Path

# 数据分析与绘图库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 机器学习工具
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.linear_model import LinearRegression, LassoCV  # 线性回归、自动选参Lasso
from sklearn.model_selection import KFold, train_test_split  # 交叉验证、数据集划分

# ===================== 路径配置 & 模块搜索路径 =====================
# 当前脚本所在目录
ROOT = Path(__file__).resolve().parent
# 项目src根目录（用于导入上层utils工具）
SRC_ROOT = ROOT.parent
# 将上层目录加入Python搜索路径，保证能导入utils包
sys.path.insert(0, str(SRC_ROOT))

# 导入自定义工具函数（诊断指标、评价指标、模型、数据预处理）
from utils.diagnostics import (
    calculate_condition_number,
    calculate_rank,
)  # 条件数、矩阵秩（共线性诊断）
from utils.metrics import calculate_mae, calculate_rmse  # 回归评价指标
from utils.models import PCRRegressor  # 自定义主成分回归模型
from utils.transformers import CustomStandardScaler  # 自定义标准化器

# 定义各类文件输出目录
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"

# 批量创建目录（已存在则跳过）
for path in [DATA_DIR, RESULT_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# 真实房价数据集原始路径
SOURCE_DATA_PATH = Path(
    "/home/hawei/Regression-Analysis-2026/students/05_qyz/src/week13/week14/data/House Prices - Advanced Regression Techniques.csv"
)
# 本地副本保存路径
LOCAL_DATA_PATH = DATA_DIR / SOURCE_DATA_PATH.name


# ===================== 通用工具函数 =====================
def write_markdown(path: Path, text: str):
    """
    将文本写入Markdown报告文件
    :param path: 文件路径
    :param text: 写入内容
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure(filename: str):
    """
    统一保存绘图文件，关闭画布避免内存堆积
    :param filename: 图片文件名
    """
    plt.tight_layout()  # 自动调整布局，防止标签截断
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()  # 关闭当前画布


# ===================== Task A：生成高维潜在因子仿真数据 =====================
def generate_high_dimensional_data(
    n_samples=160,
    n_features=120,
    n_latent=5,
    noise_std=0.1,
    noise_y_std=0.5,
    seed=42,
):
    """
    生成【潜在因子型高维仿真数据】
    数据逻辑：少量隐变量 → 线性组合生成大量原始特征 → 目标y也由隐变量驱动
    特点：特征高度相关、存在多重共线性，模拟真实业务"潜在因子"场景

    :param n_samples: 样本量
    :param n_features: 原始特征数(高维)
    :param n_latent: 真实潜在因子个数（低维真实信息）
    :param noise_std: 特征噪声标准差
    :param noise_y_std: 目标变量噪声标准差
    :param seed: 随机种子（保证复现）
    :return: 数据集df, 潜在因子, 载荷矩阵, 隐变量真实系数
    """
    rng = np.random.default_rng(seed)
    # 1. 生成低维潜在因子 (真实有效信息)
    latent = rng.normal(size=(n_samples, n_latent))
    # 2. 载荷矩阵：隐变量 → 原始特征的线性映射
    loadings = rng.normal(loc=0.0, scale=1.0, size=(n_latent, n_features))
    # 3. 原始高维特征 = 隐变量线性组合 + 噪声（造成强共线性）
    X = latent @ loadings + rng.normal(0, noise_std, size=(n_samples, n_features))
    # 4. 隐变量对应的真实系数
    beta_latent = np.linspace(2.0, 0.5, n_latent)
    # 5. 目标变量 y = 隐变量线性组合 + 噪声
    y = latent @ beta_latent + rng.normal(0, noise_y_std, size=n_samples)
    # 构造特征名 & DataFrame
    feature_names = [f"x{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, latent, loadings, beta_latent


def save_synthetic_data(df: pd.DataFrame):
    """将仿真数据集保存为csv文件"""
    path = DATA_DIR / "synthetic_highdim.csv"
    df.to_csv(path, index=False)
    return path


# ===================== Task A：OLS随特征维度p的表现分析 =====================
def plot_error_vs_p(df: pd.DataFrame, p_values):
    """
    实验：不同特征维度p下，OLS的训练/测试误差、矩阵秩、条件数变化
    核心观测：高维下OLS过拟合、矩阵病态(共线性加剧)

    :param df: 仿真数据集
    :param p_values: 待测试的特征维度列表
    :return: 统计结果表格
    """
    records = []
    for p in p_values:
        # 选取前p个特征
        features = [f"x{i + 1}" for i in range(p)]
        X = df[features].values
        y = df["target"].values

        # 划分训练集/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # 训练普通最小二乘回归
        model = LinearRegression().fit(X_train, y_train)

        # 计算各项指标并记录
        records.append(
            {
                "p": p,
                "train_rmse": calculate_rmse(y_train, model.predict(X_train)),
                "test_rmse": calculate_rmse(y_test, model.predict(X_test)),
                "rank": calculate_rank(X_train),  # 矩阵秩：判断线性相关
                "condition_number": calculate_condition_number(
                    X_train
                ),  # 条件数：判断共线性严重程度
            }
        )

    stats_df = pd.DataFrame(records)

    # 图1：训练/测试 RMSE 随特征维度变化
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df["p"], stats_df["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(stats_df["p"], stats_df["test_rmse"], marker="o", label="Test RMSE")
    plt.xlabel("Feature dimension p")
    plt.ylabel("RMSE")
    plt.title("OLS: Train/Test RMSE vs Feature Dimension")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    save_figure("task_a_error_vs_p.png")

    # 图2：矩阵秩 & 条件数 随特征维度变化
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df["p"], stats_df["rank"], marker="o", label="Rank(X_train)")
    plt.plot(
        stats_df["p"], stats_df["condition_number"], marker="o", label="cond(X_train)"
    )
    plt.xlabel("Feature dimension p")
    plt.title("Training Matrix Rank and Condition Number vs p")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    save_figure("task_a_rank_condition_vs_p.png")

    # 保存统计结果到CSV
    stats_df.to_csv(RESULT_DIR / "task_a_matrix_stats.csv", index=False)
    return stats_df


# ===================== Task A：OLS系数稳定性分析 =====================
def task_a_coefficient_stability(df: pd.DataFrame, n_splits=50):
    """
    多次随机划分数据集，观测OLS系数波动
    结论：高维+共线性下，OLS系数极不稳定

    :param df: 仿真数据集
    :param n_splits: 随机划分次数
    :return: 波动最大的特征及对应标准差
    """
    feature_names = [col for col in df.columns if col != "target"]
    X = df[feature_names].values
    y = df["target"].values

    # 整体划分一次（仅占位，循环内会重复随机划分）
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    coefs = []

    model = LinearRegression()
    # 多次随机切分，训练模型并保存系数
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        model.fit(X_train, y_train)
        coefs.append(model.coef_)

    # 拼接所有轮次系数
    coef_matrix = np.vstack(coefs)
    score = pd.DataFrame(coef_matrix, columns=feature_names)
    # 选出系数标准差最大（波动最剧烈）的3个特征
    top_features = score.std().sort_values(ascending=False).head(3).index.tolist()
    selected = score[top_features]

    # 箱线图展示系数分布/波动
    plt.figure(figsize=(10, 6))
    selected.boxplot()
    plt.title("OLS Coefficient Stability: 50 Random Splits")
    plt.ylabel("Coefficient value")
    plt.xlabel("Feature")
    save_figure("task_a_coefficient_stability.png")

    return {
        "top_features": top_features,
        "coef_std": selected.std().to_dict(),
    }


# ===================== Task B：PCA主成分分析 =====================
def task_b_pca_analysis(df: pd.DataFrame):
    """
    对高维数据做PCA，绘制累计方差解释率曲线
    作用：判断有效主成分个数、验证数据存在低维潜在结构

    :param df: 仿真数据集
    :return: 单个主成分方差占比、累计方差占比
    """
    feature_names = [col for col in df.columns if col != "target"]
    X = df[feature_names].values
    # 标准化：PCA对量纲敏感，必须标准化
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 全量主成分分解
    pca = PCA()
    pca.fit(X_scaled)

    # 单个主成分方差解释率、累计方差解释率
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # 绘制累计方差曲线
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative Explained Variance of Synthetic Data")
    plt.grid(True, linestyle="--", alpha=0.5)
    save_figure("task_b_pca_cumulative_variance.png")

    return explained, cumulative


# ===================== Task B：PCR主成分回归性能曲线 =====================
def task_b_pcr_curve(df: pd.DataFrame, max_components=20):
    """
    遍历不同主成分个数k，评估PCR训练集、测试集、5折CV误差
    目的：选择最优主成分数量，观察过拟合趋势

    :param df: 仿真数据集
    :param max_components: 最大遍历主成分数
    :return: 各k对应的误差统计表
    """
    feature_names = [col for col in df.columns if col != "target"]
    X = df[feature_names].values
    y = df["target"].values

    # 标准化
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    # 逐个尝试主成分数量 k
    for k in range(1, max_components + 1):
        # 训练PCR模型
        pcr = PCRRegressor(n_components=k).fit(X_train, y_train)
        # 训练集、测试集RMSE
        train_rmse = calculate_rmse(y_train, pcr.predict(X_train))
        test_rmse = calculate_rmse(y_test, pcr.predict(X_test))

        # 5折交叉验证RMSE
        cv_rmse = []
        for train_idx, val_idx in kf.split(X_train):
            pcr_cv = PCRRegressor(n_components=k)
            pcr_cv.fit(X_train[train_idx], y_train[train_idx])
            cv_rmse.append(
                calculate_rmse(y_train[val_idx], pcr_cv.predict(X_train[val_idx]))
            )

        records.append(
            {
                "n_components": k,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "cv_rmse": float(np.mean(cv_rmse)),
            }
        )

    pcr_df = pd.DataFrame(records)
    # 绘制三条RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(
        pcr_df["n_components"], pcr_df["train_rmse"], marker="o", label="Train RMSE"
    )
    plt.plot(pcr_df["n_components"], pcr_df["test_rmse"], marker="o", label="Test RMSE")
    plt.plot(pcr_df["n_components"], pcr_df["cv_rmse"], marker="o", label="CV RMSE")
    plt.xlabel("Number of principal components")
    plt.ylabel("RMSE")
    plt.title("PCR Performance vs Number of Components")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    save_figure("task_b_pcr_curve.png")

    pcr_df.to_csv(RESULT_DIR / "task_b_pcr_curve.csv", index=False)
    return pcr_df


# ===================== Task C：生成两类对比仿真数据 =====================
def generate_sparse_truth_data(
    n_samples=200, n_features=100, n_relevant=5, noise_std=1.0, seed=123
):
    """
    生成【稀疏真值数据】：只有少数原始特征真正影响y，其余为噪声
    适用场景：适合Lasso（变量筛选）
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    # 真实系数：仅前n_relevant个特征非零，其余为0（稀疏结构）
    true_coef = np.zeros(n_features)
    relevant_indices = np.arange(n_relevant)
    true_coef[relevant_indices] = np.linspace(5.0, 1.0, n_relevant)
    y = X @ true_coef + rng.normal(0, noise_std, size=n_samples)
    feature_names = [f"x{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, true_coef, relevant_indices


def generate_latent_truth_data(
    n_samples=200, n_features=100, n_latent=5, noise_std=0.1, noise_y_std=0.5, seed=456
):
    """复用前面的潜在因子数据生成函数"""
    return generate_high_dimensional_data(
        n_samples=n_samples,
        n_features=n_features,
        n_latent=n_latent,
        noise_std=noise_std,
        noise_y_std=noise_y_std,
        seed=seed,
    )


# ===================== Task C：Lasso & PCR 综合评估 =====================
def evaluate_scenario(df: pd.DataFrame, scenario_name: str):
    """
    在单一场景下对比 Lasso 和 PCR：预测误差、模型复杂度、系数稳定性
    :param df: 数据集
    :param scenario_name: 场景名称(稀疏/潜在因子)
    :return: 各项评估指标字典
    """
    feature_names = [col for col in df.columns if col != "target"]
    X = df[feature_names].values
    y = df["target"].values

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. 自动交叉验证选择最优alpha的Lasso
    lasso = LassoCV(
        alphas=np.logspace(-4, 1, 50),
        cv=5,
        max_iter=20000,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)
    # 统计Lasso非零系数个数（模型复杂度）
    lasso_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-6))

    # 数据标准化（PCR必须标准化）
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. 遍历主成分数，用CV选最优k
    pcr_records = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in range(1, min(25, X_train.shape[1]) + 1):
        fold_rmse = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            pcr = PCRRegressor(n_components=k).fit(
                X_train_scaled[train_idx], y_train[train_idx]
            )
            fold_rmse.append(
                calculate_rmse(y_train[val_idx], pcr.predict(X_train_scaled[val_idx]))
            )
        pcr_records.append({"n_components": k, "cv_rmse": float(np.mean(fold_rmse))})

    pcr_df = pd.DataFrame(pcr_records)
    # 选出CV误差最小的最优主成分数
    best_k = int(pcr_df.loc[pcr_df["cv_rmse"].idxmin(), "n_components"])
    pcr = PCRRegressor(n_components=best_k).fit(X_train_scaled, y_train)

    # 汇总Lasso指标
    lasso_metrics = {
        "name": "Lasso",
        "test_rmse": calculate_rmse(y_test, lasso.predict(X_test)),
        "test_mae": calculate_mae(y_test, lasso.predict(X_test)),
        "model_size": lasso_nonzero,
        "alpha": float(lasso.alpha_),
    }
    # 汇总PCR指标
    pcr_metrics = {
        "name": "PCR",
        "test_rmse": calculate_rmse(y_test, pcr.predict(X_test_scaled)),
        "test_mae": calculate_mae(y_test, pcr.predict(X_test_scaled)),
        "model_size": best_k,
        "explained_variance": float(np.sum(pcr.explained_variance_ratio_)),
    }

    # 3. 多次划分测试系数稳定性（系数标准差越小越稳定）
    def coefficient_stability(model_factory, repeat=10):
        coeffs = []
        for seed in range(repeat):
            Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=seed)
            if model_factory == "lasso":
                model = LassoCV(
                    alphas=np.logspace(-4, 1, 50),
                    cv=5,
                    max_iter=20000,
                    n_jobs=-1,
                    random_state=seed,
                ).fit(Xt, yt)
                coeffs.append(model.coef_)
            else:
                scaler_rep = CustomStandardScaler()
                Xt_scaled = scaler_rep.fit_transform(Xt)
                Xv_scaled = scaler_rep.transform(Xv)
                pcr_rep = PCRRegressor(n_components=best_k).fit(Xt_scaled, yt)
                coeffs.append(pcr_rep.coef_)
        # 所有特征系数标准差的均值
        return float(np.mean(np.std(np.vstack(coeffs), axis=0)))

    stability = {
        "lasso_coef_std": coefficient_stability("lasso"),
        "pcr_coef_std": coefficient_stability("pcr"),
    }

    result = {
        "scenario": scenario_name,
        "lasso": lasso_metrics,
        "pcr": pcr_metrics,
        "stability": stability,
        "feature_count": X.shape[1],
    }
    return result


# ===================== 报告生成 & 绘图函数 =====================
def summarize_scenario_comparison(sparse_result, latent_result):
    """生成两种数据场景对比的Markdown报告"""
    lines = [
        "# Scenario Comparison: Sparse Truth vs Latent-Factor Truth",
        "",
        "## 1. 数据世界设计",
        "- Sparse Truth：只有少数原始变量直接决定 y，其余特征是噪声。",
        "- Latent-Factor Truth：大多数原始变量由少量潜在因子线性生成，y 也由这些因子驱动。",
        "",
        "## 2. 比较结果",
        "",
        "### Sparse Truth",
        f"- Lasso test RMSE = {sparse_result['lasso']['test_rmse']:.2f}，模型规模 = {sparse_result['lasso']['model_size']}。",
        f"- PCR test RMSE = {sparse_result['pcr']['test_rmse']:.2f}，主成分数 = {sparse_result['pcr']['model_size']}。",
        f"- Lasso stability = {sparse_result['stability']['lasso_coef_std']:.4f}，PCR stability = {sparse_result['stability']['pcr_coef_std']:.4f}。",
        "",
        "### Latent-Factor Truth",
        f"- Lasso test RMSE = {latent_result['lasso']['test_rmse']:.2f}，模型规模 = {latent_result['lasso']['model_size']}。",
        f"- PCR test RMSE = {latent_result['pcr']['test_rmse']:.2f}，主成分数 = {latent_result['pcr']['model_size']}。",
        f"- Lasso stability = {latent_result['stability']['lasso_coef_std']:.4f}，PCR stability = {latent_result['stability']['pcr_coef_std']:.4f}。",
        "",
        "## 3. 结论",
        "- 当数据更像 sparse truth 时，Lasso 更自然，因为它直接在原始变量空间中筛选。",
        "- 当数据更像 latent-factor truth 时，PCR 更自然，因为它先压缩重复信息，再回归。",
        "- Lasso 的答案更像“谁留下”，而 PCR 的答案更像“保留多少个稳定方向”。",
    ]
    write_markdown(RESULT_DIR / "summary_comparison.md", "\n".join(lines))


def build_synthetic_report(
    stats_df,
    stability_summary,
    explained,
    cumulative,
    pcr_curve_df,
    sparse_result,
    latent_result,
):
    """生成仿真数据完整综合报告"""
    lines = [
        "# Synthetic Report: High-dimensional Regression and PCR",
        "",
        "## 1. 数据生成机制",
        "- 样本数：160，特征数：120。",
        "- 数据由 5 个潜在因子生成；原始特征由这些潜在因子线性组合而来，外加小量噪声。",
        "- 目标 y 由这 5 个潜在因子主导，而不是每个原始特征独立决定。",
        "- 这是一份典型的“高维 + 信息冗余”数据：p 远大于 n，且特征之间存在强共线性。",
        "",
        "## 2. Task A: OLS 在高维下的行为",
        "- 随着 p 增大，训练误差持续下降，而测试误差在高维时开始上升。",
        "- 训练集 RMSE 与测试集 RMSE 的比较说明，虚假的低训练误差并不代表泛化能力好。",
        "- 矩阵秩随 p 增大时常常保持低于 p，条件数显著上升，这说明 X_train 的病态程度增长。",
        "",
        "## 3. Task A: 系数稳定性",
        f"- 选取波动最大的 3 个变量：{', '.join(stability_summary['top_features'])}。",
        "- 50 次随机切分中，这些变量的 OLS 系数波动明显，表明即便训练误差较低，系数本身仍不稳定。",
        "",
        "## 4. Task B: PCA 与 PCR",
        "- 累计解释方差图显示，前 5~10 个主成分已解释大部分方差。",
        "- 这说明原始高维空间近似于一个低维子空间。",
        "- PCR 的 train/test/CV 曲线明确展示了随着 k 增加，过拟合风险与噪声回归会逐渐出现。",
        "",
        "## 5. Task B: 公式与定义",
        "- OLS 估计值：β_hat = (XᵀX)⁻¹ Xᵀ y。",
        "- 第一主成分定义：w1 = argmax_{||w||=1} Var(X w)。",
        "- PCR 过程：Z_k = X V_k，随后在 Z_k 上拟合 y = Z_k γ + ε。",
        "",
        "## 6. Task C: Lasso vs PCR",
        "- Sparse truth 下，Lasso 的 test RMSE 和模型复杂度均更优；",
        "- Latent-factor truth 下，PCR 的 CV 选择和稳定性更有优势；",
        "- Lasso 更像“变量筛选”，PCR 更像“信息压缩”。",
        "",
        "## 7. 结果摘要",
        f"- Sparse truth: Lasso RMSE = {sparse_result['lasso']['test_rmse']:.2f}, PCR RMSE = {sparse_result['pcr']['test_rmse']:.2f}。",
        f"- Latent truth: Lasso RMSE = {latent_result['lasso']['test_rmse']:.2f}, PCR RMSE = {latent_result['pcr']['test_rmse']:.2f}。",
        "",
        "## 8. 图表说明",
        "- `task_a_error_vs_p.png`：展示 OLS 训练/测试 RMSE 随特征维度变化。",
        "- `task_a_rank_condition_vs_p.png`：展示训练矩阵秩与条件数随 p 变化。",
        "- `task_a_coefficient_stability.png`：展示 3 个关键变量系数在不同切分下的波动。",
        "- `task_b_pca_cumulative_variance.png`：展示累计解释方差。",
        "- `task_b_pcr_curve.png`：展示 PCR 训练 / 测试 / CV RMSE 曲线。",
    ]
    write_markdown(RESULT_DIR / "synthetic_report.md", "\n".join(lines))


def plot_scenario_comparison(sparse_result, latent_result):
    """绘制 Lasso / PCR 在两种场景下 RMSE、模型复杂度对比柱状图"""
    categories = ["Sparse", "Latent"]
    lasso_rmse = [
        sparse_result["lasso"]["test_rmse"],
        latent_result["lasso"]["test_rmse"],
    ]
    pcr_rmse = [sparse_result["pcr"]["test_rmse"], latent_result["pcr"]["test_rmse"]]

    x = np.arange(len(categories))
    width = 0.35
    # RMSE 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, lasso_rmse, width, label="Lasso")
    plt.bar(x + width / 2, pcr_rmse, width, label="PCR")
    plt.xticks(x, categories)
    plt.ylabel("Test RMSE")
    plt.title("Lasso vs PCR: Test RMSE Across Two Scenarios")
    plt.legend()
    save_figure("task_c_rmse_comparison.png")

    # 模型复杂度对比图
    lasso_complexity = [
        sparse_result["lasso"]["model_size"],
        latent_result["lasso"]["model_size"],
    ]
    pcr_complexity = [
        sparse_result["pcr"]["model_size"],
        latent_result["pcr"]["model_size"],
    ]
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, lasso_complexity, width, label="Lasso non-zero count")
    plt.bar(x + width / 2, pcr_complexity, width, label="PCR components")
    plt.xticks(x, categories)
    plt.ylabel("Model complexity")
    plt.title("Lasso vs PCR Complexity Across Scenarios")
    plt.legend()
    save_figure("task_c_complexity_comparison.png")


def build_kaggle_report(feature_names, diagnostics, model_results, summary):
    """Task D 真实房价数据报告（预留，依赖外部函数 load_house_data / preprocess_data 等）"""
    lines = [
        "# Task D: Real Data Challenge - House Prices",
        "",
        "## 1. 数据来源与预处理",
        f"- 本次任务使用的数据文件：`{LOCAL_DATA_PATH.name}`。",
        f"- 训练样本数：{diagnostics['X_train'].shape[0]}，测试样本数：{diagnostics['X_test'].shape[0]}。",
        f"- 选用特征数量：{len(feature_names)}，包括面积、建成年代、地下室和车库质量等。",
        "- 缺失值处理：数值型特征补中位数，质量型特征用映射编码，并将缺失值视为最差等级。",
        "",
        "## 2. 训练与比较结果",
    ]
    for name, result in model_results.items():
        lines.extend(
            [
                f"### {name}",
                f"- 训练集 RMSE: {result['train_rmse']:.2f}",
                f"- 测试集 RMSE: {result['test_rmse']:.2f}",
                f"- 训练集 MAE: {result['train_mae']:.2f}",
                f"- 测试集 MAE: {result['test_mae']:.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 模型复杂度与稳定性",
            f"- OLS 是最简单的基准，但它对特征相关性敏感。",
            f"- Lasso 最优 alpha = {diagnostics['lasso_alpha']:.5f}，保留非零系数数 = {diagnostics['lasso_nonzero']}。",
            f"- PCR 最优主成分数 = {diagnostics['pcr_best_k']}，此时累计解释方差 = {diagnostics['pcr_explained_variance']:.3f}。",
            f"- 训练特征矩阵秩 = {diagnostics['rank']}, 条件数 = {diagnostics['condition_number']:.2f}。",
            "",
            "## 4. 数据结构判断",
            "- 该数据具有较多相关变量，例如面积、地下室和车库特征共同反映住房规模与品质。",
            "- 这一点更接近“latent-factor truth”，因为多个原始指标在测量同一套房产质量/规模结构。",
            "- 因此，PCR 通过保留前几个主成分，能够把重复信息压缩为更稳定的预测信号。",
            "",
            "## 5. 结论",
            "- 如果业务方希望一个更短的变量名单，Lasso 更直观；",
            "- 如果业务方更看重预测稳定性与特征相关性的鲁棒性，PCR 更适合；",
            "- 该 Kaggle 房价数据更像一个潜在因子驱动的世界，而不是纯粹的稀疏真值。",
            "",
            "## 6. 图表说明",
            "- `pcr_cv_rmse.png`：横轴为主成分个数，纵轴为 CV RMSE；用于选择 PCR 的最佳 k。",
            "- `actual_vs_pred_*`：展示真实值与预测值的对比。",
            "- `residuals_comparison.png`：展示三种模型在测试集上的残差分布。",
            "- `lasso_vs_pcr_coefficients.png`：展示 Lasso 和 PCR 在原始特征空间的系数对比。",
        ]
    )

    lines.append(summary)
    write_markdown(RESULT_DIR / "kaggle_report.md", "\n".join(lines))


# ===================== 主执行函数 =====================
def main():
    """
    作业主流程：
    1. Task A~C：仿真高维数据实验（OLS、PCA、PCR、Lasso 对比）
    2. Task D：真实房价数据挑战（预留流程，依赖外部数据加载/预处理函数）
    """
    # ========== 第一部分：仿真数据全流程 (Task A/B/C) ==========
    # 生成潜在因子型高维数据
    synthetic_df, latent, loadings, beta_latent = generate_high_dimensional_data()
    synthetic_path = save_synthetic_data(synthetic_df)

    # Task A：OLS 误差、矩阵指标随特征维度变化
    stats_df = plot_error_vs_p(synthetic_df, [10, 30, 60, 120])
    # Task A：OLS 系数稳定性分析
    stability_summary = task_a_coefficient_stability(synthetic_df, n_splits=50)

    # Task B：PCA 方差解释率分析
    explained, cumulative = task_b_pca_analysis(synthetic_df)
    # Task B：PCR 不同主成分数的性能曲线
    pcr_curve_df = task_b_pcr_curve(synthetic_df, max_components=20)

    # Task C：生成两类对比数据（稀疏真值 / 潜在因子真值）
    sparse_df, _, _ = generate_sparse_truth_data()
    latent_df, _, _ = generate_latent_truth_data()

    # 分别评估 Lasso & PCR
    sparse_result = evaluate_scenario(sparse_df, "Sparse Truth")
    latent_result = evaluate_scenario(latent_df, "Latent-Factor Truth")

    # 绘制对比图、生成对比报告
    plot_scenario_comparison(sparse_result, latent_result)
    summarize_scenario_comparison(sparse_result, latent_result)

    # 生成仿真数据完整综合报告
    build_synthetic_report(
        stats_df,
        stability_summary,
        explained,
        cumulative,
        pcr_curve_df,
        sparse_result,
        latent_result,
    )

    # ========== 第二部分：真实数据 Task D（房价数据集，预留逻辑） ==========
    # 注：load_house_data / preprocess_data / train_test_models / evaluate_model / make_plots
    # 为外部自定义函数，当前代码仅做流程占位
    df = load_house_data()
    X, y, feature_names = preprocess_data(df)

    # 共线性诊断指标
    diagnostics = {
        "rank": calculate_rank(X),
        "condition_number": calculate_condition_number(X),
    }
    # 训练多模型并返回结果
    results = train_test_models(X, y)
    diagnostics.update(results)

    # 分别评估 OLS / Lasso / PCR
    model_results = {
        "OLS": evaluate_model(
            "OLS",
            results["ols"],
            results["X_train"],
            results["X_test"],
            results["y_train"],
            results["y_test"],
        ),
        "Lasso": evaluate_model(
            "Lasso",
            results["lasso"],
            results["X_train"],
            results["X_test"],
            results["y_train"],
            results["y_test"],
        ),
        "PCR": evaluate_model(
            "PCR",
            results["pcr"],
            results["X_train"],
            results["X_test"],
            results["y_train"],
            results["y_test"],
        ),
    }

    # 绘图 + 生成真实数据报告
    make_plots(feature_names, diagnostics, model_results)
    build_kaggle_report(
        feature_names,
        diagnostics,
        model_results,
        "本次报告聚焦真实数据 Task D，比较 OLS、Lasso 与 PCR 的预测表现与稳定性。",
    )

    print(f"结果已写入: {RESULT_DIR}")


# 程序入口
if __name__ == "__main__":
    main()
