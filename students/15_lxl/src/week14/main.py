"""模块: week14.main
用途: 第十四周作业 - 高维回归、PCA 与 PCR。
      Task A: 观察高维/共线性如何破坏 OLS
      Task B: PCA 复习与 PCR 工作流
      Task C: Lasso vs PCR - 变量筛选 vs 信息压缩

核心思路:
    1. 生成带有潜在低秩结构的高维模拟数据 (少数 latent factors 生成大量原始变量)
    2. 展示 OLS 在高维下的虚假低训练误差和系数不稳定性
    3. 用 PCA 压缩维度, 在主成分空间中做回归 (PCR)
    4. 对比 Lasso 和 PCR 在 sparse truth / latent-factor truth 下的表现

运行方式:
    cd students/15_lxl
    uv run src/week14/main.py
"""

# ===== 标准库导入 ============================================================
import shutil   # 用于清空并重建 results 目录
import sys      # 用于修改模块搜索路径
from pathlib import Path  # 路径管理 (跨平台兼容)

# ===== 第三方库导入 ==========================================================
import matplotlib
matplotlib.use("Agg")  # 设置无 GUI 后端 (WSL2/服务器环境下无法弹窗显示图片)
import matplotlib.pyplot as plt        # 绑图
import matplotlib.font_manager as fm   # 字体管理 (用于加载中文字体)

# --- 中文字体配置 ---
# WSL2 环境下通过 /mnt/c/ 访问 Windows 字体目录
_FONT_PATH = "/mnt/c/Windows/Fonts/msyh.ttc"       # 微软雅黑字体路径
_CN_FONT = fm.FontProperties(fname=_FONT_PATH)      # 可复用的 FontProperties 对象
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False    # 解决负号显示为方块的问题

import numpy as np   # 数值计算
import pandas as pd  # 数据处理 (DataFrame)

# --- sklearn 模型与工具 ---
from sklearn.linear_model import LinearRegression, LassoCV
# LinearRegression: 普通最小二乘法 (OLS), 作为基准模型
# LassoCV: Lasso 回归 + 内置交叉验证自动选 alpha

from sklearn.decomposition import PCA  # 主成分分析
from sklearn.model_selection import train_test_split, KFold
# train_test_split: 训练集/测试集划分
# KFold: K 折交叉验证 (用于 PCR 选 k)

# ===== 自定义 utils 导入 ======================================================
# 将 src/ 加入搜索路径, 复用自己维护的 utils/ 组件
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import calculate_rmse, calculate_mae   # 评估指标
from utils.transformers import CustomStandardScaler       # 自定义标准化器

# ---------------------------------------------------------------------------
# 路径配置 (基于当前文件位置, 不硬编码绝对路径)
# ---------------------------------------------------------------------------
WEEK14_ROOT = Path(__file__).resolve().parent   # week14/ 目录
DATA_DIR = WEEK14_ROOT / "data"                 # 数据文件存放目录
RESULTS_DIR = WEEK14_ROOT / "results"           # 输出结果目录 (每次运行清空重建)
FIGURES_DIR = RESULTS_DIR / "figures"           # 图片存放子目录


# ===== 数据生成 =============================================================

def generate_highdim_data(n_samples=200, n_features=80, n_factors=5, seed=42):
    """生成带有潜在低秩结构的高维模拟回归数据。

    设计思路:
        真实数据由少数 latent factors 线性组合生成大量原始变量。
        这是 PCA/PCR 的理想场景 - 原始变量虽然多, 但信息维度很低。

    数据生成机制 (DGP):
        Step 1: Z ~ N(0, I)          # 生成 n_factors 个潜在因子
        Step 2: X = Z @ W + 噪声     # 因子载荷矩阵 W 将因子映射到高维空间
        Step 3: y = Z @ true_coefs   # y 只由前 3 个因子驱动

    参数:
        n_samples: 样本量 (默认 200)
        n_features: 特征数 (默认 80, 远大于 n_factors)
        n_factors: 潜在因子数 (默认 5)
        seed: 随机种子

    返回: X (n_samples, n_features), y (n_samples,), Z, W, true_coefs
    """
    rng = np.random.default_rng(seed)

    # Step 1: 生成潜在因子矩阵 Z (n_samples x n_factors)
    # 每行是一个样本的因子值, 每列是一个潜在因子
    Z = rng.normal(0, 1, (n_samples, n_factors))

    # Step 2: 生成因子载荷矩阵 W (n_factors x n_features)
    # W 的每一列定义了一个原始变量如何由潜在因子线性组合
    # 例如: x1 = w11*z1 + w12*z2 + ... + w15*z5 + 小噪声
    W = rng.normal(0, 1, (n_factors, n_features))
    X = Z @ W + rng.normal(0, 0.1, (n_samples, n_features))

    # Step 3: y 只由前 3 个 latent factors 驱动
    # true_coefs = [3, -2, 1.5, 0, 0] 表示:
    #   z1 的系数 = 3 (正向影响)
    #   z2 的系数 = -2 (负向影响)
    #   z3 的系数 = 1.5 (正向影响)
    #   z4, z5 的系数 = 0 (对 y 无贡献)
    true_coefs = np.array([3.0, -2.0, 1.5, 0, 0])
    y = Z @ true_coefs + rng.normal(0, 0.5, n_samples)

    return X, y, Z, W, true_coefs


def generate_sparse_data(n_samples=200, n_features=80, n_signal=5, seed=42):
    """生成 sparse truth 场景的数据。

    设计思路:
        只有少数原始变量 (前 5 个) 直接决定 y, 其余 75 个都是纯噪声。
        这是 Lasso 的理想场景 - 变量筛选可以精确识别信号变量。

    DGP:
        y = X @ true_beta + 噪声
        true_beta = [3, -2, 1.5, 4, -1, 0, 0, ..., 0]  (只有前 5 个非零)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    true_beta = np.zeros(n_features)
    true_beta[:n_signal] = rng.normal(0, 3, n_signal)  # 前 5 个有贡献
    y = X @ true_beta + rng.normal(0, 0.5, n_samples)
    return X, y, true_beta


def generate_latent_factor_data(n_samples=200, n_features=80, n_factors=5, seed=42):
    """生成 latent-factor truth 场景的数据。

    设计思路:
        原始变量由少数潜在因子线性组合生成 (与 generate_highdim_data 相同机制)。
        y 由这些潜在因子驱动。
        这是 PCR 的理想场景 - 信息压缩可以恢复潜在结构。

    DGP:
        X = Z @ W + 噪声 (5 个因子生成 80 个变量)
        y = Z @ [3, -2, 1.5, 0, 0] + 噪声
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n_samples, n_factors))
    W = rng.normal(0, 1, (n_factors, n_features))
    X = Z @ W + rng.normal(0, 0.1, (n_samples, n_features))
    true_coefs = np.array([3.0, -2.0, 1.5, 0, 0])
    y = Z @ true_coefs + rng.normal(0, 0.5, n_samples)
    return X, y, true_coefs


# ===== 辅助函数 =============================================================

def compute_condition_number(X):
    """计算矩阵的条件数 (衡量病态程度)。

    条件数 = 最大奇异值 / 最小奇异值
    条件数越大, 矩阵越接近奇异, OLS 的系数估计越不稳定。
    条件数 = inf 表示矩阵奇异 (不可逆)。
    """
    try:
        return np.linalg.cond(X)
    except np.linalg.LinAlgError:
        return np.inf


def compute_matrix_rank(X):
    """计算矩阵的数值秩。

    数值秩: 奇异值大于阈值 tol 的个数。
    当 p > n 时, rank(X) <= n, 说明矩阵是秩亏的。
    """
    try:
        s = np.linalg.svd(X, compute_uv=False)
        # 阈值: max(m,n) * max(singular_value) * epsilon
        tol = max(X.shape) * np.max(s) * np.finfo(float).eps
        return int(np.sum(s > tol))
    except np.linalg.LinAlgError:
        return min(X.shape)


def stability_index(coef_matrix):
    """计算系数稳定性指标 (平均变异系数)。

    对每个特征, 计算其系数在多次切分下的变异系数 (CV = std / |mean|)。
    返回所有特征的平均变异系数。
    CV 越小 → 系数越稳定 → 模型对训练样本越不敏感。

    参数:
        coef_matrix: (n_splits, n_features) 的系数矩阵

    返回: 平均变异系数 (标量)
    """
    means = np.mean(coef_matrix, axis=0)  # 每个特征的系数均值
    stds = np.std(coef_matrix, axis=0)    # 每个特征的系数标准差
    # 避免除以 0: 如果 |mean| < 1e-6, 直接用 std 作为不稳定度量
    cv = np.where(np.abs(means) > 1e-6, stds / np.abs(means), stds)
    return np.mean(cv)


# ===== Task A ===============================================================

def run_task_a():
    """Task A: 观察高维/共线性如何破坏 OLS。

    包含:
        A1/A2: 生成高维模拟数据并保存
        A3: OLS 随特征维度 p 变化的误差和矩阵结构
        A4: 50 次随机切分下的系数不稳定性
    """
    print("=" * 60)
    print("Task A: 高维/共线性如何破坏 OLS")
    print("=" * 60)

    # =========================================================================
    # A1/A2: 生成高维数据
    # n_samples=200, n_features=80, n_factors=5
    # 这是一个 p < n 但 p 接近 n 的场景 (80 < 200, 但信息维度只有 5)
    # =========================================================================
    print("\n[A1/A2] 生成高维模拟数据...")
    X, y, Z, W, true_coefs = generate_highdim_data()
    n_samples, n_features = X.shape
    print(f"  样本量: {n_samples}, 特征数: {n_features}")
    print(f"  潜在因子数: 5 (只有前 3 个驱动 y)")

    # 保存数据到 CSV
    cols = [f"x{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    csv_path = DATA_DIR / "synthetic_highdim.csv"
    df.to_csv(csv_path, index=False)
    print(f"  数据已保存: {csv_path}")

    # =========================================================================
    # A3: OLS 随特征维度变化
    # 固定 n=200, 改变 p = [10, 30, 60, 80, 120, 200]
    # 观察: 训练误差趋近 0, 测试误差上升, 条件数急剧增大
    # =========================================================================
    print("\n[A3] OLS 随特征维度变化...")
    p_values = [10, 30, 60, 80, 120, 200]
    train_rmses = []  # 各 p 下的训练 RMSE
    test_rmses = []   # 各 p 下的测试 RMSE
    ranks = []        # 各 p 下的矩阵秩
    conds = []        # 各 p 下的条件数

    rng = np.random.default_rng(42)
    for p in p_values:
        # 用相同的 DGP, 但只取前 p 个特征
        # 如果 p > n_features (80), 补充随机噪声列 (模拟更多无关变量)
        X_p = X[:, :min(p, n_features)]
        if p > n_features:
            extra = rng.normal(0, 1, (n_samples, p - n_features))
            X_p = np.hstack([X_p, extra])

        # 70/30 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_p, y, test_size=0.3, random_state=42
        )

        # 标准化 (用自定义 Scaler, 只在训练集上 fit)
        scaler = CustomStandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # OLS 拟合
        model = LinearRegression()
        model.fit(X_train_s, y_train)

        # 记录指标
        train_rmse = calculate_rmse(y_train, model.predict(X_train_s))
        test_rmse = calculate_rmse(y_test, model.predict(X_test_s))
        rank = compute_matrix_rank(X_train_s)    # 矩阵秩
        cond = compute_condition_number(X_train_s)  # 条件数

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        ranks.append(rank)
        conds.append(cond)

        print(f"    p={p:4d} -> train RMSE={train_rmse:.4f}, test RMSE={test_rmse:.4f}, "
              f"rank={rank}, cond={cond:.2e}")

    # --- 画图 1: 误差随特征维度变化 ---
    # 横轴: 特征维度 p (对数刻度不适用, 用线性)
    # 纵轴: RMSE
    # 两条线: 训练 RMSE (蓝色) 和 测试 RMSE (橙色)
    # 红色虚线: n=200 的位置 (p = n 的分界线)
    # 结论: p 接近 n 时训练 RMSE 趋近 0, 但测试 RMSE 大幅上升 (过拟合)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_values, train_rmses, "o-", color="steelblue", linewidth=2, label="训练 RMSE")
    ax.plot(p_values, test_rmses, "s-", color="orange", linewidth=2, label="测试 RMSE")
    ax.axvline(x=n_samples, color="red", linestyle="--", linewidth=1, label=f"n={n_samples}")
    ax.set_xlabel("Feature Dimension (p)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Task A3: OLS 误差随特征维度变化", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=11, prop=_CN_FONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a3_ols_error_vs_p.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/a3_ols_error_vs_p.png")

    # --- 画图 2: 矩阵结构随特征维度变化 ---
    # 左图: 横轴=p, 纵轴=矩阵秩 rank(X_train)
    #   - 红色虚线=n, 表示秩的上界
    #   - 当 p > n 时, 秩不再增长 (被 n 截断)
    # 右图: 横轴=p, 纵轴=条件数 (log scale)
    #   - 条件数急剧上升说明矩阵越来越病态
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(p_values, ranks, "o-", color="steelblue", linewidth=2)
    axes[0].axhline(y=n_samples, color="red", linestyle="--", linewidth=1, label=f"n={n_samples}")
    axes[0].set_xlabel("Feature Dimension (p)", fontsize=12)
    axes[0].set_ylabel("Rank of X_train", fontsize=12)
    axes[0].set_title("矩阵秩随 p 变化", fontsize=13, fontproperties=_CN_FONT)
    axes[0].legend(fontsize=10, prop=_CN_FONT)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p_values, conds, "s-", color="orange", linewidth=2)
    axes[1].set_xlabel("Feature Dimension (p)", fontsize=12)
    axes[1].set_ylabel("Condition Number (log)", fontsize=12)
    axes[1].set_title("条件数随 p 变化", fontsize=13, fontproperties=_CN_FONT)
    axes[1].set_yscale("log")  # 条件数跨越多个数量级, 用对数刻度
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Task A3: 矩阵结构随特征维度变化", fontsize=14, fontproperties=_CN_FONT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a3_matrix_structure.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/a3_matrix_structure.png")

    # =========================================================================
    # A4: 系数不稳定性
    # 固定数据 (p=80), 做 50 次不同随机切分
    # 每次用 OLS 拟合, 收集系数, 观察波动
    # =========================================================================
    print("\n[A4] 系数不稳定性 (50 次随机切分)...")
    n_splits = 50
    coef_matrix = []  # (50, 80) 的系数矩阵

    for i in range(n_splits):
        # 每次用不同的 random_state 做切分 -> 得到不同的训练集
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            X, y, test_size=0.3, random_state=i
        )
        scaler = CustomStandardScaler()
        X_train_s = scaler.fit_transform(X_train_i)
        model = LinearRegression()
        model.fit(X_train_s, y_train_i)
        coef_matrix.append(model.coef_)

    coef_matrix = np.array(coef_matrix)  # shape: (50, 80)

    # 选取 3 个关键变量 (x1, x2, x3) 观察其系数波动
    key_vars = [0, 1, 2]
    var_names = [f"x{i+1}" for i in key_vars]

    # --- 画箱线图 ---
    # 横轴: 变量名 (x1, x2, x3)
    # 纵轴: 该变量在 50 次切分下的系数值
    # 每个箱线: 中位数、四分位距、异常值
    # 结论: 系数波动剧烈, OLS 在高维下极不稳定
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [coef_matrix[:, i] for i in key_vars]
    bp = ax.boxplot(data, positions=range(len(key_vars)), widths=0.5, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["steelblue", "orange", "green"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(key_vars)))
    ax.set_xticklabels(var_names)
    ax.set_ylabel("Coefficient Value", fontsize=12)
    ax.set_title("Task A4: OLS 系数在 50 次随机切分下的波动", fontsize=14, fontproperties=_CN_FONT)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "a4_coefficient_instability.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/a4_coefficient_instability.png")

    # 计算稳定性指标 (平均变异系数)
    stab_idx = stability_index(coef_matrix)
    print(f"  系数稳定性指标 (平均变异系数): {stab_idx:.4f}")

    # 汇总 A3 和 A4 的数据, 供报告生成使用
    a3_data = {
        "p_values": p_values, "train_rmses": train_rmses, "test_rmses": test_rmses,
        "ranks": ranks, "conds": conds,
    }
    a4_data = {"coef_matrix": coef_matrix, "key_vars": key_vars, "stability_index": stab_idx}

    return X, y, a3_data, a4_data


# ===== Task B ===============================================================

def run_pca_analysis(X):
    """Task B1: PCA 分析, 画累计解释方差曲线。

    PCA 的核心思想:
        找到原始空间中使投影后方差最大的方向 (主成分)。
        前几个主成分通常包含了数据的大部分信息。

    返回: cumvar (累计解释方差比例), k_90, k_95
    """
    print("\n[B1] PCA 分析...")

    # 标准化 (PCA 对量纲敏感, 必须先标准化)
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 做全量 PCA (保留所有主成分)
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # 累计解释方差比例
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = len(cumvar)

    # 找到解释 90% 和 95% 方差所需的主成分数
    k_90 = int(np.argmax(cumvar >= 0.90) + 1)
    k_95 = int(np.argmax(cumvar >= 0.95) + 1)
    print(f"  解释 90% 方差需要 {k_90} 个主成分")
    print(f"  解释 95% 方差需要 {k_95} 个主成分")

    # --- 画累计解释方差曲线 ---
    # 横轴: 主成分个数 (1, 2, ..., 80)
    # 纵轴: 累计解释方差比例 (0 ~ 1)
    # 橙色虚线: 90% 方差线
    # 红色虚线: 95% 方差线
    # 结论: 前 5 个主成分已解释 >95% 方差, 原始 80 维空间贴近 5 维子空间
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_components + 1), cumvar, "o-", color="steelblue", linewidth=2, markersize=3)
    ax.axhline(y=0.90, color="orange", linestyle="--", linewidth=1, label="90% 方差")
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1, label="95% 方差")
    ax.axvline(x=k_90, color="orange", linestyle=":", linewidth=1)
    ax.axvline(x=k_95, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("Task B1: PCA 累计解释方差曲线", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=11, prop=_CN_FONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "b1_pca_cumvar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/b1_pca_cumvar.png")

    return cumvar, k_90, k_95


def run_pcr_comparison(X, y, k_max=20):
    """Task B2: PCR 工作流, 比较不同 k 下的误差。

    PCR 流程:
        标准化 -> PCA(k) -> 保留前 k 个主成分 -> 线性回归

    与 OLS 的区别:
        OLS 在原始 p 维空间中拟合, 当 p 接近 n 时过拟合。
        PCA 先将 p 维压缩到 k 维 (k << p), 再在低维空间中拟合,
        从而控制模型复杂度, 避免过拟合。

    返回: dict 包含各 k 下的 train/test/CV RMSE
    """
    print("\n[B2] PCR 工作流...")

    # 70/30 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 标准化
    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ks = list(range(1, k_max + 1))
    train_rmses = []  # 各 k 下的训练 RMSE
    test_rmses = []   # 各 k 下的测试 RMSE
    cv_rmses = []     # 各 k 下的 CV RMSE

    # 5 折交叉验证 (用于选 k)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for k in ks:
        # PCA 降维: 将 80 维压缩到 k 维
        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_s)  # 训练集投影到 k 维
        Z_test = pca.transform(X_test_s)        # 测试集用同样的投影

        # 在 k 维主成分空间中做线性回归
        model = LinearRegression()
        model.fit(Z_train, y_train)

        train_rmses.append(calculate_rmse(y_train, model.predict(Z_train)))
        test_rmses.append(calculate_rmse(y_test, model.predict(Z_test)))

        # CV RMSE: 在每个折中独立做 PCA + 回归
        # 注意: PCA 必须在每个折的训练集上 fit, 不能用全量训练集的 PCA
        # 否则会造成数据泄漏
        fold_rmses = []
        for tr_idx, val_idx in kf.split(X_train_s):
            X_tr, X_val = X_train_s[tr_idx], X_train_s[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            pca_cv = PCA(n_components=k)
            Z_tr = pca_cv.fit_transform(X_tr)
            Z_val = pca_cv.transform(X_val)
            m = LinearRegression()
            m.fit(Z_tr, y_tr)
            fold_rmses.append(calculate_rmse(y_val, m.predict(Z_val)))
        cv_rmses.append(np.mean(fold_rmses))

    # OLS 基准 (在原始 80 维空间中拟合, 不降维)
    ols = LinearRegression()
    ols.fit(X_train_s, y_train)
    ols_train_rmse = calculate_rmse(y_train, ols.predict(X_train_s))
    ols_test_rmse = calculate_rmse(y_test, ols.predict(X_test_s))

    print(f"  OLS 基准 -> train RMSE={ols_train_rmse:.4f}, test RMSE={ols_test_rmse:.4f}")
    best_k_idx = np.argmin(cv_rmses)
    print(f"  PCR 最优 k={ks[best_k_idx]} -> CV RMSE={cv_rmses[best_k_idx]:.4f}")

    # --- 画图 ---
    # 横轴: 保留主成分数 k (1, 2, ..., 20)
    # 纵轴: RMSE
    # 三条实线: PCR 的训练/测试/CV RMSE
    # 两条虚线: OLS 的训练/测试 RMSE (作为基准)
    # 红色竖线: 最优 k 的位置
    # 结论: CV RMSE 先降后升, 存在最优 k=5; OLS 训练误差低但泛化差
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, train_rmses, "o-", color="steelblue", linewidth=2, label="PCR 训练 RMSE")
    ax.plot(ks, test_rmses, "s-", color="orange", linewidth=2, label="PCR 测试 RMSE")
    ax.plot(ks, cv_rmses, "^-", color="green", linewidth=2, label="PCR CV RMSE")
    ax.axhline(y=ols_train_rmse, color="steelblue", linestyle="--", alpha=0.5, label="OLS 训练 RMSE")
    ax.axhline(y=ols_test_rmse, color="orange", linestyle="--", alpha=0.5, label="OLS 测试 RMSE")
    ax.axvline(x=ks[best_k_idx], color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of Components (k)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Task B2: PCR 误差随主成分数 k 变化", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=10, prop=_CN_FONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "b2_pcr_error_vs_k.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/b2_pcr_error_vs_k.png")

    return {
        "ks": ks, "train_rmses": train_rmses, "test_rmses": test_rmses,
        "cv_rmses": cv_rmses, "best_k": ks[best_k_idx],
        "ols_train_rmse": ols_train_rmse, "ols_test_rmse": ols_test_rmse,
    }


# ===== Task C ===============================================================

def run_task_c():
    """Task C: Lasso vs PCR 在两种场景下的对比。

    核心对比: selection (选择) vs compression (压缩)
        - Sparse Truth: 只有少数变量决定 y -> Lasso 更自然
        - Latent-Factor Truth: y 由潜在因子驱动 -> PCR 更自然

    返回: dict 包含两种场景下 Lasso 和 PCR 的指标
    """
    print("\n" + "=" * 60)
    print("Task C: Lasso vs PCR - 变量筛选 vs 信息压缩")
    print("=" * 60)

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================================
    # 场景 1: Sparse Truth (只有少数变量决定 y)
    # 真实 DGP: y = X @ true_beta + 噪声, true_beta 只有前 5 个非零
    # 这是 Lasso 的理想场景
    # =========================================================================
    print("\n[C1] Sparse Truth 场景...")
    X_sp, y_sp, true_beta = generate_sparse_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X_sp, y_sp, test_size=0.3, random_state=42
    )

    # 标准化
    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Lasso (用 LassoCV 自动选 alpha) ---
    # LassoCV 内置 5 折 CV, 自动在对数空间中搜索最优 alpha
    lasso = LassoCV(cv=5, random_state=42, max_iter=50000)
    lasso.fit(X_train_s, y_train)
    lasso_pred = lasso.predict(X_test_s)
    lasso_rmse = calculate_rmse(y_test, lasso_pred)
    lasso_mae = calculate_mae(y_test, lasso_pred)
    # 统计非零系数个数 (Lasso 的变量选择结果)
    lasso_nz = int(np.sum(np.abs(lasso.coef_) > 1e-6))
    print(f"  Lasso -> RMSE={lasso_rmse:.4f}, MAE={lasso_mae:.4f}, 非零系数={lasso_nz}")

    # --- PCR (用 CV 选 k) ---
    # 遍历 k=1~20, 选 CV RMSE 最低的 k
    best_k_sp, best_cv_sp = 1, 999.0
    for k in range(1, 21):
        fold_rmses = []
        for tr_idx, val_idx in kf.split(X_train_s):
            X_tr, X_val = X_train_s[tr_idx], X_train_s[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            pca = PCA(n_components=k)
            Z_tr = pca.fit_transform(X_tr)
            Z_val = pca.transform(X_val)
            m = LinearRegression()
            m.fit(Z_tr, y_tr)
            fold_rmses.append(calculate_rmse(y_val, m.predict(Z_val)))
        cv = np.mean(fold_rmses)
        if cv < best_cv_sp:
            best_cv_sp = cv
            best_k_sp = k

    # 用最优 k 在全量训练集上训练 PCR
    pca_sp = PCA(n_components=best_k_sp)
    Z_train_sp = pca_sp.fit_transform(X_train_s)
    Z_test_sp = pca_sp.transform(X_test_s)
    pcr_model = LinearRegression()
    pcr_model.fit(Z_train_sp, y_train)
    pcr_pred = pcr_model.predict(Z_test_sp)
    pcr_rmse = calculate_rmse(y_test, pcr_pred)
    pcr_mae = calculate_mae(y_test, pcr_pred)
    print(f"  PCR (k={best_k_sp}) -> RMSE={pcr_rmse:.4f}, MAE={pcr_mae:.4f}")

    results["sparse"] = {
        "lasso_rmse": lasso_rmse, "lasso_mae": lasso_mae, "lasso_nz": lasso_nz,
        "pcr_rmse": pcr_rmse, "pcr_mae": pcr_mae, "pcr_k": best_k_sp,
    }

    # =========================================================================
    # 场景 2: Latent-Factor Truth (y 由潜在因子驱动)
    # 真实 DGP: X = Z @ W + 噪声, y = Z @ true_coefs + 噪声
    # 这是 PCR 的理想场景
    # =========================================================================
    print("\n[C2] Latent-Factor Truth 场景...")
    X_lf, y_lf, true_coefs = generate_latent_factor_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X_lf, y_lf, test_size=0.3, random_state=42
    )

    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Lasso ---
    lasso = LassoCV(cv=5, random_state=42, max_iter=50000)
    lasso.fit(X_train_s, y_train)
    lasso_pred = lasso.predict(X_test_s)
    lasso_rmse = calculate_rmse(y_test, lasso_pred)
    lasso_mae = calculate_mae(y_test, lasso_pred)
    lasso_nz = int(np.sum(np.abs(lasso.coef_) > 1e-6))
    print(f"  Lasso -> RMSE={lasso_rmse:.4f}, MAE={lasso_mae:.4f}, 非零系数={lasso_nz}")

    # --- PCR ---
    best_k_lf, best_cv_lf = 1, 999.0
    for k in range(1, 21):
        fold_rmses = []
        for tr_idx, val_idx in kf.split(X_train_s):
            X_tr, X_val = X_train_s[tr_idx], X_train_s[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            pca = PCA(n_components=k)
            Z_tr = pca.fit_transform(X_tr)
            Z_val = pca.transform(X_val)
            m = LinearRegression()
            m.fit(Z_tr, y_tr)
            fold_rmses.append(calculate_rmse(y_val, m.predict(Z_val)))
        cv = np.mean(fold_rmses)
        if cv < best_cv_lf:
            best_cv_lf = cv
            best_k_lf = k

    pca_lf = PCA(n_components=best_k_lf)
    Z_train_lf = pca_lf.fit_transform(X_train_s)
    Z_test_lf = pca_lf.transform(X_test_s)
    pcr_model = LinearRegression()
    pcr_model.fit(Z_train_lf, y_train)
    pcr_pred = pcr_model.predict(Z_test_lf)
    pcr_rmse = calculate_rmse(y_test, pcr_pred)
    pcr_mae = calculate_mae(y_test, pcr_pred)
    print(f"  PCR (k={best_k_lf}) -> RMSE={pcr_rmse:.4f}, MAE={pcr_mae:.4f}")

    results["latent_factor"] = {
        "lasso_rmse": lasso_rmse, "lasso_mae": lasso_mae, "lasso_nz": lasso_nz,
        "pcr_rmse": pcr_rmse, "pcr_mae": pcr_mae, "pcr_k": best_k_lf,
    }

    # --- 画对比图 ---
    # 左图: Sparse Truth 场景
    #   横轴: 方法 (Lasso, PCR)
    #   纵轴: 测试 RMSE
    #   柱上标注: 模型复杂度 (非零系数/主成分数)
    #   结论: Lasso 大幅领先 (0.536 vs 5.473)
    # 右图: Latent-Factor Truth 场景
    #   同上
    #   结论: 两者接近, PCR 略优
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = ["Lasso", "PCR"]
    rmses_sp = [results["sparse"]["lasso_rmse"], results["sparse"]["pcr_rmse"]]
    bars = axes[0].bar(methods, rmses_sp, color=["orange", "steelblue"], alpha=0.8, width=0.5)
    axes[0].set_ylabel("Test RMSE", fontsize=12)
    axes[0].set_title(f"Sparse Truth\nLasso: {results['sparse']['lasso_nz']} 非零系数, "
                     f"PCR: k={results['sparse']['pcr_k']}", fontsize=12, fontproperties=_CN_FONT)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, rmses_sp):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=11)

    rmses_lf = [results["latent_factor"]["lasso_rmse"], results["latent_factor"]["pcr_rmse"]]
    bars = axes[1].bar(methods, rmses_lf, color=["orange", "steelblue"], alpha=0.8, width=0.5)
    axes[1].set_ylabel("Test RMSE", fontsize=12)
    axes[1].set_title(f"Latent-Factor Truth\nLasso: {results['latent_factor']['lasso_nz']} 非零系数, "
                     f"PCR: k={results['latent_factor']['pcr_k']}", fontsize=12, fontproperties=_CN_FONT)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, rmses_lf):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=11)

    fig.suptitle("Task C: Lasso vs PCR - 两种场景对比", fontsize=14, fontproperties=_CN_FONT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c_lasso_vs_pcr.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/c_lasso_vs_pcr.png")

    return results


# ===== 报告生成 =============================================================

def write_synthetic_report(a3_data, a4_data, cumvar, k_90, k_95, pcr_data):
    """生成 synthetic_report.md - Task A + B 的完整报告。

    报告结构:
        A1/A2. 数据生成与潜在因子结构
        A3. OLS 随特征维度变化
        A4. 系数不稳定性
        B1. PCA 累计解释方差
        B2. PCR 工作流
        B3. CV 曲线解释
        B4. 严格定义与公式
    """
    print("\n[报告] 生成 synthetic_report.md...")

    lines = [
        "# 第十四周 - Task A & B: 高维回归、PCA 与 PCR 实验报告",
        "",
        "## A1/A2. 数据生成与潜在因子结构",
        "",
        "### 数据规格",
        "",
        "- **样本量**: 200",
        "- **特征数**: 80",
        "- **潜在因子数**: 5 (只有前 3 个驱动 y)",
        "- **DGP**: y = 3*z1 - 2*z2 + 1.5*z3 + e",
        "",
        "### 潜在因子结构",
        "",
        "原始变量 X 由 5 个潜在因子 Z 通过线性组合生成:",
        "",
        "```",
        "X = Z @ W + 小噪声",
        "Z ~ N(0, I_5),  W ~ N(0, I) 是 5x80 的因子载荷矩阵",
        "```",
        "",
        "这是一个典型的低秩+噪声结构: 80 个原始变量的信息维度其实只有 5。",
        "这就是为什么说这是一份高维+信息冗余的数据。",
        "",
        "---",
        "",
        "## A3. OLS 随特征维度变化",
        "",
        "### 误差变化",
        "",
        "| p | Train RMSE | Test RMSE | Rank | Condition Number |",
        "|---|---|---|---|---|",
    ]
    for i, p in enumerate(a3_data["p_values"]):
        lines.append(
            f"| {p} | {a3_data['train_rmses'][i]:.4f} | {a3_data['test_rmses'][i]:.4f} | "
            f"{a3_data['ranks'][i]} | {a3_data['conds'][i]:.2e} |"
        )

    lines += [
        "",
        "### 为什么训练误差接近 0 是危险信号?",
        "",
        "当 p 接近或超过 n 时, OLS 可以完美拟合训练数据 (训练 RMSE -> 0),",
        "但这不是因为模型学到了真实规律, 而是因为参数太多、自由度太高,",
        "模型在记忆训练数据的噪声。此时测试误差反而会大幅上升 - 这就是过拟合。",
        "",
        "同时, 条件数随 p 增大而急剧上升, 说明矩阵越来越病态,",
        "OLS 的系数估计变得极度不稳定。",
        "",
        "---",
        "",
        "## A4. 系数不稳定性",
        "",
        f"- **稳定性指标** (平均变异系数): {a4_data['stability_index']:.4f}",
        "- 观察: 系数在不同随机切分下剧烈波动, 不仅是误差在波动, 系数本身也在波动",
        "- 系数不稳定意味着: 换一批数据, 模型给出的结论可能完全不同",
        "- 这在业务上是重大风险 - 如果模型每次训练后最重要变量都不同,",
        "  业务方无法信任模型的解释",
        "",
        "---",
        "",
        "## B1. PCA 累计解释方差",
        "",
        f"- 解释 90% 方差需要 **{k_90}** 个主成分",
        f"- 解释 95% 方差需要 **{k_95}** 个主成分",
        "- 总特征数: 80, 潜在因子数: 5",
        "",
        "前几个主成分已经解释了大部分方差, 说明原始 80 维空间",
        "实际上贴近一个约 5 维的低维子空间。这正是 PCA 降维的基础。",
        "",
        "---",
        "",
        "## B2. PCR 工作流",
        "",
        "### 流程",
        "",
        "```",
        "标准化 -> PCA(k) -> 保留前 k 个主成分 -> 线性回归",
        "```",
        "",
        "### 误差随 k 变化",
        "",
        "| k | Train RMSE | Test RMSE | CV RMSE |",
        "|---|---|---|---|",
    ]
    for i, k in enumerate(pcr_data["ks"]):
        lines.append(
            f"| {k} | {pcr_data['train_rmses'][i]:.4f} | "
            f"{pcr_data['test_rmses'][i]:.4f} | {pcr_data['cv_rmses'][i]:.4f} |"
        )

    lines += [
        "",
        f"- **最优 k**: {pcr_data['best_k']} (CV RMSE 最低)",
        f"- **OLS 基准**: train RMSE={pcr_data['ols_train_rmse']:.4f}, "
        f"test RMSE={pcr_data['ols_test_rmse']:.4f}",
        "",
        "---",
        "",
        "## B3. CV 曲线解释",
        "",
        "### PCR CV RMSE 代表什么?",
        "",
        "PCR CV RMSE 是在交叉验证中, 对每个折的训练集做 PCA + 回归,",
        "然后在验证集上计算的平均 RMSE。它反映了模型在未见过数据上的预期表现。",
        "",
        "### 与 train/test 曲线的关系",
        "",
        "- 训练 RMSE 随 k 增大持续下降 (更多主成分 = 更多自由度)",
        "- 测试 RMSE 和 CV RMSE 先降后升 (存在最优 k)",
        "- CV RMSE 通常介于 train 和 test RMSE 之间",
        "",
        "### 为什么 OLS 训练误差低但不代表更好?",
        "",
        "OLS 在原始 80 维空间中训练误差极低, 但这是因为 p 约等于 n 时",
        "模型有足够的自由度完美拟合训练数据。这种低训练误差是虚假的 -",
        "它来自过拟合而非真正的学习。PCR 通过限制 k 来控制模型复杂度,",
        "虽然训练误差略高, 但泛化能力更强。",
        "",
        "---",
        "",
        "## B4. 严格定义与公式",
        "",
        "### 1. OLS 估计式",
        "",
        "```",
        "beta_hat = (X'X)^{-1} X'y",
        "```",
        "最小化 ||y - X*beta||^2 的解析解。当 X'X 不可逆 (p > n 或共线性) 时无唯一解。",
        "",
        "### 2. 第一主成分的方差最大化定义",
        "",
        "```",
        "v1 = argmax_{||v||=1} Var(Xv) = argmax v'(X'X)v",
        "```",
        "在所有单位方向中, 找到使投影后方差最大的方向。",
        "",
        "### 3. PCR 流程的符号表达",
        "",
        "```",
        "Z_k = X @ V_k          # 将 X 投影到前 k 个主成分方向",
        "beta_pcr = (Z_k'Z_k)^{-1} Z_k' y   # 在主成分空间中做 OLS",
        "```",
        "其中 V_k 是 PCA 得到的前 k 个主成分方向矩阵 (p x k)。",
    ]

    path = RESULTS_DIR / "synthetic_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/synthetic_report.md")


def write_summary_report(c_results):
    """生成 summary_comparison.md - Task C 总结。

    报告结构:
        实验结果表格 (Sparse Truth + Latent-Factor Truth)
        核心问题讨论 (5 个问题)
        C4. 为什么主线是 Lasso vs PCR
        三条核心结论
    """
    print("[报告] 生成 summary_comparison.md...")

    sp = c_results["sparse"]
    lf = c_results["latent_factor"]

    lines = [
        "# 第十四周 - Task C: Lasso vs PCR - 变量筛选 vs 信息压缩",
        "",
        "## 实验结果",
        "",
        "### Sparse Truth (少数变量决定 y)",
        "",
        "| 方法 | Test RMSE | Test MAE | 模型复杂度 |",
        "|---|---|---|---|",
        f"| Lasso | {sp['lasso_rmse']:.4f} | {sp['lasso_mae']:.4f} | {sp['lasso_nz']} 个非零系数 |",
        f"| PCR | {sp['pcr_rmse']:.4f} | {sp['pcr_mae']:.4f} | {sp['pcr_k']} 个主成分 |",
        "",
        "### Latent-Factor Truth (y 由潜在因子驱动)",
        "",
        "| 方法 | Test RMSE | Test MAE | 模型复杂度 |",
        "|---|---|---|---|",
        f"| Lasso | {lf['lasso_rmse']:.4f} | {lf['lasso_mae']:.4f} | {lf['lasso_nz']} 个非零系数 |",
        f"| PCR | {lf['pcr_rmse']:.4f} | {lf['pcr_mae']:.4f} | {lf['pcr_k']} 个主成分 |",
        "",
        "---",
        "",
        "## 核心问题讨论",
        "",
        "### 1. Sparse Truth 时为什么 Lasso 更自然?",
        "",
        "当真实模型只有少数变量直接决定 y 时, Lasso 的 L1 惩罚可以",
        "自动识别出这些变量并将其他变量的系数压缩为 0。",
        "这正是变量筛选的用武之地 - 它直接回答了谁留下的问题。",
        "PCR 在这种场景下会把所有变量的信息都混在一起,",
        "无法区分哪些变量是信号、哪些是噪声。",
        "",
        "值得注意的是, Lasso 在本实验中选出了 13 个非零系数 (真实信号变量只有 5 个),",
        "说明 Lasso 存在一定的误选 (false positive)。这在实践中是常见的 -",
        "Lasso 的变量选择不是完美的, 尤其当特征之间存在微弱相关性时,",
        "它可能会多选一些变量。但即便如此, Lasso 的预测精度 (RMSE=0.536)",
        "远优于 PCR (RMSE=5.473), 因为它至少抓住了信号变量。",
        "",
        "### 2. Latent-Factor Truth 时为什么 PCR 更自然?",
        "",
        "当原始变量由少数潜在因子线性组合生成时,",
        "PCR 的 PCA 步骤正好能恢复这些潜在因子 (主成分 = 潜在因子)。",
        "它不关心哪个原始变量重要, 而是关心哪个方向的信息最重要。",
        "Lasso 在这种场景下可能会在高度相关的变量中随机选一个,",
        "丢失了其他相关变量中蕴含的同一份信息。",
        "",
        "### 3. Lasso 回答谁留下, PCR 回答什么?",
        "",
        "Lasso 回答的是哪些原始变量应该保留 - 这是一个**选择**问题。",
        "PCR 回答的是原始高维空间中最重要的是哪些方向 - 这是一个**压缩**问题。",
        "选择是在原变量空间中做减法, 压缩是在新空间中做投影。",
        "",
        "### 4. 业务方要更短的变量名单 -> Lasso",
        "",
        "如果业务方需要知道哪几个变量最重要, 用 Lasso。",
        "它直接输出非零系数对应的变量名单, 可解释性最强。",
        "",
        "### 5. 业务方要更稳的预测器 -> PCR",
        "",
        "如果业务方只关心预测准确性和稳定性, 用 PCR。",
        "它通过降维消除了共线性, 系数估计更稳定, 泛化能力更强。",
        "",
        "---",
        "",
        "## C4. 为什么这周主线是 Lasso vs PCR, 不是前向/后向选择?",
        "",
        "前向/后向选择和 Lasso 都属于**选择**路线 (selection),",
        "它们的目标都是从原始变量中挑出一个子集。",
        "本周的核心对比是 **selection vs compression** -",
        "两种完全不同的应对高维问题的思路。",
        "",
        "如果把前向/后向选择拉回来, 就变成了三种选择方法的比较,",
        "偏离了选择 vs 压缩的主线。",
        "",
        "前向/后向选择更接近 **selection** 路线 - 它在原始变量空间中做加减法,",
        "不会像 PCA 那样构造新的方向。",
        "",
        "---",
        "",
        "## 本周三条核心结论",
        "",
        "1. **高维下 OLS 的低训练误差是虚假的**: 当 p 接近或超过 n 时,",
        "   OLS 可以完美拟合训练数据, 但这只是过拟合。系数估计变得极度不稳定,",
        "   条件数急剧上升。这是一个危险信号, 不是好消息。",
        "",
        "2. **PCA/PCR 是信息压缩, 不是变量选择**: PCA 找到原始空间中最重要的方向,",
        "   PCR 在这些方向上做回归。它不关心哪个原始变量重要,",
        "   而是关心哪个方向的信息最重要。这在 latent-factor 结构下特别有效。",
        "",
        "3. **选择 vs 压缩取决于数据生成机制**: Sparse truth -> Lasso (选择),",
        "   Latent-factor truth -> PCR (压缩)。理解数据的真实结构,",
        "   才能选择正确的方法。",
    ]

    path = RESULTS_DIR / "summary_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/summary_comparison.md")


# ===== 主入口 ================================================================

def main():
    """主入口: Task A -> Task B -> Task C -> 报告。

    执行流程:
        1. 清空并重建 results/ 目录 (保证每次运行结果可复现)
        2. Task A: 生成高维数据, 展示 OLS 的失败
           - A1/A2: 生成 200x80 的模拟数据 (5 个潜在因子)
           - A3: OLS 随 p 变化的误差和矩阵结构
           - A4: 50 次随机切分下的系数不稳定性
        3. Task B: PCA + PCR
           - B1: PCA 累计解释方差曲线
           - B2: PCR 误差随 k 变化, 与 OLS 基准对比
        4. Task C: Lasso vs PCR
           - Sparse Truth 场景: Lasso 更优
           - Latent-Factor Truth 场景: 两者接近
        5. 生成两份中文报告
    """

    # ---- 动态清理: 新建或清空 results/ ----
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True)
    print(f"results/ 已清空并重建: {RESULTS_DIR}\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Task A ----
    X, y, a3_data, a4_data = run_task_a()

    # ---- Task B ----
    cumvar, k_90, k_95 = run_pca_analysis(X)
    pcr_data = run_pcr_comparison(X, y)

    # ---- Task C ----
    c_results = run_task_c()

    # ---- 生成报告 ----
    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)
    write_synthetic_report(a3_data, a4_data, cumvar, k_90, k_95, pcr_data)
    write_summary_report(c_results)

    # ---- 完成 ----
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  {DATA_DIR / 'synthetic_highdim.csv'}")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f}")
    print(f"  {RESULTS_DIR / 'synthetic_report.md'}")
    print(f"  {RESULTS_DIR / 'summary_comparison.md'}")


if __name__ == "__main__":
    main()
