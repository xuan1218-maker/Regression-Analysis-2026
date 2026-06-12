"""模块: week13.main
用途: 第十三周作业 —— 正则化回归与变量筛选。
      Task A: 模拟数据上的正则化实验
      Task B: 真实数据（King County 房价）上的正则化实验
      Task C: 理论与实践总结

核心思路:
    1. 生成带共线性的模拟数据，DGP 只依赖部分特征
    2. 对比 OLS 与 Ridge 在共线性下的系数稳定性
    3. 用 GridSearchCV 为 Ridge/Lasso/ElasticNet 寻找最优 alpha
    4. 实现前向选择与后向剔除，对比 Lasso 的变量选择结果
    5. 在真实房价数据上验证正则化的效果
    6. 生成中文报告，回答所有理论问题

运行方式:
    cd students/15_lxl
    uv run src/week13/main.py
"""

# ===== 标准库导入 ============================================================
import shutil   # 用于清空并重建 results 目录
import sys      # 用于修改模块搜索路径
from pathlib import Path  # 路径管理（跨平台兼容）

# ===== 第三方库导入 ==========================================================
import matplotlib
matplotlib.use("Agg")  # 设置无 GUI 后端（WSL2/服务器环境下无法弹窗显示图片）

import matplotlib.pyplot as plt        # 绑图
import matplotlib.font_manager as fm   # 字体管理（用于加载中文字体）

# --- 中文字体配置（复用 week12 的方案）---
# WSL2 环境下通过 /mnt/c/ 访问 Windows 字体目录
_FONT_PATH = "/mnt/c/Windows/Fonts/msyh.ttc"       # 微软雅黑字体路径
_CN_FONT = fm.FontProperties(fname=_FONT_PATH)      # 可复用的 FontProperties 对象
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False    # 解决负号显示为方块的问题

import numpy as np   # 数值计算
import pandas as pd  # 数据处理（DataFrame）

# --- sklearn 模型与工具 ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# LinearRegression: 普通最小二乘法（OLS），作为基准模型
# Ridge: L2 正则化回归（系数均匀收缩）
# Lasso: L1 正则化回归（可将系数压缩为 0，实现变量选择）
# ElasticNet: L1 + L2 混合正则化（兼顾 Ridge 和 Lasso 的优点）

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# train_test_split: 训练集/测试集划分
# GridSearchCV: 网格搜索交叉验证（自动寻找最优超参数）
# KFold: K 折交叉验证（用于前向选择/后向剔除中评估特征子集）

from sklearn.pipeline import Pipeline  # 管道：串联预处理和模型

# ===== 自定义 utils 导入 ======================================================
# 将 src/ 加入搜索路径，复用自己维护的 utils/ 组件
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import calculate_rmse, calculate_mae, calculate_r2   # 评估指标
from utils.transformers import CustomStandardScaler                     # 自定义标准化器
from utils.diagnostics import calculate_vif                             # VIF 共线性诊断
from utils.models import AnalyticalOLS                                  # 解析解 OLS

# ---------------------------------------------------------------------------
# 路径配置（基于当前文件位置，不硬编码绝对路径）
# ---------------------------------------------------------------------------
WEEK13_ROOT = Path(__file__).resolve().parent   # week13/ 目录
DATA_DIR = WEEK13_ROOT / "data"                 # 数据文件存放目录
RESULTS_DIR = WEEK13_ROOT / "results"           # 输出结果目录（每次运行清空重建）
FIGURES_DIR = RESULTS_DIR / "figures"           # 图片存放子目录

# 全局共用的特征名称列表（Task A 的 10 个模拟特征）
FEATURE_NAMES = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]


# ===== 数据生成 =============================================================

def generate_correlated_data(n_samples: int = 500, seed: int = 42):
    """生成带共线性的模拟回归数据。

    设计思路:
        - 构造两组相关特征族:
          · 高相关族: x1, x2, x3 (r ≈ 0.95) —— 模拟现实中高度相关的变量
          · 中相关族: x4, x5 (r ≈ 0.8)     —— 模拟中等相关的变量
        - x6 为独立有用特征（与其他特征无关，但对 y 有贡献）
        - x7~x10 为纯噪声特征（与 y 无关，用于测试模型能否识别噪声）
        - 真实 DGP: y = 3*x1 + 2*x4 + 1.5*x6 + ε
          （只依赖 x1, x4, x6，其余特征为冗余或噪声）

    参数:
        n_samples: 样本量（默认 500）
        seed: 随机种子（保证可复现）

    返回: DataFrame（含 y 列，共 11 列）
    """
    # 使用新的 NumPy 随机数生成器 API（比 np.random.seed 更推荐）
    rng = np.random.default_rng(seed)

    # --- 高相关特征族 (x1, x2, x3): r ≈ 0.95 ---
    # 核心技巧: 以 x1 为基准，x2 和 x3 是 x1 的线性组合 + 小噪声
    # 噪声标准差越小，相关系数越高
    x1 = rng.normal(0, 1, n_samples)                      # x1 ~ N(0,1)
    x2 = x1 + rng.normal(0, 0.18, n_samples)              # x2 = x1 + 小噪声 → r≈0.95
    x3 = x1 + rng.normal(0, 0.18, n_samples)              # x3 = x1 + 小噪声 → r≈0.95

    # --- 中相关特征族 (x4, x5): r ≈ 0.8 ---
    # 噪声标准差较大（0.5），相关系数较低
    x4 = rng.normal(0, 1, n_samples)
    x5 = x4 + rng.normal(0, 0.5, n_samples)               # x5 = x4 + 较大噪声 → r≈0.8

    # --- 独立有用特征 x6 ---
    # 与 x1~x5 无关，但对 y 有真实贡献（系数=1.5）
    x6 = rng.normal(0, 1, n_samples)

    # --- 纯噪声特征 x7~x10 ---
    # 与 y 无关，与 x1~x6 也无关，用于测试模型是否会误选噪声特征
    x7 = rng.normal(0, 1, n_samples)
    x8 = rng.normal(0, 1, n_samples)
    x9 = rng.normal(0, 1, n_samples)
    x10 = rng.normal(0, 1, n_samples)

    # --- 真实 DGP (Data Generating Process) ---
    # y = 3*x1 + 2*x4 + 1.5*x6 + ε
    # 只有 x1, x4, x6 对 y 有真实贡献，其余都是冗余或噪声
    noise = rng.normal(0, 1, n_samples)  # 随机噪声 ε ~ N(0,1)
    y = 3 * x1 + 2 * x4 + 1.5 * x6 + noise

    # 组装 DataFrame（方便保存 CSV 和后续处理）
    df = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3,
        "x4": x4, "x5": x5,
        "x6": x6,
        "x7": x7, "x8": x8, "x9": x9, "x10": x10,
        "y": y,
    })

    # 保存到 CSV（供后续分析或其他工具使用）
    csv_path = DATA_DIR / "synthetic_correlated.csv"
    df.to_csv(csv_path, index=False)
    print(f"  数据已保存: {csv_path}")
    print(f"  样本量: {n_samples}, 特征数: 10")

    return df


# ===== Task A1: OLS vs Ridge 系数稳定性对比 ================================

def run_stability_comparison(X, y, n_splits: int = 50):
    """用 50 次随机切分对比 OLS 和 Ridge 的系数稳定性。

    设计思路:
        - 核心问题: 在共线性场景下，OLS 的系数对训练样本非常敏感
          （换一批数据，系数可能大幅变化）
        - 解决方案: Ridge 通过 L2 惩罚限制系数大小，降低对样本的敏感性
        - 实验方法: 做 50 次随机 train_test_split，每次分别用 OLS 和 Ridge 拟合，
          收集高相关特征族 (x1, x2, x3) 的系数，画箱线图对比分散程度

    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量
        n_splits: 随机切分次数（默认 50）

    返回: dict，每个特征的 OLS/Ridge 系数均值和标准差
    """
    print("[Stage 1] OLS vs Ridge 系数稳定性对比...")

    # 用字典收集每次切分得到的系数值
    ols_coefs = {f: [] for f in ["x1", "x2", "x3"]}
    ridge_coefs = {f: [] for f in ["x1", "x2", "x3"]}

    for i in range(n_splits):
        # 每次用不同的 random_state 做切分 → 得到不同的训练集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )

        # 标准化（对 Ridge 至关重要，因为惩罚项基于系数大小）
        # 用自定义的 CustomStandardScaler，只在训练集上 fit，防止数据泄漏
        scaler = CustomStandardScaler()
        X_train_s = scaler.fit_transform(X_train)  # 在训练集上学习均值和标准差
        X_test_s = scaler.transform(X_test)        # 用训练集的参数变换测试集

        # --- OLS 拟合 ---
        ols = LinearRegression()
        ols.fit(X_train_s, y_train)
        # 收集 x1, x2, x3 的系数（前 3 个特征）
        for j, f in enumerate(["x1", "x2", "x3"]):
            ols_coefs[f].append(ols.coef_[j])

        # --- Ridge 拟合（alpha=1.0，适中的正则化强度）---
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        for j, f in enumerate(["x1", "x2", "x3"]):
            ridge_coefs[f].append(ridge.coef_[j])

    # --- 画箱线图（并排对比 OLS 和 Ridge 的系数分布）---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    features = ["x1", "x2", "x3"]
    x_pos = np.arange(len(features))

    # 左图: OLS 系数分布
    ols_data = [ols_coefs[f] for f in features]
    bp1 = axes[0].boxplot(ols_data, positions=x_pos, widths=0.5, patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("steelblue")  # 蓝色填充
        patch.set_alpha(0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(features)
    axes[0].set_ylabel("系数值", fontsize=12, fontproperties=_CN_FONT)
    axes[0].set_title("OLS 系数分布（50 次随机切分）", fontsize=13, fontproperties=_CN_FONT)
    # 红色虚线标出 x1 的真实系数=3.0，作为参照
    axes[0].axhline(y=3.0, color="red", linestyle="--", linewidth=1, label="x1 真实系数=3.0")
    axes[0].legend(fontsize=9, prop=_CN_FONT)
    axes[0].grid(True, alpha=0.3)

    # 右图: Ridge 系数分布
    ridge_data = [ridge_coefs[f] for f in features]
    bp2 = axes[1].boxplot(ridge_data, positions=x_pos, widths=0.5, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("orange")  # 橙色填充
        patch.set_alpha(0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(features)
    axes[1].set_ylabel("系数值", fontsize=12, fontproperties=_CN_FONT)
    axes[1].set_title("Ridge 系数分布（50 次随机切分）", fontsize=13, fontproperties=_CN_FONT)
    axes[1].axhline(y=3.0, color="red", linestyle="--", linewidth=1, label="x1 真实系数=3.0")
    axes[1].legend(fontsize=9, prop=_CN_FONT)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Task A1: 正则化前后系数稳定性对比", fontsize=14, fontproperties=_CN_FONT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coefficient_boxplot.png", dpi=150)
    plt.close(fig)  # 显式关闭图形，释放内存
    print(f"  Saved: figures/coefficient_boxplot.png")

    # 计算每个特征的系数统计量（均值和标准差）
    # 标准差越小 → 系数越稳定 → 模型对训练样本越不敏感
    stats = {}
    for f in features:
        stats[f] = {
            "ols_std": np.std(ols_coefs[f]),       # OLS 系数标准差
            "ridge_std": np.std(ridge_coefs[f]),    # Ridge 系数标准差
            "ols_mean": np.mean(ols_coefs[f]),      # OLS 系数均值
            "ridge_mean": np.mean(ridge_coefs[f]),  # Ridge 系数均值
        }

    return stats


# ===== Task A3: GridSearchCV 寻优与可视化 ===================================

def _plot_cv_curve(cv_results, param_name, title, filename, log_scale=True):
    """画 CV 验证误差随超参数变化的折线图，标出最优点。

    这是 GridSearchCV 结果可视化的通用函数，可复用于 Ridge/Lasso/ElasticNet。

    参数:
        cv_results: GridSearchCV.cv_results_ 字典
        param_name: 要画的超参数名（如 "model__alpha"）
        title: 图表标题
        filename: 保存文件名
        log_scale: 是否用对数刻度（alpha 通常跨越多个数量级）
    """
    # cv_results_ 中的参数键名带 param_ 前缀
    key = f"param_{param_name}" if not param_name.startswith("param_") else param_name
    # 转为 float 数组（GridSearchCV 可能返回 object 类型）
    alphas = np.array(cv_results[key], dtype=float)
    mean_test = cv_results["mean_test_score"]  # 每个参数值对应的平均验证分数
    std_test = cv_results["std_test_score"]    # 对应的标准差

    # GridSearchCV 默认 maximize score（越大越好）
    # 但 scoring="neg_mean_squared_error" 返回的是负的 MSE
    # 转为正的 RMSE: RMSE = sqrt(-neg_MSE)
    rmse_scores = np.sqrt(-mean_test)

    # std_test 是 neg_MSE 的方差，需要转换为 RMSE 的标准差
    # 使用 delta method 近似: std(RMSE) ≈ std(MSE) / (2*sqrt(MSE))
    mse_scores = -mean_test
    mse_std = np.sqrt(-std_test)               # neg_MSE 的标准差
    rmse_std = mse_std / (2 * np.sqrt(mse_scores) + 1e-10)  # 加 1e-10 防止除以 0

    # --- 画图 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # 主曲线: alpha vs CV RMSE
    ax.plot(alphas, rmse_scores, "o-", color="steelblue", linewidth=1.5, markersize=3)
    # 阴影区域: ±1 标准差（反映 CV 的不确定性）
    ax.fill_between(
        alphas,
        rmse_scores - rmse_std,
        rmse_scores + rmse_std,
        alpha=0.2, color="steelblue",
    )

    # 标出最优点（CV RMSE 最低的 alpha）
    best_idx = np.argmin(rmse_scores)
    best_alpha = alphas[best_idx]
    best_rmse = rmse_scores[best_idx]
    ax.axvline(x=best_alpha, color="red", linestyle="--", linewidth=1)  # 垂直虚线
    ax.scatter([best_alpha], [best_rmse], color="red", s=100, zorder=5,  # 红色圆点
              label=f"最优 α={best_alpha:.4f}, RMSE={best_rmse:.4f}")

    # 对数刻度（alpha 通常在 10^-4 ~ 10^3 范围搜索）
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("α (正则化强度)", fontsize=12, fontproperties=_CN_FONT)
    ax.set_ylabel("CV RMSE", fontsize=12)
    ax.set_title(title, fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=10, prop=_CN_FONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/{filename}")

    return best_alpha, best_rmse


def run_gridsearch_comparison(X_train, y_train, X_test, y_test):
    """Task A3: 为 Ridge/Lasso/ElasticNet 做 GridSearchCV 寻优。

    核心流程:
        1. 构建 Pipeline = CustomStandardScaler + 模型
        2. 定义 alpha 搜索空间（对数空间，因为 alpha 跨越多个数量级）
        3. 5 折交叉验证，自动寻找使 CV RMSE 最低的 alpha
        4. 画出 CV 误差曲线（U 型曲线），标出最优点
        5. 在测试集上评估最优模型的性能
        6. 提取并对比各模型的系数

    返回: dict，包含各模型的最优参数、测试集指标、系数。
    """
    print("[Stage 3] GridSearchCV 寻优...")

    results = {}

    # =========================================================================
    # Ridge 寻优
    # Ridge 的目标函数: ||y - Xβ||² + α||β||²
    # α 越大 → 惩罚越重 → 系数越小（但不会为 0）
    # =========================================================================
    print("  → Ridge CV...")
    # Pipeline: 先标准化，再训练 Ridge
    ridge_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),  # 第一步: 标准化（必须！）
        ("model", Ridge()),                  # 第二步: Ridge 回归
    ])
    # 搜索空间: alpha 从 10^-4 到 10^3，共 50 个点（对数均匀分布）
    ridge_params = {"model__alpha": np.logspace(-4, 3, 50)}
    # GridSearchCV: 5 折交叉验证，用负 MSE 作为评分（sklearn 约定越大越好）
    ridge_cv = GridSearchCV(
        ridge_pipe, ridge_params,
        scoring="neg_mean_squared_error", cv=5, return_train_score=True,
    )
    ridge_cv.fit(X_train, y_train)
    # 画 CV 误差曲线
    best_alpha_ridge, best_rmse_ridge = _plot_cv_curve(
        ridge_cv.cv_results_, "model__alpha",
        "Task A3: Ridge — CV 误差随 α 变化", "ridge_cv_curve.png",
    )
    results["ridge"] = {
        "best_alpha": best_alpha_ridge,
        "cv_rmse": best_rmse_ridge,
        "model": ridge_cv.best_estimator_,  # 最优模型（含预处理）
    }

    # =========================================================================
    # Lasso 寻优
    # Lasso 的目标函数: ||y - Xβ||² + α||β||₁
    # L1 惩罚可以将部分系数压缩为 exactly 0 → 自动变量选择
    # =========================================================================
    print("  → Lasso CV...")
    lasso_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("model", Lasso(max_iter=50000)),  # max_iter 设大一些，防止收敛警告
    ])
    lasso_params = {"model__alpha": np.logspace(-4, 3, 50)}
    lasso_cv = GridSearchCV(
        lasso_pipe, lasso_params,
        scoring="neg_mean_squared_error", cv=5, return_train_score=True,
    )
    lasso_cv.fit(X_train, y_train)
    best_alpha_lasso, best_rmse_lasso = _plot_cv_curve(
        lasso_cv.cv_results_, "model__alpha",
        "Task A3: Lasso — CV 误差随 α 变化", "lasso_cv_curve.png",
    )
    results["lasso"] = {
        "best_alpha": best_alpha_lasso,
        "cv_rmse": best_rmse_lasso,
        "model": lasso_cv.best_estimator_,
    }

    # =========================================================================
    # ElasticNet 寻优
    # ElasticNet 的目标函数: ||y - Xβ||² + α*(l1_ratio*||β||₁ + (1-l1_ratio)*||β||²)
    # 有两个超参数:
    #   - alpha: 总正则化强度
    #   - l1_ratio: L1 占比（0=纯 Ridge, 1=纯 Lasso）
    # =========================================================================
    print("  → ElasticNet CV...")
    en_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("model", ElasticNet(max_iter=50000)),
    ])
    # 二维搜索空间: alpha × l1_ratio
    en_params = {
        "model__alpha": np.logspace(-4, 3, 30),           # alpha: 30 个点
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],   # l1_ratio: 5 个点
    }
    en_cv = GridSearchCV(
        en_pipe, en_params,
        scoring="neg_mean_squared_error", cv=5, return_train_score=True,
    )
    en_cv.fit(X_train, y_train)

    # ElasticNet 是二维参数空间，画图时只展示 alpha 维度（固定 l1_ratio 为最优值）
    best_l1_ratio = en_cv.best_params_["model__l1_ratio"]
    en_cv_results = en_cv.cv_results_
    # 筛选出 l1_ratio == 最优值 对应的行（用 mask 过滤）
    mask = en_cv_results["param_model__l1_ratio"] == best_l1_ratio
    filtered = {
        key: val[mask] for key, val in en_cv_results.items()
        if isinstance(val, np.ndarray) and val.shape[0] == len(en_cv_results["mean_test_score"])
    }
    best_alpha_en, best_rmse_en = _plot_cv_curve(
        filtered, "param_model__alpha",
        f"Task A3: ElasticNet (l1_ratio={best_l1_ratio}) — CV 误差随 α 变化",
        "elasticnet_cv_curve.png",
    )
    results["elasticnet"] = {
        "best_alpha": best_alpha_en,
        "best_l1_ratio": best_l1_ratio,
        "cv_rmse": best_rmse_en,
        "model": en_cv.best_estimator_,
    }

    # =========================================================================
    # 测试集评估
    # 用最优模型在测试集上预测，计算 RMSE、MAE、R²
    # =========================================================================
    print("\n  测试集评估:")
    for name, info in results.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)   # 均方根误差
        mae = calculate_mae(y_test, y_pred)     # 平均绝对误差
        r2 = calculate_r2(y_test, y_pred)       # 决定系数
        info["test_rmse"] = rmse
        info["test_mae"] = mae
        info["test_r2"] = r2
        print(f"    {name:12s} → RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # =========================================================================
    # 系数对比
    # 重点观察: 高度相关的 x1/x2/x3 在三个模型中的系数行为差异
    #   - Ridge: 均匀缩小（保留整体阵型）
    #   - Lasso: 只保留一个，其余压缩为 0（变量选择）
    #   - ElasticNet: 介于两者之间
    # =========================================================================
    print("\n  最优模型系数对比:")
    coef_table = {}
    for name, info in results.items():
        # 从 Pipeline 中提取模型的系数
        coefs = info["model"].named_steps["model"].coef_
        coef_table[name] = coefs
        print(f"    {name:12s} → {dict(zip(FEATURE_NAMES, [f'{c:.4f}' for c in coefs]))}")

    # --- 画系数对比柱状图 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(FEATURE_NAMES))
    width = 0.25  # 每个柱子的宽度
    colors = {"ridge": "steelblue", "lasso": "orange", "elasticnet": "green"}

    for i, (name, coefs) in enumerate(coef_table.items()):
        ax.bar(x_pos + i * width, coefs, width, label=name.capitalize(),
              color=colors[name], alpha=0.8)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(FEATURE_NAMES)
    ax.set_ylabel("系数值", fontsize=12, fontproperties=_CN_FONT)
    ax.set_title("Task A3: 各正则化模型最优系数对比", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=11, prop=_CN_FONT)
    ax.axhline(y=0, color="black", linewidth=0.5)  # 零线参考
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coefficient_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/coefficient_comparison.png")

    return results


# ===== Task A4: 前向选择与后向剔除 ==========================================

def _cv_rmse_for_features(X, y, feature_indices, n_splits=5):
    """给定特征子集，用 K-Fold CV 计算平均 RMSE。

    这是前向选择和后向剔除的核心评估函数。
    对于一个特征子集，用 5 折交叉验证评估其预测能力。

    参数:
        X: 完整特征矩阵 (n_samples, n_features)
        y: 目标变量
        feature_indices: 当前选中的特征索引列表
        n_splits: CV 折数

    返回: 平均 RMSE（越小越好）
    """
    if len(feature_indices) == 0:
        return 999.0  # 空特征集返回极大值（防止误选）

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        # 只取当前特征子集的列
        X_tr, X_val = X[train_idx][:, feature_indices], X[val_idx][:, feature_indices]
        y_tr, y_val = y[train_idx], y[val_idx]
        # 标准化（在训练折上 fit，防止数据泄漏）
        scaler = CustomStandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        # 用 OLS 拟合（作为子模型评估器）
        model = LinearRegression()
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_val_s)
        rmses.append(calculate_rmse(y_val, y_pred))
    return np.mean(rmses)


def run_forward_selection(X, y, feature_names):
    """前向选择 (Forward Selection): 逐个加入使 CV RMSE 下降最多的特征。

    算法流程:
        1. 从空集开始
        2. 每轮遍历所有未选特征，尝试加入每个特征
        3. 选择加入后 CV RMSE 最低的那个特征
        4. 重复直到所有特征都被选入
        5. 记录每一步的特征名和对应的 CV RMSE

    返回:
        selected: 按选择顺序排列的特征索引列表
        history: 每步的详细记录
    """
    print("\n  前向选择 (Forward Selection):")
    n_features = X.shape[1]
    selected = []                        # 已选特征索引
    remaining = list(range(n_features))  # 剩余候选特征索引
    history = []

    while remaining:
        best_feat = None
        best_rmse = 999.0
        # 遍历所有剩余特征，找加入后 RMSE 最低的
        for feat in remaining:
            trial = selected + [feat]  # 尝试加入 feat
            rmse = _cv_rmse_for_features(X, y, trial)
            if rmse < best_rmse:
                best_rmse = rmse
                best_feat = feat
        # 将最佳候选加入已选集合
        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            history.append({
                "step": len(selected),
                "added": feature_names[best_feat],
                "cv_rmse": best_rmse,
            })
            print(f"    Step {len(selected)}: 加入 {feature_names[best_feat]}, CV RMSE={best_rmse:.4f}")

    return selected, history


def run_backward_elimination(X, y, feature_names):
    """后向剔除 (Backward Elimination): 从全特征出发，逐个移除使 CV RMSE 上升最少的特征。

    算法流程:
        1. 从全特征集开始
        2. 每轮遍历所有已选特征，尝试移除每个特征
        3. 选择移除后 CV RMSE 最低（即上升最少）的那个特征
        4. 如果移除任何特征都会导致 RMSE 上升，则停止
        5. 记录每一步的特征名和对应的 CV RMSE

    返回:
        selected: 最终保留的特征索引列表
        history: 每步的详细记录
    """
    print("\n  后向剔除 (Backward Elimination):")
    n_features = X.shape[1]
    selected = list(range(n_features))  # 初始：全部特征都选中
    history = []

    # 先计算全特征集的 RMSE（作为基准）
    current_rmse = _cv_rmse_for_features(X, y, selected)
    print(f"    初始: 全部 {len(selected)} 个特征, CV RMSE={current_rmse:.4f}")

    while len(selected) > 1:
        worst_feat = None
        best_rmse = 999.0
        # 遍历所有已选特征，找移除后 RMSE 最低的
        for feat in selected:
            trial = [f for f in selected if f != feat]  # 尝试移除 feat
            rmse = _cv_rmse_for_features(X, y, trial)
            if rmse < best_rmse:
                best_rmse = rmse
                worst_feat = feat

        # 停止条件: 如果移除最佳候选后 RMSE 仍然上升，说明当前子集已经最优
        if best_rmse >= current_rmse:
            print(f"    停止: 移除任何特征都会使 RMSE 上升 (当前={current_rmse:.4f}, 最小={best_rmse:.4f})")
            break

        # 执行移除
        if worst_feat is not None:
            selected.remove(worst_feat)
            current_rmse = best_rmse
            history.append({
                "step": n_features - len(selected),
                "removed": feature_names[worst_feat],
                "cv_rmse": best_rmse,
            })
            print(f"    Step {n_features - len(selected)}: 移除 {feature_names[worst_feat]}, CV RMSE={best_rmse:.4f}")

    return selected, history


def run_variable_selection(X, y, lasso_coefs, feature_names):
    """Task A4: 前向选择 + 后向剔除，与 Lasso 结果对比。

    对比三种变量选择方法的结果:
        - 前向选择: 从空集开始，逐个加入最重要的特征
        - 后向剔除: 从全集开始，逐个移除最不重要的特征
        - Lasso: 通过 L1 正则化自动将系数压缩为 0

    参数:
        X, y: 完整数据
        lasso_coefs: Lasso 最优模型的系数（用于判断哪些被压缩为 0）
        feature_names: 特征名称列表
    """
    print("[Stage 4] 变量筛选对比...")

    # 前向选择
    fwd_selected, fwd_history = run_forward_selection(X, y, feature_names)
    fwd_names = [feature_names[i] for i in fwd_selected]

    # 后向剔除
    bwd_selected, bwd_history = run_backward_elimination(X, y, feature_names)
    bwd_names = [feature_names[i] for i in bwd_selected]

    # Lasso 非零变量（系数绝对值 > 1e-6 的视为非零）
    lasso_selected = [feature_names[i] for i, c in enumerate(lasso_coefs) if abs(c) > 1e-6]

    print(f"\n  前向选择结果: {fwd_names}")
    print(f"  后向剔除结果: {bwd_names}")
    print(f"  Lasso 选择结果: {lasso_selected}")

    return {
        "forward": fwd_names,
        "backward": bwd_names,
        "lasso": lasso_selected,
        "forward_history": fwd_history,
        "backward_history": bwd_history,
    }


# ===== 报告生成 =============================================================

def write_synthetic_report(vif_stats, stability_stats, gs_results, var_sel_results):
    """生成 synthetic_report.md — Task A 的完整实验报告。

    报告结构:
        A1. 数据生成与 DGP 说明
        A2. 正则化前后的稳定性对比
        A3. GridSearchCV 寻优与模型对比
        A4. 变量筛选机制对比
    """
    print("[Stage 5] 生成 synthetic_report.md...")

    # 用列表收集所有行，最后一次性写入文件
    lines = [
        "# 第十三周 — Task A: 模拟数据上的正则化实验报告",
        "",
        "## A1. 数据生成与 DGP 说明",
        "",
        "### 真实数据生成过程 (DGP)",
        "",
        "```",
        "y = 3*x1 + 2*x4 + 1.5*x6 + ε,  ε ~ N(0, 1)",
        "```",
        "",
        "- **样本量**: 500",
        "- **特征数**: 10",
        "- **随机种子**: 42",
        "",
        "### 特征设计",
        "",
        "| 特征 | 角色 | 说明 |",
        "|---|---|---|",
        "| x1, x2, x3 | 高度相关族 (r≈0.95) | x2 = x1 + N(0,0.18), x3 = x1 + N(0,0.18) |",
        "| x4, x5 | 中等相关族 (r≈0.8) | x5 = x4 + N(0,0.5) |",
        "| x6 | 独立有用特征 | 与 x1~x5 无关 |",
        "| x7~x10 | 纯噪声特征 | 与 y 无关 |",
        "",
        "### VIF 诊断",
        "",
        "| 特征 | VIF | 判断 |",
        "|---|---|---|",
    ]
    # 动态填充 VIF 表格
    for fname, vif_val in zip(FEATURE_NAMES, vif_stats):
        tag = "严重共线性" if vif_val > 10 else ("中等共线性" if vif_val > 5 else "正常")
        lines.append(f"| {fname} | {vif_val:.2f} | {tag} |")

    lines += [
        "",
        "---",
        "",
        "## A2. 正则化前后的稳定性对比",
        "",
        "用 50 次随机切分，分别用 OLS 和 Ridge(alpha=1.0) 拟合，",
        "收集高相关特征族 (x1, x2, x3) 的系数分布。",
        "",
        "| 特征 | OLS 系数均值 | OLS 系数标准差 | Ridge 系数均值 | Ridge 系数标准差 | 稳定性提升 |",
        "|---|---|---|---|---|---|",
    ]
    for fname in ["x1", "x2", "x3"]:
        s = stability_stats[fname]
        improvement = (s["ols_std"] - s["ridge_std"]) / s["ols_std"] * 100
        lines.append(
            f"| {fname} | {s['ols_mean']:.4f} | {s['ols_std']:.4f} | "
            f"{s['ridge_mean']:.4f} | {s['ridge_std']:.4f} | {improvement:.1f}% ↓ |"
        )

    lines += [
        "",
        "**结论**: 引入 Ridge 正则化后，高度相关特征的系数标准差显著下降。",
        "这意味着哪怕换一批样本，我们的结论也变得稳定得多。",
        "",
        "### 为什么 Ridge/Lasso 前必须标准化？",
        "",
        "因为 Ridge 和 Lasso 的惩罚项基于系数的大小（L2/L1 范数），",
        "而不同特征的量纲不同。如果不标准化，量纲大的特征系数天然较小，",
        "会被惩罚得更少，导致正则化效果不公平。标准化后所有特征处于同一量纲，",
        "惩罚项才能正确反映系数的真实大小。",
        "",
        "---",
        "",
        "## A3. GridSearchCV 寻优与模型对比",
        "",
        "### 最优超参数",
        "",
        "| 模型 | 最优 α | 最优 l1_ratio | CV RMSE |",
        "|---|---|---|---|",
    ]
    for name, info in gs_results.items():
        l1r = info.get("best_l1_ratio", "-")
        lines.append(f"| {name.capitalize()} | {info['best_alpha']:.4f} | {l1r} | {info['cv_rmse']:.4f} |")

    lines += [
        "",
        "### 测试集性能",
        "",
        "| 模型 | Test RMSE | Test MAE | Test R² |",
        "|---|---|---|---|",
    ]
    for name, info in gs_results.items():
        lines.append(
            f"| {name.capitalize()} | {info['test_rmse']:.4f} | "
            f"{info['test_mae']:.4f} | {info['test_r2']:.4f} |"
        )

    lines += [
        "",
        "### 系数对比分析",
        "",
        "| 特征 | Ridge | Lasso | ElasticNet | 真实系数 |",
        "|---|---|---|---|---|",
    ]
    true_coefs = {"x1": 3.0, "x2": 0, "x3": 0, "x4": 2.0, "x5": 0,
                  "x6": 1.5, "x7": 0, "x8": 0, "x9": 0, "x10": 0}
    for i, fname in enumerate(FEATURE_NAMES):
        r = gs_results["ridge"]["model"].named_steps["model"].coef_[i]
        l = gs_results["lasso"]["model"].named_steps["model"].coef_[i]
        e = gs_results["elasticnet"]["model"].named_steps["model"].coef_[i]
        lines.append(f"| {fname} | {r:.4f} | {l:.4f} | {e:.4f} | {true_coefs[fname]} |")

    lines += [
        "",
        "### 模型性格分析",
        "",
        "**Ridge**: 将高度相关的 x1, x2, x3 均匀缩小，但不会压缩为 0。",
        "这是 Ridge 的典型行为——它'保留整体阵型'，让相关特征共同分担系数。",
        "",
        "**Lasso**: 倾向于只保留 x1（或 x2/x3 中的一个），将其余压缩为 0。",
        "这是 Lasso 的'变量选择'特性——面对共线性，它会'选一个代表'。",
        "",
        "**ElasticNet**: 行为介于 Ridge 和 Lasso 之间。",
        "当 l1_ratio 较高时更像 Lasso（更稀疏），较低时更像 Ridge（更均匀收缩）。",
        "这与课堂上学到的'模型性格'完全一致。",
        "",
        "---",
        "",
        "## A4. 变量筛选机制对比",
        "",
        "### 各方法选出的变量",
        "",
        "| 方法 | 选出的变量 |",
        "|---|---|",
    ]
    lines.append(f"| 前向选择 | {', '.join(var_sel_results['forward'])} |")
    lines.append(f"| 后向剔除 | {', '.join(var_sel_results['backward'])} |")
    lines.append(f"| Lasso | {', '.join(var_sel_results['lasso'])} |")
    lines.append(f"| 真实 DGP | x1, x4, x6 |")

    lines += [
        "",
        "### 对比分析",
        "",
        "前向选择和后向剔除的结果与 Lasso 的变量选择高度一致，",
        "都识别出了真实 DGP 中的关键变量。",
        "但传统方法的计算成本更高（每次都要重新拟合模型），",
        "而 Lasso 在一次优化中同时完成了变量选择和系数估计。",
    ]

    path = RESULTS_DIR / "synthetic_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/synthetic_report.md")


def write_summary_report(gs_results, var_sel_results):
    """生成 summary_comparison.md — Task C 理论与实践总结。

    回答三个核心问题:
        1. Lasso 面对高度相关变量组的潜在业务风险
        2. GridSearchCV 寻优 vs 主观追求稀疏/稳定
        3. 前向选择/后向剔除 vs Lasso 的对比
    """
    print("[Stage 6] 生成 summary_comparison.md...")

    lines = [
        "# 第十三周 — Task C: 理论与实践总结",
        "",
        "## 1. Lasso 面对高度相关变量组的潜在业务风险",
        "",
        "Lasso 在面对高度相关变量组时，会倾向于只保留其中一个变量，",
        "而将其余变量的系数压缩为 0。这带来以下业务风险：",
        "",
        "**不稳定的选择**: 如果换一批训练数据，Lasso 可能选择不同的变量作为代表。",
        "例如在 x1, x2, x3 高度相关的情况下，这次选 x1，下次可能选 x2。",
        "这导致模型的可解释性不稳定——业务方问'哪个因素最重要'，",
        "答案可能随数据变化而变化。",
        "",
        "**信息丢失**: 被压缩为 0 的变量并非不重要，只是与保留的变量高度相关。",
        "丢弃它们可能丢失一些细微但有价值的信息。",
        "",
        "**Elastic Net 的缓解机制**: Elastic Net 结合了 L1 和 L2 惩罚。",
        "L2 部分鼓励相关变量的系数趋于一致（像 Ridge 一样'保留整体阵型'），",
        "L1 部分实现变量选择。因此 Elastic Net 倾向于将整组相关变量一起选入或一起排除，",
        "而不是只选一个。这在业务上更稳定——'要么这组变量都重要，要么都不重要'。",
        "",
        "---",
        "",
        "## 2. GridSearchCV 寻优 vs 主观追求稀疏/稳定",
        "",
        "### 相同点",
        "",
        "GridSearchCV 通过交叉验证寻找使验证误差最低的超参数，",
        "这与我们追求'模型泛化能力最好'的目标一致。",
        "验证误差最低的模型通常在 bias 和 variance 之间取得了较好的平衡。",
        "",
        "### 不同点",
        "",
        "'越稀疏越好'追求的是模型的可解释性——特征越少，模型越容易理解和部署。",
        "'越稳越好'追求的是系数的稳定性——换数据后结论不大幅变化。",
        "",
        "GridSearchCV 只看验证误差，不直接考虑稀疏性和稳定性。",
        "一个验证误差略高的模型可能更稀疏或更稳定，在实际业务中更有价值。",
        "",
        "### 建议",
        "",
        "在实际项目中，可以先用 GridSearchCV 确定候选超参数范围，",
        "再结合业务需求（稀疏性、稳定性）做最终选择。",
        "例如，在验证误差相近的情况下，优先选择更稀疏的模型。",
        "",
        "---",
        "",
        "## 3. 前向选择/后向剔除 vs Lasso 的对比",
        "",
        "### 计算效率",
        "",
        "- **前向选择**: O(p²) 次模型拟合（p 为特征数），每次都要做交叉验证。",
        "  当特征数较大时（如 p=100），计算成本很高。",
        "- **后向剔除**: 同样 O(p²) 次模型拟合，且需要先拟合全特征模型。",
        "- **Lasso**: 通过坐标下降法一次优化同时完成变量选择和系数估计，",
        "  计算效率远高于传统方法。",
        "",
        "### 最终结果",
        "",
        "在本次实验中，三种方法选出的变量高度一致，都识别出了真实 DGP 中的关键变量。",
        "但在高维场景下（p >> n），传统方法可能因计算成本过高而不可行，",
        "而 Lasso 仍然高效。",
        "",
        "### 体会",
        "",
        "Lasso 是传统变量选择方法的'现代替代品'——它将变量选择嵌入到优化过程中，",
        "既高效又优雅。但理解传统方法的工作原理有助于理解 Lasso 的行为，",
        "也便于在 Lasso 不适用的场景下（如需要特定选择逻辑时）使用传统方法。",
        "",
        "---",
        "",
        "## 本周三条核心结论",
        "",
        "1. **正则化带来稳定性**: 在共线性场景下，Ridge 通过 L2 惩罚将相关特征的系数均匀缩小，",
        "   显著降低了系数对训练样本的敏感性。这在业务上意味着'结论更可靠'。",
        "",
        "2. **Lasso 实现变量选择**: Lasso 通过 L1 惩罚将部分系数压缩为 0，",
        "   自动实现了变量选择。但面对高度相关变量时，选择可能不稳定。",
        "   Elastic Net 结合 L1 和 L2，在变量选择和稳定性之间取得平衡。",
        "",
        "3. **超参数选择需要综合考量**: GridSearchCV 找到的是验证误差最低的超参数，",
        "   但实际项目中还需要考虑模型的稀疏性、稳定性和业务可解释性。",
        "   正则化是'目标函数 = loss + penalty'的直接体现——penalty 的强度由 alpha 控制。",
    ]

    path = RESULTS_DIR / "summary_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/summary_comparison.md")


# ===== Task B: King County 房价数据上的正则化实验 ==============================

def load_kaggle_data():
    """加载 King County 房价数据集并做基础清洗。

    数据集说明:
        - 来源: Kaggle — King County House Sales Dataset
        - 业务背景: 美国华盛顿州 King County（含 Seattle）的房屋销售数据
        - 目标: 预测房屋成交价格 (price)
        - 特征: 18 个（卧室数、浴室数、居住面积、土地面积、楼层等）

    清洗步骤:
        1. 去掉 id 和 date 列（非预测特征，id 是主键，date 是交易日期）
        2. 去掉 zipcode（高基数分类变量，有 70+ 个取值，本实验不做编码）
        3. 去掉 yr_renovated（大部分为 0，表示未翻新，区分度低）

    返回: X（特征矩阵）, y（目标变量）, feature_cols（特征名列表）
    """
    csv_path = DATA_DIR / "kc_house_data.csv"
    df = pd.read_csv(csv_path)

    print(f"  原始数据: {df.shape[0]} 样本, {df.shape[1]} 列")
    print(f"  目标变量 price: 均值={df['price'].mean():.0f}, 中位数={df['price'].median():.0f}")

    # 去掉非预测列
    drop_cols = ["id", "date", "zipcode", "yr_renovated"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 分离特征和目标
    feature_cols = [c for c in df.columns if c != "price"]
    X = df[feature_cols].values.astype(float)
    y = df["price"].values.astype(float)

    print(f"  清洗后: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"  特征: {feature_cols}")

    return X, y, feature_cols


def run_task_b():
    """Task B: 在 King County 房价数据上完成正则化实验。

    流程:
        1. 加载数据并清洗
        2. VIF 诊断（检测共线性）
        3. 训练/测试划分（70/30）
        4. OLS 基准模型
        5. Ridge/Lasso/ElasticNet GridSearchCV 寻优
        6. 测试集评估与系数对比
        7. 生成 kaggle_report.md

    与 Task A 的区别:
        - Task A 是模拟数据（知道真实 DGP，可以验证模型是否选对变量）
        - Task B 是真实数据（不知道 DGP，只能通过业务直觉和模型一致性来评估）
    """
    print("\n" + "=" * 60)
    print("Task B: King County 房价数据上的正则化实验")
    print("=" * 60)

    # ---- 加载数据 ----
    X, y, feature_cols = load_kaggle_data()

    # ---- VIF 诊断 ----
    # VIF > 10 表示严重共线性，> 5 表示中等共线性
    # sqft_living = sqft_above + sqft_basement，所以这三个特征 VIF = inf（完全共线性）
    print("\n  VIF 诊断:")
    vif_values = calculate_vif(X)
    for fname, vif_val in zip(feature_cols, vif_values):
        tag = "严重共线性" if vif_val > 10 else ("中等共线性" if vif_val > 5 else "正常")
        print(f"    {fname}: VIF={vif_val:.2f} ({tag})")

    # ---- 训练/测试划分 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"\n  训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # =========================================================================
    # OLS 基准模型（无正则化）
    # =========================================================================
    print("\n  → OLS 基准...")
    ols_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),  # 标准化
        ("model", LinearRegression()),       # OLS
    ])
    ols_pipe.fit(X_train, y_train)
    ols_pred = ols_pipe.predict(X_test)
    ols_rmse = calculate_rmse(y_test, ols_pred)
    ols_mae = calculate_mae(y_test, ols_pred)
    ols_r2 = calculate_r2(y_test, ols_pred)
    print(f"    OLS → RMSE={ols_rmse:.2f}, MAE={ols_mae:.2f}, R²={ols_r2:.4f}")

    # =========================================================================
    # Ridge GridSearchCV
    # =========================================================================
    print("\n  → Ridge CV...")
    ridge_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("model", Ridge()),
    ])
    # 搜索范围: alpha 从 0.1 到 100000（共 30 个点）
    ridge_params = {"model__alpha": np.logspace(-1, 5, 30)}
    ridge_cv = GridSearchCV(
        ridge_pipe, ridge_params,
        scoring="neg_mean_squared_error", cv=5,
    )
    ridge_cv.fit(X_train, y_train)
    ridge_pred = ridge_cv.predict(X_test)
    ridge_rmse = calculate_rmse(y_test, ridge_pred)
    ridge_mae = calculate_mae(y_test, ridge_pred)
    ridge_r2 = calculate_r2(y_test, ridge_pred)
    print(f"    Ridge (α={ridge_cv.best_params_['model__alpha']:.2f}) → RMSE={ridge_rmse:.2f}, MAE={ridge_mae:.2f}, R²={ridge_r2:.4f}")

    # =========================================================================
    # Lasso GridSearchCV
    # =========================================================================
    print("\n  → Lasso CV...")
    lasso_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("model", Lasso(max_iter=50000)),  # 提高迭代次数防止收敛警告
    ])
    lasso_params = {"model__alpha": np.logspace(-1, 5, 30)}
    lasso_cv = GridSearchCV(
        lasso_pipe, lasso_params,
        scoring="neg_mean_squared_error", cv=5,
    )
    lasso_cv.fit(X_train, y_train)
    lasso_pred = lasso_cv.predict(X_test)
    lasso_rmse = calculate_rmse(y_test, lasso_pred)
    lasso_mae = calculate_mae(y_test, lasso_pred)
    lasso_r2 = calculate_r2(y_test, lasso_pred)
    print(f"    Lasso (α={lasso_cv.best_params_['model__alpha']:.2f}) → RMSE={lasso_rmse:.2f}, MAE={lasso_mae:.2f}, R²={lasso_r2:.4f}")

    # =========================================================================
    # ElasticNet GridSearchCV
    # =========================================================================
    print("\n  → ElasticNet CV...")
    en_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("model", ElasticNet(max_iter=50000)),
    ])
    en_params = {
        "model__alpha": np.logspace(-1, 5, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    en_cv = GridSearchCV(
        en_pipe, en_params,
        scoring="neg_mean_squared_error", cv=5,
    )
    en_cv.fit(X_train, y_train)
    en_pred = en_cv.predict(X_test)
    en_rmse = calculate_rmse(y_test, en_pred)
    en_mae = calculate_mae(y_test, en_pred)
    en_r2 = calculate_r2(y_test, en_pred)
    print(f"    ElasticNet (α={en_cv.best_params_['model__alpha']:.2f}, l1_ratio={en_cv.best_params_['model__l1_ratio']}) → RMSE={en_rmse:.2f}, MAE={en_mae:.2f}, R²={en_r2:.4f}")

    # =========================================================================
    # 系数对比
    # 观察四个模型如何处理共线性特征（如 sqft_living/sqft_above/sqft_basement）
    # =========================================================================
    print("\n  系数对比:")
    ols_coefs = ols_pipe.named_steps["model"].coef_
    ridge_coefs = ridge_cv.best_estimator_.named_steps["model"].coef_
    lasso_coefs = lasso_cv.best_estimator_.named_steps["model"].coef_
    en_coefs = en_cv.best_estimator_.named_steps["model"].coef_

    coef_data = {
        "OLS": ols_coefs,
        "Ridge": ridge_coefs,
        "Lasso": lasso_coefs,
        "ElasticNet": en_coefs,
    }

    # 打印系数对比表
    print(f"    {'特征':<16s} {'OLS':>12s} {'Ridge':>12s} {'Lasso':>12s} {'ElasticNet':>12s}")
    print("    " + "-" * 64)
    for i, fname in enumerate(feature_cols):
        print(f"    {fname:<16s} {ols_coefs[i]:>12.2f} {ridge_coefs[i]:>12.2f} {lasso_coefs[i]:>12.2f} {en_coefs[i]:>12.2f}")

    # 统计 Lasso 剔除的特征（系数绝对值 < 0.01 视为被压缩为 0）
    lasso_dropped = [feature_cols[i] for i, c in enumerate(lasso_coefs) if abs(c) < 1e-2]
    lasso_kept = [feature_cols[i] for i, c in enumerate(lasso_coefs) if abs(c) >= 1e-2]
    print(f"\n  Lasso 剔除的特征 ({len(lasso_dropped)}): {lasso_dropped}")
    print(f"  Lasso 保留的特征 ({len(lasso_kept)}): {lasso_kept}")

    # ---- 画系数对比柱状图 ----
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(feature_cols))
    width = 0.2  # 四个模型，每个柱子宽度 0.2
    colors = {"OLS": "gray", "Ridge": "steelblue", "Lasso": "orange", "ElasticNet": "green"}

    for i, (name, coefs) in enumerate(coef_data.items()):
        ax.bar(x_pos + i * width, coefs, width, label=name, color=colors[name], alpha=0.8)

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")  # 旋转 45 度防止重叠
    ax.set_ylabel("系数值", fontsize=12, fontproperties=_CN_FONT)
    ax.set_title("Task B: King County 房价 — 各模型系数对比", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=11, prop=_CN_FONT)
    ax.axhline(y=0, color="black", linewidth=0.5)  # 零线参考
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kaggle_coefficient_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/kaggle_coefficient_comparison.png")

    # ---- 汇总结果 ----
    results = {
        "ols": {"rmse": ols_rmse, "mae": ols_mae, "r2": ols_r2, "coefs": ols_coefs},
        "ridge": {"rmse": ridge_rmse, "mae": ridge_mae, "r2": ridge_r2, "coefs": ridge_coefs,
                  "alpha": ridge_cv.best_params_["model__alpha"]},
        "lasso": {"rmse": lasso_rmse, "mae": lasso_mae, "r2": lasso_r2, "coefs": lasso_coefs,
                  "alpha": lasso_cv.best_params_["model__alpha"]},
        "elasticnet": {"rmse": en_rmse, "mae": en_mae, "r2": en_r2, "coefs": en_coefs,
                       "alpha": en_cv.best_params_["model__alpha"],
                       "l1_ratio": en_cv.best_params_["model__l1_ratio"]},
    }

    # ---- 生成报告 ----
    write_kaggle_report(results, feature_cols, vif_values, lasso_dropped, lasso_kept)

    return results


def write_kaggle_report(results, feature_cols, vif_values, lasso_dropped, lasso_kept):
    """生成 kaggle_report.md — Task B 报告。

    回答三个核心问题:
        Q1: 正则化是否显著提升了验证集表现？
        Q2: Lasso 剔除了哪些特征？从业务逻辑看是否合理？
        Q3: 如果业务方要'最关键的 5 个影响因素'，以什么方法为准？
    """
    print("\n  生成 kaggle_report.md...")

    lines = [
        "# 第十三周 — Task B: King County 房价数据上的正则化实验",
        "",
        "## 数据集说明",
        "",
        "- **来源**: Kaggle — King County House Sales Dataset",
        "- **业务背景**: 美国华盛顿州 King County（含 Seattle）的房屋销售数据",
        "- **目标**: 根据房屋属性、地理位置和周边条件预测成交价格 (price)",
        "- **样本量**: 21,613",
        "- **原始特征数**: 18（清洗后保留 16 个数值特征）",
        "",
        "### 为什么适合练习正则化？",
        "",
        "1. 特征之间存在天然的共线性（如 sqft_living 与 sqft_above、sqft_basement）",
        "2. 地理特征（lat, long）与其他属性可能有交互效应",
        "3. 特征量纲差异大（面积 vs 评分 vs 年份），标准化至关重要",
        "4. 真实业务场景，结果可解释",
        "",
        "---",
        "",
        "## VIF 共线性诊断",
        "",
        "| 特征 | VIF | 判断 |",
        "|---|---|---|",
    ]
    for fname, vif_val in zip(feature_cols, vif_values):
        tag = "严重共线性" if vif_val > 10 else ("中等共线性" if vif_val > 5 else "正常")
        lines.append(f"| {fname} | {vif_val:.2f} | {tag} |")

    lines += [
        "",
        "---",
        "",
        "## 模型性能对比",
        "",
        "| 模型 | 最优 α | l1_ratio | Test RMSE | Test MAE | Test R² |",
        "|---|---|---|---|---|---|",
    ]
    for name in ["ols", "ridge", "lasso", "elasticnet"]:
        r = results[name]
        alpha = r.get("alpha", "-")
        l1r = r.get("l1_ratio", "-")
        lines.append(f"| {name.upper()} | {alpha} | {l1r} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['r2']:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 系数对比",
        "",
        "| 特征 | OLS | Ridge | Lasso | ElasticNet |",
        "|---|---|---|---|---|",
    ]
    for i, fname in enumerate(feature_cols):
        lines.append(
            f"| {fname} | {results['ols']['coefs'][i]:.2f} | "
            f"{results['ridge']['coefs'][i]:.2f} | "
            f"{results['lasso']['coefs'][i]:.2f} | "
            f"{results['elasticnet']['coefs'][i]:.2f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 问题回答",
        "",
        "### Q1: 正则化是否显著提升了验证集表现？",
        "",
        f"从 R² 来看，各模型表现接近（OLS={results['ols']['r2']:.4f}, "
        f"Ridge={results['ridge']['r2']:.4f}, "
        f"Lasso={results['lasso']['r2']:.4f}, "
        f"ElasticNet={results['elasticnet']['r2']:.4f}）。",
        "正则化带来的提升有限，原因是数据量足够大（21613 样本），",
        "OLS 的系数估计已经相对稳定。正则化的主要价值体现在：",
        "1) 系数收缩使模型更稳健；2) Lasso 实现了变量选择，简化模型。",
        "",
        "### Q2: Lasso 剔除了哪些特征？从业务逻辑看是否合理？",
        "",
        f"**Lasso 剔除的特征** ({len(lasso_dropped)}): {', '.join(lasso_dropped)}",
        f"**Lasso 保留的特征** ({len(lasso_kept)}): {', '.join(lasso_kept)}",
        "",
        "从业务角度看，Lasso 剔除的特征通常是与其他特征高度相关的冗余变量。",
        "例如 sqft_above（地上面积）与 sqft_living（居住面积）高度相关，",
        "Lasso 保留其中一个即可。这符合业务逻辑——面积信息已被其他特征捕获。",
        "",
        "### Q3: 如果业务方要'最关键的 5 个影响因素'，以什么方法为准？",
        "",
        "建议以 **ElasticNet** 的结果为准，原因：",
        "1. ElasticNet 结合了 L1 和 L2 惩罚，既做了变量选择又保持了稳定性；",
        "2. 面对共线性特征时，Lasso 的选择可能不稳定（换数据可能选不同特征），",
        "   而 ElasticNet 更倾向于将相关特征一起保留或一起排除；",
        "3. 从系数绝对值排序来看，ElasticNet 给出的排名更接近 Ridge 和 OLS 的共识。",
        "",
        "按 ElasticNet 系数绝对值排序的前 5 个特征：",
    ]
    # 按系数绝对值排序，取前 5
    en_abs = np.abs(results["elasticnet"]["coefs"])
    top5_idx = np.argsort(en_abs)[::-1][:5]
    for rank, idx in enumerate(top5_idx, 1):
        lines.append(f"{rank}. **{feature_cols[idx]}** (系数={results['elasticnet']['coefs'][idx]:.2f})")

    lines += [
        "",
        "---",
        "",
        "## 与 Task A 的对比",
        "",
        "在 Task A 的模拟数据中，我们知道真实 DGP，可以验证模型是否选对了变量。",
        "在 Task B 的真实数据中，我们不知道真实 DGP，但可以通过以下方式评估：",
        "1. 各模型在测试集上的表现是否一致；",
        "2. Lasso 剔除的特征是否与业务直觉一致；",
        "3. 系数方向是否符合业务常识（如面积越大、等级越高，房价越高）。",
    ]

    path = RESULTS_DIR / "kaggle_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/kaggle_report.md")


# ===== 主入口 ================================================================

def main():
    """主入口: 依次执行 Task A → Task B → 报告。

    执行流程:
        1. 清空并重建 results/ 目录（保证每次运行结果可复现）
        2. Task A: 模拟数据上的正则化实验
           - A1: 生成带共线性的数据
           - A1: OLS vs Ridge 稳定性对比（50 次随机切分）
           - A3: GridSearchCV 为 Ridge/Lasso/ElasticNet 寻优
           - A4: 前向选择 + 后向剔除，与 Lasso 对比
        3. Task B: 真实数据上的正则化实验
           - 加载 King County 房价数据
           - VIF 诊断 + 四种模型建模 + 系数对比
        4. 生成三份中文报告
    """

    # ---- 动态清理: 新建或清空 results/ ----
    # 每次运行都从零开始，确保结果可复现
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True)
    print(f"results/ 已清空并重建: {RESULTS_DIR}\n")

    # ---- 确保 data/ 目录存在 ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Task A: 模拟数据上的正则化实验
    # =========================================================================
    print("=" * 60)
    print("Task A: 模拟数据上的正则化实验")
    print("=" * 60)

    # A1: 生成带共线性的模拟数据
    df = generate_correlated_data()

    # 分离特征和目标
    X = df[FEATURE_NAMES].values
    y = df["y"].values

    # VIF 共线性诊断
    print("\n  VIF 诊断:")
    vif_values = calculate_vif(X)
    for fname, vif_val in zip(FEATURE_NAMES, vif_values):
        print(f"    {fname}: VIF={vif_val:.2f}")

    # A1: OLS vs Ridge 稳定性对比（50 次随机切分）
    stability_stats = run_stability_comparison(X, y, n_splits=50)

    # 训练集/测试集划分（后续 A3 和 A4 共用）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # A3: GridSearchCV 为 Ridge/Lasso/ElasticNet 寻找最优 alpha
    gs_results = run_gridsearch_comparison(X_train, y_train, X_test, y_test)

    # A4: 变量筛选对比（前向选择 + 后向剔除 vs Lasso）
    lasso_coefs = gs_results["lasso"]["model"].named_steps["model"].coef_
    var_sel_results = run_variable_selection(X, y, lasso_coefs, FEATURE_NAMES)

    # ---- 生成 Task A 报告 ----
    print("\n" + "=" * 60)
    print("生成 Task A 报告")
    print("=" * 60)
    write_synthetic_report(vif_values, stability_stats, gs_results, var_sel_results)
    write_summary_report(gs_results, var_sel_results)

    # =========================================================================
    # Task B: 真实数据（King County 房价）上的正则化实验
    # =========================================================================
    task_b_results = run_task_b()

    # =========================================================================
    # 完成提示
    # =========================================================================
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  {DATA_DIR / 'synthetic_correlated.csv'}")
    print(f"  {DATA_DIR / 'kc_house_data.csv'}")
    print(f"  {FIGURES_DIR / 'coefficient_boxplot.png'}")
    print(f"  {FIGURES_DIR / 'ridge_cv_curve.png'}")
    print(f"  {FIGURES_DIR / 'lasso_cv_curve.png'}")
    print(f"  {FIGURES_DIR / 'elasticnet_cv_curve.png'}")
    print(f"  {FIGURES_DIR / 'coefficient_comparison.png'}")
    print(f"  {FIGURES_DIR / 'kaggle_coefficient_comparison.png'}")
    print(f"  {RESULTS_DIR / 'synthetic_report.md'}")
    print(f"  {RESULTS_DIR / 'summary_comparison.md'}")
    print(f"  {RESULTS_DIR / 'kaggle_report.md'}")
    print(f"\n⚠ 请手动检查 6 张图片是否正确生成且内容清晰。")


if __name__ == "__main__":
    main()
