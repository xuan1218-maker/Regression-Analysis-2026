"""模块: week12.main
用途: 第十二周作业 —— 偏差-方差可视化实验。
      Task A: 三位候选模型对比
      Task B: 复杂度-误差曲线
      Task C: repeated sampling 展示 variance
      Task D: RMSE vs MAE 对异常值的敏感度
      Task E: 可讲授的实验工作流
      Task F: 输出总结报告

核心思路:
    1. 用非线性真实函数 y=sin(1.5x)+0.5x 生成带噪声数据
    2. 用不同阶数的多项式回归拟合，观察欠拟合/过拟合现象
    3. 通过 repeated sampling 可视化 variance
    4. 对比 RMSE 和 MAE 对异常值的敏感度

运行方式:
    cd students/15_lxl
    uv run src/week12/main.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端（服务器/WSL 环境下无法弹窗显示图片）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 加载 Windows 微软雅黑字体，确保中文不乱码
# WSL2 环境下通过 /mnt/c/ 访问 Windows 字体目录
_FONT_PATH = "/mnt/c/Windows/Fonts/msyh.ttc"
_CN_FONT = fm.FontProperties(fname=_FONT_PATH)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

import numpy as np
from sklearn.linear_model import LinearRegression  # 线性回归（OLS 解析解）
from sklearn.pipeline import Pipeline  # 管道：串联多个处理步骤
from sklearn.preprocessing import PolynomialFeatures  # 多项式特征扩展
from sklearn.model_selection import train_test_split  # 训练集/测试集划分

# 将 src/ 加入搜索路径，复用自己维护的 utils/
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import calculate_rmse, calculate_mae  # 复用自己的指标函数

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
WEEK12_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = WEEK12_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ===== 数据生成 =============================================================

def generate_data(n_samples: int = 150, noise_std: float = 0.3, seed: int = 42):
    """生成一维非线性回归数据。

    设计思路:
        - 真实函数 y = sin(1.5x) + 0.5x 是一个非线性函数，
          既有周期性（sin）又有趋势性（0.5x），适合展示多项式拟合的行为
        - 加入高斯噪声模拟真实数据的随机性
        - 用 train_test_split 划分，确保评估时用未见过的数据

    参数:
        n_samples: 总样本数（默认 150）
        noise_std: 噪声标准差（默认 0.3，信噪比适中）
        seed: 随机种子（保证可复现）

    返回: X_train, X_test, y_train, y_test, X_all, y_true_all, true_func
    """
    rng = np.random.default_rng(seed)

    # 生成均匀分布的 x 值，范围 [-3, 3]
    X = rng.uniform(-3, 3, n_samples).reshape(-1, 1)

    # 真实函数（非线性）：sin(1.5x) 提供周期性，0.5x 提供线性趋势
    def true_func(x):
        return np.sin(1.5 * x.ravel()) + 0.5 * x.ravel()

    # 生成带噪声的目标值: y = true_func(x) + ε, ε ~ N(0, 0.3²)
    y = true_func(X) + rng.normal(0, noise_std, n_samples)

    # 划分训练集和测试集 (70/30)
    # 测试集用于评估模型泛化能力，不参与训练
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # 用于画真实曲线的平滑 x 值（300 个点，足够画出平滑曲线）
    X_all = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_true_all = true_func(X_all)

    return X_train, X_test, y_train, y_test, X_all, y_true_all, true_func


# ===== Task A: 三位候选模型对比 ============================================

def run_model_complexity_demo(X_train, y_train, X_test, y_test, X_all, y_true_all):
    """Task A: 比较 degree=1, 4, 15 三个多项式模型。

    设计思路:
        - degree=1: 直线，预期欠拟合（高 bias，低 variance）
        - degree=4: 中等复杂度，预期较平衡
        - degree=15: 高阶多项式，预期过拟合（低 bias，高 variance）
        通过三个典型例子，直观展示 bias-variance tradeoff

    生成 candidate_models.png，展示:
    - 训练点、测试点
    - 真实函数曲线
    - 三个模型的拟合曲线
    - 各自的 train RMSE 和 test RMSE
    """
    print("[Stage 1] Comparing candidate polynomial models...")

    degrees = [1, 4, 15]  # 三个典型复杂度
    colors = ["blue", "green", "red"]  # 对应颜色：蓝=欠拟合，绿=平衡，红=过拟合
    results = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    # 画训练点和测试点
    ax.scatter(X_train, y_train, c="steelblue", s=20, alpha=0.5, label="训练数据", zorder=5)
    ax.scatter(X_test, y_test, c="orange", s=20, alpha=0.5, label="测试数据", zorder=5)

    # 画真实函数（黑色虚线，作为参照基准）
    ax.plot(X_all, y_true_all, "k--", linewidth=2, label="真实函数", zorder=4)

    # 对每个 degree 拟合并画曲线
    for deg, color in zip(degrees, colors):
        # 用 Pipeline 组合多项式特征 + 线性回归
        # PolynomialFeatures: 将 x 扩展为 [x, x², x³, ..., x^deg]
        # LinearRegression: 在扩展后的特征上做 OLS 拟合
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("lr", LinearRegression()),
        ])
        model.fit(X_train, y_train)

        # 在训练集、测试集、全范围上分别预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_all = model.predict(X_all)

        # 计算指标（复用 utils/metrics.py）
        train_rmse = calculate_rmse(y_train, y_pred_train)
        test_rmse = calculate_rmse(y_test, y_pred_test)

        results[deg] = {"train_rmse": train_rmse, "test_rmse": test_rmse}

        # 画拟合曲线，标注 train/test RMSE
        label = f"degree={deg} (训练RMSE={train_rmse:.3f}, 测试RMSE={test_rmse:.3f})"
        ax.plot(X_all, y_pred_all, color=color, linewidth=1.5, label=label, zorder=3)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Task A: 三位候选多项式模型对比", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=9, loc="upper left", prop=_CN_FONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_models.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/candidate_models.png")

    return results


# ===== Task B: 复杂度-误差曲线 =============================================

def run_error_curve_sweep(X_train, y_train, X_test, y_test):
    """Task B: 扫描 degree 1~18，画出 train RMSE 和 test RMSE 曲线。

    设计思路:
        - 从 degree=1 到 degree=18 逐一拟合，记录训练/测试 RMSE
        - 训练 RMSE 会随复杂度增加持续下降（模型越来越能'记住'训练数据）
        - 测试 RMSE 会先降后升，形成 U 形曲线（过拟合的典型特征）
        - 两条曲线的 gap 反映泛化能力

    生成 error_curves.png 和返回各 degree 的指标数据。
    """
    print("[Stage 2] Sweeping model complexity (degree 1~18)...")

    max_degree = 18
    degrees = list(range(1, max_degree + 1))
    train_rmses = []
    test_rmses = []

    for deg in degrees:
        # 每个 degree 独立训练一个模型
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("lr", LinearRegression()),
        ])
        model.fit(X_train, y_train)

        # 分别在训练集和测试集上预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 记录两个集合上的 RMSE
        train_rmses.append(calculate_rmse(y_train, y_pred_train))
        test_rmses.append(calculate_rmse(y_test, y_pred_test))

    # 画误差曲线：横轴=复杂度，纵轴=RMSE，两条线分别代表训练/测试
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(degrees, train_rmses, "o-", color="steelblue", label="训练 RMSE", linewidth=2)
    ax.plot(degrees, test_rmses, "o-", color="orange", label="测试 RMSE", linewidth=2)
    ax.set_xlabel("多项式阶数（模型复杂度）", fontsize=12, fontproperties=_CN_FONT)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Task B: 偏差-方差权衡 — 误差 vs 复杂度", fontsize=14, fontproperties=_CN_FONT)
    ax.legend(fontsize=11, prop=_CN_FONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/error_curves.png")

    return degrees, train_rmses, test_rmses


# ===== Task C: Repeated Sampling 展示 Variance ==============================

def run_variance_demo(X_all, y_true_all, true_func, noise_std=0.3, n_repeats=15):
    """Task C: 固定真实函数，重复抽样训练集，展示不同复杂度的 variance。

    设计思路:
        - variance（方差）衡量的是：换一批训练数据，模型的预测会变化多少
        - 低复杂度模型（degree=2）：不同训练集拟合出的曲线很接近（低 variance）
        - 高复杂度模型（degree=15）：不同训练集拟合出的曲线差异很大（高 variance）
        - 通过重复抽样 n_repeats 次，把多条曲线叠加在同一张图上，直观展示差异

    对 degree=2 和 degree=15 各重复拟合 n_repeats 次，
    把多条拟合曲线画在同一张图上。
    """
    print("[Stage 3] Demonstrating variance via repeated sampling...")

    rng = np.random.default_rng(0)
    degrees_to_demo = [2, 15]  # 低复杂度 vs 高复杂度
    variance_stats = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, deg in enumerate(degrees_to_demo):
        ax = axes[idx]
        predictions = []  # 收集每次拟合在 X_all 上的预测值

        for i in range(n_repeats):
            # 每次从相同分布中抽取不同的训练集（80 个样本）
            # 这模拟了'从同一个总体中反复抽样'的场景
            X_train_i = rng.uniform(-3, 3, 80).reshape(-1, 1)
            y_train_i = true_func(X_train_i) + rng.normal(0, noise_std, 80)

            # 用当前训练集拟合模型
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
                ("lr", LinearRegression()),
            ])
            model.fit(X_train_i, y_train_i)
            y_pred_i = model.predict(X_all)
            predictions.append(y_pred_i)

            # 画每条拟合曲线（半透明，方便观察多条曲线的分散程度）
            ax.plot(X_all, y_pred_i, alpha=0.3, linewidth=1, color="steelblue")

        predictions = np.array(predictions)  # shape: (n_repeats, 300)

        # 画真实函数（黑色虚线，作为参照）
        ax.plot(X_all, y_true_all, "k--", linewidth=2, label="真实函数")

        # 画所有拟合的平均预测（红色实线）
        # 平均预测接近真实函数，但单次拟合可能偏离很远（这就是 variance）
        mean_pred = np.mean(predictions, axis=0)
        ax.plot(X_all, mean_pred, "r-", linewidth=2, label="平均预测")

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"degree={deg}（{n_repeats} 次拟合）", fontsize=13, fontproperties=_CN_FONT)
        ax.legend(fontsize=9, prop=_CN_FONT)
        ax.grid(True, alpha=0.3)

        # 计算统计量：每个 x 位置上预测值的标准差
        # mean_std: 平均标准差（整体波动大小）
        # max_std: 最大标准差（最不稳定的位置）
        pred_std = np.std(predictions, axis=0)
        variance_stats[deg] = {
            "mean_std": np.mean(pred_std),
            "max_std": np.max(pred_std),
        }

    fig.suptitle("Task C: 方差可视化 — 相同数据分布，不同训练样本", fontsize=14, fontproperties=_CN_FONT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_demo.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/variance_demo.png")

    return variance_stats


# ===== Task D: RMSE vs MAE 对异常值的敏感度 =================================

def run_loss_comparison_demo():
    """Task D: 构造干净预测和含异常值的预测，对比 RMSE 和 MAE 的行为。

    设计思路:
        - RMSE = sqrt(mean(error²)) — 对大误差取平方，放大效应
        - MAE = mean(|error|) — 对所有误差线性加权
        - 当存在异常值时，RMSE 会远高于 MAE（因为大误差被平方放大）
        - 这反映了两种指标不同的'风险偏好'
    """
    print("[Stage 4] Comparing RMSE vs MAE sensitivity to outliers...")

    rng = np.random.default_rng(42)

    # 构造干净预测场景：50 个样本，真实值在 50~150 之间，预测误差 ~N(0,5)
    n = 50
    y_true = rng.uniform(50, 150, n)
    y_pred_clean = y_true + rng.normal(0, 5, n)  # 小误差（标准差 5）

    # 构造含异常值的预测：只改动 2 个样本，制造巨大误差
    # 这模拟了现实中偶尔出现的极端预测错误
    y_pred_outlier = y_pred_clean.copy()
    outlier_idx = [0, 1]
    y_pred_outlier[outlier_idx[0]] = y_true[outlier_idx[0]] + 200  # 大误差 +200
    y_pred_outlier[outlier_idx[1]] = y_true[outlier_idx[1]] - 180  # 大误差 -180

    # 计算指标（复用 utils/metrics.py）
    clean_rmse = calculate_rmse(y_true, y_pred_clean)
    clean_mae = calculate_mae(y_true, y_pred_clean)
    outlier_rmse = calculate_rmse(y_true, y_pred_outlier)
    outlier_mae = calculate_mae(y_true, y_pred_outlier)

    # 画对比图：左图=干净预测，右图=含异常值
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 干净预测的误差分布（近似正态，集中在 0 附近）
    errors_clean = y_pred_clean - y_true
    axes[0].hist(errors_clean, bins=15, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)  # 零误差参考线
    axes[0].set_xlabel("预测误差", fontsize=11, fontproperties=_CN_FONT)
    axes[0].set_ylabel("频数", fontsize=11, fontproperties=_CN_FONT)
    axes[0].set_title(f"干净预测\nRMSE={clean_rmse:.2f}, MAE={clean_mae:.2f}", fontsize=12, fontproperties=_CN_FONT)
    axes[0].grid(True, alpha=0.3)

    # 右图: 含异常值的误差分布（两个极端值在 -180 和 +200 处）
    errors_outlier = y_pred_outlier - y_true
    axes[1].hist(errors_outlier, bins=15, color="salmon", alpha=0.7, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("预测误差", fontsize=11, fontproperties=_CN_FONT)
    axes[1].set_ylabel("频数", fontsize=11, fontproperties=_CN_FONT)
    axes[1].set_title(f"含异常值（2 个样本）\nRMSE={outlier_rmse:.2f}, MAE={outlier_mae:.2f}", fontsize=12, fontproperties=_CN_FONT)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Task D: RMSE vs MAE — 对异常值的敏感度", fontsize=14, fontproperties=_CN_FONT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_outlier_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/loss_outlier_comparison.png")

    return {
        "clean_rmse": clean_rmse, "clean_mae": clean_mae,
        "outlier_rmse": outlier_rmse, "outlier_mae": outlier_mae,
    }


# ===== Task F: 输出总结报告 ================================================

def write_summary_report(task_a_results, degrees, train_rmses, test_rmses,
                         variance_stats, loss_results):
    """生成中文总结报告 summary.md。"""

    # 找到测试误差最低的 degree
    best_degree = degrees[np.argmin(test_rmses)]
    best_test_rmse = min(test_rmses)

    # 找到泛化 gap 最大的 degree
    gaps = [t - tr for t, tr in zip(test_rmses, train_rmses)]
    max_gap_degree = degrees[np.argmax(gaps)]
    max_gap = max(gaps)

    lines = [
        "# 第十二周 — 偏差-方差可视化实验报告",
        "",
        "## Task A: 三位候选模型对比",
        "",
        "| Degree | Train RMSE | Test RMSE | 判断 |",
        "|---|---|---|---|",
    ]
    for deg, m in task_a_results.items():
        if deg == 1:
            判断 = "欠拟合（过于简单）"
        elif deg == 15:
            判断 = "过拟合（过于复杂）"
        else:
            判断 = "较平衡"
        lines.append(f"| {deg} | {m['train_rmse']:.4f} | {m['test_rmse']:.4f} | {判断} |")

    lines += [
        "",
        "**Q1: 谁最像欠拟合？**",
        "degree=1。它用一条直线拟合非线性数据，训练误差和测试误差都很高，说明模型太简单，无法捕捉数据的真实模式。",
        "",
        "**Q2: 谁最像过拟合？**",
        "degree=15。它的训练误差极低（几乎为 0），但测试误差明显更高。模型记住了训练数据的噪声，泛化能力差。",
        "",
        "**Q3: 如果必须选一个上线？**",
        f"选 degree=4。它的训练误差和测试误差都较低，泛化 gap 最小。虽然 degree=1 的测试误差也不算太高，",
        "但它是因为'什么都学不到'所以误差均匀，而非真正拟合了数据。",
        "",
        "---",
        "",
        "## Task B: 复杂度-误差曲线",
        "",
        "| Degree | Train RMSE | Test RMSE | 泛化 Gap |",
        "|---|---|---|---|",
    ]
    for d, tr, te in zip(degrees, train_rmses, test_rmses):
        lines.append(f"| {d} | {tr:.4f} | {te:.4f} | {te-tr:+.4f} |")

    lines += [
        "",
        f"**测试误差最低的复杂度**: degree={best_degree} (test RMSE = {best_test_rmse:.4f})",
        f"**泛化 gap 最大的复杂度**: degree={max_gap_degree} (gap = {max_gap:+.4f})",
        "",
        "**为什么训练误差最低的模型不一定是最好的？**",
        "因为训练误差会随着模型复杂度增加而持续下降（模型越来越'记住'训练数据），",
        "但测试误差会在某个点之后开始上升——这就是过拟合。最好的模型是测试误差最低的那个，",
        "而不是训练误差最低的那个。",
        "",
        "---",
        "",
        "## Task C: Variance 可视化",
        "",
        "| Degree | 平均预测标准差 | 最大预测标准差 |",
        "|---|---|---|",
    ]
    for deg, stats in variance_stats.items():
        lines.append(f"| {deg} | {stats['mean_std']:.4f} | {stats['max_std']:.4f} |")

    lines += [
        "",
        "**high variance model 的危险，不是它不会拟合训练集，而是它对训练样本的微小变化过于敏感。**",
        "",
        "degree=2 的模型在不同训练集上得到的曲线非常一致（低 variance），",
        "而 degree=15 的模型在不同训练集上得到的曲线差异巨大（高 variance）。",
        "这意味着高复杂度模型的预测结果'不稳定'——换一批训练数据，预测就完全不同。",
        "",
        "---",
        "",
        "## Task D: RMSE vs MAE 对异常值的敏感度",
        "",
        "| 场景 | RMSE | MAE |",
        "|---|---|---|",
        f"| 干净预测 | {loss_results['clean_rmse']:.2f} | {loss_results['clean_mae']:.2f} |",
        f"| 含 2 个异常值 | {loss_results['outlier_rmse']:.2f} | {loss_results['outlier_mae']:.2f} |",
        "",
        "**Q1: 为什么 RMSE 更容易被大错拉高？**",
        "因为 RMSE 对误差取平方——大误差的平方会被放大。例如误差 200 的平方是 40000，",
        "而误差 5 的平方只有 25。一个大误差对 RMSE 的贡献相当于 800 个小误差。",
        "MAE 对误差取绝对值，大误差和小误差的权重是线性的。",
        "",
        "**Q2: 如果线上系统偶尔一次大错的代价极高，更想看哪个指标？**",
        "看 RMSE。因为 RMSE 对大误差更敏感，能更好地反映'最坏情况'的风险。",
        "如果 RMSE 远高于 MAE，说明存在少量但严重的大误差，需要重点关注。",
        "",
        "**Q3: 如果数据天然包含较多异常值，会不会重新考虑指标选择？**",
        "会。如果数据中异常值是'正常的'（如医疗费用中的高费用群体），",
        "用 MAE 更合适，因为它不会被少数极端值主导。",
        "如果异常值是'错误的'（如数据录入错误），应该先清洗数据再选指标。",
        "",
        "---",
        "",
        "## 本周三条核心结论",
        "",
        "1. **过拟合是可见现象，不是抽象概念**: degree=15 的拟合曲线在训练点附近剧烈震荡，",
        "   而 degree=4 的曲线平滑地逼近真实函数。看图就能直觉理解过拟合。",
        "2. **模型复杂度需要平衡**: 训练误差随复杂度单调下降，但测试误差呈 U 形。",
        "   最佳模型是测试误差最低的那个，而不是训练误差最低的。",
        "3. **损失函数的选择反映风险偏好**: RMSE 放大大误差的风险，MAE 对所有误差一视同仁。",
        "   选择哪个指标取决于业务场景——对大错零容忍用 RMSE，数据含异常值用 MAE。",
        "",
        "---",
        "",
        "## 与下一周的连接",
        "",
        "如果模型复杂度过高会带来 high variance，那么下一步我们为什么自然会想到正则化（Ridge / Lasso）？",
        "",
        "因为正则化通过在损失函数中添加系数惩罚项（如 L2 范数），",
        "迫使模型的系数不能太大，从而限制模型的'灵活度'。",
        "这等价于在 bias 和 variance 之间做显式的权衡——",
        "稍微增加一点 bias（训练误差略升），换取 variance 大幅下降（测试误差下降）。",
        "Ridge 用 L2 惩罚收缩系数，Lasso 用 L1 惩罚还能做特征选择。",
    ]

    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/summary.md")


# ===== 主入口 ================================================================

def main():
    """主入口: 依次执行 Task A → B → C → D → F。

    执行流程（具有叙事性）:
        1. Task A: 先看三个典型模型的对比，建立直觉
        2. Task B: 再扫描完整复杂度范围，看到 U 形曲线
        3. Task C: 用 repeated sampling 把 variance 可视化
        4. Task D: 最后对比 RMSE 和 MAE 对异常值的行为
        5. Task F: 收束到总结报告
    """

    # ---- 动态清理: 新建或清空 results/ ----
    # 每次运行都从零开始，确保结果可复现
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True)
    print(f"results/ 已清空并重建: {RESULTS_DIR}\n")

    # ---- 生成数据 ----
    # 真实函数 y=sin(1.5x)+0.5x，150 个样本，噪声标准差 0.3
    X_train, X_test, y_train, y_test, X_all, y_true_all, true_func = generate_data()
    print(f"数据: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本\n")

    # ---- Task A: 三位候选模型 ----
    task_a_results = run_model_complexity_demo(X_train, y_train, X_test, y_test, X_all, y_true_all)

    # ---- Task B: 复杂度-误差曲线 ----
    degrees, train_rmses, test_rmses = run_error_curve_sweep(X_train, y_train, X_test, y_test)

    # ---- Task C: Variance 可视化 ----
    variance_stats = run_variance_demo(X_all, y_true_all, true_func)

    # ---- Task D: RMSE vs MAE ----
    loss_results = run_loss_comparison_demo()

    # ---- Task F: 生成总结报告 ----
    print("\n[Stage 5] Writing summary report...")
    write_summary_report(task_a_results, degrees, train_rmses, test_rmses,
                         variance_stats, loss_results)

    # ---- 提示用户手动检查图片 ----
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  {FIGURES_DIR / 'candidate_models.png'}")
    print(f"  {FIGURES_DIR / 'error_curves.png'}")
    print(f"  {FIGURES_DIR / 'variance_demo.png'}")
    print(f"  {FIGURES_DIR / 'loss_outlier_comparison.png'}")
    print(f"  {RESULTS_DIR / 'summary.md'}")
    print(f"\n⚠ 请手动检查 4 张图片是否正确生成且内容清晰。")


if __name__ == "__main__":
    main()
