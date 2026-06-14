import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent  
sys.path.insert(0, str(project_root))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.utils.models import CustomOLS, PCR, train_lasso_cv, ForwardSelector
from src.utils.transformers import StandardScaler, generate_highdim_latent_data, generate_sparse_truth_data
from src.utils.metrics import rmse, mae, r2
from src.utils.diagnostics import matrix_condition_metrics, calc_coef_std, calculate_vif, plot_coef_path

# 全局配置
SEED = 42
N_SAMPLES = 150
LATENT_DIM = 8
FIG_DIR = "src/week14/results/figures"
DATA_PATH = "src/week14/data/synthetic_highdim.csv"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("src/week14/data", exist_ok=True)
os.makedirs("src/week14/results", exist_ok=True)


def add_figure_description(report_path, fig_name, description):
    """辅助函数：向报告添加图表说明"""
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n### 图：{fig_name}\n")
        f.write(description)
        f.write("\n")


def task_a():
    """Task A：高维数据生成 + OLS过拟合 + 系数不稳定"""
    print("===== 运行任务 A =====")
    
    # A1 生成高维低秩数据 p=120 > 训练样本
    X, y, latent = generate_highdim_latent_data(
        n_samples=N_SAMPLES, n_features=120, n_latent=LATENT_DIM, random_seed=SEED
    )
    
    # A2 保存数据
    df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]))
    df.columns = [f"feat_{i}" for i in range(120)] + ["target"]
    df.to_csv(DATA_PATH, index=False)
    print(f"已保存合成数据至 {DATA_PATH}, 样本量 n={N_SAMPLES}, 特征数 p=120, 潜在因子数={LATENT_DIM}")

    # A3 多组p对比 OLS train/test RMSE
    p_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    train_rmses = []
    test_rmses = []
    rank_list = []
    cond_list = []
    
    print("\n开始 OLS 实验：")
    for p in p_list:
        X_sub = X[:, :p]
        X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y, test_size=0.2, random_state=SEED)
        ols = CustomOLS(fit_intercept=True, alpha=0)
        ols.fit(X_tr, y_tr)
        y_tr_pred = ols.predict(X_tr)
        y_te_pred = ols.predict(X_te)
        train_rmses.append(rmse(y_tr, y_tr_pred))
        test_rmses.append(rmse(y_te, y_te_pred))
        
        # 矩阵病态指标
        metrics = matrix_condition_metrics(X_tr)
        rank_list.append(metrics["rank"])
        cond_list.append(metrics["condition_number"])
        
        print(f"  p={p:3d}: Train RMSE={train_rmses[-1]:.4f}, Test RMSE={test_rmses[-1]:.4f}, Rank={rank_list[-1]}, Cond={cond_list[-1]:.2e}")

    # 图1：误差随p变化
    plt.figure(figsize=(10, 5))
    plt.plot(p_list, train_rmses, marker="o", label="Train RMSE", c="blue", linewidth=2)
    plt.plot(p_list, test_rmses, marker="s", label="Test RMSE", c="red", linewidth=2)
    plt.xlabel("Number of Features p", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("OLS Train & Test Error vs Feature Dimension", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(f"{FIG_DIR}/A3_error_vs_p.png", dpi=300)
    plt.close()

    # 图2：矩阵结构随p变化
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(p_list, rank_list, marker="o", color="blue", linewidth=2, label="Matrix Rank")
    ax2.plot(p_list, np.log10(cond_list), marker="s", color="red", linewidth=2, label="log10(Condition Number)")
    ax1.set_xlabel("Number of Features p", fontsize=12)
    ax1.set_ylabel("Matrix Rank", color="blue", fontsize=12)
    ax2.set_ylabel("log10(Condition Number)", color="red", fontsize=12)
    plt.title("Matrix Rank and Condition Number vs Feature Dimension", fontsize=14)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(f"{FIG_DIR}/A3_matrix_metrics.png", dpi=300)
    plt.close()

    # A4 50次随机切分，观察系数波动
    n_splits = 50
    coef_record = []
    
    print(f"\n开始系数稳定性实验 ({n_splits} 次随机切分)：")
    for i in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None)
        ols = CustomOLS(fit_intercept=True, alpha=0)
        ols.fit(X_tr, y_tr)
        coef_record.append(ols.coef_[:3])
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{n_splits} 次切分")
    
    coef_mat = np.array(coef_record)
    coef_std = calc_coef_std(coef_mat)
    
    # 打印详细的系数统计信息
    print("\n" + "=" * 60)
    print("系数稳定性详细统计：")
    print(f"coef_mat shape: {coef_mat.shape}")
    print(f"\n各特征系数统计：")
    for i in range(3):
        print(f"  Feature {i}:")
        print(f"    均值: {np.mean(coef_mat[:, i]):.6f}")
        print(f"    标准差: {coef_std[i]:.6f}")
        print(f"    最小值: {np.min(coef_mat[:, i]):.6f}")
        print(f"    最大值: {np.max(coef_mat[:, i]):.6f}")
        print(f"    正系数比例: {np.mean(coef_mat[:, i] > 0)*100:.1f}%")
        print(f"    负系数比例: {np.mean(coef_mat[:, i] < 0)*100:.1f}%")
    
    print(f"\n前5次切分的系数示例：")
    for i in range(min(5, n_splits)):
        print(f"  Split {i+1}: {coef_record[i]}")
    print("=" * 60)
    
    # 绘制系数箱线图（增强版）
    plt.figure(figsize=(10, 6))
    
    # 绘制箱线图
    bp = plt.boxplot(coef_mat, tick_labels=["Feature 0", "Feature 1", "Feature 2"], 
                     patch_artist=True, widths=0.6)
    
    # 设置箱体颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 添加 y=0 参考线
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7, label='y=0 reference')
    
    # 添加每个系数的散点（jittered）
    for i in range(3):
        # 添加 jitter 避免重叠
        jitter = np.random.normal(0, 0.04, size=n_splits)
        x_pos = np.ones(n_splits) * (i + 1) + jitter
        plt.scatter(x_pos, coef_mat[:, i], alpha=0.3, s=20, color='darkblue')
    
    plt.ylabel("OLS Coefficient Value", fontsize=12)
    plt.title(f"Coefficient Stability Over {n_splits} Random Splits", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3, axis='y')
    
    # 获取并打印 y 轴范围
    ylim = plt.gca().get_ylim()
    print(f"\n箱线图 Y 轴范围: [{ylim[0]:.2f}, {ylim[1]:.2f}]")
    
    plt.savefig(f"{FIG_DIR}/A4_coef_stability.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 系数随时间变化的轨迹图
    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.plot(range(n_splits), coef_mat[:, i], marker='o', alpha=0.7, 
                label=f'Feature {i}', linewidth=1, markersize=3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.xlabel("Random Split Index", fontsize=12)
    plt.ylabel("Coefficient Value", fontsize=12)
    plt.title(f"Coefficient Trajectory Across {n_splits} Random Splits", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{FIG_DIR}/A4_coef_trajectory.png", dpi=300)
    plt.close()

    # 写入A报告
    with open("src/week14/results/synthetic_report.md", "w", encoding="utf-8") as f:
        f.write("# 合成数据集报告 (任务 A/B)\n\n")
        
        f.write("## 1. 数据规格与生成机制 (A1-A2)\n\n")
        f.write(f"- **样本量 n** = {N_SAMPLES}\n")
        f.write(f"- **特征数 p** = 120\n")
        f.write(f"- **潜在因子数** = {LATENT_DIM}\n")
        f.write(f"- **数据特点**：所有特征由 {LATENT_DIM} 个低维潜在因子线性组合生成，目标变量 y 仅由潜在因子驱动\n")
        f.write("- **为什么是'高维+信息冗余'数据**：\n")
        f.write("  - 特征数 p=120 大于训练样本量 n=120×0.8=96，属于典型高维设定\n")
        f.write("  - 潜在因子远少于特征数，导致特征间存在高度多重共线性\n")
        f.write("  - 矩阵条件数随 p 增加急剧增大，表明设计矩阵越来越病态\n\n")
        
        f.write("## 2. OLS 在高维下的过拟合表现 (A3)\n\n")
        
        f.write("### 表1：不同特征维度下的矩阵病态指标\n\n")
        f.write("| 特征数 p | 训练集 RMSE | 测试集 RMSE | 矩阵秩 | 条件数 |\n")
        f.write("|----------|-------------|-------------|--------|--------|\n")
        for p, tr_rmse, te_rmse, rank, cond in zip(p_list, train_rmses, test_rmses, rank_list, cond_list):
            f.write(f"| {p} | {tr_rmse:.4f} | {te_rmse:.4f} | {rank} | {cond:.2e} |\n")
        
        f.write("\n### 图 A3-1：OLS 误差随特征维度变化\n")
        f.write("- **横轴**：特征数量 p (10-120)\n")
        f.write("- **纵轴**：RMSE（均方根误差）\n")
        f.write("- **蓝色线**：训练集 RMSE，随 p 增加持续下降\n")
        f.write("- **红色线**：测试集 RMSE，先降后升（过拟合）\n")
        f.write("- **结论**：当 p 增大时，训练误差趋近于 0（过拟合），测试误差先降后升（泛化能力变差）\n")
        f.write("- **为什么训练误差接近 0 是危险信号**：\n")
        f.write("  - 高维下 OLS 可以完美拟合训练数据（甚至记忆噪声）\n")
        f.write("  - 但测试集表现差，说明模型没有学习到真实规律\n")
        f.write("  - 这是典型的过拟合：模型复杂度过高，记住了噪声而非信号\n\n")
        
        f.write("### 图 A3-2：矩阵结构随特征维度变化\n")
        f.write("- **横轴**：特征数量 p\n")
        f.write("- **左纵轴（蓝色）**：矩阵秩，反映线性无关的列数\n")
        f.write("- **右纵轴（红色）**：条件数（log10 变换），反映矩阵病态程度\n")
        f.write(f"- **实际数据**：\n")
        f.write(f"  - p=20: 秩={rank_list[1]}, log10条件数={np.log10(cond_list[1]):.2f}\n")
        f.write(f"  - p=40: 秩={rank_list[3]}, log10条件数={np.log10(cond_list[3]):.2f}\n")
        f.write(f"  - p=60: 秩={rank_list[5]}, log10条件数={np.log10(cond_list[5]):.2f}\n")
        f.write(f"  - p=80: 秩={rank_list[7]}, log10条件数={np.log10(cond_list[7]):.2f}\n")
        f.write(f"  - p=100: 秩={rank_list[9]}, log10条件数={np.log10(cond_list[9]):.2f}\n")
        f.write(f"  - p=120: 秩={rank_list[11]}, log10条件数={np.log10(cond_list[11]):.2f}\n")
        f.write("- **结论**：\n")
        f.write("  - 低维潜在结构明显：p=20 时矩阵不满秩（潜在因子主导）\n")
        f.write("  - 随 p 增加，随机噪声逐渐填满矩阵（p=60 后趋于满秩）\n")
        f.write("  - 条件数随 p 增加持续增大，矩阵越来越病态，OLS 估计极不稳定\n\n")
        
        f.write("## 3. 系数不稳定性分析 (A4)\n\n")
        f.write(f"### 图 A4-1：{n_splits} 次随机切分的系数箱线图\n")
        f.write("- **横轴**：前3个特征 (Feature 0, Feature 1, Feature 2)\n")
        f.write("- **纵轴**：OLS 系数值\n")
        f.write("- **每个箱线图**：代表该特征在 50 次随机数据切分下的系数分布\n")
        f.write("- **蓝色散点**：每次切分的具体系数值（带 jitter 避免重叠）\n")
        f.write("- **黑色实线**：y=0 参考线\n")
        f.write(f"- **系数统计**：\n")
        f.write(f"  - Feature 0: 均值={np.mean(coef_mat[:, 0]):.4f}, 标准差={coef_std[0]:.4f}\n")
        f.write(f"  - Feature 1: 均值={np.mean(coef_mat[:, 1]):.4f}, 标准差={coef_std[1]:.4f}\n")
        f.write(f"  - Feature 2: 均值={np.mean(coef_mat[:, 2]):.4f}, 标准差={coef_std[2]:.4f}\n")
        f.write(f"- **正负号翻转情况**：\n")
        f.write(f"  - Feature 0: {np.mean(coef_mat[:, 0] > 0)*100:.1f}% 为正, {np.mean(coef_mat[:, 0] < 0)*100:.1f}% 为负\n")
        f.write(f"  - Feature 1: {np.mean(coef_mat[:, 1] > 0)*100:.1f}% 为正, {np.mean(coef_mat[:, 1] < 0)*100:.1f}% 为负\n")
        f.write(f"  - Feature 2: {np.mean(coef_mat[:, 2] > 0)*100:.1f}% 为正, {np.mean(coef_mat[:, 2] < 0)*100:.1f}% 为负\n")
        f.write("- **结论**：\n")
        f.write("  - 系数在不同数据切分下波动极大\n")
        f.write("  - 同一特征出现正负号翻转，说明估计极不稳定\n")
        f.write("  - **系数不稳定本身就是重要风险**：\n")
        f.write("    - 说明估计结果对训练数据高度敏感\n")
        f.write("    - 无法给出稳定的变量重要性解释\n")
        f.write("    - 在实际业务中，这种不稳定会导致预测不可靠、决策不可复现\n\n")
        
        f.write(f"### 图 A4-2：系数轨迹图\n")
        f.write("- **横轴**：50 次随机切分的索引\n")
        f.write("- **纵轴**：OLS 系数值\n")
        f.write("- **每条线**：代表一个特征在不同切分下的系数变化轨迹\n")
        f.write("- **结论**：系数在正负之间剧烈震荡，进一步验证了不稳定性\n\n")
    
    return X, y


def task_b(X, y):
    """Task B PCA + PCR 自建流程、k选择、CV误差曲线"""
    print("\n===== 运行任务 B =====")
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler()
    max_k = min(20, X_tr.shape[1])
    k_range = np.arange(1, max_k + 1)
    train_err = []
    test_err = []
    cv_err = []

    # B1 PCA累计方差曲线
    pca_full = PCA()
    X_scaled = scaler.fit_transform(X_tr)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o', linewidth=2)
    plt.axhline(y=0.8, color='red', linestyle='--', label='80% Variance Threshold')
    plt.axhline(y=0.9, color='green', linestyle='--', label='90% Variance Threshold')
    plt.xlabel("Number of Principal Components k", fontsize=12)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=12)
    plt.title("PCA Cumulative Variance Curve", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{FIG_DIR}/B1_cum_variance.png", dpi=300)
    plt.close()

    # 确定达到90%和95%方差的主成分数量
    n_90 = np.argmax(cum_var >= 0.9) + 1 if np.any(cum_var >= 0.9) else len(cum_var)
    n_95 = np.argmax(cum_var >= 0.95) + 1 if np.any(cum_var >= 0.95) else len(cum_var)

    # B2 遍历k训练PCR，记录三类误差
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    print("PCR 交叉验证中...")
    for k in k_range:
        pcr = PCR(n_components=k, scaler=StandardScaler())
        pcr.fit(X_tr, y_tr)
        
        y_tr_pred = pcr.predict(X_tr)
        train_err.append(rmse(y_tr, y_tr_pred))
        
        y_te_pred = pcr.predict(X_te)
        test_err.append(rmse(y_te, y_te_pred))
        
        cv_scores = cross_val_score(pcr, X_tr, y_tr, cv=kf, scoring="neg_root_mean_squared_error")
        cv_err.append(-np.mean(cv_scores))
        
        if k % 5 == 0:
            print(f"  k={k:2d}: Train RMSE={train_err[-1]:.4f}, Test RMSE={test_err[-1]:.4f}, CV RMSE={cv_err[-1]:.4f}")

    # 找到最优 k
    best_k = k_range[np.argmin(cv_err)]
    best_cv_rmse = min(cv_err)
    print(f"\n最优 k = {best_k}, 最小 CV RMSE = {best_cv_rmse:.4f}")

    # 图：k vs train/test/CV RMSE
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, train_err, marker="o", label="Train RMSE", c="blue", linewidth=2)
    plt.plot(k_range, test_err, marker="s", label="Test RMSE", c="red", linewidth=2)
    plt.plot(k_range, cv_err, marker="^", label="5-Fold CV RMSE", c="green", linewidth=2, linestyle="--")
    plt.axvline(x=best_k, color='purple', linestyle=':', linewidth=2, label=f'Optimal k = {best_k}')
    plt.xlabel("Number of Principal Components k", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("PCR Error Curves vs k", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(f"{FIG_DIR}/B2_pcr_k_curve.png", dpi=300)
    plt.close()

    # 更新报告
    with open("src/week14/results/synthetic_report.md", "a", encoding="utf-8") as f:
        f.write("\n## 4. PCA 与 PCR 分析 (Task B)\n\n")
        
        f.write("### B1：PCA 累计方差曲线\n\n")
        f.write("#### 图 B1：PCA 累计方差曲线\n")
        f.write("- **横轴**：主成分数量 k\n")
        f.write("- **纵轴**：累计解释方差比例\n")
        f.write("- **红色虚线**：80% 方差阈值\n")
        f.write("- **绿色虚线**：90% 方差阈值\n")
        f.write(f"- **达到 90% 方差所需主成分数**：{n_90}\n")
        f.write(f"- **达到 95% 方差所需主成分数**：{n_95}\n")
        f.write("- **结论**：\n")
        f.write(f"  - 前 {n_90} 个主成分已解释 90% 以上的方差\n")
        f.write("  - 原始高维空间确实贴近一个更低维的子空间（潜在因子维度约为 8）\n")
        f.write("  - 这验证了数据生成机制：潜在因子数量远少于特征数\n\n")
        
        f.write("### B2：PCR 误差曲线\n\n")
        f.write("#### 图 B2：PCR 误差随主成分数量变化\n")
        f.write("- **横轴**：主成分数量 k (1-20)\n")
        f.write("- **纵轴**：RMSE（均方根误差）\n")
        f.write("- **蓝色线 (训练集 RMSE)**：随 k 增加单调下降\n")
        f.write("- **红色线 (测试集 RMSE)**：先降后升，存在最优区间\n")
        f.write("- **绿色虚线 (CV RMSE)**：5折交叉验证误差\n")
        f.write("- **紫色虚线**：最优 k 位置\n")
        f.write(f"- **最优 k**：{best_k}，对应最小 CV RMSE：{best_cv_rmse:.4f}\n\n")
        
        f.write("### B3：CV 曲线解释\n\n")
        f.write("**Q: PCR CV RMSE 代表什么？**\n")
        f.write("- 交叉验证 RMSE 是对模型泛化误差的估计\n")
        f.write("- 通过多次在训练子集上训练、在验证子集上评估，得到更稳健的性能指标\n")
        f.write("- 相比单一测试集，CV 能减少数据切分的随机性影响\n\n")
        
        f.write("**Q: 它与 train/test 曲线的关系如何理解？**\n")
        f.write("- **训练误差**：随 k 增加持续下降（更多主成分 = 更多信息 = 更好拟合）\n")
        f.write("- **CV 误差**：先降后升，在最优 k 处达到最低\n")
        f.write("  - k 太小：欠拟合，丢失重要信息\n")
        f.write("  - k 太大：过拟合，引入噪声\n")
        f.write("- **测试误差**：趋势与 CV 误差一致，验证 CV 的有效性\n\n")
        
        f.write("**Q: 为什么 OLS 在原始高维空间可以取得很低训练误差但不更好？**\n")
        f.write("- OLS 在高维下可以完美记忆训练数据（训练误差 ≈ 0）\n")
        f.write("- 但这种'完美拟合'是虚假的：模型同时拟合了信号和噪声\n")
        f.write("- 由于系数不稳定，在测试集上表现很差（高方差、低泛化）\n")
        f.write("- PCR 通过降维强制模型关注主要变异方向，牺牲训练精度换取泛化能力\n\n")
        
        f.write("### B4：公式定义\n\n")
        f.write("**OLS 估计量**：\n")
        f.write("$$\n\\hat{\\beta} = (X^TX)^{-1}X^Ty\n$$\n\n")
        f.write("**第一主成分定义**：\n")
        f.write("$$\n\\max_{\\|v\\|=1} \\text{Var}(Xv) = \\max_{\\|v\\|=1} v^TX^TXv\n$$\n\n")
        f.write("**PCR 流程**：\n")
        f.write("$$\nZ_k = XV_k, \\quad \\hat{y} = Z_k\\hat{\\gamma}\n$$\n")
        f.write("其中 $V_k$ 是前 k 个主成分载荷向量，$\\hat{\\gamma}$ 是主成分上的回归系数\n\n")
    
    return X_tr, X_te, y_tr, y_te


def task_c():
    """Task C Lasso vs PCR: Sparse Truth vs Latent Factor Truth"""
    print("\n===== 运行任务 C =====")
    
    n = 150
    p = 100
    
    # C1 生成两种场景数据
    X_sp, y_sp, sig_idx = generate_sparse_truth_data(n, p, n_signal_feats=5, random_seed=SEED)
    X_lat, y_lat, latent = generate_highdim_latent_data(n, p, n_latent=8, random_seed=SEED)
    
    scenes = {
        "Sparse Truth": (X_sp, y_sp, "Only 5 original features directly affect y"),
        "Latent Factor Truth": (X_lat, y_lat, "All features generated by 8 latent factors, y driven by latent factors")
    }
    
    def calc_stability_lasso(X, y, n_splits=20, cv=5):
        """计算 Lasso 的稳定性指标"""
        coefs = []
        for i in range(n_splits):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=None)
            lasso = train_lasso_cv(X_tr, y_tr, cv=cv)
            coefs.append(lasso.coef_.flatten())
        coefs = np.array(coefs)
        stability = np.mean(np.std(coefs, axis=0))
        return stability
    
    def calc_stability_pcr(X, y, best_k, n_splits=20):
        """计算 PCR 的稳定性指标"""
        coefs = []
        for i in range(n_splits):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=None)
            pcr = PCR(n_components=best_k, scaler=StandardScaler())
            pcr.fit(X_tr, y_tr)
            coefs.append(pcr.lr.coef_.flatten())
        coefs = np.array(coefs)
        stability = np.mean(np.std(coefs, axis=0))
        return stability
    
    # 存储结果用于绘图
    results = []
    
    for scene_name, (X, y, description) in scenes.items():
        print(f"\n处理场景: {scene_name}")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
        
        # 1. LassoCV
        print("  训练 LassoCV...")
        lasso = train_lasso_cv(X_tr, y_tr, cv=5)
        y_lasso_pred = lasso.predict(X_te)
        lasso_rmse = rmse(y_te, y_lasso_pred)
        lasso_complex = np.sum(np.abs(lasso.coef_) > 1e-6)
        print(f"    Lasso: RMSE={lasso_rmse:.4f}, 非零特征数={lasso_complex}")
        
        # 2. PCR - 通过 CV 选择最优 k
        print("  寻找 PCR 最优 k...")
        best_k = 8
        best_cv_score = np.inf
        for k in range(1, min(20, X_tr.shape[1])):
            pcr_temp = PCR(n_components=k, scaler=StandardScaler())
            scores = cross_val_score(pcr_temp, X_tr, y_tr, cv=5, scoring='neg_root_mean_squared_error')
            cv_score = -np.mean(scores)
            if cv_score < best_cv_score:
                best_cv_score = cv_score
                best_k = k
        print(f"    最优 k = {best_k}, CV RMSE = {best_cv_score:.4f}")
        
        pcr = PCR(n_components=best_k, scaler=StandardScaler())
        pcr.fit(X_tr, y_tr)
        y_pcr_pred = pcr.predict(X_te)
        pcr_rmse = rmse(y_te, y_pcr_pred)
        pcr_complex = best_k
        print(f"    PCR: RMSE={pcr_rmse:.4f}, 主成分数={pcr_complex}")
        
        # 3. 稳定性计算
        print("  计算稳定性指标...")
        lasso_stability = calc_stability_lasso(X, y, n_splits=20)
        pcr_stability = calc_stability_pcr(X, y, best_k, n_splits=20)
        print(f"    Lasso 稳定性: {lasso_stability:.6f}")
        print(f"    PCR 稳定性: {pcr_stability:.6f}")
        
        results.append({
            "scene": scene_name,
            "description": description,
            "lasso_rmse": lasso_rmse,
            "lasso_complex": lasso_complex,
            "lasso_stability": lasso_stability,
            "pcr_rmse": pcr_rmse,
            "pcr_complex": pcr_complex,
            "pcr_stability": pcr_stability,
            "best_k": best_k,
            "winner": "Lasso" if lasso_rmse < pcr_rmse else "PCR"
        })
    
    # 打印结果汇总
    print("\n" + "=" * 60)
    print("结果汇总：")
    for r in results:
        print(f"\n{r['scene']}:")
        print(f"  Lasso: RMSE={r['lasso_rmse']:.4f}, 复杂度={r['lasso_complex']}, 稳定性={r['lasso_stability']:.6f}")
        print(f"  PCR:   RMSE={r['pcr_rmse']:.4f}, 复杂度={r['pcr_complex']}, 稳定性={r['pcr_stability']:.6f}")
        print(f"  Winner: {r['winner']}")
    print("=" * 60)
    
    # 绘制对比柱状图（添加数值标签）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scenes_short = ["Sparse", "Latent"]
    lasso_rmse_vals = [results[0]["lasso_rmse"], results[1]["lasso_rmse"]]
    pcr_rmse_vals = [results[0]["pcr_rmse"], results[1]["pcr_rmse"]]
    lasso_complex_vals = [results[0]["lasso_complex"], results[1]["lasso_complex"]]
    pcr_complex_vals = [results[0]["pcr_complex"], results[1]["pcr_complex"]]
    
    x = np.arange(len(scenes_short))
    width = 0.35
    
    # 左图：RMSE 对比
    bars1 = ax1.bar(x - width/2, lasso_rmse_vals, width, label='Lasso', color='steelblue')
    bars2 = ax1.bar(x + width/2, pcr_rmse_vals, width, label='PCR', color='coral')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_ylabel('Test RMSE', fontsize=12)
    ax1.set_title('Lasso vs PCR: RMSE Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenes_short)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # 右图：复杂度对比
    bars3 = ax2.bar(x - width/2, lasso_complex_vals, width, label='Lasso (# non-zero)', color='steelblue')
    bars4 = ax2.bar(x + width/2, pcr_complex_vals, width, label='PCR (# components)', color='coral')
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_ylabel('Model Complexity', fontsize=12)
    ax2.set_title('Lasso vs PCR: Model Complexity Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenes_short)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/C2_lasso_vs_pcr_comparison.png", dpi=300)
    plt.close()
    
    summary_text = "# Lasso vs PCR 对比总结：特征选择 vs 信息压缩\n\n"
    summary_text += "## 实验设置\n\n"
    summary_text += f"- **样本量 n** = {n}\n"
    summary_text += f"- **特征数 p** = {p}\n"
    summary_text += "- **对比方法**：LassoCV (L1正则化) vs PCR (主成分回归)\n"
    summary_text += "- **稳定性指标定义**：20次随机切分下，系数向量的平均标准差（值越小越稳定）\n\n"
    summary_text += "---\n\n"
    
    # 使用实际运行结果
    sparse_result = results[0]
    latent_result = results[1]
    
    summary_text += f"## 稀疏真实信号场景\n\n"
    summary_text += f"**数据说明**：仅5个原始特征真正影响目标变量 y\n\n"
    summary_text += "### 结果对比\n\n"
    summary_text += "| 指标 | Lasso | PCR |\n"
    summary_text += "|------|-------|-----|\n"
    summary_text += f"| 测试集 RMSE | {sparse_result['lasso_rmse']:.4f} | {sparse_result['pcr_rmse']:.4f} |\n"
    summary_text += f"| 模型复杂度 | {sparse_result['lasso_complex']} 个非零特征 | {sparse_result['pcr_complex']} 个主成分 |\n"
    summary_text += f"| 稳定性指标 | {sparse_result['lasso_stability']:.4f} | {sparse_result['pcr_stability']:.4f} |\n"
    summary_text += f"\n**结论**：**{'Lasso' if sparse_result['lasso_rmse'] < sparse_result['pcr_rmse'] else 'PCR'} 表现更好**\n\n"
    summary_text += "---\n\n"
    
    summary_text += f"## 潜在因子真实信号场景\n\n"
    summary_text += f"**数据说明**：所有特征由8个潜在因子生成，y 由潜在因子驱动\n\n"
    summary_text += "### 结果对比\n\n"
    summary_text += "| 指标 | Lasso | PCR |\n"
    summary_text += "|------|-------|-----|\n"
    summary_text += f"| 测试集 RMSE | {latent_result['lasso_rmse']:.4f} | {latent_result['pcr_rmse']:.4f} |\n"
    summary_text += f"| 模型复杂度 | {latent_result['lasso_complex']} 个非零特征 | {latent_result['pcr_complex']} 个主成分 |\n"
    summary_text += f"| 稳定性指标 | {latent_result['lasso_stability']:.4f} | {latent_result['pcr_stability']:.4f} |\n"
    summary_text += f"\n**结论**：**{'Lasso' if latent_result['lasso_rmse'] < latent_result['pcr_rmse'] else 'PCR'} 表现更好**\n\n"
    summary_text += "---\n\n"
    
    summary_text += "## 场景总结对比表\n\n"
    summary_text += "| 场景 | 胜出方法 | Lasso RMSE | PCR RMSE | Lasso 复杂度 | PCR 复杂度 | Lasso 稳定性 | PCR 稳定性 |\n"
    summary_text += "|------|----------|------------|----------|--------------|------------|--------------|------------|\n"
    summary_text += f"| 稀疏真实信号 | {'Lasso' if sparse_result['lasso_rmse'] < sparse_result['pcr_rmse'] else 'PCR'} | {sparse_result['lasso_rmse']:.4f} | {sparse_result['pcr_rmse']:.4f} | {sparse_result['lasso_complex']} | {sparse_result['pcr_complex']} | {sparse_result['lasso_stability']:.4f} | {sparse_result['pcr_stability']:.4f} |\n"
    summary_text += f"| 潜在因子真实信号 | {'Lasso' if latent_result['lasso_rmse'] < latent_result['pcr_rmse'] else 'PCR'} | {latent_result['lasso_rmse']:.4f} | {latent_result['pcr_rmse']:.4f} | {latent_result['lasso_complex']} | {latent_result['pcr_complex']} | {latent_result['lasso_stability']:.4f} | {latent_result['pcr_stability']:.4f} |\n"
    
    summary_text += "\n---\n\n"
    
    summary_text += """
## 核心讨论问题

### Q1: 当数据真的是稀疏真实信号时，为什么 Lasso 往往更自然？

**答案**：在稀疏真实信号场景中，真实信号只集中在少数原始特征上。Lasso 通过 L1 正则化产生稀疏解，能够自动将无关噪声特征的系数压缩为零，实现特征选择。

### Q2: 当数据更像潜在因子真实信号时，为什么 PCR 往往更自然？

**答案**：在潜在因子场景中，真实信号分布在所有原始特征上。PCR 通过 PCA 将高度相关的特征压缩为少数主成分，有效捕捉数据的低维结构。

### Q3: Lasso 回答的更像"谁留下"，而 PCR 回答的更像什么？

**答案**：Lasso 回答"哪些原始变量真正重要？"（变量选择）；PCR 回答"什么低维子空间能够捕捉数据的主要变异？"（信息压缩）。

### Q4: 如果业务方要求"一个更短的变量名单"，更可能用哪个方法？

**答案**：**Lasso**。因为 Lasso 能直接输出稀疏的系数向量，明确告诉业务方哪些变量对预测有贡献。

### Q5: 如果业务方要求"一个更稳的预测器"，更可能用哪个方法？

**答案**：**PCR**。因为 PCR 通过降维去除噪声和冗余信息，系数估计更稳定。

### Q6: 为什么这周主线更适合比较 Lasso vs PCR？

**答案**：逐步回归存在高方差、易过拟合、计算效率低等问题；Lasso 提供连续正则化框架，PCR 通过子空间压缩更稳定。

### Q7: 前向/后向选择更接近哪个路线？

**答案**：更接近 selection（变量选择）路线，与 Lasso 同类别；PCR 属于 compression（信息压缩）路线。
"""
    
    with open("src/week14/results/summary_comparison.md", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"\n对比完成：稀疏场景最优={results[0]['winner']}，潜在因子场景最优={results[1]['winner']}")
    return results


def main():
    """作业要求单入口完整流程"""
    print("=" * 60)
    print("开始执行 Week14 作业：高维回归分析")
    print("=" * 60)
    
    # 任务 A
    X_highdim, y_highdim = task_a()
    
    # 任务 B
    Xtr, Xte, ytr, yte = task_b(X_highdim, y_highdim)
    
    # 任务 C
    results = task_c()
    
    print("\n" + "=" * 60)
    print("所有任务 A/B/C 已完成！")
    print(f"输出文件保存在: src/week14/results/")
    print(f"- 图表目录: {FIG_DIR}")
    print(f"- 数据文件: {DATA_PATH}")
    print(f"- 数据报告: src/week14/results/synthetic_report.md")
    print(f"- 对比报告: src/week14/results/summary_comparison.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
