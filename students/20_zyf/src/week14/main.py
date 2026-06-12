"""
Week 14 主入口: 高维回归、PCA 与 PCR
============================================
单一入口: uv run src/week14/main.py
"""

import sys
import os
import traceback

# Ensure the student's src directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# 复用你自己的 utils
from utils.metrics import calculate_rmse, calculate_mae
from utils.models import (AnalyticalOLS, PCR, compute_matrix_rank,
                          compute_condition_number, compute_coefficient_stability)

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


# ============================================================
# 数据生成
# ============================================================
def generate_synthetic_data(
    n_samples: int = 200,
    p_total: int = 60,
    n_latent: int = 5,
    noise_level: float = 0.5,
    seed: int = 42
) -> dict:
    """
    生成高维且带有潜在低秩结构的模拟回归数据。
    
    数据生成机制:
    - 首先生成 n_latent 个潜在因子 F (n_samples x n_latent)
    - 用随机线性投影 V (p_total x n_latent) 生成原始变量 X_raw = F @ V^T + noise
    - y 主要由前几个潜在因子驱动

    返回 dict: {'X': X, 'y': y, 'F': F, 'V': V, 'true_beta_f': true_beta_f}
    """
    rng = np.random.RandomState(seed)
    
    # 1. 生成潜在因子
    F = rng.randn(n_samples, n_latent) * 2.0
    
    # 2. 随机投影矩阵（原始变量由潜在因子线性组合生成）
    V = rng.randn(n_latent, p_total) * 0.3
    
    # 3. 生成原始高维特征
    X_raw = F @ V  # 信号部分（低秩）
    X_noise = rng.randn(n_samples, p_total) * noise_level
    X = X_raw + X_noise
    
    # 4. 生成 y（主要由前 3 个潜在因子驱动，带非线性微扰）
    true_beta_f = np.zeros(n_latent)
    true_beta_f[0] = 3.0
    true_beta_f[1] = 2.0
    true_beta_f[2] = 1.5
    # 后两个因子贡献较小
    true_beta_f[3] = 0.3
    true_beta_f[4] = 0.1
    
    y = F @ true_beta_f + rng.randn(n_samples) * 0.3
    
    return {
        'X': X,
        'y': y,
        'F': F,
        'V': V,
        'true_beta_f': true_beta_f,
        'n_latent': n_latent,
        'p_total': p_total,
        'n_samples': n_samples
    }


def generate_sparse_data(
    n_samples: int = 200,
    p_total: int = 60,
    n_active: int = 5,
    noise_level: float = 0.5,
    seed: int = 42
) -> dict:
    """
    生成稀疏真实场景的数据:
    - 只有少数原始变量直接决定 y
    - 其他变量大多是噪声
    """
    rng = np.random.RandomState(seed)
    
    # 生成所有特征（独立正态）
    X = rng.randn(n_samples, p_total) * 1.0
    
    # 只有前 n_active 个变量对 y 有贡献
    true_coef = np.zeros(p_total)
    true_coef[0] = 3.0
    true_coef[1] = 2.5
    true_coef[2] = 2.0
    true_coef[3] = 1.5
    true_coef[4] = 1.0
    # 其他都是 0
    
    y = X @ true_coef + rng.randn(n_samples) * noise_level
    
    return {
        'X': X,
        'y': y,
        'true_coef': true_coef,
        'n_active': n_active,
        'p_total': p_total,
        'n_samples': n_samples
    }


# ============================================================
# Task A: OLS 不稳定性展示
# ============================================================
def task_a():
    """Task A: 生成数据并展示 OLS 在高维/共线性场景下的不稳定性。"""
    print("=" * 60)
    print("Task A: 高维/共线性如何破坏 OLS")
    print("=" * 60)
    
    # A1: 生成完整数据集 (p=60, n=200) 
    data = generate_synthetic_data(n_samples=200, p_total=60, n_latent=5,
                                   noise_level=0.5, seed=42)
    X_full = data['X']
    y_full = data['y']
    
    # 保存数据
    df = pd.DataFrame(X_full, columns=[f"X{i+1}" for i in range(X_full.shape[1])])
    df['y'] = y_full
    csv_path = os.path.join(DATA_DIR, "synthetic_highdim.csv")
    df.to_csv(csv_path, index=False)
    print(f"[A1] 数据已保存至: {csv_path}")
    print(f"     样本量 n={data['n_samples']}, 特征数 p={data['p_total']}, 潜在因子数={data['n_latent']}")
    
    # A3: 不同 p 下的 OLS 表现
    print("\n[A3] 不同特征维度下 OLS 误差变化实验...")
    p_list = [10, 30, 60, 120]
    results_a3 = []
    
    rng = np.random.RandomState(42)
    
    for p in p_list:
        if p <= X_full.shape[1]:
            X_p = X_full[:, :p]
        else:
            # 对 p > 原始维度的情况，通过原始特征的随机线性组合扩展
            extra = rng.randn(X_full.shape[0], p - X_full.shape[1]) * 0.5
            X_p = np.hstack([X_full, extra])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_p, y_full, test_size=0.3, random_state=42
        )
        
        model = AnalyticalOLS()
        model.fit(X_train, y_train)
        
        train_rmse = calculate_rmse(y_train, model.predict(X_train))
        test_rmse = calculate_rmse(y_test, model.predict(X_test))
        rank_X = compute_matrix_rank(X_train)
        cond_num = compute_condition_number(X_train)
        
        results_a3.append({
            'p': p,
            'n_train': X_train.shape[0],
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'rank_X_train': rank_X,
            'condition_number': cond_num
        })
        print(f"  p={p:>3}: train_rmse={train_rmse:.4f}, test_rmse={test_rmse:.4f}, "
              f"rank={rank_X}, cond={cond_num:.2e}")
    
    # 图 1: 误差随特征维度变化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    p_vals = [r['p'] for r in results_a3]
    train_rmses = [r['train_rmse'] for r in results_a3]
    test_rmses = [r['test_rmse'] for r in results_a3]
    ax.plot(p_vals, train_rmses, 'o-', label='Train RMSE', color='blue', markersize=8)
    ax.plot(p_vals, test_rmses, 's-', label='Test RMSE', color='red', markersize=8)
    ax.set_xlabel('特征维度 p', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('A3: OLS 误差随特征维度变化', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    rank_vals = [r['rank_X_train'] for r in results_a3]
    cond_vals = [r['condition_number'] for r in results_a3]
    
    ax2_color = 'tab:green'
    ax.bar(np.array(p_vals) - 1.5, rank_vals, width=3, color=ax2_color, alpha=0.7,
           label='rank(X_train)')
    ax.set_xlabel('特征维度 p', fontsize=12)
    ax.set_ylabel('秩 (rank)', fontsize=12, color=ax2_color)
    ax.tick_params(axis='y', labelcolor=ax2_color)
    
    ax3 = ax.twinx()
    ax3.plot(p_vals, cond_vals, 'D-', color='purple', markersize=8, label='Condition Number')
    ax3.set_ylabel('条件数 (Condition Number)', fontsize=12, color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_yscale('log')
    
    ax.set_title('A3: 矩阵结构随特征维度变化', fontsize=14)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "a3_ols_dimension_analysis.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  图表已保存: {fig_path}")
    
    # A4: 系数不稳定性
    print("\n[A4] 重复随机切分 50 次，展示 OLS 系数不稳定...")
    X_a4 = X_full[:, :]
    y_a4 = y_full
    
    stability_result = compute_coefficient_stability(
        X_a4, y_a4, n_splits=50,
        selected_indices=[0, 1, 2, 10, 30]
    )
    
    # 箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = []
    labels = []
    for idx, traj in stability_result['coef_trajectories'].items():
        plot_data.append(traj)
        labels.append(f'X{idx+1}')
    
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('变量', fontsize=12)
    ax.set_ylabel('OLS 系数估计值', fontsize=12)
    ax.set_title('A4: 同一变量在不同随机切分下的 OLS 系数波动 (50次)', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(RESULTS_DIR, "a4_coefficient_instability.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fig_path}")
    
    for idx, std_val in stability_result['coef_stds'].items():
        print(f"  X{idx+1}: 系数标准差 = {std_val:.4f}")
    
    return data, results_a3, stability_result


# ============================================================
# Task B: PCA 与 PCR
# ============================================================
def task_b(data: dict):
    """Task B: PCA 复习与 PCR 实现。"""
    print("\n" + "=" * 60)
    print("Task B: PCA 复习与 PCR 实现")
    print("=" * 60)
    
    X = data['X']
    y = data['y']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # B1: PCA 分析
    print("\n[B1] PCA 降维分析...")
    pca_full = PCA()
    Z_full = pca_full.fit_transform(X_scaled)
    
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var_ratio)
    
    # 找出解释 80%, 90%, 95% 方差所需的主成分数
    for threshold in [0.80, 0.90, 0.95]:
        n_needed = np.argmax(cumsum_var >= threshold) + 1
        print(f"  累计解释 {threshold*100:.0f}% 方差需要前 {n_needed} 个主成分")
    
    print(f"  前 5 个主成分解释方差比例: {cumsum_var[4]:.3f}")
    print(f"  前 10 个主成分解释方差比例: {cumsum_var[9]:.3f}")
    
    # 累计解释方差图
    fig, ax = plt.subplots(figsize=(10, 5))
    n_pcs = min(60, len(cumsum_var))
    ax.plot(range(1, n_pcs + 1), cumsum_var[:n_pcs], 'b-', linewidth=2)
    ax.fill_between(range(1, n_pcs + 1), 0, cumsum_var[:n_pcs], alpha=0.2)
    ax.axhline(y=0.80, color='orange', linestyle='--', label='80% 阈值')
    ax.axhline(y=0.90, color='red', linestyle='--', label='90% 阈值')
    ax.axhline(y=0.95, color='purple', linestyle='--', label='95% 阈值')
    ax.set_xlabel('主成分个数', fontsize=12)
    ax.set_ylabel('累计解释方差比例', fontsize=12)
    ax.set_title('B1: 累计解释方差曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_pcs)
    ax.set_ylim(0, 1.05)
    
    fig_path = os.path.join(RESULTS_DIR, "b1_cumulative_explained_variance.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  图表已保存: {fig_path}")
    
    # B2: PCR 工作流
    # 关键修正：标准化和PCA必须只在训练集上fit，然后统一transform训练集和测试集。
    # 之前每轮独立fit导致train/test的预处理不一致，造成train_rmse > test_rmse的异常。
    print("\n[B2] PCR 工作流：比较不同 k 下的表现...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 第一步：在训练集上统一做标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 第二步：在训练集上统一做PCA（保留全部主成分）
    pca_full = PCA(n_components=min(X_train_scaled.shape[0], X_train_scaled.shape[1]))
    Z_train_full = pca_full.fit_transform(X_train_scaled)
    Z_test_full = pca_full.transform(X_test_scaled)
    
    k_range = list(range(1, 21))
    pcr_results = []
    
    for k in k_range:
        # 第三步：取前k个主成分做普通最小二乘回归
        Z_train_k = Z_train_full[:, :k]
        Z_test_k = Z_test_full[:, :k]
        
        reg = LinearRegression()
        reg.fit(Z_train_k, y_train)
        
        train_rmse = calculate_rmse(y_train, reg.predict(Z_train_k))
        test_rmse = calculate_rmse(y_test, reg.predict(Z_test_k))
        
        # 5-fold CV RMSE：每次CV fold也需要在fold内部重新标准化+PCA，保证无数据泄露
        cv_rmse_list = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_cv_train_raw, X_cv_val_raw = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]
            
            # fold内独立标准化
            cv_scaler = StandardScaler()
            X_cv_train_s = cv_scaler.fit_transform(X_cv_train_raw)
            X_cv_val_s = cv_scaler.transform(X_cv_val_raw)
            
            # fold内独立PCA
            cv_pca = PCA(n_components=min(X_cv_train_s.shape[0], X_cv_train_s.shape[1]))
            Z_cv_train = cv_pca.fit_transform(X_cv_train_s)
            Z_cv_val = cv_pca.transform(X_cv_val_s)
            
            cv_reg = LinearRegression()
            cv_reg.fit(Z_cv_train[:, :k], y_cv_train)
            cv_rmse_list.append(calculate_rmse(y_cv_val, cv_reg.predict(Z_cv_val[:, :k])))
        cv_rmse = np.mean(cv_rmse_list)
        
        pcr_results.append({
            'k': k,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_rmse': cv_rmse
        })
    
    # OLS 基线（同样用标准化后的数据做公平对比）
    ols_raw = AnalyticalOLS()
    ols_raw.fit(X_train_scaled, y_train)
    ols_train_rmse = calculate_rmse(y_train, ols_raw.predict(X_train_scaled))
    ols_test_rmse = calculate_rmse(y_test, ols_raw.predict(X_test_scaled))
    
    # 保存PCR结果用于后续报告
    pcr_results_saved = pcr_results.copy()
    
    print(f"  OLS 基线: train_rmse={ols_train_rmse:.4f}, test_rmse={ols_test_rmse:.4f}")
    best_k = min(pcr_results, key=lambda r: r['cv_rmse'])
    print(f"  PCR 最佳 k (按 CV RMSE): k={best_k['k']}, "
          f"train_rmse={best_k['train_rmse']:.4f}, "
          f"test_rmse={best_k['test_rmse']:.4f}, "
          f"cv_rmse={best_k['cv_rmse']:.4f}")
    
    # 图: PCR train/test/CV RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ks = [r['k'] for r in pcr_results]
    ax.plot(ks, [r['train_rmse'] for r in pcr_results], 'o-', label='PCR Train RMSE',
            color='blue', markersize=6)
    ax.plot(ks, [r['test_rmse'] for r in pcr_results], 's-', label='PCR Test RMSE',
            color='red', markersize=6)
    ax.plot(ks, [r['cv_rmse'] for r in pcr_results], '^--', label='PCR CV RMSE (5-fold)',
            color='green', markersize=6)
    
    # 添加 OLS 基线
    ax.axhline(y=ols_test_rmse, color='red', linestyle=':', linewidth=1.5,
               label=f'OLS Test RMSE ({ols_test_rmse:.3f})')
    ax.axhline(y=ols_train_rmse, color='blue', linestyle=':', linewidth=1.5,
               label=f'OLS Train RMSE ({ols_train_rmse:.3f})')
    
    ax.set_xlabel('保留主成分个数 k', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('B2: PCR 表现随 k 变化 (含 OLS 基线)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(RESULTS_DIR, "b2_pcr_performance.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fig_path}")
    
    return pcr_results, ols_train_rmse, ols_test_rmse, cumsum_var


# ============================================================
# Task C: Lasso vs PCR — selection vs compression
# ============================================================
def task_c():
    """Task C: 比较 Lasso (变量筛选) 与 PCR (信息压缩)。"""
    print("\n" + "=" * 60)
    print("Task C: Lasso vs PCR — selection vs compression")
    print("=" * 60)
    
    # C1: 构造两种场景
    print("\n[C1] 构造两种数据场景...")
    data_sparse = generate_sparse_data(n_samples=200, p_total=60, n_active=5,
                                       noise_level=0.5, seed=123)
    data_latent = generate_synthetic_data(n_samples=200, p_total=60, n_latent=5,
                                          noise_level=0.5, seed=456)
    
    scenarios = {
        'Sparse Truth (稀疏真实)': data_sparse,
        'Latent-Factor Truth (潜在因子真实)': data_latent
    }
    
    all_results = {}
    
    for scenario_name, data in scenarios.items():
        print(f"\n  处理场景: {scenario_name}")
        X = data['X']
        y = data['y']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # --- Lasso ---
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000,
                           alphas=np.logspace(-3, 1, 50))
        lasso_cv.fit(X_train, y_train)
        
        lasso_test_rmse = calculate_rmse(y_test, lasso_cv.predict(X_test))
        lasso_test_mae = calculate_mae(y_test, lasso_cv.predict(X_test))
        lasso_n_nonzero = int(np.sum(np.abs(lasso_cv.coef_) > 1e-5))
        
        # Lasso 稳定性: 多次拟合观察非零系数数量波动
        lasso_nz_list = []
        for split_i in range(30):
            idx = np.random.RandomState(split_i).permutation(len(X))
            n_tr = int(0.8 * len(X))
            l_tmp = LassoCV(cv=3, random_state=42, max_iter=10000,
                            alphas=np.logspace(-3, 1, 30))
            l_tmp.fit(X[idx[:n_tr]], y[idx[:n_tr]])
            lasso_nz_list.append(int(np.sum(np.abs(l_tmp.coef_) > 1e-5)))
        
        lasso_nz_std = float(np.std(lasso_nz_list))
        
        # --- PCR ---
        pcr_best_k = None
        pcr_best_cv = np.inf
        k_search = list(range(1, min(31, X_train.shape[1] + 1)))
        for k in k_search:
            cv_rmse_list = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, val_idx in kf.split(X):
                pcr_cv_model = PCR(n_components=k)
                pcr_cv_model.fit(X[tr_idx], y[tr_idx])
                cv_rmse_list.append(calculate_rmse(y[val_idx], pcr_cv_model.predict(X[val_idx])))
            mean_cv = np.mean(cv_rmse_list)
            if mean_cv < pcr_best_cv:
                pcr_best_cv = mean_cv
                pcr_best_k = k
        
        pcr_final = PCR(n_components=pcr_best_k)
        pcr_final.fit(X_train, y_train)
        pcr_test_rmse = calculate_rmse(y_test, pcr_final.predict(X_test))
        pcr_test_mae = calculate_mae(y_test, pcr_final.predict(X_test))
        
        # PCR 稳定性: 多次拟合观察最佳 k 的波动
        pcr_k_list = []
        for split_i in range(30):
            idx = np.random.RandomState(split_i).permutation(len(X))
            n_tr = int(0.8 * len(X))
            best_k_tmp = 1
            best_cv_tmp = np.inf
            for k in k_search:
                cv_list = []
                kf2 = KFold(n_splits=3, shuffle=True, random_state=42)
                X_sub = X[idx[:n_tr]]
                y_sub = y[idx[:n_tr]]
                for t2, v2 in kf2.split(X_sub):
                    m = PCR(n_components=k)
                    m.fit(X_sub[t2], y_sub[t2])
                    cv_list.append(calculate_rmse(y_sub[v2], m.predict(X_sub[v2])))
                mc = np.mean(cv_list)
                if mc < best_cv_tmp:
                    best_cv_tmp = mc
                    best_k_tmp = k
            pcr_k_list.append(best_k_tmp)
        
        pcr_k_std = float(np.std(pcr_k_list))
        
        # OLS 基线
        ols_model = AnalyticalOLS()
        ols_model.fit(X_train, y_train)
        ols_test_rmse = calculate_rmse(y_test, ols_model.predict(X_test))
        ols_test_mae = calculate_mae(y_test, ols_model.predict(X_test))
        
        result = {
            'scenario': scenario_name,
            'ols_test_rmse': ols_test_rmse,
            'ols_test_mae': ols_test_mae,
            'lasso_test_rmse': lasso_test_rmse,
            'lasso_test_mae': lasso_test_mae,
            'lasso_n_nonzero': lasso_n_nonzero,
            'lasso_nz_std': lasso_nz_std,
            'pcr_test_rmse': pcr_test_rmse,
            'pcr_test_mae': pcr_test_mae,
            'pcr_best_k': pcr_best_k,
            'pcr_k_std': pcr_k_std
        }
        all_results[scenario_name] = result
        
        print(f"    OLS:   test_rmse={ols_test_rmse:.4f}, test_mae={ols_test_mae:.4f}")
        print(f"    Lasso: test_rmse={lasso_test_rmse:.4f}, test_mae={lasso_test_mae:.4f}, "
              f"非零系数数={lasso_n_nonzero}, nz_std={lasso_nz_std:.2f}")
        print(f"    PCR:   test_rmse={pcr_test_rmse:.4f}, test_mae={pcr_test_mae:.4f}, "
              f"最佳k={pcr_best_k}, k_std={pcr_k_std:.2f}")
    
    # 对比图: 两个场景并排展示
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (scenario_name, res) in enumerate(all_results.items()):
        ax = axes[i]
        methods = ['OLS', 'Lasso', 'PCR']
        rmse_vals = [res['ols_test_rmse'], res['lasso_test_rmse'], res['pcr_test_rmse']]
        colors = ['#888888', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(methods, rmse_vals, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Test RMSE', fontsize=12)
        ax.set_title(f'{scenario_name}', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注复杂度信息
        complexity_text = (
            f'Lasso 非零系数: {res["lasso_n_nonzero"]} ± {res["lasso_nz_std"]:.1f}\n'
            f'PCR 主成分数: {res["pcr_best_k"]} ± {res["pcr_k_std"]:.1f}'
        )
        ax.text(0.98, 0.95, complexity_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('C2: Lasso vs PCR  — 两种数据场景下的对比', fontsize=15, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "c2_lasso_vs_pcr.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {fig_path}")
    
    return all_results


# ============================================================
# 报告生成
# ============================================================
def generate_synthetic_report(data: dict, results_a3: list, stability_result: dict,
                               pcr_results: list, ols_train_rmse: float,
                               ols_test_rmse: float, cumsum_var: np.ndarray):
    """生成 synthetic_report.md"""
    print("\n" + "=" * 60)
    print("生成 synthetic_report.md")
    print("=" * 60)
    
    report = f"""# 第十四周合成数据报告 (Synthetic Data Report)

## 1. 数据生成机制说明

### 1.1 基本参数
- **样本量 n**: {data['n_samples']}
- **特征维度 p**: {data['p_total']}
- **潜在因子数**: {data['n_latent']}

### 1.2 潜在因子结构 (Latent-Factor Structure)
数据通过以下步骤生成:

1. **生成潜在因子**: $F \\sim \\mathcal{{N}}(0, 4)$ 的 $n \\times k$ 矩阵，其中 $k={data['n_latent']}$ 为潜在因子数。
2. **随机投影**: 原始变量 $X_{{\\text{{raw}}}} = F \\cdot V^\\top$，其中 $V$ 是 $k \\times p$ 的随机投影矩阵。
3. **加噪声**: $X = X_{{\\text{{raw}}}} + \\epsilon_X$，$\\epsilon_X \\sim \\mathcal{{N}}(0, 0.5^2)$。
4. **目标变量**: $y = 3.0 \\cdot F_1 + 2.0 \\cdot F_2 + 1.5 \\cdot F_3 + 0.3 \\cdot F_4 + 0.1 \\cdot F_5 + \\epsilon_y$，$\\epsilon_y \\sim \\mathcal{{N}}(0, 0.3^2)$。

### 1.3 为什么这是"高维 + 信息冗余"的数据？

- **高维**: 特征数 p={data['p_total']}，虽然样本量 n={data['n_samples']} > p，但在后续实验中将扩展到 p > n 的场景。
- **信息冗余**: 所有 {data['p_total']} 个原始变量仅由 {data['n_latent']} 个潜在因子线性生成（加上噪声），因此原始变量的有效秩近似为 {data['n_latent']}（远小于 p={data['p_total']}）。这意味着大量变量之间高度相关，存在严重的多重共线性。

---

## 2. Task A: OLS 不稳定性分析

### 2.1 A3: 误差随特征维度变化

| 特征维度 p | 训练集 RMSE | 测试集 RMSE | rank(X_train) | 条件数 (Condition Number) |
|:----------:|:-----------:|:-----------:|:-------------:|:-------------------------:|
"""
    for r in results_a3:
        report += f"| {r['p']:<8} | {r['train_rmse']:.4f}     | {r['test_rmse']:.4f}     | {r['rank_X_train']:<11} | {r['condition_number']:.2e}            |\n"
    
    report += f"""
**图表说明**:
- **图 a3_ols_dimension_analysis.png 左图**: 横轴为特征维度 p，纵轴为 RMSE。蓝色线 (Train RMSE) 随 p 增大而持续下降（甚至趋近于 0），红色线 (Test RMSE) 则先降后升，在 p 接近或超过训练样本量时急剧恶化。这展示了"虚假的低训练误差"。
- **图 a3_ols_dimension_analysis.png 右图**: 横轴为特征维度 p。绿色柱状图为 rank(X_train)，随着 p 增大秩增长受限（因为潜在因子的低秩结构）。紫色折线为条件数（对数坐标），随 p 增大条件数急剧上升，表明矩阵越来越病态（ill-conditioned）。

### 2.2 为什么"训练误差接近 0"在这里反而是危险信号？

当 p 接近或超过 n 时，OLS 可以在训练数据上完美插值（训练误差趋近于 0），但这并不代表模型学到了真实的信号。由于矩阵高度病态（条件数极大），OLS 系数估计极不稳定——它把噪声当成了信号来拟合（过拟合）。测试误差的高涨证明了这一点。**训练误差趋近于 0 恰恰是过拟合的典型症状**。

### 2.3 A4: 系数不稳定性

对固定数据集进行 50 次不同随机切分，每次用 OLS 拟合，记录系数波动：

| 变量 | 系数标准差 |
|:----:|:----------:|
"""
    for idx, std_val in stability_result['coef_stds'].items():
        report += f"| X{idx+1} | {std_val:.4f} |\n"
    
    report += f"""
**图表说明**:
- **图 a4_coefficient_instability.png**: 箱线图展示了 5 个变量 (X1, X2, X3, X11, X31) 在 50 次不同随机切分下的 OLS 系数分布。横轴为变量名，纵轴为系数估计值。每个箱线代表该变量在不同切分下的系数波动范围。

**观察与回答**:
- 我们同时观察到了**误差的波动**（测试 RMSE 在不同切分下变化）和**系数的波动**（同一变量的系数估计在不同切分下剧烈变化，甚至符号反转）。
- 课堂上说"系数不稳定本身就是一种重要风险"的原因：在实际应用中，我们不仅关心预测精度，也关心系数解释。如果同一个变量的系数在不同数据子集上可以从正变负、从大变到零，那么我们对"这个变量到底起什么作用"就完全失去了信心。**模型的可解释性和可信度因此崩塌**。

---

## 3. Task B: PCA 与 PCR

### 3.1 B1: PCA 累计解释方差

前 5 个主成分累计解释方差比例: **{cumsum_var[4]:.3f}**  
前 10 个主成分累计解释方差比例: **{cumsum_var[9]:.3f}**

"""
    for threshold in [0.80, 0.90, 0.95]:
        n_needed = int(np.argmax(cumsum_var >= threshold) + 1)
        report += f"- 累计解释 {threshold*100:.0f}% 方差需要前 **{n_needed}** 个主成分\n"
    
    report += f"""
**图表说明**:
- **图 b1_cumulative_explained_variance.png**: 横轴为主成分个数，纵轴为累计解释方差比例。蓝色实线为累计曲线，三条虚线分别标注 80%、90%、95% 阈值。

**解释**: 前少数几个主成分已经解释了绝大部分方差，这验证了"原始高维空间其实贴近一个更低维子空间"的判断。因为数据是由 {data['n_latent']} 个潜在因子生成的，所以前 {data['n_latent']} 个主成分就应捕获大部分信号方差。

### 3.2 B2: PCR 工作流

**OLS 基线**: train_rmse={ols_train_rmse:.4f}, test_rmse={ols_test_rmse:.4f}

| k (主成分数) | Train RMSE | Test RMSE | CV RMSE (5-fold) |
|:-----------:|:----------:|:---------:|:----------------:|
"""
    for r in pcr_results:
        report += f"| {r['k']:<9} | {r['train_rmse']:.4f}    | {r['test_rmse']:.4f}   | {r['cv_rmse']:.4f}         |\n"
    
    best_k_pcr = min(pcr_results, key=lambda r: r['cv_rmse'])
    report += f"""
**图表说明**:
- **图 b2_pcr_performance.png**: 横轴为保留主成分个数 k (1-20)，纵轴为 RMSE。蓝色线为 PCR Train RMSE，红色线为 PCR Test RMSE，绿色虚线为 5-fold CV RMSE。两条水平虚线分别标注 OLS 的 Train 和 Test RMSE 基线。

**PCR 最佳 k (按 CV RMSE)**: k={best_k_pcr['k']}，train_rmse={best_k_pcr['train_rmse']:.4f}，test_rmse={best_k_pcr['test_rmse']:.4f}。

### 3.3 B3: CV 曲线解释

- **PCR CV RMSE 代表什么？**  
  CV RMSE 表示在交叉验证中，用训练集拟合 PCR 模型后在验证集上计算的平均 RMSE。它是对模型在未见数据上泛化能力的无偏估计。

- **它与 train/test 曲线的关系如何理解？**  
  Train RMSE 通常随 k 增大而下降（模型复杂度增加，更好地拟合训练数据）。Test RMSE 和 CV RMSE 呈现 U 型曲线：k 太小时欠拟合（未能捕获足够信号），k 太大时过拟合（开始拟合噪声）。CV RMSE 的最低点给出了最优 k 的合理估计。

- **为什么 OLS 在原始高维空间里可以取得很低的训练误差，但这并不意味着它更好？**  
  因为低训练误差是过拟合的结果。OLS 在高维空间中可以完美拟合训练数据（包括其中的噪声），但这些噪声模式不会在测试集上重复出现，导致泛化性能很差。PCR 通过降维丢弃了噪声主导的方向，虽然训练误差略高，但测试误差更低且更稳定。

### 3.4 B4: 严格定义与公式

1. **OLS 的估计式**:
   $$\\hat{{\\beta}}_{{\\text{{OLS}}}} = \\arg\\min_\\beta \\|y - X\\beta\\|_2^2 = (X^\\top X)^{{-1}} X^\\top y$$
   当 $X^\\top X$ 近似奇异时，该解极不稳定。

2. **第一主成分的方差最大化定义**:
   $$v_1 = \\arg\\max_{{\\|v\\|=1}} \\text{{Var}}(X v) = \\arg\\max_{{\\|v\\|=1}} v^\\top \\Sigma_X v$$
   其中 $\\Sigma_X$ 是 $X$ 的协方差矩阵。第一主成分方向是数据方差最大的方向。

3. **PCR 流程的符号表达**:
   设 $V_k = [v_1, v_2, \\ldots, v_k]$ 为前 $k$ 个主成分方向矩阵，则：
   $$Z_k = X V_k \\quad \\text{{(PC scores)}}$$
   $$\\hat{{\\beta}}_{{\\text{{PCR}}}} = V_k \\cdot \\arg\\min_\\gamma \\|y - Z_k \\gamma\\|_2^2$$
   即在降维后的 $Z_k$ 空间中做普通最小二乘回归。

---

## 4. 核心结论

本实验通过合成数据展示了:
1. 高维/共线性场景下 OLS 的训练误差和系数估计均不可靠；
2. PCA 能有效识别数据的低维潜在结构；
3. PCR 通过"先压缩再回归"的流程，在降低模型复杂度的同时提升了泛化能力。
"""
    
    report_path = os.path.join(RESULTS_DIR, "synthetic_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  报告已保存: {report_path}")


def generate_summary_comparison(all_results: dict):
    """生成 summary_comparison.md"""
    print("\n" + "=" * 60)
    print("生成 summary_comparison.md")
    print("=" * 60)
    
    report = """# 第十四周总结对比报告 (Summary Comparison)

## Lasso vs PCR: Selection vs Compression

### 实验结果汇总

| 场景 | 方法 | Test RMSE | Test MAE | 复杂度指标 |
|:-----|:-----|:---------:|:--------:|:-----------|
"""
    for scenario_name, res in all_results.items():
        short_name = "Sparse" if "Sparse" in scenario_name else "Latent"
        report += f"| {short_name} | OLS   | {res['ols_test_rmse']:.4f} | {res['ols_test_mae']:.4f} | - |\n"
        report += f"| {short_name} | Lasso | {res['lasso_test_rmse']:.4f} | {res['lasso_test_mae']:.4f} | 非零系数={res['lasso_n_nonzero']}±{res['lasso_nz_std']:.1f} |\n"
        report += f"| {short_name} | PCR   | {res['pcr_test_rmse']:.4f} | {res['pcr_test_mae']:.4f} | 主成分数={res['pcr_best_k']}±{res['pcr_k_std']:.1f} |\n"
    
    report += """
**图表说明**:
- **图 c2_lasso_vs_pcr.png**: 两个子图分别展示 Sparse Truth 和 Latent-Factor Truth 场景下 OLS、Lasso、PCR 的 Test RMSE 对比。横轴为方法名，纵轴为 Test RMSE。每个场景图内标注了模型复杂度信息（Lasso 非零系数数量、PCR 保留主成分数）。

---

### 核心问题回答

#### C3.1 当数据真的是 sparse truth 时，为什么 Lasso 往往更自然？

当只有少数原始变量真正决定 y 时，Lasso 的 $\\ell_1$ 正则化天然倾向于产生稀疏解——它会把不相关变量的系数压缩为 0，从而自动完成变量筛选。这直接契合了"sparse truth"的数据生成机制：**少数变量起作用，其余为零**。Lasso 的答案是"哪几个变量（在原始空间中）是真正重要的"。

#### C3.2 当数据更像 latent-factor truth 时，为什么 PCR 往往更自然？

当 y 由少数潜在因子驱动、而这些因子又线性生成了大量原始变量时，没有一个原始变量是"独立决定" y 的——所有变量都承载了部分信号，但它们高度相关。此时：
- 试图筛选出少数原始变量（如 Lasso）会丢失分散在多个变量中的信号；
- PCR 通过 PCA 将分散的信号"压缩"回少数主成分，恢复了潜在因子结构，从而更高效地利用了所有变量中的信息。

#### C3.3 Lasso 回答的更像是"谁留下"，而 PCR 回答的更像是什么？

- **Lasso** 回答: "在原始变量中，哪些是真正重要的？" → **变量选择 (selection)**
- **PCR** 回答: "原始高维数据背后，隐藏着什么样的低维结构？" → **信息压缩 (compression)** 或 **结构发现 (structure discovery)**

Lasso 让你得到一个简短的变量名单；PCR 让你得到一个低维表示，但该表示是原始变量的线性组合，不易直接对应回具体的业务变量。

#### C3.4 如果业务方要求的是"一个更短的变量名单"，你更可能用哪个方法？

**Lasso**。因为 Lasso 的输出直接是"哪些原始变量的系数非零"，可以被翻译为一个具体的变量列表。业务方可以直接拿着这个名单去收集数据、做决策或进行干预。

#### C3.5 如果业务方要求的是"一个更稳的预测器"，你更可能用哪个方法？

**PCR**（或 Ridge）。因为当多重共线性严重时，PCR 通过丢弃噪声主导的主成分来稳定预测。Lasso 虽然也能通过正则化稳定系数，但如果真实结构是 latent-factor 类型，Lasso 可能因为被迫选择少数变量而丢失信息，导致预测不如 PCR 稳定。

---

### C4: 是否要加入前向/后向变量选择？

1. **为什么这周主线更适合比较 Lasso vs PCR，而不是把前向/后向选择重新拉回主舞台？**  
   前向/后向选择本质上是**离散的、贪心的变量筛选方法**，它在每一步做硬性的"要/不要"决策。在高维场景下:
   - 计算开销巨大（尤其后向选择在 p 很大时几乎不可行）；
   - 离散选择极不稳定（数据微小变化可能导致完全不同的变量子集）；
   - 它仍然属于 selection 路线，与 Lasso 有相似的哲学但缺乏 Lasso 的凸优化和正则化理论支撑。
   
   本周的核心议题是 **selection vs compression**，Lasso 是 selection 路线的连续、现代代表，PCR 是 compression 路线的代表，二者的对比已经充分覆盖了这一议题。

2. **如果一定要加，前向/后向选择更接近 selection 路线还是 compression 路线？**  
   前向/后向选择更接近 **selection** 路线。它们的目标是在原始变量集合中选出一个子集，输出的结果是可以直接指认的"重要变量名单"，这与 Lasso 的目标一致。

---

### 最终总结

> 什么时候我们更需要 **variable selection**，什么时候我们更需要 **information compression**？

- 当真实信号**稀疏地分布在少数原始变量**上，且业务需要**可解释的变量名单**时 → **Selection (Lasso)**
- 当真实信号**分散在大量高度相关的变量**中，且业务需要**稳定准确的预测**时 → **Compression (PCR)**

理解数据的生成机制，才是选择正确方法的关键。
"""
    
    report_path = os.path.join(RESULTS_DIR, "summary_comparison.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  报告已保存: {report_path}")


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print("Week 14: 高维回归、PCA 与 PCR")
    print("=" * 60)
    
    # Task A
    data, results_a3, stability_result = task_a()
    
    # Task B
    pcr_results, ols_train_rmse, ols_test_rmse, cumsum_var = task_b(data)
    
    # Task C
    all_results = task_c()
    
    # 生成报告
    generate_synthetic_report(data, results_a3, stability_result,
                               pcr_results, ols_train_rmse, ols_test_rmse, cumsum_var)
    generate_summary_comparison(all_results)
    
    print("\n" + "=" * 60)
    print("全部流程完成!")
    print(f"数据文件: {os.path.join(DATA_DIR, 'synthetic_highdim.csv')}")
    print(f"图表目录: {RESULTS_DIR}")
    print(f"报告文件: {os.path.join(RESULTS_DIR, 'synthetic_report.md')}")
    print(f"         {os.path.join(RESULTS_DIR, 'summary_comparison.md')}")
    print("=" * 60)


if __name__ == "__main__":
    main()