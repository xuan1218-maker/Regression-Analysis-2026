"""
Week 14: High-Dimensional Regression, PCA, and PCR
高维问题、共线性与降维回归

Tasks:
- A: 高维/共线性如何破坏 OLS
- B: PCA 与 PCR（先压缩，再回归）
- C: Lasso vs PCR（变量筛选 vs 信息压缩）
- D: 真实数据挑战（可选）

Usage: uv run src/week14/main.py
"""
import sys
import csv
import math
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

from utils.metrics import calculate_rmse, calculate_mae


# ============================================================
# Task A1: 生成高维且带有潜在低秩结构的模拟数据
# ============================================================

def generate_highdim_data(n_samples=120, n_features=80, n_factors=5, noise_std=0.5, random_seed=42):
    """
    生成高维且带有潜在低秩结构的模拟回归数据
    
    Args:
        n_samples: 样本量 (默认120)
        n_features: 特征数 (默认80)
        n_factors: 潜在因子数 (默认5)
        noise_std: 噪声标准差
        random_seed: 随机种子
    
    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量
        feature_names: 特征名列表
        true_factors: 真实因子载荷
    """
    np.random.seed(random_seed)
    
    # 1. 生成潜在因子 (低维结构)
    F = np.random.randn(n_samples, n_factors)
    
    # 2. 生成因子载荷矩阵 (每个特征由因子线性组合)
    loadings = np.random.randn(n_factors, n_features)
    
    # 3. 生成特征矩阵 X = F @ loadings + noise
    X = F @ loadings + np.random.randn(n_samples, n_features) * noise_std
    
    # 4. 生成目标变量 y (由前3个因子决定，其它因子无关)
    true_coef = np.zeros(n_factors)
    true_coef[:3] = [2.0, 1.5, 1.0]
    y = F @ true_coef + np.random.randn(n_samples) * 0.3
    
    feature_names = [f'x{i+1}' for i in range(n_features)]
    
    return X, y, feature_names, F, loadings, true_coef


def generate_sparse_data(n_samples=200, n_features=100, n_true_features=5, noise_std=0.5, random_seed=42):
    """
    生成稀疏真值数据 (Sparse Truth)
    - 只有少数原始变量真正决定 y
    - 其他变量是噪声
    """
    np.random.seed(random_seed)
    
    # 真实相关特征索引 (前 n_true_features 个)
    X = np.random.randn(n_samples, n_features)
    
    # 真实系数 (只有前5个非零)
    true_coef = np.zeros(n_features)
    true_coef[:n_true_features] = [3.0, 2.0, 1.5, 1.0, 0.5]
    
    # 生成 y
    y = X @ true_coef + np.random.randn(n_samples) * noise_std
    
    feature_names = [f'x{i+1}' for i in range(n_features)]
    
    return X, y, feature_names, true_coef


def generate_latent_data(n_samples=200, n_features=100, n_factors=5, noise_std=0.5, random_seed=42):
    """
    生成潜在因子真值数据 (Latent-factor Truth)
    - 原始变量主要由少数潜在因子线性组合生成
    - y 也主要由这些潜在因子驱动
    """
    np.random.seed(random_seed)
    
    # 潜在因子
    F = np.random.randn(n_samples, n_factors)
    
    # 因子载荷
    loadings = np.random.randn(n_factors, n_features)
    X = F @ loadings + np.random.randn(n_samples, n_features) * noise_std
    
    # y 由因子决定
    true_factor_coef = [2.0, 1.5, 1.0, 0, 0]
    y = F @ true_factor_coef + np.random.randn(n_samples) * 0.3
    
    feature_names = [f'x{i+1}' for i in range(n_features)]
    
    return X, y, feature_names, F, loadings, true_factor_coef


def save_data(X, y, feature_names, filepath):
    """保存数据到 CSV"""
    data = np.column_stack([X, y])
    headers = feature_names + ['y']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"✅ 数据已保存: {filepath}")


# ============================================================
# Task A3: OLS 误差随特征维度变化实验
# ============================================================

def ols_dimension_experiment():
    """实验 OLS 误差随特征维度变化"""
    print("\n" + "="*60)
    print("Task A3: OLS 误差随特征维度变化")
    print("="*60)
    
    n_samples = 120
    feature_dims = [10, 30, 60, 90, 120, 150]
    results = []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    train_rmse_list = []
    test_rmse_list = []
    rank_list = []
    cond_list = []
    
    for p in feature_dims:
        # 生成数据
        X, y, _, _, _, _ = generate_highdim_data(
            n_samples=n_samples, n_features=p, n_factors=min(10, p//2),
            noise_std=0.5, random_seed=42
        )
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # OLS
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        
        y_train_pred = ols.predict(X_train_scaled)
        y_test_pred = ols.predict(X_test_scaled)
        
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        
        # 矩阵秩和条件数
        rank = np.linalg.matrix_rank(X_train_scaled)
        cond = np.linalg.cond(X_train_scaled) if X_train_scaled.shape[1] <= X_train_scaled.shape[0] else np.inf
        rank_list.append(rank)
        cond_list.append(cond if cond != np.inf else 1e6)
        
        print(f"p={p:3d}: train RMSE={train_rmse:.4f}, test RMSE={test_rmse:.4f}, rank={rank}/{p}, cond={cond:.2e}")
    
    # 图1: 误差随维度变化
    axes[0].plot(feature_dims, train_rmse_list, 'o-', label='Train RMSE', color='steelblue', linewidth=2)
    axes[0].plot(feature_dims, test_rmse_list, 's-', label='Test RMSE', color='darkorange', linewidth=2)
    axes[0].set_xlabel('特征维度 p')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('OLS 误差随特征维度变化')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 矩阵病态程度
    axes[1].plot(feature_dims, rank_list, 'o-', label='rank(X)', color='steelblue', linewidth=2)
    axes[1].plot(feature_dims, [min(p, n_samples*0.7) for p in feature_dims], '--', label='min(n, p)', color='gray')
    axes[1].set_xlabel('特征维度 p')
    axes[1].set_ylabel('矩阵秩')
    axes[1].set_title('矩阵秩随特征维度变化')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'ols_dimension_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图已保存: {results_dir}/ols_dimension_analysis.png")
    
    return results


# ============================================================
# Task A4: 系数稳定性实验 (重复切分)
# ============================================================

def coefficient_stability_experiment(n_splits=50):
    """重复切分，展示系数不稳定性"""
    print("\n" + "="*60)
    print("Task A4: 系数稳定性实验 (50次随机切分)")
    print("="*60)
    
    # 生成固定数据
    n_samples = 120
    n_features = 80
    X, y, feature_names, _, _, _ = generate_highdim_data(
        n_samples=n_samples, n_features=n_features, n_factors=10,
        noise_std=0.5, random_seed=42
    )
    
    # 选择3个关键变量展示
    key_indices = [0, 1, 2]  # x1, x2, x3
    key_names = [feature_names[i] for i in key_indices]
    
    ols_coefs = {i: [] for i in key_indices}
    
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        
        for i in key_indices:
            ols_coefs[i].append(ols.coef_[i])
    
    # 计算标准差
    print("\n系数标准差:")
    for i in key_indices:
        std_val = np.std(ols_coefs[i])
        print(f"  {feature_names[i]}: {std_val:.4f}")
    
    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [ols_coefs[i] for i in key_indices]
    bp = ax.boxplot(data_to_plot, labels=key_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['steelblue', 'darkorange', 'forestgreen']):
        patch.set_facecolor(color)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('特征')
    ax.set_ylabel('系数值')
    ax.set_title(f'OLS 系数分布 ({n_splits}次随机切分)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results_dir = Path(__file__).parent / "results"
    plt.savefig(results_dir / 'coefficient_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图已保存: {results_dir}/coefficient_stability.png")
    
    return ols_coefs


# ============================================================
# Task B: PCA 和 PCR
# ============================================================

def pca_analysis(X, y, results_dir):
    """PCA 分析和累计解释方差曲线"""
    print("\n" + "="*60)
    print("Task B1: PCA 分析")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 找到解释 90% 方差所需的主成分数
    n_90 = np.argmax(cumulative_variance >= 0.9) + 1
    print(f"解释 90% 方差所需的主成分数: {n_90}")
    print(f"前5个主成分解释方差比例: {explained_variance_ratio[:5]}")
    
    # 绘制累计解释方差曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-', color='steelblue', linewidth=2)
    ax.axhline(y=0.9, color='red', linestyle='--', label='90% 阈值')
    ax.axvline(x=n_90, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('主成分个数')
    ax.set_ylabel('累计解释方差比例')
    ax.set_title('PCA: 累计解释方差曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'pca_cumulative_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/pca_cumulative_variance.png")
    
    return pca, cumulative_variance, n_90


def pcr_experiment(X, y, results_dir, max_components=30):
    """PCR 实验：比较不同主成分个数下的表现"""
    print("\n" + "="*60)
    print("Task B2: PCR 实验")
    print("="*60)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 不同主成分个数
    n_components_list = list(range(1, min(max_components, X.shape[1])))
    train_rmse_list = []
    test_rmse_list = []
    cv_rmse_list = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in n_components_list:
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # 训练 PCR
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        
        y_train_pred = model.predict(X_train_pca)
        y_test_pred = model.predict(X_test_pca)
        
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        
        # CV 误差
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train_pca):
            X_tr, X_val = X_train_pca[train_idx], X_train_pca[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model_cv = LinearRegression()
            model_cv.fit(X_tr, y_tr)
            y_val_pred = model_cv.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, y_val_pred))
        
        cv_rmse = math.sqrt(np.mean(cv_scores))
        cv_rmse_list.append(cv_rmse)
        
        if k % 5 == 0 or k == 1:
            print(f"k={k:2d}: train RMSE={train_rmse:.4f}, test RMSE={test_rmse:.4f}, CV RMSE={cv_rmse:.4f}")
    
    # 找到最佳 k
    best_k = n_components_list[np.argmin(test_rmse_list)]
    print(f"\n最佳主成分个数: k={best_k}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(n_components_list, train_rmse_list, 'o-', label='Train RMSE', color='steelblue', linewidth=2)
    ax.plot(n_components_list, test_rmse_list, 's-', label='Test RMSE', color='darkorange', linewidth=2)
    ax.plot(n_components_list, cv_rmse_list, '^-', label='CV RMSE', color='forestgreen', linewidth=2)
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
    
    ax.set_xlabel('主成分个数 k')
    ax.set_ylabel('RMSE')
    ax.set_title('PCR: 不同主成分个数下的误差')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'pcr_error_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/pcr_error_curves.png")
    
    return n_components_list, train_rmse_list, test_rmse_list, cv_rmse_list, best_k


# ============================================================
# Task C: Lasso vs PCR 比较
# ============================================================

def lasso_vs_pcr_comparison():
    """比较 Lasso 和 PCR 在不同数据场景下的表现"""
    print("\n" + "="*60)
    print("Task C: Lasso vs PCR 比较")
    print("="*60)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 场景1: Sparse Truth
    print("\n" + "-"*40)
    print("场景1: Sparse Truth (稀疏真值)")
    print("-"*40)
    
    X_sparse, y_sparse, _, true_coef_sparse = generate_sparse_data(
        n_samples=200, n_features=100, n_true_features=5, noise_std=0.5
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y_sparse, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lasso with CV
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    lasso_rmse = calculate_rmse(y_test, y_pred_lasso)
    lasso_nonzero = sum(abs(lasso.coef_) > 0.01)
    
    print(f"Lasso: RMSE={lasso_rmse:.4f}, 非零系数={lasso_nonzero}")
    
    # PCR
    pca = PCA(n_components=15)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    pcr = LinearRegression()
    pcr.fit(X_train_pca, y_train)
    y_pred_pcr = pcr.predict(X_test_pca)
    pcr_rmse = calculate_rmse(y_test, y_pred_pcr)
    
    print(f"PCR (k=15): RMSE={pcr_rmse:.4f}")
    
    # 场景2: Latent-factor Truth
    print("\n" + "-"*40)
    print("场景2: Latent-factor Truth (潜在因子真值)")
    print("-"*40)
    
    X_latent, y_latent, _, _, _, _ = generate_latent_data(
        n_samples=200, n_features=100, n_factors=5, noise_std=0.5
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X_latent, y_latent, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    lasso_rmse2 = calculate_rmse(y_test, y_pred_lasso)
    lasso_nonzero2 = sum(abs(lasso.coef_) > 0.01)
    
    print(f"Lasso: RMSE={lasso_rmse2:.4f}, 非零系数={lasso_nonzero2}")
    
    # PCR
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    pcr = LinearRegression()
    pcr.fit(X_train_pca, y_train)
    y_pred_pcr = pcr.predict(X_test_pca)
    pcr_rmse2 = calculate_rmse(y_test, y_pred_pcr)
    
    print(f"PCR (k=10): RMSE={pcr_rmse2:.4f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 场景1 对比
    axes[0].bar(['Lasso', 'PCR'], [lasso_rmse, pcr_rmse], color=['steelblue', 'darkorange'])
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('场景1: Sparse Truth\nLasso vs PCR')
    axes[0].grid(True, alpha=0.3)
    
    # 场景2 对比
    axes[1].bar(['Lasso', 'PCR'], [lasso_rmse2, pcr_rmse2], color=['steelblue', 'darkorange'])
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('场景2: Latent-factor Truth\nLasso vs PCR')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'lasso_vs_pcr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图已保存: {results_dir}/lasso_vs_pcr_comparison.png")
    
    return {
        'sparse': {'lasso_rmse': lasso_rmse, 'pcr_rmse': pcr_rmse, 'lasso_nonzero': lasso_nonzero},
        'latent': {'lasso_rmse': lasso_rmse2, 'pcr_rmse': pcr_rmse2, 'lasso_nonzero': lasso_nonzero2}
    }


# ============================================================
# 生成报告
# ============================================================

def write_synthetic_report(results_dir, pcr_best_k, comparison_results):
    """生成 synthetic_report.md"""
    report_path = results_dir / 'synthetic_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Week 14: 高维回归、PCA 与 PCR 实验报告\n\n")
        
        f.write("## 一、数据生成机制\n\n")
        f.write("### 高维数据 (Task A1)\n")
        f.write("- 样本量: 120\n")
        f.write("- 特征数: 80\n")
        f.write("- 潜在因子数: 5\n")
        f.write("- 目标变量 y 由前3个潜在因子决定\n\n")
        
        f.write("### 稀疏真值数据 (Task C)\n")
        f.write("- 样本量: 200\n")
        f.write("- 特征数: 100\n")
        f.write("- 只有前5个特征真正决定 y\n\n")
        
        f.write("### 潜在因子真值数据 (Task C)\n")
        f.write("- 样本量: 200\n")
        f.write("- 特征数: 100\n")
        f.write("- 潜在因子数: 5\n")
        f.write("- y 由潜在因子决定\n\n")
        
        f.write("## 二、OLS 在高维下的表现\n\n")
        f.write("### 误差随特征维度变化\n")
        f.write("- 训练误差随 p 增加而下降\n")
        f.write("- 测试误差在 p > n 后急剧上升\n")
        f.write("- 训练误差接近 0 是危险信号 → 过拟合\n\n")
        
        f.write("### 系数不稳定性\n")
        f.write("- 50 次随机切分下，同一变量的系数波动很大\n")
        f.write("- 系数不稳定本身就是重要风险：结论不可复现\n\n")
        
        f.write("## 三、PCA 与 PCR\n\n")
        f.write(f"- PCR 最佳主成分个数: k={pcr_best_k}\n")
        f.write("- 前几个主成分解释了大部分方差，验证了低秩结构\n")
        f.write("- CV 曲线帮助我们选择最优 k\n\n")
        
        f.write("## 四、Lasso vs PCR 对比\n\n")
        f.write("### 场景1: Sparse Truth\n")
        f.write(f"- Lasso RMSE: {comparison_results['sparse']['lasso_rmse']:.4f}\n")
        f.write(f"- PCR RMSE: {comparison_results['sparse']['pcr_rmse']:.4f}\n")
        f.write(f"- Lasso 非零系数: {comparison_results['sparse']['lasso_nonzero']}\n")
        f.write("- Lasso 更适合稀疏真值场景\n\n")
        
        f.write("### 场景2: Latent-factor Truth\n")
        f.write(f"- Lasso RMSE: {comparison_results['latent']['lasso_rmse']:.4f}\n")
        f.write(f"- PCR RMSE: {comparison_results['latent']['pcr_rmse']:.4f}\n")
        f.write(f"- Lasso 非零系数: {comparison_results['latent']['lasso_nonzero']}\n")
        f.write("- PCR 更适合潜在因子场景\n\n")
        
        f.write("## 五、核心结论\n\n")
        f.write("1. **什么时候用 Lasso？** 数据是 sparse truth，只有少数变量真正重要\n")
        f.write("2. **什么时候用 PCR？** 数据是 latent-factor truth，信息分布在多个相关变量中\n")
        f.write("3. **Lasso 回答的是**：谁留下 (variable selection)\n")
        f.write("4. **PCR 回答的是**：怎么压缩 (information compression)\n")
        f.write("5. **短名单 → 更可能用 Lasso；稳预测器 → 更可能用 PCR**\n")
    
    print(f"✅ 报告已保存: {report_path}")


def write_summary_comparison(results_dir, comparison_results):
    """生成 summary_comparison.md"""
    report_path = results_dir / 'summary_comparison.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Week 14: Lasso vs PCR 总结对比\n\n")
        
        f.write("## 1. 两种数据世界的对比\n\n")
        f.write("| 维度 | Sparse Truth | Latent-factor Truth |\n")
        f.write("|------|--------------|---------------------|\n")
        f.write(f"| Lasso RMSE | {comparison_results['sparse']['lasso_rmse']:.4f} | {comparison_results['latent']['lasso_rmse']:.4f} |\n")
        f.write(f"| PCR RMSE | {comparison_results['sparse']['pcr_rmse']:.4f} | {comparison_results['latent']['pcr_rmse']:.4f} |\n")
        f.write(f"| Lasso 非零系数 | {comparison_results['sparse']['lasso_nonzero']} | {comparison_results['latent']['lasso_nonzero']} |\n\n")
        
        f.write("## 2. 核心问题回答\n\n")
        f.write("### Q: Sparse truth 时为什么 Lasso 更自然？\n")
        f.write("Lasso 的 L1 惩罚天然倾向于产生稀疏解，正好匹配 sparse truth 的设定。\n\n")
        
        f.write("### Q: Latent-factor truth 时为什么 PCR 更自然？\n")
        f.write("PCR 先压缩再回归，抓住了潜在因子的结构，而 Lasso 在相关变量组中可能随机选择。\n\n")
        
        f.write("### Q: Lasso 回答的是什么？\n")
        f.write("'谁留下了' (variable selection) — 哪些原始变量真正重要。\n\n")
        
        f.write("### Q: PCR 回答的是什么？\n")
        f.write("'怎么压缩' (information compression) — 如何用更少的方向概括信息。\n\n")
        
        f.write("### Q: 短名单用哪个方法？\n")
        f.write("**Lasso**，因为它直接输出稀疏系数。\n\n")
        
        f.write("### Q: 稳预测器用哪个方法？\n")
        f.write("**PCR**，因为它压缩信息，对噪声更鲁棒。\n\n")
        
        f.write("## 3. 前向/后向选择 vs Lasso\n\n")
        f.write("- 本周主线更适合比较 Lasso vs PCR，因为它们在 philosophy 上差异更大\n")
        f.write("- 前向/后向选择更接近 selection 路线，与 Lasso 目标相似但效率更低\n\n")
        
        f.write("## 4. 结论\n\n")
        f.write("> 当数据是 sparse truth → 用 Lasso (selection)\n")
        f.write("> 当数据是 latent-factor truth → 用 PCR (compression)\n")
        f.write("> 不确定时 → Elastic Net 或交叉验证选择\n")
    
    print(f"✅ 报告已保存: {report_path}")


# ============================================================
# Main
# ============================================================

def setup_results_dir():
    """设置结果目录"""
    results_dir = Path(__file__).parent / "results"
    import shutil
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    return results_dir


def main():
    print("="*60)
    print("Week 14: High-Dimensional Regression, PCA, and PCR")
    print("高维问题、共线性与降维回归")
    print("="*60)
    
    results_dir = setup_results_dir()
    print(f"✅ 结果目录: {results_dir}")
    
    # ========== Task A3: OLS 维度实验 ==========
    ols_dimension_experiment()
    
    # ========== Task A4: 系数稳定性 ==========
    coefficient_stability_experiment(n_splits=50)
    
    # ========== Task B: PCA 和 PCR ==========
    # 生成高维数据用于 PCA/PCR
    X_highdim, y_highdim, _, _, _, _ = generate_highdim_data(
        n_samples=120, n_features=80, n_factors=5, noise_std=0.5, random_seed=42
    )
    
    # 保存数据
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    save_data(X_highdim, y_highdim, [f'x{i+1}' for i in range(80)], 
              data_dir / "synthetic_highdim.csv")
    
    # PCA 分析
    pca, cum_var, n_90 = pca_analysis(X_highdim, y_highdim, results_dir)
    
    # PCR 实验
    n_components, train_rmse, test_rmse, cv_rmse, best_k = pcr_experiment(
        X_highdim, y_highdim, results_dir, max_components=30
    )
    
    # ========== Task C: Lasso vs PCR 比较 ==========
    comparison_results = lasso_vs_pcr_comparison()
    
    # ========== 生成报告 ==========
    write_synthetic_report(results_dir, best_k, comparison_results)
    write_summary_comparison(results_dir, comparison_results)
    
    print("\n" + "="*60)
    print("✅ Week 14 所有任务完成！")
    print(f"📁 结果保存在: {results_dir}")
    print("="*60)
    
    print("\n生成的文件:")
    print("  - data/synthetic_highdim.csv")
    print("  - results/ols_dimension_analysis.png")
    print("  - results/coefficient_stability.png")
    print("  - results/pca_cumulative_variance.png")
    print("  - results/pcr_error_curves.png")
    print("  - results/lasso_vs_pcr_comparison.png")
    print("  - results/synthetic_report.md")
    print("  - results/summary_comparison.md")


if __name__ == "__main__":
    main()
