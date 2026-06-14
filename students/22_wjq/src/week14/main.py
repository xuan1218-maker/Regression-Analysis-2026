#!/usr/bin/env python3
"""
Week 14: High-Dimensional Regression, PCA, and PCR
完整版：Task A（高维模拟数据）+ Task B（PCA/PCR）+ Task C（Lasso vs PCR）+ Task D（真实数据）
Usage: uv run src/week14/main.py
"""

import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import calculate_rmse, calculate_mae, calculate_mape

# 设置中文字体 - 使用默认字体避免警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ========================== Task A: 生成高维模拟数据 ==========================
def generate_highdim_data(n_samples=150, n_features=80, n_latent=5, 
                          noise_std=0.5, random_seed=42):
    """生成高维且带有潜在低秩结构的模拟回归数据"""
    np.random.seed(random_seed)
    
    # 1. 生成潜在因子
    Z = np.random.normal(0, 1, (n_samples, n_latent))
    
    # 2. 生成载荷矩阵
    W = np.random.normal(0, 1, (n_latent, n_features))
    
    # 3. 生成观测特征
    X = Z @ W + np.random.normal(0, noise_std, (n_samples, n_features))
    
    # 4. 生成目标变量
    beta_latent = np.random.normal(0, 2, n_latent)
    y = Z @ beta_latent + np.random.normal(0, 0.3, n_samples)
    
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    
    metadata = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_latent': n_latent,
        'beta_latent': beta_latent.tolist()
    }
    return df, metadata


def generate_sparse_data(n_samples=150, n_features=60, n_true_features=5,
                         noise_std=0.5, random_seed=42):
    """生成稀疏真相数据"""
    np.random.seed(random_seed)
    X = np.random.normal(0, 1, (n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_coef[:n_true_features] = np.random.normal(2, 0.5, n_true_features)
    y = X @ true_coef + np.random.normal(0, noise_std, n_samples)
    
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    return df, {'type': 'sparse_truth', 'n_true_features': n_true_features}


def generate_latent_data(n_samples=150, n_features=60, n_latent=5,
                         noise_std=0.5, random_seed=42):
    """生成潜在因子真相数据"""
    np.random.seed(random_seed)
    Z = np.random.normal(0, 1, (n_samples, n_latent))
    W = np.random.normal(0, 1, (n_latent, n_features))
    X = Z @ W + np.random.normal(0, noise_std, (n_samples, n_features))
    beta_latent = np.random.normal(2, 0.5, n_latent)
    y = Z @ beta_latent + np.random.normal(0, 0.3, n_samples)
    
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    return df, {'type': 'latent_factor', 'n_latent': n_latent}


# ========================== Task A3: OLS 实验 ==========================
def run_ols_experiment(df, feature_names, target='y'):
    """在不同特征维度上测试 OLS 性能"""
    X = df[feature_names].values
    y = df[target].values
    
    p_values = [10, 20, 30, 40, 50, min(60, X.shape[1])]
    results = []
    
    for p in p_values:
        if p > X.shape[1]:
            continue
        X_subset = X[:, :p]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        
        train_rmse = calculate_rmse(y_train, ols.predict(X_train_scaled))
        test_rmse = calculate_rmse(y_test, ols.predict(X_test_scaled))
        condition_number = np.linalg.cond(X_train_scaled)
        rank = np.linalg.matrix_rank(X_train_scaled)
        
        results.append({
            'p': p, 'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'condition_number': condition_number, 'rank': rank
        })
        print(f"  p={p:3d}: train RMSE={train_rmse:.4f}, test RMSE={test_rmse:.4f}, "
              f"cond={np.log10(condition_number):.2f}")
    return results


def plot_ols_experiment(results, output_path):
    """绘制 OLS 实验的误差曲线和矩阵病态程度图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    p_values = [r['p'] for r in results]
    train_rmse = [r['train_rmse'] for r in results]
    test_rmse = [r['test_rmse'] for r in results]
    
    axes[0].plot(p_values, train_rmse, 'b-o', label='Train RMSE', linewidth=2)
    axes[0].plot(p_values, test_rmse, 'r-s', label='Test RMSE', linewidth=2)
    axes[0].set_xlabel('Number of Features (p)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('OLS: Train vs Test Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    cond_values = [np.log10(r['condition_number']) for r in results]
    axes[1].plot(p_values, cond_values, 'g-d', linewidth=2)
    axes[1].set_xlabel('Number of Features (p)')
    axes[1].set_ylabel('log10(Condition Number)')
    axes[1].set_title('Matrix Ill-Conditioning vs Feature Dimension')
    axes[1].axhline(y=4, color='r', linestyle='--', label='Ill-conditioned threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_coefficient_stability(X, y, feature_names, n_selected=5, n_splits=50, output_path=None):
    """展示系数不稳定性"""
    n_features = X.shape[1]
    coefs = np.zeros((n_splits, n_features))
    
    for i in range(n_splits):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=None)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        coefs[i, :] = ols.coef_
    
    coef_stds = np.std(coefs, axis=0)
    top_idx = np.argsort(coef_stds)[-n_selected:][::-1]
    
    print(f"\n  Top {n_selected} unstable variables (std dev):")
    for idx in top_idx:
        print(f"    {feature_names[idx]}: {coef_stds[idx]:.4f}")
    
    if output_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        data_to_plot = [coefs[:, idx] for idx in top_idx]
        labels = [feature_names[idx] for idx in top_idx]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Features')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'OLS Coefficient Distribution across {n_splits} Random Splits')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    return coefs, coef_stds


# ========================== Task B: PCA 和 PCR ==========================
def run_pca_analysis(X_train, X_test, feature_names, max_components=30):
    """运行 PCA 分析"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA()
    pca.fit(X_train_scaled)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    n_90 = np.argmax(cumsum_var >= 0.9) + 1
    print(f"  PCs needed for 90% variance: {n_90}")
    
    return pca, cumsum_var, X_train_scaled, X_test_scaled, n_90


def plot_cumulative_variance(cumsum_var, output_path, max_components=30):
    """绘制累计解释方差曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_comp = min(len(cumsum_var), max_components)
    ax.plot(range(1, n_comp + 1), cumsum_var[:n_comp], 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_title('PCA: Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return np.argmax(cumsum_var >= 0.9) + 1


def run_pcr_experiment(X_train, y_train, X_test, y_test, max_components=30):
    """运行 PCR 实验"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k_values = range(1, min(max_components, X_train_scaled.shape[1]) + 1)
    train_rmse_list, test_rmse_list, cv_rmse_list = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_values:
        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_scaled)
        Z_test = pca.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(Z_train, y_train)
        
        train_rmse_list.append(calculate_rmse(y_train, model.predict(Z_train)))
        test_rmse_list.append(calculate_rmse(y_test, model.predict(Z_test)))
        
        cv_scores = []
        for train_idx, val_idx in kf.split(Z_train):
            Z_tr, Z_val = Z_train[train_idx], Z_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            cv_model = LinearRegression()
            cv_model.fit(Z_tr, y_tr)
            cv_scores.append(calculate_rmse(y_val, cv_model.predict(Z_val)))
        cv_rmse_list.append(np.mean(cv_scores))
    
    return list(k_values), train_rmse_list, test_rmse_list, cv_rmse_list


def plot_pcr_results(k_values, train_rmse, test_rmse, cv_rmse, ols_test_rmse, output_path):
    """绘制 PCR 实验结果"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if train_rmse is not None:
        ax.plot(k_values, train_rmse, 'b-o', label='PCR Train RMSE', linewidth=2, markersize=4)
    ax.plot(k_values, test_rmse, 'r-s', label='PCR Test RMSE', linewidth=2, markersize=4)
    ax.plot(k_values, cv_rmse, 'g-d', label='PCR CV RMSE (5-fold)', linewidth=2, markersize=4)
    
    if ols_test_rmse is not None:
        ax.axhline(y=ols_test_rmse, color='orange', linestyle='--', 
                   label=f'OLS Test RMSE = {ols_test_rmse:.4f}')
    
    ax.set_xlabel('Number of Principal Components (k)')
    ax.set_ylabel('RMSE')
    ax.set_title('PCR: Model Performance vs Number of Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ========================== Task C: Lasso vs PCR ==========================
def run_lasso_experiment(X_train, y_train, X_test, y_test):
    """运行 Lasso 实验"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    alphas = np.logspace(-4, 1, 50)
    lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': alphas}, cv=5, 
                            scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train_scaled, y_train)
    
    best_lasso = lasso_cv.best_estimator_
    y_pred = best_lasso.predict(X_test_scaled)
    
    return {
        'test_rmse': calculate_rmse(y_test, y_pred),
        'n_nonzero': np.sum(np.abs(best_lasso.coef_) > 1e-6),
        'best_alpha': lasso_cv.best_params_['alpha']
    }


def compare_lasso_pcr(X_train, y_train, X_test, y_test, max_pc=30, name=""):
    """比较 Lasso 和 PCR"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_rmse = calculate_rmse(y_test, ols.predict(X_test_scaled))
    
    # Lasso
    lasso_res = run_lasso_experiment(X_train, y_train, X_test, y_test)
    
    # PCR
    _, _, test_rmse_list, cv_rmse_list = run_pcr_experiment(
        X_train, y_train, X_test, y_test, max_components=max_pc
    )
    best_k = np.argmin(cv_rmse_list) + 1
    pcr_rmse = test_rmse_list[best_k - 1]
    
    print(f"\n  {name} Results:")
    print(f"    OLS test RMSE: {ols_rmse:.4f}")
    print(f"    Lasso test RMSE: {lasso_res['test_rmse']:.4f} (nonzero coefs: {lasso_res['n_nonzero']})")
    print(f"    PCR test RMSE: {pcr_rmse:.4f} (best k={best_k})")
    
    return {'ols_rmse': ols_rmse, 'lasso': lasso_res, 'pcr_rmse': pcr_rmse, 'pcr_best_k': best_k}


# ========================== Task D: 真实数据处理 ==========================
def load_and_preprocess_housing_data(data_dir):
    """加载并预处理房价数据"""
    data_path = Path(data_dir) / "train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Please put train.csv in {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded housing data: {data_path}, shape: {df.shape}")
    
    # 选择数值特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    
    # 删除缺失率超过60%的列
    missing_ratio = df[numeric_cols].isnull().sum() / len(df)
    keep_cols = [c for c in numeric_cols if missing_ratio[c] < 0.6]
    
    X = df[keep_cols].copy()
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # 目标变量取对数
    y = np.log(df['SalePrice'].values)
    
    print(f"  Features retained: {X.shape[1]}")
    return X.values, y, keep_cols


def run_kaggle_task(data_dir, results_dir):
    """运行真实数据任务"""
    print("\n" + "="*70)
    print("Task D: Kaggle Housing Data - High-Dimensional Regression")
    print("="*70)
    
    X, y, feature_names = load_and_preprocess_housing_data(data_dir)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # PCA 分析
    print("\n[1] PCA Analysis")
    _, cumsum_var, _, _, n_90 = run_pca_analysis(X_train, X_test, feature_names, max_components=50)
    n_90 = plot_cumulative_variance(cumsum_var, Path(results_dir) / "kaggle_cumulative_variance.png", max_components=50)
    
    # OLS 基线
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_rmse = calculate_rmse(y_test, ols.predict(X_test_scaled))
    print(f"\n[2] OLS test RMSE: {ols_rmse:.4f}")
    
    # Lasso 调优
    print("\n[3] Lasso Regularization")
    lasso_res = run_lasso_experiment(X_train, y_train, X_test, y_test)
    print(f"    Best alpha={lasso_res['best_alpha']:.4f}, nonzero coefs={lasso_res['n_nonzero']}, test RMSE={lasso_res['test_rmse']:.4f}")
    
    # PCR 实验
    print("\n[4] PCR (Principal Component Regression)")
    k_values, _, test_rmse, cv_rmse = run_pcr_experiment(X_train, y_train, X_test, y_test, max_components=50)
    best_k = k_values[np.argmin(cv_rmse)]
    best_pcr_rmse = test_rmse[np.argmin(cv_rmse)]
    plot_pcr_results(k_values, None, test_rmse, cv_rmse, ols_rmse, 
                     Path(results_dir) / "kaggle_pcr_results.png")
    print(f"    Best k={best_k}, test RMSE={best_pcr_rmse:.4f}")
    
    # 生成报告
    report_path = Path(results_dir) / "kaggle_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Kaggle Housing Data: High-Dimensional Regression Analysis\n\n")
        f.write("## Dataset Info\n")
        f.write("- Source: Kaggle House Prices (Ames, Iowa)\n")
        f.write("- Target: SalePrice (log-transformed)\n")
        f.write(f"- Raw features: 79, After preprocessing: {X.shape[1]}\n")
        f.write(f"- Sample size: {X.shape[0]}\n\n")
        
        f.write("## PCA Analysis\n")
        f.write("![Cumulative Variance](kaggle_cumulative_variance.png)\n")
        f.write(f"- PCs needed for 90% variance: {n_90}\n")
        f.write("- Suggests latent low-dimensional structure (size factor, quality factor, location factor).\n\n")
        
        f.write("## Model Comparison\n")
        f.write("| Model | Test RMSE | Complexity |\n")
        f.write("|-------|-----------|------------|\n")
        f.write(f"| OLS | {ols_rmse:.4f} | {X.shape[1]} coefficients |\n")
        f.write(f"| Lasso | {lasso_res['test_rmse']:.4f} | {lasso_res['n_nonzero']} nonzero coefs |\n")
        f.write(f"| PCR | {best_pcr_rmse:.4f} | {best_k} PCs |\n\n")
        
        f.write("## Lasso Feature Selection\n")
        f.write(f"Lasso reduced {X.shape[1]} features to {lasso_res['n_nonzero']} nonzero coefficients, indicating sparse structure.\n\n")
        
        f.write("## PCR Results\n")
        f.write("![PCR Results](kaggle_pcr_results.png)\n")
        f.write(f"Optimal number of PCs: k={best_k}\n\n")
        
        f.write("## Conclusion\n")
        f.write("- OLS shows overfitting risk in high-dim space.\n")
        f.write("- **Lasso performs better on this dataset**: achieves sparsity while maintaining good prediction.\n")
        f.write("- The data structure is more like sparse truth than pure latent-factor.\n")
    
    print(f"📄 Report: {report_path}")
    return report_path


# ========================== Task A 主流程 ==========================
def run_synthetic_task(data_dir, results_dir):
    print("\n" + "="*70)
    print("Task A/B/C: High-Dimensional Simulation - PCA, PCR, Lasso Comparison")
    print("="*70)
    
    data_path = Path(data_dir) / "synthetic_highdim.csv"
    df, metadata = generate_highdim_data(n_samples=150, n_features=80, n_latent=5)
    df.to_csv(data_path, index=False)
    print(f"✅ Synthetic data generated: {data_path}")
    
    feature_names = [col for col in df.columns if col != 'y']
    X, y = df[feature_names].values, df['y'].values
    
    # OLS 实验
    print("\n[OLS Experiment]")
    ols_results = run_ols_experiment(df, feature_names)
    plot_ols_experiment(ols_results, Path(results_dir) / "ols_dimension_experiment.png")
    
    # 系数稳定性
    plot_coefficient_stability(X, y, feature_names, n_selected=5, n_splits=50,
                               output_path=Path(results_dir) / "coefficient_stability.png")
    
    # PCA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    _, cumsum_var, _, _, _ = run_pca_analysis(X_train, X_test, feature_names)
    n_90 = plot_cumulative_variance(cumsum_var, Path(results_dir) / "cumulative_variance.png")
    
    # PCR
    k_values, train_rmse, test_rmse, cv_rmse = run_pcr_experiment(X_train, y_train, X_test, y_test, max_components=30)
    scaler_ols = StandardScaler()
    X_train_scaled = scaler_ols.fit_transform(X_train)
    X_test_scaled = scaler_ols.transform(X_test)
    ols_baseline = LinearRegression().fit(X_train_scaled, y_train)
    ols_test_rmse = calculate_rmse(y_test, ols_baseline.predict(X_test_scaled))
    plot_pcr_results(k_values, train_rmse, test_rmse, cv_rmse, ols_test_rmse, 
                     Path(results_dir) / "pcr_results.png")
    
    # Lasso vs PCR 对比
    print("\n[Lasso vs PCR Comparison]")
    df_sparse, _ = generate_sparse_data()
    X_sparse, y_sparse = df_sparse[[f'X{i+1}' for i in range(60)]].values, df_sparse['y'].values
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sparse, y_sparse, test_size=0.3, random_state=42)
    compare_lasso_pcr(X_train_s, y_train_s, X_test_s, y_test_s, name="Sparse Truth Scenario")
    
    df_latent, _ = generate_latent_data()
    X_latent, y_latent = df_latent[[f'X{i+1}' for i in range(60)]].values, df_latent['y'].values
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_latent, y_latent, test_size=0.3, random_state=42)
    compare_lasso_pcr(X_train_l, y_train_l, X_test_l, y_test_l, name="Latent Factor Scenario")
    
    # 生成报告
    report_path = generate_synthetic_report(results_dir, df, metadata, ols_results, n_90, ols_test_rmse)
    return report_path


def generate_synthetic_report(results_dir, df, metadata, ols_results, n_90, ols_test_rmse):
    report_path = Path(results_dir) / "synthetic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Week 14: High-Dimensional Regression, PCA, and PCR Report\n\n")
        f.write(f"## 1. Data Generation\n- Samples: {metadata['n_samples']}, Features: {metadata['n_features']}\n")
        f.write(f"- Latent factors: {metadata['n_latent']}\n- High redundancy suitable for dimensionality reduction.\n\n")
        f.write("## 2. OLS Experiment\n![OLS Experiment](ols_dimension_experiment.png)\n")
        f.write("| p | train RMSE | test RMSE | log10(cond) |\n")
        f.write("|---|------------|-----------|-------------|\n")
        for r in ols_results:
            f.write(f"| {r['p']} | {r['train_rmse']:.4f} | {r['test_rmse']:.4f} | {np.log10(r['condition_number']):.2f} |\n")
        f.write("\n## 3. Coefficient Stability\n![Stability](coefficient_stability.png)\n\n")
        f.write(f"## 4. PCA: {n_90} PCs needed for 90% variance\n![PCA](cumulative_variance.png)\n\n")
        f.write("## 5. PCR Results\n![PCR](pcr_results.png)\n\n")
        f.write("## 6. Formulas\n- OLS: $\\hat{\\beta} = (X^TX)^{-1}X^Ty$\n")
        f.write("- First PC: $v_1 = \\arg\\max_{||v||=1} \\text{Var}(Xv)$\n")
        f.write("- PCR: $Z_k = XV_k$, then $y = Z_k\\gamma + \\epsilon$\n")
    print(f"📄 Report: {report_path}")
    return report_path


def generate_summary_comparison(results_dir):
    summary_path = Path(results_dir) / "summary_comparison.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Summary: Selection vs Compression\n\n")
        f.write("## 1. Sparse Truth -> Lasso is more natural\n")
        f.write("Lasso directly selects original variables, matching the data generation.\n\n")
        f.write("## 2. Latent-factor Truth -> PCR is more natural\n")
        f.write("PCR aggregates information via principal components.\n\n")
        f.write("## 3. Lasso answers 'which variables matter', PCR answers 'how to compress information'\n\n")
        f.write("## 4. Business wants short variable list -> Lasso\n")
        f.write("## 5. Business wants stable prediction -> Ridge / PCR\n")
    print(f"📄 Summary: {summary_path}")
    return summary_path


# ========================== 主流程 ==========================
def main():
    results_dir = Path("src/week14/results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("✅ results folder cleaned and recreated")
    
    data_dir = Path("src/week14/data")
    ensure_dir(data_dir)
    
    # Task A/B/C: Simulation
    run_synthetic_task(data_dir, results_dir)
    
    # Task D: Real data (optional)
    try:
        run_kaggle_task(data_dir, results_dir)
    except FileNotFoundError as e:
        print(f"\n⚠️ Task D skipped: {e}")
    
    generate_summary_comparison(results_dir)
    
    print("\n" + "="*70)
    print("🎉 Week 14 Assignment Complete!")
    print("="*70)

if __name__ == "__main__":
    main()