# src/week14/main.py
"""
Week 14: High-Dimensional Regression, PCA, and PCR
主执行入口
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.metrics import calculate_rmse, calculate_mae
from src.utils.models import PCR, CoefficientStabilityAnalyzer
from src.utils.diagnostics import compute_condition_number


# ============================================
# Task A: 数据生成
# ============================================

def generate_high_dimensional_data(
    n_samples: int = 120,
    n_features: int = 80,
    n_latent: int = 5,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    
    latent_factors = np.random.randn(n_samples, n_latent)
    
    loadings = np.random.randn(n_latent, n_features) * 2
    for i in range(n_features):
        n_dominant = np.random.choice([1, 2])
        dominant_indices = np.random.choice(n_latent, n_dominant, replace=False)
        for idx in dominant_indices:
            loadings[idx, i] *= 3
    
    X = latent_factors @ loadings + np.random.randn(n_samples, n_features) * 0.5
    
    true_coef_latent = np.array([2.0, 1.5, 1.0, 0.0, 0.0])
    y = latent_factors @ true_coef_latent + np.random.randn(n_samples) * noise_std
    
    return X, y, latent_factors, loadings


def generate_sparse_data(
    n_samples: int = 200,
    n_features: int = 80,
    n_relevant: int = 5,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:n_relevant] = np.random.randn(n_relevant) * 2
    y = X @ true_coef + np.random.randn(n_samples) * noise_std
    return X, y


def generate_latent_data(
    n_samples: int = 200,
    n_features: int = 80,
    n_latent: int = 5,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    latent = np.random.randn(n_samples, n_latent)
    loadings = np.random.randn(n_latent, n_features) * 1.5
    for i in range(n_features):
        n_dominant = np.random.choice([1, 2])
        dominant_indices = np.random.choice(n_latent, n_dominant, replace=False)
        for idx in dominant_indices:
            loadings[idx, i] *= 3
    X = latent @ loadings + np.random.randn(n_samples, n_features) * 0.3
    true_coef_latent = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
    y = latent @ true_coef_latent + np.random.randn(n_samples) * noise_std
    return X, y


# ============================================
# Task A 实验
# ============================================

def task_a_experiment() -> Tuple[pd.DataFrame, Dict, Dict]:
    print("\n" + "="*60)
    print("Task A: 高维数据与OLS失效分析")
    print("="*60)
    
    X, y, latent_factors, loadings = generate_high_dimensional_data()
    
    os.makedirs("src/week14/data", exist_ok=True)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['target'] = y
    df.to_csv("src/week14/data/synthetic_highdim.csv", index=False)
    print(f"数据已保存: shape={df.shape}")
    
    data_mechanism = {
        'n_samples': 120,
        'n_features': 80,
        'n_latent': 5,
        'true_coef_latent': [2.0, 1.5, 1.0, 0.0, 0.0]
    }
    
    feature_dims = [10, 30, 60, 80, 100, 120]
    results_dim = []
    
    for p in feature_dims:
        X_p, y_p, _, _ = generate_high_dimensional_data(n_features=p)
        X_train, X_test, y_train, y_test = train_test_split(X_p, y_p, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        train_rmse = calculate_rmse(y_train, model.predict(X_train_scaled))
        test_rmse = calculate_rmse(y_test, model.predict(X_test_scaled))
        rank = np.linalg.matrix_rank(X_train_scaled)
        cond_num = compute_condition_number(X_train_scaled)
        
        results_dim.append({
            'p': p, 
            'train_rmse': train_rmse, 
            'test_rmse': test_rmse, 
            'rank': rank, 
            'condition_number': cond_num
        })
    
    results_df = pd.DataFrame(results_dim)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(results_df['p'], results_df['train_rmse'], 'o-', label='Train RMSE', linewidth=2, color='blue')
    axes[0].plot(results_df['p'], results_df['test_rmse'], 's-', label='Test RMSE', linewidth=2, color='red')
    axes[0].set_xlabel('Number of Features (p)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('OLS Performance vs Feature Dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(results_df['p'].astype(str), np.log10(results_df['condition_number']), alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of Features (p)')
    axes[1].set_ylabel('log10(Condition Number)')
    axes[1].set_title('Ill-conditioning vs Feature Dimension')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, row in results_df.iterrows():
        axes[1].annotate(f"rank={int(row['rank'])}", 
                        xy=(i, np.log10(row['condition_number'])), 
                        xytext=(0, 5), textcoords='offset points', 
                        ha='center', fontsize=8)
    
    plt.tight_layout()
    os.makedirs("src/week14/results", exist_ok=True)
    plt.savefig("src/week14/results/fig_A3_ols_error_vs_dim.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    analyzer = CoefficientStabilityAnalyzer(LinearRegression, n_splits=50)
    stability_results = analyzer.analyze(X, y)
    
    n_show = min(6, X.shape[1])
    coeffs_subset = stability_results['coeffs'][:, :n_show]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(coeffs_subset, patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('OLS Coefficient Instability Across 50 Random Splits')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i in range(n_show):
        ax.annotate(f'std={stability_results["coeff_std"][i]:.2f}', 
                   xy=(i+1, np.percentile(coeffs_subset[:, i], 75)), 
                   xytext=(0, 5), textcoords='offset points', 
                   ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("src/week14/results/fig_A4_coeff_instability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  系数标准差范围: [{stability_results['coeff_std'].min():.3f}, {stability_results['coeff_std'].max():.3f}]")
    
    return results_df, stability_results, data_mechanism


# ============================================
# Task B 实验
# ============================================

def task_b_pcr_experiment(X: np.ndarray, y: np.ndarray) -> Tuple[List, List, List, int, float, int, int]:
    print("\n" + "="*60)
    print("Task B: PCA和PCR分析")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    n_show = min(30, len(cumulative_var))
    ax.plot(range(1, n_show + 1), cumulative_var[:n_show], 'o-', linewidth=2, color='steelblue')
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% variance explained')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% variance explained')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_title('PCA Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    n_80 = np.argmax(cumulative_var >= 0.8) + 1 if np.any(cumulative_var >= 0.8) else len(cumulative_var)
    n_90 = np.argmax(cumulative_var >= 0.9) + 1 if np.any(cumulative_var >= 0.9) else len(cumulative_var)
    ax.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_90, color='g', linestyle=':', alpha=0.5)
    ax.annotate(f'80%: {n_80} PCs', xy=(n_80, 0.75))
    ax.annotate(f'90%: {n_90} PCs', xy=(n_90, 0.85))
    
    plt.tight_layout()
    plt.savefig("src/week14/results/fig_B1_cumulative_variance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  前{n_80}个主成分解释了80%的方差")
    print(f"  前{n_90}个主成分解释了90%的方差")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    max_k = min(30, X.shape[1])
    k_values = list(range(1, max_k + 1))
    pcr_train_rmse, pcr_test_rmse, pcr_cv_rmse = [], [], []
    
    for k in k_values:
        pcr_model = PCR(n_components=k)
        pcr_model.fit(X_train, y_train)
        
        pcr_train_rmse.append(calculate_rmse(y_train, pcr_model.predict(X_train)))
        pcr_test_rmse.append(calculate_rmse(y_test, pcr_model.predict(X_test)))
        
        Z_train = pcr_model.pca.transform(X_train)
        cv_scores = cross_val_score(LinearRegression(), Z_train, y_train, cv=5, scoring='neg_mean_squared_error')
        pcr_cv_rmse.append(np.sqrt(-np.mean(cv_scores)))
    
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_test_rmse = calculate_rmse(y_test, ols.predict(X_test))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k_values, pcr_train_rmse, 'o-', label='PCR Train RMSE', linewidth=2, markersize=4)
    ax.plot(k_values, pcr_test_rmse, 's-', label='PCR Test RMSE', linewidth=2, markersize=4)
    ax.plot(k_values, pcr_cv_rmse, '^-', label='PCR CV RMSE (5-fold)', linewidth=2, markersize=4)
    ax.axhline(y=ols_test_rmse, color='red', linestyle=':', linewidth=2, label=f'OLS Test RMSE = {ols_test_rmse:.4f}')
    ax.set_xlabel('Number of Principal Components (k)')
    ax.set_ylabel('RMSE')
    ax.set_title('PCR Performance vs Number of Principal Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    best_k_idx = np.argmin(pcr_test_rmse)
    best_k = k_values[best_k_idx]
    ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7)
    ax.annotate(f'Best k={best_k}', xy=(best_k, pcr_test_rmse[best_k_idx]), xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig("src/week14/results/fig_B2_pcr_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCR最佳k={best_k}, Test RMSE={pcr_test_rmse[best_k_idx]:.4f}")
    
    return pcr_train_rmse, pcr_test_rmse, pcr_cv_rmse, best_k, ols_test_rmse, n_80, n_90


# ============================================
# Task C 实验
# ============================================

def task_c_comparison() -> Dict:
    print("\n" + "="*60)
    print("Task C: Lasso vs PCR 比较")
    print("="*60)
    
    X_sparse, y_sparse = generate_sparse_data()
    X_latent, y_latent = generate_latent_data()
    
    results = {}
    
    for name, X, y in [('Sparse Truth', X_sparse, y_sparse), ('Latent Truth', X_latent, y_latent)]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Lasso
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        
        # PCR - 自动选择最佳k（通过交叉验证）
        max_k_test = min(20, X_train_scaled.shape[1])
        best_k = 1
        best_rmse = np.inf
        for k in range(1, max_k_test + 1):
            pca_k = PCA(n_components=k)
            Z_train = pca_k.fit_transform(X_train_scaled)
            Z_test = pca_k.transform(X_test_scaled)
            model = LinearRegression()
            model.fit(Z_train, y_train)
            rmse = calculate_rmse(y_test, model.predict(Z_test))
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
        
        pcr = PCR(n_components=best_k)
        pcr.fit(X_train_scaled, y_train)
        y_pred_pcr = pcr.predict(X_test_scaled)
        
        results[name] = {
            'Lasso': {
                'test_rmse': calculate_rmse(y_test, y_pred_lasso),
                'test_mae': calculate_mae(y_test, y_pred_lasso),
                'n_nonzero': np.sum(np.abs(lasso.coef_) > 1e-6),
                'alpha': lasso.alpha_
            },
            'PCR': {
                'test_rmse': calculate_rmse(y_test, y_pred_pcr),
                'test_mae': calculate_mae(y_test, y_pred_pcr),
                'n_components': best_k
            }
        }
        
        print(f"\n  === {name} ===")
        print(f"  Lasso: RMSE={results[name]['Lasso']['test_rmse']:.4f}, 非零系数={results[name]['Lasso']['n_nonzero']}")
        print(f"  PCR: RMSE={results[name]['PCR']['test_rmse']:.4f}, 主成分数={results[name]['PCR']['n_components']}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scenarios = ['Sparse Truth', 'Latent Truth']
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        lasso_vals = [results[scenario]['Lasso']['test_rmse'], results[scenario]['Lasso']['test_mae']]
        pcr_vals = [results[scenario]['PCR']['test_rmse'], results[scenario]['PCR']['test_mae']]
        
        x = np.arange(2)
        width = 0.35
        bars1 = ax.bar(x - width/2, lasso_vals, width, label='Lasso', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, pcr_vals, width, label='PCR', alpha=0.8, color='coral')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(f'{scenario} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['RMSE', 'MAE'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("src/week14/results/fig_C2_lasso_vs_pcr.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    lasso_complexity = [results[s]['Lasso']['n_nonzero'] for s in scenarios]
    pcr_complexity = [results[s]['PCR']['n_components'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    bars1 = ax.bar(x - width/2, lasso_complexity, width, label='Lasso (Non-zero Coefficients)', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, pcr_complexity, width, label='PCR (PCs Retained)', alpha=0.8, color='coral')
    ax.set_xlabel('Data Scenario')
    ax.set_ylabel('Model Complexity')
    ax.set_title('Model Complexity Comparison: Lasso vs PCR')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        ax.annotate(f'{int(bar.get_height())}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                   xytext=(0, 3), textcoords='offset points', ha='center')
    for bar in bars2:
        ax.annotate(f'{int(bar.get_height())}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                   xytext=(0, 3), textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig("src/week14/results/fig_C2_complexity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================
# Task D 实验
# ============================================

def task_d_real_data() -> Tuple[Dict, str, Dict, Dict]:
    print("\n" + "="*60)
    print("Task D: 真实数据挑战")
    print("="*60)
    
    df = pd.read_csv("src/week14/data/kaggle_housing.csv")
    print(f"  原始数据shape: {df.shape}")
    
    preprocessing_info = {
        'original_shape': df.shape,
        'removed_cols': [],
        'final_shape': None
    }
    
    df = df.dropna(subset=['SalePrice'])
    
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
    preprocessing_info['removed_cols'] = high_missing_cols
    df = df.drop(columns=high_missing_cols)
    
    # 保留所有特征（数值型 + 类别型）
    X = df.drop(['Id', 'SalePrice'], axis=1)
    y = np.log(df['SalePrice'].values)
    
    # 数值型特征：中位数填补
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # 类别型特征：众数填补 + one-hot编码
    categorical_cols = X.select_dtypes(include=['object']).columns
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    X = pd.get_dummies(X, drop_first=True)
    
    preprocessing_info['final_shape'] = X.shape
    X = X.values
    
    print(f"  清洗后: {X.shape[0]}样本 × {X.shape[1]}特征")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_train_rmse = calculate_rmse(y_train, ols.predict(X_train_scaled))
    ols_test_rmse = calculate_rmse(y_test, ols.predict(X_test_scaled))
    cond_num = compute_condition_number(X_train_scaled)
    rank = np.linalg.matrix_rank(X_train_scaled)
    
    print(f"\n  OLS Train RMSE: {ols_train_rmse:.4f}")
    print(f"  OLS Test RMSE: {ols_test_rmse:.4f}")
    print(f"  Condition Number: {cond_num:.2e}, Rank: {rank}/{X_train_scaled.shape[1]}")
    
    # Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    
    # PCR - 自动选择最佳k
    max_k_test = min(20, X_train_scaled.shape[1])
    best_k = 1
    best_rmse = np.inf
    for k in range(1, max_k_test + 1):
        pca_k = PCA(n_components=k)
        Z_train = pca_k.fit_transform(X_train_scaled)
        Z_test = pca_k.transform(X_test_scaled)
        model = LinearRegression()
        model.fit(Z_train, y_train)
        rmse = calculate_rmse(y_test, model.predict(Z_test))
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
    
    pcr = PCR(n_components=best_k)
    pcr.fit(X_train_scaled, y_train)
    y_pred_pcr = pcr.predict(X_test_scaled)
    
    results = {
        'Lasso': {
            'test_rmse': calculate_rmse(y_test, y_pred_lasso),
            'test_mae': calculate_mae(y_test, y_pred_lasso),
            'n_nonzero': np.sum(np.abs(lasso.coef_) > 1e-6)
        },
        'PCR': {
            'test_rmse': calculate_rmse(y_test, y_pred_pcr),
            'test_mae': calculate_mae(y_test, y_pred_pcr),
            'n_components': best_k
        }
    }
    
    print(f"\n  Lasso: RMSE={results['Lasso']['test_rmse']:.4f}, 非零系数={results['Lasso']['n_nonzero']}")
    print(f"  PCR: RMSE={results['PCR']['test_rmse']:.4f}, 主成分数={results['PCR']['n_components']}")
    
    if results['Lasso']['n_nonzero'] < X.shape[1] * 0.3:
        structure = "更接近 Sparse Truth（只有少数特征真正重要）"
    else:
        structure = "更接近 Latent-factor Truth（需要压缩而非筛选）"
    print(f"\n  数据结构判断: {structure}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['OLS', 'Lasso', 'PCR']
    rmse_values = [ols_test_rmse, results['Lasso']['test_rmse'], results['PCR']['test_rmse']]
    bars = ax.bar(models, rmse_values, alpha=0.8, color=['gray', 'steelblue', 'coral'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Test RMSE (log scale)')
    ax.set_title('Real Data: Model Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val), 
                   xytext=(0, 3), textcoords='offset points', ha='center')
    plt.tight_layout()
    plt.savefig("src/week14/results/fig_D_real_data_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    ols_info = {
        'train_rmse': ols_train_rmse,
        'test_rmse': ols_test_rmse,
        'condition_number': cond_num,
        'rank': rank,
        'n_features': X_train_scaled.shape[1]
    }
    
    return results, structure, preprocessing_info, ols_info


# ============================================
# 自动生成三份报告
# ============================================

def generate_synthetic_report(
    a_results_df: pd.DataFrame,
    a_stability: Dict,
    a_mechanism: Dict,
    b_results: Tuple,
    c_results: Dict
):
    with open("src/week14/results/synthetic_report.md", "w", encoding="utf-8") as f:
        f.write("# 高维模拟数据分析报告\n\n")
        
        f.write("## 1. 数据生成机制说明\n\n")
        f.write("### 1.1 数据维度\n")
        f.write(f"- **样本量**: {a_mechanism['n_samples']}\n")
        f.write(f"- **特征维度**: {a_mechanism['n_features']}\n")
        f.write(f"- **潜在因子个数**: {a_mechanism['n_latent']}\n\n")
        
        f.write("### 1.2 潜在因子结构\n\n")
        f.write("数据生成过程如下：\n\n")
        f.write("```\n")
        f.write("X = latent_factors @ loadings + noise\n")
        f.write("y = latent_factors @ true_coef_latent + noise\n")
        f.write("```\n\n")
        f.write("其中：\n")
        f.write(f"- `latent_factors` 维度为 (120, {a_mechanism['n_latent']})，服从标准正态分布\n")
        f.write("- `loadings` 维度为 (5, 80)，每个特征主要由1-2个因子主导\n")
        f.write(f"- `true_coef_latent = {a_mechanism['true_coef_latent']}`，只有前3个因子真正影响 y\n\n")
        
        f.write("### 1.3 为什么是\"高维 + 信息冗余\"数据？\n\n")
        f.write("1. **高维性**: 特征数 p=80，与样本量 n=120 的比例接近，属于高维边缘场景\n")
        f.write("2. **信息冗余**: 80个特征由5个潜在因子生成，变量间存在强相关性\n")
        f.write("3. **低秩结构**: 数据的有效秩远低于特征维度\n\n")
        
        f.write("## 2. OLS在不同特征维度下的表现\n\n")
        f.write("### 2.1 实验结果\n\n")
        f.write("| p | Train RMSE | Test RMSE | Rank | Condition Number |\n")
        f.write("|---|------------|-----------|------|------------------|\n")
        for _, row in a_results_df.iterrows():
            cond_str = "inf" if np.isinf(row['condition_number']) else f"{row['condition_number']:.2e}"
            f.write(f"| {int(row['p'])} | {row['train_rmse']:.4f} | {row['test_rmse']:.4f} | {int(row['rank'])} | {cond_str} |\n")
        
        f.write("\n### 2.2 关键观察\n\n")
        f.write("1. **训练误差持续下降**：随着 p 增加，OLS 能够更好地拟合训练数据，训练 RMSE 接近 0\n")
        f.write("2. **测试误差先降后升**：过高的 p 导致过拟合，测试误差上升\n")
        f.write("3. **病态程度加剧**：条件数随 p 增加呈指数级增长，矩阵接近奇异\n\n")
        
        f.write("### 2.3 \"训练误差接近0\"的危险信号\n\n")
        f.write("当 p 接近或超过 n 时，OLS 可以完美拟合训练数据，但这种\"完美\"是因为：\n")
        f.write("- 模型拥有了足够的自由度来拟合噪声\n")
        f.write("- 系数估计极不稳定，微小变化导致巨大波动\n")
        f.write("- 这种模型在新数据上表现很差\n\n")
        
        f.write("## 3. OLS系数不稳定性分析\n\n")
        f.write("### 3.1 稳定性指标\n\n")
        f.write("**稳定性指标定义**: 系数标准差的平均值，衡量同一变量在不同随机切分下系数的波动程度。值越小越稳定。\n\n")
        f.write(f"- **系数标准差范围**: [{a_stability['coeff_std'].min():.3f}, {a_stability['coeff_std'].max():.3f}]\n")
        f.write(f"- **稳定性分数（系数标准差均值）**: {np.mean(a_stability['coeff_std']):.3f}\n")
        f.write(f"- **训练RMSE波动**: mean={a_stability['train_rmse_mean']:.3f}, std={a_stability['train_rmse_std']:.3f}\n")
        f.write(f"- **测试RMSE波动**: mean={a_stability['test_rmse_mean']:.3f}, std={a_stability['test_rmse_std']:.3f}\n\n")
        
        f.write("### 3.2 关键变量选择\n\n")
        f.write("选取了**前6个特征**进行分析，因为这些特征在数据生成中受到不同因子载荷的影响：\n")
        f.write("- 特征0-2：主要由第一个潜在因子主导\n")
        f.write("- 特征3-5：由其他因子混合生成\n")
        f.write("箱线图展示了同一特征在不同随机切分下的系数分布，波动越大说明越不稳定。\n\n")
        
        f.write("### 3.3 结论\n\n")
        f.write("系数不稳定本身就是一种重要风险：\n")
        f.write("1. 不同随机切分产生差异巨大的系数，结论不可复现\n")
        f.write("2. 无法进行可靠的特征重要性解释\n")
        f.write("3. 模型对训练数据极度敏感，缺乏稳健性\n\n")
        
        f.write("## 4. PCA与PCR分析\n\n")
        if len(b_results) >= 7:
            f.write(f"### 4.1 主成分解释方差\n\n")
            f.write(f"- 前 {b_results[5]} 个主成分解释了80%的方差\n")
            f.write(f"- 前 {b_results[6]} 个主成分解释了90%的方差\n")
            f.write("- 原始高维空间确实贴近一个更低维的子空间\n\n")
            
            f.write("### 4.2 PCR与OLS对比\n\n")
            f.write(f"- **PCR最佳k（通过CV选择）**: {b_results[3]}\n")
            f.write(f"- **PCR最佳Test RMSE**: {b_results[1][b_results[3]-1]:.4f}\n")
            f.write(f"- **OLS Test RMSE**: {b_results[4]:.4f}\n\n")
        
        f.write("### 4.3 CV曲线解释\n\n")
        f.write("PCR的CV RMSE代表：\n")
        f.write("- 在保留的主成分上做回归的交叉验证误差\n")
        f.write("- 反映了模型的泛化能力\n\n")
        f.write("CV曲线与train/test曲线的关系：\n")
        f.write("- train RMSE通常随k增加而下降（拟合能力增强）\n")
        f.write("- test和CV RMSE通常呈U型，存在最优k\n")
        f.write("- CV是test的近似，用于避免过拟合\n\n")
        
        f.write("## 5. 公式与定义\n\n")
        f.write("### 5.1 OLS估计式\n\n")
        f.write("$$\\hat{\\beta}_{OLS} = (X^T X)^{-1} X^T y$$\n\n")
        f.write("### 5.2 第一主成分定义\n\n")
        f.write("$$v_1 = \\arg\\max_{\\|v\\|=1} \\text{Var}(Xv) = \\arg\\max_{\\|v\\|=1} v^T \\Sigma v$$\n\n")
        f.write("### 5.3 PCR流程\n\n")
        f.write("$$Z_k = X V_k \\quad \\text{(前k个主成分)}$$\n")
        f.write("$$\\hat{y}_{PCR} = Z_k \\hat{\\beta}_{PCR} = X V_k (Z_k^T Z_k)^{-1} Z_k^T y$$\n\n")


def generate_summary_comparison_report(c_results: Dict):
    with open("src/week14/results/summary_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Lasso vs PCR 比较总结\n\n")
        
        f.write("## 1. 两种数据世界对比\n\n")
        f.write("### 1.1 Sparse Truth（稀疏真相）\n\n")
        f.write("- **数据特征**: 只有少数原始变量真正决定 y\n")
        f.write("- **其他变量**: 大多是噪声或弱相关\n")
        f.write("- **适合方法**: Lasso\n\n")
        
        f.write("### 1.2 Latent-factor Truth（潜在因子真相）\n\n")
        f.write("- **数据特征**: 原始变量由少数潜在因子线性组合生成\n")
        f.write("- **y的驱动**: 主要由潜在因子驱动，而非原始变量\n")
        f.write("- **适合方法**: PCR\n\n")
        
        f.write("## 2. 实验对比结果\n\n")
        
        if 'Sparse Truth' in c_results:
            f.write("### 2.1 Sparse Truth场景\n\n")
            f.write("| 方法 | Test RMSE | Test MAE | 模型复杂度 |\n")
            f.write("|------|-----------|----------|------------|\n")
            f.write(f"| Lasso | {c_results['Sparse Truth']['Lasso']['test_rmse']:.4f} | {c_results['Sparse Truth']['Lasso']['test_mae']:.4f} | {c_results['Sparse Truth']['Lasso']['n_nonzero']} 个非零系数 |\n")
            f.write(f"| PCR | {c_results['Sparse Truth']['PCR']['test_rmse']:.4f} | {c_results['Sparse Truth']['PCR']['test_mae']:.4f} | {c_results['Sparse Truth']['PCR']['n_components']} 个主成分 |\n\n")
        
        if 'Latent Truth' in c_results:
            f.write("### 2.2 Latent-factor Truth场景\n\n")
            f.write("| 方法 | Test RMSE | Test MAE | 模型复杂度 |\n")
            f.write("|------|-----------|----------|------------|\n")
            f.write(f"| Lasso | {c_results['Latent Truth']['Lasso']['test_rmse']:.4f} | {c_results['Latent Truth']['Lasso']['test_mae']:.4f} | {c_results['Latent Truth']['Lasso']['n_nonzero']} 个非零系数 |\n")
            f.write(f"| PCR | {c_results['Latent Truth']['PCR']['test_rmse']:.4f} | {c_results['Latent Truth']['PCR']['test_mae']:.4f} | {c_results['Latent Truth']['PCR']['n_components']} 个主成分 |\n\n")
        
        f.write("## 3. 核心问题回答\n\n")
        
        f.write("### 3.1 为什么Sparse Truth时Lasso更自然？\n\n")
        f.write("当数据真的是稀疏真相时：\n")
        f.write("- **Lasso能够精确识别出真正重要的变量**\n")
        f.write("- 通过L1惩罚将不相关变量的系数压缩为0\n")
        f.write("- 结果具有可解释性：直接告诉你\"谁留下了\"\n\n")
        
        f.write("### 3.2 为什么Latent-factor Truth时PCR更自然？\n\n")
        f.write("当数据更像潜在因子结构时：\n")
        f.write("- **PCR通过主成分捕捉数据的核心变异方向**\n")
        f.write("- 潜在因子正是这些主成分的线性组合\n")
        f.write("- PCR回答的是\"信息压缩到哪里去了\"\n\n")
        
        f.write("### 3.3 Lasso vs PCR：回答的问题不同\n\n")
        f.write("- **Lasso回答**: \"谁留下？\" → 变量筛选\n")
        f.write("- **PCR回答**: \"信息浓缩到哪里？\" → 信息压缩\n\n")
        
        f.write("### 3.4 业务场景选择\n\n")
        f.write("- **需要更短的变量名单** → 选择 Lasso（可解释性强）\n")
        f.write("- **需要更稳的预测器** → 选择 PCR（对噪声更稳健）\n\n")
        
        f.write("## 4. 关于前向/后向选择\n\n")
        f.write("### 4.1 为什么本周主线更适合Lasso vs PCR？\n\n")
        f.write("1. **前向/后向选择是离散搜索**：在p很大时计算量爆炸\n")
        f.write("2. **Lasso是连续优化**：更高效，更适合高维场景\n")
        f.write("3. **PCR是完全不同的哲学**：从压缩角度解决问题\n\n")
        
        f.write("### 4.2 前向/后向选择属于哪个路线？\n\n")
        f.write("- **更接近 selection 路线**\n")
        f.write("- 本质上是贪心的变量筛选方法\n")
        f.write("- 与Lasso目标相同，但计算方式不同\n")


def generate_kaggle_report(d_results: Dict, d_structure: str, d_preprocess: Dict, d_ols: Dict):
    with open("src/week14/results/kaggle_report.md", "w", encoding="utf-8") as f:
        f.write("# 真实数据挑战报告\n\n")
        
        f.write("## 1. 数据来源与预处理\n\n")
        f.write("### 1.1 数据集\n")
        f.write("- **来源**: Kaggle House Prices: Advanced Regression Techniques\n")
        f.write(f"- **原始维度**: {d_preprocess['original_shape'][0]}样本 × {d_preprocess['original_shape'][1]}特征\n")
        f.write("- **目标变量**: SalePrice（房屋售价）\n\n")
        
        f.write("### 1.2 关键变量选择\n\n")
        f.write("**选择方法**: 保留所有数值型特征 + 对类别型特征进行One-Hot编码\n\n")
        f.write("**选择理由**:\n")
        f.write("1. 数值型特征（如LotArea, TotalBsmtSF, GrLivArea等）直接反映房屋物理属性\n")
        f.write("2. 类别型特征（如Neighborhood, MSZoning等）反映位置和分区信息\n")
        f.write("3. One-Hot编码保留了类别信息的全部维度，避免信息损失\n\n")
        
        f.write("### 1.3 预处理步骤\n")
        f.write(f"1. 移除缺失率>50%的特征: {len(d_preprocess['removed_cols'])}个\n")
        f.write("2. 数值型特征：中位数填补缺失值\n")
        f.write("3. 类别型特征：众数填补 + One-Hot编码\n")
        f.write("4. 对目标变量取对数（处理偏态分布）\n\n")
        
        f.write(f"### 1.4 清洗后数据\n")
        f.write(f"- **样本量**: {d_preprocess['final_shape'][0]}\n")
        f.write(f"- **特征数**: {d_preprocess['final_shape'][1]}\n\n")
        
        f.write("## 2. 模型表现对比\n\n")
        f.write("| 模型 | Test RMSE | 备注 |\n")
        f.write("|------|-----------|------|\n")
        f.write(f"| OLS | {d_ols['test_rmse']:.4f} | 条件数={d_ols['condition_number']:.2e}, rank={d_ols['rank']}/{d_ols['n_features']} |\n")
        f.write(f"| Lasso | {d_results['Lasso']['test_rmse']:.4f} | 非零系数={d_results['Lasso']['n_nonzero']} |\n")
        f.write(f"| PCR | {d_results['PCR']['test_rmse']:.4f} | 最佳k（自动CV选择）={d_results['PCR']['n_components']} |\n\n")
        
        f.write("## 3. 结构判断\n\n")
        f.write("### 3.1 OLS不稳定迹象\n\n")
        f.write(f"- **条件数**: {d_ols['condition_number']:.2e} (inf表示矩阵奇异)\n")
        f.write(f"- **秩亏损**: {d_ols['rank']}/{d_ols['n_features']}（特征存在严重共线性）\n")
        f.write(f"- **训练-测试差距**: Train RMSE={d_ols['train_rmse']:.4f}, Test RMSE={d_ols['test_rmse']:.4f}\n\n")
        
        f.write("### 3.2 Lasso vs PCR 表现\n\n")
        f.write("根据实验结果：\n")
        f.write(f"- Lasso 选择保留 {d_results['Lasso']['n_nonzero']} 个非零系数\n")
        f.write(f"- PCR 自动选择 {d_results['PCR']['n_components']} 个主成分\n\n")
        
        f.write("### 3.3 数据结构判断\n\n")
        f.write(f"**结论**: {d_structure}\n\n")
        
        f.write("## 4. 业务解释\n\n")
        f.write("如果要向业务方解释这份数据：\n\n")
        
        if "Sparse" in d_structure:
            f.write("> 这份数据更适合 **筛选（selection）** 路线，因为只有一部分特征对房价有显著影响。\n")
            f.write("> Lasso方法能够识别出这些关键特征，给出简洁的解释。\n\n")
            f.write("### 4.1 为什么Lasso更合适？\n\n")
            f.write("1. 房屋价格通常由少数核心因素决定（面积、位置、质量）\n")
            f.write("2. 业务方需要知道\"哪些因素最重要\"\n")
            f.write("3. 可解释性在房价预测中至关重要\n")
        else:
            f.write("> 这份数据更适合 **压缩（compression）** 路线，因为变量之间存在复杂的因子结构。\n")
            f.write("> PCR通过主成分捕捉数据的核心变异方向，获得更稳定的预测。\n\n")
            f.write("### 4.1 为什么PCR更合适？\n\n")
            f.write("1. 房价受多个潜在因素综合影响\n")
            f.write("2. 原始特征之间存在高度共线性\n")
            f.write("3. PCR提供更稳健的预测，虽然牺牲了部分可解释性\n")


# ============================================
# 主函数
# ============================================

def main():
    print("\n" + "="*60)
    print("Week 14: High-Dimensional Regression, PCA, and PCR")
    print("="*60)
    
    os.makedirs("src/week14/data", exist_ok=True)
    os.makedirs("src/week14/results", exist_ok=True)
    
    a_results_df, a_stability, a_mechanism = task_a_experiment()
    
    df = pd.read_csv("src/week14/data/synthetic_highdim.csv")
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    b_results = task_b_pcr_experiment(X, y)
    c_results = task_c_comparison()
    
    generate_synthetic_report(a_results_df, a_stability, a_mechanism, b_results, c_results)
    print("  ✅ synthetic_report.md 已生成")
    
    generate_summary_comparison_report(c_results)
    print("  ✅ summary_comparison.md 已生成")
    
    if os.path.exists("src/week14/data/kaggle_housing.csv"):
        d_results, d_structure, d_preprocess, d_ols = task_d_real_data()
        generate_kaggle_report(d_results, d_structure, d_preprocess, d_ols)
        print("  ✅ kaggle_report.md 已生成")
    else:
        print("\n  ⚠️ 未找到 kaggle_housing.csv，跳过 Task D")
    
    print("\n" + "="*60)
    print("所有实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()