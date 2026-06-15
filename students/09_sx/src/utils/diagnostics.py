"""
模块：工具.诊断
用途：模型诊断工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# 图表使用英文，避免中文字体问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_vif(X: np.ndarray) -> List[float]:
    """计算每个特征的方差膨胀因子（VIF）"""
    n_samples, n_features = X.shape
    vif_values = []
    
    for i in range(n_features):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        X_others_with_intercept = np.column_stack([np.ones(n_samples), X_others])
        
        XTX = X_others_with_intercept.T @ X_others_with_intercept
        XTX_inv = np.linalg.inv(XTX + 1e-10 * np.eye(XTX.shape[0]))
        beta = XTX_inv @ (X_others_with_intercept.T @ y)
        
        y_pred = X_others_with_intercept @ beta
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float('inf')
        
        vif_values.append(vif)
    
    return vif_values


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """绘制相关矩阵热力图（图中使用英文）"""
    
    plot_cols = feature_cols + [target_col]
    corr_matrix = df[plot_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=10)
    
    labels = [col.replace('_', ' ') for col in plot_cols]
    ax.set_xticks(range(len(plot_cols)))
    ax.set_yticks(range(len(plot_cols)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    for i in range(len(plot_cols)):
        for j in range(len(plot_cols)):
            text_color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                   ha="center", va="center", color=text_color, fontsize=7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_coefficient_stability(coeffs_dict: dict, feature_names: List[str], 
                                figsize: tuple = (12, 6), save_path: str = None):
    """绘制系数稳定性对比箱线图（图中使用英文）"""
    
    fig, axes = plt.subplots(1, len(coeffs_dict), figsize=figsize)
    
    if len(coeffs_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, coeffs_array) in enumerate(coeffs_dict.items()):
        n_features_to_show = min(6, coeffs_array.shape[1])
        
        data_to_plot = []
        labels_to_use = []
        
        for i in range(n_features_to_show):
            data_to_plot.append(coeffs_array[:, i])
            labels_to_use.append(feature_names[i])
        
        bp = axes[idx].boxplot(data_to_plot, labels=labels_to_use, patch_artist=True)
        
        for box in bp['boxes']:
            box.set_facecolor('lightblue')
            box.set_alpha(0.7)
        
        axes[idx].set_title(f'{model_name} - Coefficient Stability', fontsize=12)
        axes[idx].set_xlabel('Features', fontsize=10)
        axes[idx].set_ylabel('Coefficient Value', fontsize=10)
        axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Coefficient Stability Across Random Splits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cv_curves(cv_results_dict: dict, param_name: str = 'alpha', 
                   figsize: tuple = (10, 6), save_path: str = None):
    """
    绘制交叉验证曲线（U型曲线）
    注意：scoring='neg_mean_squared_error'，分数越高越好（越接近0）
    所以 -neg_mse = mse，用于显示真实的MSE
    """
    plt.figure(figsize=figsize)
    
    for model_name, results in cv_results_dict.items():
        params = results['params']
        mean_scores = results['mean_scores']  # 这是 neg_mean_squared_error
        
        # 转换为正的MSE：MSE = -neg_mean_squared_error
        mse_values = -mean_scores
        
        plt.plot(params, mse_values, marker='o', label=f'{model_name}', linewidth=2)
    
    plt.xscale('log')
    plt.xlabel(f'{param_name} (log scale)', fontsize=12)
    plt.ylabel('Cross-validation MSE', fontsize=12)
    plt.title('Regularization Strength vs Model Performance (U-shaped Curve)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_coefficient_comparison(coeffs_dict: dict, feature_names: List[str],
                                figsize: tuple = (12, 6), save_path: str = None):
    """绘制不同模型的系数对比图（图中使用英文）"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    for idx, (model_name, coeffs) in enumerate(coeffs_dict.items()):
        offset = (idx - len(coeffs_dict)/2 + 0.5) * width
        ax.bar(x + offset, coeffs, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Coefficient Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig




# src/utils/diagnostics.py (末尾添加)
"""
模块：工具.诊断
用途：模型诊断工具 - 新增矩阵病态诊断
"""

def compute_condition_number(X: np.ndarray) -> float:
    """计算矩阵的条件数，评估病态程度"""
    X_center = X - np.mean(X, axis=0)
    _, S, _ = np.linalg.svd(X_center, full_matrices=False)
    return S[0] / S[-1] if S[-1] > 1e-10 else np.inf

def compute_rank_deficiency(X: np.ndarray) -> dict:
    """计算矩阵的秩亏缺程度"""
    rank = np.linalg.matrix_rank(X)
    n_samples, n_features = X.shape
    return {
        'rank': rank,
        'n_samples': n_samples,
        'n_features': n_features,
        'deficiency': n_features - rank,
        'rank_ratio': rank / min(n_samples, n_features)
    }

def diagnose_multicollinearity(X: np.ndarray, threshold: float = 0.8) -> dict:
    """诊断多重共线性"""
    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []
    
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    return {
        'max_correlation': np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])),
        'high_correlation_pairs': len(high_corr_pairs),
        'pairs': high_corr_pairs[:10]
    }