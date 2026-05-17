"""
模块：工具.诊断
用途：模型统计诊断工具
包含：方差膨胀因子计算
"""

import numpy as np
from typing import List


def calculate_vif(X: np.ndarray) -> List[float]:
    """
    计算每个特征的方差膨胀因子（VIF）
    
    公式：VIF_j = 1 / (1 - R_j^2)
    其中 R_j^2 是将第 j 个特征作为因变量对其他所有特征做回归得到的拟合优度
    
    参数：
        X: 特征矩阵，形状为 (n_samples, n_features)
        
    返回：
        vif_values: 每个特征的 VIF 值列表
    """
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