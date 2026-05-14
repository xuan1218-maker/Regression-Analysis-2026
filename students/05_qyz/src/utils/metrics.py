"""
评估指标库 (Metrics Library)
包含 RMSE, MAE, MAPE 的计算函数
"""

import numpy as np


def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差 (Root Mean Square Error)
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    对大误差敏感，常用于回归模型评估
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    
    Returns:
    --------
    float : RMSE 值
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    MAE = mean(|y_true - y_pred|)
    对异常值不敏感，更稳健
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    
    Returns:
    --------
    float : MAE 值
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    以百分比形式表示预测误差，便于业务理解
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    epsilon : float
        极小值，防止除以零
    
    Returns:
    --------
    float : MAPE 百分比值（例如 5.2 表示 5.2%）
            如果所有真实值都为零，返回 inf
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 找出真实值不为零的位置
    mask = y_true != 0
    
    if not mask.any():
        return float('inf')
    
    # 计算百分比误差
    percentage_error = np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))
    
    return np.mean(percentage_error) * 100