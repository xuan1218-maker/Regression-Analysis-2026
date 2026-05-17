"""
模块：工具.评估指标
用途：存放 RMSE, MAE, MAPE 的计算函数
"""

import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """计算平均绝对百分比误差"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.abs(y_true) > epsilon
    if np.sum(mask) == 0:
        return 0.0
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    mape = np.mean(percentage_errors) * 100.0
    return mape