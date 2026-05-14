"""
模块：utils.metrics
用途：手写回归评估指标
"""

import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    均方根误差 Root Mean Squared Error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均绝对误差 Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    平均绝对百分比误差 Mean Absolute Percentage Error
    注意：处理真实值接近 0 的情况，避免除以零
    """
    # 避免分母为 0 或极小值
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0