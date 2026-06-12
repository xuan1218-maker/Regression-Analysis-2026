"""
模块：utils.metrics
用途：手写回归评估指标与稳定性指标
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


def coefficient_cv(coef_matrix: np.ndarray) -> np.ndarray:
    """
    计算系数的变异系数 (Coefficient of Variation) 作为稳定性指标。

    CV = std / |mean|，值越大说明系数在不同切分下越不稳定。

    Parameters
    ----------
    coef_matrix : np.ndarray, shape (n_splits, n_features)
        每行是一次切分得到的系数向量

    Returns
    -------
    np.ndarray
        每个特征的 CV 值
    """
    means = np.mean(coef_matrix, axis=0)
    stds = np.std(coef_matrix, axis=0)
    # 避免除零
    safe_means = np.where(np.abs(means) < 1e-10, 1e-10, means)
    return stds / np.abs(safe_means)