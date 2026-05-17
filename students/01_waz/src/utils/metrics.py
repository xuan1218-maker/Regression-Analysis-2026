"""
Module: utils.metrics
Purpose: Regression evaluation metrics — RMSE, MAE, MAPE.
"""
import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (in %).

    处理分母为 0 或极小值的异常情况：
    当 |y_true| <= epsilon 时跳过该样本，避免除零导致无穷大 MAPE。
    若所有样本都被跳过则返回 NaN。
    """
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
