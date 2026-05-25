import numpy as np

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差 RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差 MAE"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对百分比误差 MAPE，处理分母为0的情况"""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100