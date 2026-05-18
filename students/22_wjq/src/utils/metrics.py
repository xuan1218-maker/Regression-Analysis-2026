import numpy as np

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """平均绝对百分比误差，处理 y_true 接近 0 的情况"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # 避免除零，将很小的真实值替换为 epsilon
    mask = np.abs(y_true) < epsilon
    y_true_safe = np.where(mask, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100