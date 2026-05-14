import numpy as np

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """平均绝对百分比误差，自动处理分母为0/极小值"""
    abs_error = np.abs(y_true - y_pred)
    # 避免除以0，加入极小值
    abs_true = np.abs(y_true) + epsilon
    mape = np.mean(abs_error / abs_true) * 100
    return mape
