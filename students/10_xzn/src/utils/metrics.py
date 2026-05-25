import numpy as np


def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差
    MAE = mean(|y_true - y_pred|)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """
    计算平均绝对百分比误差
    MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    注意：处理 y_true 为 0 或极小值的情况
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 避免除零，将 y_true 中接近 0 的值替换为 epsilon
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    ape = np.abs((y_true_safe - y_pred) / y_true_safe) * 100
    return np.mean(ape)
