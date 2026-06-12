import numpy as np

def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    """决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
