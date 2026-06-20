"""
Metrics Library (Regression Evaluation Toolkit)

包含：
- RMSE
- MAE
- MAPE（稳定版）
"""

import numpy as np


def _to_array(y):
    return np.asarray(y)


# =========================
# RMSE
# =========================
def calculate_rmse(y_true, y_pred):
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# =========================
# MAE
# =========================
def calculate_mae(y_true, y_pred):
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    return np.mean(np.abs(y_true - y_pred))


# =========================
# MAPE
# =========================
def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """
    Robust MAPE:
    避免 y_true=0 导致不稳定
    """

    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    denom = np.maximum(np.abs(y_true), epsilon)

    return np.mean(np.abs((y_true - y_pred) / denom)) * 100