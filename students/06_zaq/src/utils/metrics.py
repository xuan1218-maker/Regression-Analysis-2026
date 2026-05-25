"""
Module: utils.metrics
Purpose: Evaluation metrics
"""
import math


def calculate_rmse(y_true, y_pred):
    """Root Mean Square Error"""
    n = len(y_true)
    return math.sqrt(sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n)


def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n


def calculate_mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error"""
    n = len(y_true)
    return 100 * sum(abs((y_true[i] - y_pred[i]) / max(abs(y_true[i]), eps)) for i in range(n)) / n


def calculate_all_metrics(y_true, y_pred):
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }
