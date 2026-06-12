"""手写回归评价指标，从 Week10/Week11 开始持续复用。"""
from __future__ import annotations

import numpy as np


def _to_1d_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("metric inputs must not be empty")
    return arr


def _validate_metric_inputs(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    true = _to_1d_float_array(y_true)
    pred = _to_1d_float_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    finite_mask = np.isfinite(true) & np.isfinite(pred)
    if not finite_mask.any():
        raise ValueError("metric inputs contain no finite paired observations")
    return true[finite_mask], pred[finite_mask]


def calculate_rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """计算均方根误差 RMSE。"""
    true, pred = _validate_metric_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def calculate_mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """计算平均绝对误差 MAE。"""
    true, pred = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def calculate_mape(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    epsilon: float = 1e-8,
) -> float:
    """计算平均绝对百分比误差 MAPE，返回百分数。"""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    true, pred = _validate_metric_inputs(y_true, y_pred)
    safe_mask = np.abs(true) > epsilon
    if not safe_mask.any():
        return float("nan")

    percentage_errors = np.abs((true[safe_mask] - pred[safe_mask]) / true[safe_mask])
    return float(np.mean(percentage_errors) * 100.0)


def summarize_regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    """返回报告中使用的回归指标字典。"""
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


def calculate_mse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """计算均方误差 MSE；用于后续偏差-方差实验。"""
    true, pred = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def generalization_gap(train_error: float, test_error: float) -> float:
    """返回测试误差减训练误差，用于 Week12 模型复杂度曲线。"""
    return float(test_error - train_error)
