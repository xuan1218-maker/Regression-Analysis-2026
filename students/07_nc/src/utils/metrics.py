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


# ---------------------------------------------------------------------------
# Week15 新增：二分类指标工具
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray | list[float]) -> np.ndarray:
    """稳定计算 sigmoid 函数，用于把线性得分映射到 0 到 1 的概率。"""
    z_arr = np.asarray(z, dtype=float)
    z_clip = np.clip(z_arr, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


def binary_confusion_counts(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> dict[str, int]:
    """返回二分类混淆矩阵四个元素：TP、TN、FP、FN。"""
    true = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "TP": int(np.sum((true == 1) & (pred == 1))),
        "TN": int(np.sum((true == 0) & (pred == 0))),
        "FP": int(np.sum((true == 0) & (pred == 1))),
        "FN": int(np.sum((true == 1) & (pred == 0))),
    }


def summarize_binary_classification(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """基于给定阈值，把概率转成类别，并计算分类指标。"""
    true = np.asarray(y_true, dtype=int).ravel()
    prob = np.asarray(y_prob, dtype=float).ravel()
    if true.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    pred = (prob >= threshold).astype(int)
    counts = binary_confusion_counts(true, pred)
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        **counts,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
    }


def binary_log_loss_manual(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    eps: float = 1e-15,
) -> float:
    """手写二分类 log loss，与 Bernoulli 负对数似然对应。"""
    true = np.asarray(y_true, dtype=float).ravel()
    prob = np.asarray(y_prob, dtype=float).ravel()
    if true.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    p = np.clip(prob, eps, 1.0 - eps)
    return float(-np.mean(true * np.log(p) + (1.0 - true) * np.log(1.0 - p)))


def scan_thresholds(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    thresholds: np.ndarray | list[float],
) -> list[dict[str, float | int]]:
    """对一组阈值扫描 accuracy、precision、recall 和 F1。"""
    return [summarize_binary_classification(y_true, y_prob, float(t)) for t in thresholds]
