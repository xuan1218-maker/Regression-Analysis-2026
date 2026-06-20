"""
Module: utils.metrics
Purpose: Regression evaluation metrics — RMSE, MAE, MAPE, MSE.
"""
import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


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


def summarize_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Return a dict of RMSE, MAE, MAPE for reporting."""
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Week 15: classification metrics
# ---------------------------------------------------------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TP, TN, FP, FN from binary (0/1) labels."""
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return float(np.mean(yt == yp))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP)."""
    counts = confusion_counts(y_true, y_pred)
    denom = counts["TP"] + counts["FP"]
    return float(counts["TP"] / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN)."""
    counts = confusion_counts(y_true, y_pred)
    denom = counts["TP"] + counts["FN"]
    return float(counts["TP"] / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 at a given threshold."""
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy / log loss."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1 - eps)
    return float(-np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))
