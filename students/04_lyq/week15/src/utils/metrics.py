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

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss

# ====================== 新增分类指标 ======================
def calculate_class_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    输入真实标签、预测概率，按阈值计算全套分类指标
    返回：TP/TN/FP/FN/Accuracy/Precision/Recall/F1
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total != 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4)
    }

def scan_threshold_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, thresholds: list) -> list:
    """遍历一组阈值，批量输出各指标"""
    res_list = []
    for t in thresholds:
        met = calculate_class_metrics(y_true, y_pred_prob, t)
        met["threshold"] = t
        res_list.append(met)
    return res_list

def calculate_roc_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    return round(roc_auc_score(y_true, y_pred_prob), 4)

def calculate_logloss(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    return round(log_loss(y_true, y_pred_prob), 4)

# 新增损失函数：MSE、负对数似然(log loss)
def mse_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y - p) ** 2

def neg_log_likelihood(p: np.ndarray, y: np.ndarray, eps=1e-10) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

