"""个人 utils 库中持续维护的回归诊断工具。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.models import AnalyticalOLS


def add_intercept(X: np.ndarray) -> np.ndarray:
    """给数值设计矩阵添加最前面的截距列。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    return np.column_stack([np.ones(X.shape[0]), X])


def calculate_vif(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """使用自定义 OLS 模型计算方差膨胀因子 VIF。

    VIF_j = 1 / (1 - R_j^2)，其中 R_j^2 来自“用其他特征回归第 j 个特征”。
    这里直接使用 numpy 和自定义 AnalyticalOLS 实现，不依赖 statsmodels。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must equal number of X columns")

    rows: list[dict[str, float | str]] = []
    for j, name in enumerate(feature_names):
        y_j = X[:, j]
        other_idx = [i for i in range(X.shape[1]) if i != j]
        if len(other_idx) == 0 or np.isclose(np.var(y_j), 0.0):
            vif = np.inf
        else:
            X_other = add_intercept(X[:, other_idx])
            model = AnalyticalOLS().fit(X_other, y_j)
            r2 = model.score(X_other, y_j)
            vif = np.inf if np.isclose(1.0 - r2, 0.0) else 1.0 / max(1e-12, 1.0 - r2)
        rows.append({"feature": name, "VIF": float(vif)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def correlation_pairs(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """返回数值列中绝对相关系数超过阈值的变量对。"""
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    rows: list[dict[str, float | str]] = []
    cols = list(corr.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_1": col_a, "feature_2": col_b, "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """为报告生成基础残差诊断统计。"""
    residuals = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_p95_abs": float(np.quantile(np.abs(residuals), 0.95)),
    }
