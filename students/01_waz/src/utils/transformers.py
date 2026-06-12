"""
Module: utils.transformers
Purpose: Transformer-style preprocessing classes (fit / transform / fit_transform).
"""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer:
    """
    缺失值填补器 — 用训练集的列均值填补 NaN。

    Interface:
        fit(X)       → 计算并保存每列的均值
        transform(X) → 用保存的均值填补缺失值
        fit_transform(X) → fit + transform
    """

    def __init__(self):
        self.means_ = None

    def fit(self, X: np.ndarray):
        self.means_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("CustomImputer: 必须先调用 fit() 再调用 transform()")
        X_filled = X.copy()
        for i, mean_val in enumerate(self.means_):
            mask = np.isnan(X_filled[:, i])
            if np.any(mask):
                X_filled[mask, i] = mean_val
        return X_filled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    标准化器 — sklearn Pipeline 兼容。

    从 Week 13 开始继承 BaseEstimator/TransformerMixin 以便放在
    sklearn Pipeline 中与 GridSearchCV 配合使用。

    Interface:
        fit(X, y=None)       → 计算并保存 X 的均值 (mean_) 和标准差 (std_)
        transform(X)         → 用保存的参数对 X 做 Z-score 标准化
        fit_transform(X, y=None) → fit + transform

    Safety: 标准差为 0 的列不会被缩放（std_ 设为 1.0）。
    """

    def __init__(self, epsilon: float = 1e-12):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y=None):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0)
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CustomStandardScaler: 必须先调用 fit() 再调用 transform()")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)


class CustomNumericImputer:
    """Median/mean imputer fitted on training data only — Week 13 新增."""

    def __init__(self, strategy: str = "median"):
        if strategy not in {"median", "mean"}:
            raise ValueError("strategy must be 'median' or 'mean'")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        import pandas as pd
        numeric = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        if self.strategy == "median":
            stats = numeric.median()
        else:
            stats = numeric.mean()
        self.statistics_ = stats.fillna(0.0)
        return self

    def transform(self, X):
        import pandas as pd
        if self.statistics_ is None:
            raise RuntimeError("CustomNumericImputer: 必须先调用 fit() 再调用 transform()")
        numeric = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        return numeric.fillna(self.statistics_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)
