"""
Module: utils.transformers
Purpose: Transformer-style preprocessing classes (fit / transform / fit_transform).
"""
import numpy as np


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


class CustomStandardScaler:
    """
    标准化器 — 严格遵循 sklearn Transformer 接口规范。

    Interface:
        fit(X)       → 计算并保存 X 的均值 (mean_) 和标准差 (std_)
        transform(X) → 用保存的参数对 X 做 Z-score 标准化
        fit_transform(X) → fit + transform

    Safety: 标准差为 0 的列不会被缩放（std_ 设为 1.0）。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 防止除零：常量列不缩放
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CustomStandardScaler: 必须先调用 fit() 再调用 transform()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
