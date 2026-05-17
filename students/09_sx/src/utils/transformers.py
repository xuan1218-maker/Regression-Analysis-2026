"""
模块：工具.转换器
用途：存放手写的预处理类
"""

import numpy as np


class CustomStandardScaler:
    """自定义标准化转换器"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'CustomStandardScaler':
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")
        X = np.asarray(X)
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class SimpleImputer:
    """简单填补器"""
    
    def __init__(self, strategy: str = 'mean', fill_value: float = 0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        X = np.asarray(X)
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'constant':
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"不支持的填补策略: {self.strategy}")
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")
        X = np.asarray(X).copy()
        nan_mask = np.isnan(X)
        for col in range(X.shape[1]):
            X[nan_mask[:, col], col] = self.statistics_[col]
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)