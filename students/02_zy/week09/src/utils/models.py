"""
models.py

自定义 OLS 回归模型。
"""

import numpy as np


class CustomOLS:
    """
    手写 Ordinary Least Squares 线性回归。
    """

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        intercept = np.ones((X.shape[0], 1))
        X_design = np.hstack([intercept, X])

        try:
            self.coef_ = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("模型还没有训练，请先调用 fit。")

        X = np.asarray(X, dtype=float)

        intercept = np.ones((X.shape[0], 1))
        X_design = np.hstack([intercept, X])

        return (X_design @ self.coef_).reshape(-1)