"""
Module: utils.models
Purpose: Core machine learning estimators for regression.
"""

import numpy as np
from scipy import stats
from typing import Dict


class AnalyticalOLS:
    """解析解 OLS 回归模型"""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.residuals_ = None
        self.fitted_values_ = None
        self._X_design = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        n = X.shape[0]
        return np.column_stack([np.ones(n), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_design = self._add_intercept(X)
        self._X_design = X_design
        n, p = X_design.shape

        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        XtY = X_design.T @ y
        self.coef_ = XtX_inv @ XtY

        self.fitted_values_ = X_design @ self.coef_
        self.residuals_ = y - self.fitted_values_

        RSS = np.sum(self.residuals_ ** 2)
        self.df_resid_ = n - p
        self.sigma2_ = RSS / self.df_resid_
        self.cov_matrix_ = self.sigma2_ * XtX_inv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("必须先调用 fit()")
        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        return 1 - SSE / SST

    def f_test(self, C: np.ndarray, d: np.ndarray) -> Dict[str, float]:
        if self.coef_ is None:
            raise RuntimeError("必须先调用 fit()")
        C = np.asarray(C)
        d = np.asarray(d).reshape(-1, 1)
        q = C.shape[0]

        C_beta = C @ self.coef_.reshape(-1, 1)
        diff = C_beta - d

        XtX = self._X_design.T @ self._X_design
        XtX_inv = np.linalg.inv(XtX)

        C_XtX_inv_Ct = C @ XtX_inv @ C.T
        C_XtX_inv_Ct_inv = np.linalg.inv(C_XtX_inv_Ct)

        quad_form = diff.T @ C_XtX_inv_Ct_inv @ diff
        f_stat = quad_form.item() / (q * self.sigma2_)
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {"f_stat": f_stat, "p_value": p_value}


class GradientDescentOLS:
    """梯度下降 OLS 回归模型"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.loss_history_ = []
        self._X_design = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        n = X.shape[0]
        return np.column_stack([np.ones(n), X])

    def _compute_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = X @ self.coef_
        return np.mean((y - y_pred) ** 2)

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        y_pred = X @ self.coef_
        error = y_pred - y
        return (2 / n) * (X.T @ error)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """使用梯度下降拟合模型"""
        # 添加截距
        X_design = self._add_intercept(X)
        n_samples, n_features = X_design.shape

        # 初始化系数为 0
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

        # 确定 batch 大小
        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type 必须是 'full_batch' 或 'mini_batch'")

        print(f"开始训练: {self.gd_type}, batch_size={batch_size}")

        # 梯度下降迭代
        for epoch in range(self.max_iter):
            # 1. 采样（如果是 mini_batch）
            if self.gd_type == "mini_batch":
                # 随机采样
                indices = np.random.choice(n_samples, size=batch_size, replace=False)
                X_batch = X_design[indices]
                y_batch = y[indices]
            else:
                X_batch = X_design
                y_batch = y

            # 2. 计算梯度
            gradient = self._compute_gradient(X_batch, y_batch)

            # 3. 更新回归系数
            self.coef_ -= self.learning_rate * gradient

            # 4. 记录本轮 loss（使用全量数据）
            current_loss = self._compute_mse(X_design, y)
            self.loss_history_.append(current_loss)

            # 5. 检查收敛
            if epoch > 50:
                rel_loss_change = abs(self.loss_history_[-1] - self.loss_history_[-2]) / (abs(self.loss_history_[-2]) + 1e-8)
                if rel_loss_change < self.tol:
                    print(f"收敛于第 {epoch} 轮，loss 相对变化 = {rel_loss_change:.2e}")
                    break

        print(f"训练完成: 共 {len(self.loss_history_)} 轮，最终 loss = {self.loss_history_[-1]:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("必须先调用 fit()")
        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        return 1 - SSE / SST