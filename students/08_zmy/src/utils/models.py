import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from utils.transformers import CustomStandardScaler

class AnalyticalOLS:
    """解析解 OLS，支持截距选项"""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self._is_fitted = False

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y):
        X_design = self._add_intercept(X)
        self.coef_ = np.linalg.lstsq(X_design, y, rcond=None)[0]
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred)**2)
        sst = np.sum((y - np.mean(y))**2)
        return 1 - sse/sst if sst != 0 else 0.0


class GradientDescentOLS:
    """梯度下降 OLS，支持 full_batch 和 mini_batch，记录 loss 历史"""
    def __init__(self, learning_rate=0.01, tol=1e-5, max_iter=1000,
                 gd_type="full_batch", batch_fraction=0.1, fit_intercept=True):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.loss_history_ = []
        self._is_fitted = False

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y, seed=42):
        X_design = self._add_intercept(X)
        n, p = X_design.shape
        self.coef_ = np.zeros(p)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                idx = rng.choice(n, size=batch_size, replace=False)
                X_batch = X_design[idx]
                y_batch = y[idx]
            else:
                X_batch = X_design
                y_batch = y

            y_pred_batch = X_batch @ self.coef_
            error = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error)
            self.coef_ -= self.learning_rate * gradient

            # 全量 loss 用于监控
            y_pred_full = X_design @ self.coef_
            mse = np.mean((y - y_pred_full)**2)
            self.loss_history_.append(mse)

            if epoch > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred)**2)
        sst = np.sum((y - np.mean(y))**2)
        return 1 - sse/sst if sst != 0 else 0.0
    
class PCRRegressor(BaseEstimator, RegressorMixin):
    """主成分回归 (PCR) 封装，使用 CustomStandardScaler 标准化"""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.scaler_ = CustomStandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.pca_ = PCA(n_components=self.n_components)
        scores = self.pca_.fit_transform(X_scaled)
        self.reg_ = LinearRegression()
        self.reg_.fit(scores, y)
        # 将系数映射回原始尺度（用于解释）
        self.coef_ = (self.pca_.components_.T @ self.reg_.coef_) / self.scaler_.std_
        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        scores = self.pca_.transform(X_scaled)
        return self.reg_.predict(scores)

def repeated_ols_coefficients(X, y, n_repeats=60, test_size=0.3, random_seed=42):
    """重复切分 OLS 系数波动（用于稳定性分析）"""
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    records = []
    cond_numbers = []
    for split_seed in range(n_repeats):
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=split_seed
        )
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        singular_values = np.linalg.svd(X_train_scaled, full_matrices=False)[1]
        cond_numbers.append(singular_values[0] / singular_values[-1])
        for feature_idx, coef in enumerate(model.coef_):
            records.append({
                "split": split_seed,
                "feature": f"x{feature_idx + 1}",
                "coefficient": coef,
            })
    return pd.DataFrame(records), float(np.mean(cond_numbers))