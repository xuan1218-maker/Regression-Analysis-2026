# -*- coding: utf-8 -*-
import numpy as np

class AnalyticalOLS:
    """解析解线性回归"""
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class GradientDescentOLS:
    """梯度下降线性回归"""
    def __init__(
        self,
        learning_rate=0.01,
        tol=1e-6,
        max_iter=1000,
        gd_type="full_batch",
        batch_fraction=0.2
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X, y, seed=42):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                batch_size = max(1, int(n_samples * self.batch_fraction))
                idx = rng.choice(n_samples, batch_size, replace=False)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            y_pred = Xb @ self.coef_
            error = y_pred - yb
            grad = (2 / len(Xb)) * (Xb.T @ error)
            self.coef_ -= self.learning_rate * grad

            mse = np.mean((X @ self.coef_ - y) ** 2)
            self.loss_history_.append(mse)

            if epoch > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0



# ====================== Week13 新增：基于交叉验证的前向特征选择 ======================
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def forward_selection_cv(X, y, max_features=4, cv_splits=5):
    """
    自主实现带5折CV的前向选择（满足 Week13 工程要求）
    """
    selected = []
    remaining = list(X.columns)
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    while remaining and len(selected) < max_features:
        best_cv_mse = float('inf')
        best_feat = None

        for feat in remaining:
            current_feats = selected + [feat]
            cv_errors = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx][current_feats], X.iloc[val_idx][current_feats]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = LinearRegression().fit(X_tr, y_tr)
                cv_errors.append(mean_squared_error(y_val, model.predict(X_val)))

            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv_mse:
                best_cv_mse = mean_cv
                best_feat = feat

        selected.append(best_feat)
        remaining.remove(best_feat)

    return selected