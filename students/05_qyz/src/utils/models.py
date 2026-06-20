"""
Module: utils.models
Purpose: Core machine learning estimators for regression.
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# =============================================================================
# 1. Analytical OLS（解析解）
# =============================================================================
class AnalyticalOLS:
    """Analytical Ordinary Least Squares (stable version)"""

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
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_design = self._add_intercept(X)
        self._X_design = X_design

        n, p = X_design.shape

        XtX = X_design.T @ X_design

        # ✅ stable inverse (fix numerical instability)
        XtX_inv = np.linalg.pinv(XtX)

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
            raise RuntimeError("Must call fit() first")
        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        return 1 - SSE / SST

    def f_test(self, C: np.ndarray, d: np.ndarray):
        if self.coef_ is None:
            raise RuntimeError("Must call fit() first")

        C = np.asarray(C)
        d = np.asarray(d).reshape(-1, 1)
        q = C.shape[0]

        diff = C @ self.coef_.reshape(-1, 1) - d

        XtX_inv = np.linalg.pinv(self._X_design.T @ self._X_design)

        middle = C @ XtX_inv @ C.T
        middle_inv = np.linalg.pinv(middle)

        f_stat = (diff.T @ middle_inv @ diff).item() / (q * self.sigma2_)
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)

        return {"f_stat": f_stat, "p_value": p_value}


# =============================================================================
# 2. Gradient Descent OLS
# =============================================================================
class GradientDescentOLS:

    def __init__(self, learning_rate=0.01, tol=1e-5, max_iter=1000,
                 gd_type="full_batch", batch_fraction=0.1,
                 fit_intercept=True):

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.loss_history_ = []

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.column_stack([np.ones(X.shape[0]), X])

    def _mse(self, X, y):
        return np.mean((y - X @ self.coef_) ** 2)

    def _grad(self, X, y):
        n = X.shape[0]
        return (2 / n) * (X.T @ (X @ self.coef_ - y))

    def fit(self, X, y):

        X = self._add_intercept(X)
        n, p = X.shape

        self.coef_ = np.zeros(p)
        self.loss_history_ = []

        batch_size = n if self.gd_type == "full_batch" else max(
            1, int(n * self.batch_fraction)
        )

        prev_loss = np.inf

        for i in range(self.max_iter):

            if self.gd_type == "mini_batch":
                idx = np.random.choice(n, batch_size, replace=False)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            grad = self._grad(Xb, yb)
            self.coef_ -= self.learning_rate * grad

            loss = self._mse(X, y)
            self.loss_history_.append(loss)

            if abs(prev_loss - loss) / (prev_loss + 1e-8) < self.tol:
                break
            prev_loss = loss

        return self

    def predict(self, X):
        return self._add_intercept(X) @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        return 1 - SSE / SST


# =============================================================================
# 3. Ridge Regression
# =============================================================================
class RidgeRegression:

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):

        Xd = self._add_intercept(X)
        XtX = Xd.T @ Xd

        # stable inverse
        XtX_inv = np.linalg.pinv(XtX + self.alpha * np.eye(Xd.shape[1]))

        beta = XtX_inv @ (Xd.T @ y)

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = beta

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


# =============================================================================
# 4. PCR Regression
# =============================================================================
class PCRRegressor:

    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = None
        self.model = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)

        self.std_[self.std_ < 1e-12] = 1.0

        Xs = (X - self.mean_) / self.std_

        self.pca = PCA(n_components=self.n_components)
        Z = self.pca.fit_transform(Xs)

        self.model = LinearRegression().fit(Z, y)

        return self

    def predict(self, X):

        Xs = (X - self.mean_) / self.std_
        Z = self.pca.transform(Xs)

        return self.model.predict(Z)

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    @property
    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_

    @property
    def coef_(self):
        return self.pca.components_.T @ self.model.coef_