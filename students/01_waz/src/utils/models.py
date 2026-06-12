"""
Module: utils.models
Purpose: Core machine learning estimators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from utils.metrics import calculate_rmse


class CustomOLS:
    """自定义OLS回归器 - 与AnalyticalOLS功能相同"""
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        try:
            self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]

        return self
    
    def predict(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            return X @ np.concatenate([[self.intercept_], self.coef_])
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sse / sst) if sst != 0 else 0.0

class AnalyticalOLS:
    """
    Analytical OLS regression model.
    Supports fit, predict, score, and f_test.
    """

    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.residuals_ = None
        self.fitted_values_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx)
        self.coef_ = xtx_inv @ X.T @ y

        self.fitted_values_ = X @ self.coef_
        self.residuals_ = y - self.fitted_values_

        rss = self.residuals_ @ self.residuals_
        self.df_resid_ = n - p
        self.sigma2_ = rss / self.df_resid_
        self.cov_matrix_ = self.sigma2_ * xtx_inv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        return 1 - (rss / tss)

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        q = C.shape[0]
        diff = C @ self.coef_ - d
        cov_c = C @ self.cov_matrix_ @ C.T
        cov_c_inv = np.linalg.inv(cov_c)
        f_stat = (diff.T @ cov_c_inv @ diff) / q
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "q": q,
            "df_resid": self.df_resid_,
        }


class GradientDescentOLS:
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []
        self.full_loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        self.full_loss_history_ = []

        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)

            self.coef_ -= self.learning_rate * gradient

            updated_pred_batch = X_batch @ self.coef_
            batch_mse = np.mean((y_batch - updated_pred_batch) ** 2)
            y_pred_full = X @ self.coef_
            full_mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(batch_mse)
            self.full_loss_history_.append(full_mse)

            if epoch > 0:
                delta = abs(self.full_loss_history_[-1] - self.full_loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst


# ---------------------------------------------------------------------------
# Week 13: Custom forward selection with cross-validation
# ---------------------------------------------------------------------------

@dataclass
class SelectionStep:
    """One row of the forward-selection search history."""

    step: int
    added_feature: str
    cv_rmse: float
    selected_features: list[str]


class ForwardSelectorCV:
    """Greedy forward variable selection evaluated by K-fold CV.

    At each step, the algorithm tests every remaining candidate feature, fits a
    LinearRegression model on the current selected set plus that candidate, and
    picks the feature with the lowest mean validation RMSE.  This is the custom
    Week 13 model-selection logic requested by the assignment.
    """

    def __init__(self, max_features: int = 5, cv: int = 5, random_state: int = 42) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if cv < 2:
            raise ValueError("cv must be at least 2")
        self.max_features = max_features
        self.cv = cv
        self.random_state = random_state
        self.selected_indices_: list[int] | None = None
        self.selected_features_: list[str] | None = None
        self.history_: list[SelectionStep] = []
        self.estimator_: LinearRegression | None = None

    def _cv_rmse(self, X: np.ndarray, y: np.ndarray, indices: list[int]) -> float:
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        rmses: list[float] = []
        for train_idx, val_idx in splitter.split(X):
            model = LinearRegression()
            model.fit(X[train_idx][:, indices], y[train_idx])
            pred = model.predict(X[val_idx][:, indices])
            rmses.append(calculate_rmse(y[val_idx], pred))
        return float(np.mean(rmses))

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "ForwardSelectorCV":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        if len(feature_names) != X_arr.shape[1]:
            raise ValueError("feature_names length must match X columns")

        selected: list[int] = []
        remaining = list(range(X_arr.shape[1]))
        self.history_ = []

        for step in range(1, min(self.max_features, X_arr.shape[1]) + 1):
            scores: list[tuple[float, int]] = []
            for candidate in remaining:
                score = self._cv_rmse(X_arr, y_arr, selected + [candidate])
                scores.append((score, candidate))
            best_score, best_feature_idx = min(scores, key=lambda pair: pair[0])
            selected.append(best_feature_idx)
            remaining.remove(best_feature_idx)
            self.history_.append(
                SelectionStep(
                    step=step,
                    added_feature=feature_names[best_feature_idx],
                    cv_rmse=best_score,
                    selected_features=[feature_names[i] for i in selected],
                )
            )

        self.selected_indices_ = selected
        self.selected_features_ = [feature_names[i] for i in selected]
        self.estimator_ = LinearRegression().fit(X_arr[:, selected], y_arr)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.selected_indices_ is None or self.estimator_ is None:
            raise RuntimeError("ForwardSelectorCV must be fitted before predict")
        X_arr = np.asarray(X, dtype=float)
        return self.estimator_.predict(X_arr[:, self.selected_indices_])

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "step": [row.step for row in self.history_],
                "added_feature": [row.added_feature for row in self.history_],
                "cv_rmse": [row.cv_rmse for row in self.history_],
                "selected_features": [", ".join(row.selected_features) for row in self.history_],
            }
        )