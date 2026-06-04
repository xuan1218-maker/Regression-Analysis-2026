"""Small regression models and variable-selection tools.

This file preserves earlier OLS utilities and adds Week 13's custom
cross-validation based forward selection.  Ridge/Lasso/ElasticNet themselves
are intentionally called from sklearn in the Week 13 script because the
assignment explicitly permits them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from utils.metrics import calculate_rmse


class AnalyticalOLS:
    """Ordinary Least Squares solved by the normal equation.

    The class does not add an intercept automatically. Add a column of ones
    before fit/predict if an intercept is needed.
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnalyticalOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if np.isclose(sst, 0.0):
            return 0.0
        return float(1.0 - sse / sst)


class GradientDescentOLS:
    """OLS solved by gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-8,
        max_iter: int = 10000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.25,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if gd_type not in {"full_batch", "mini_batch"}:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")
        if not 0 < batch_fraction <= 1:
            raise ValueError("batch_fraction must be in (0, 1]")

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "GradientDescentOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n_samples, n_features = X.shape
        batch_size = n_samples
        if self.gd_type == "mini_batch":
            batch_size = max(1, int(round(n_samples * self.batch_fraction)))

        self.coef_ = np.zeros(n_features, dtype=float)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)
        previous_loss = np.inf

        for _ in range(self.max_iter):
            if self.gd_type == "mini_batch":
                idx = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
            else:
                X_batch = X
                y_batch = y

            error_batch = X_batch @ self.coef_ - y_batch
            gradient = (2.0 / X_batch.shape[0]) * (X_batch.T @ error_batch)
            self.coef_ -= self.learning_rate * gradient

            full_error = X @ self.coef_ - y
            loss = float(np.mean(full_error**2))
            self.loss_history_.append(loss)
            if abs(previous_loss - loss) < self.tol:
                break
            previous_loss = loss

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if np.isclose(sst, 0.0):
            return 0.0
        return float(1.0 - sse / sst)


CustomOLS = AnalyticalOLS


class OrdinaryLeastSquares:
    """Convenience OLS regressor that adds an intercept internally."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinaryLeastSquares":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        self.coef_ = np.linalg.lstsq(X_design, y_arr, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        return X_design @ self.coef_


LinearRegressionWithIntercept = OrdinaryLeastSquares


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
