"""Solver abstraction and pseudocode implementations.

Teaching intent:
- keep a unified fit/predict interface
- expose where analytical and iterative methods differ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class AnalyticalSolver:
    """Normal-equation solver pseudocode."""

    coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnalyticalSolver":
        # Pseudocode:
        # A = X.T @ X
        # b = X.T @ y
        # coef = solve(A, b)
        # self.coef_ = coef
        A = X.T @ X
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Call fit before predict.")
        return X @ self.coef_


@dataclass
class GradientDescentSolver:
    """Batch gradient descent pseudocode."""

    learning_rate: float
    max_iter: int
    tol: float
    coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientDescentSolver":
        n_samples, n_features = X.shape
        coef = np.zeros(n_features)

        # Pseudocode loop:
        # for step in range(max_iter):
        #   y_pred = X @ coef
        #   grad = (2 / n_samples) * X.T @ (y_pred - y)
        #   coef = coef - learning_rate * grad
        #   if norm(grad) < tol: break
        for _ in range(self.max_iter):
            y_pred = X @ coef
            grad = (2.0 / n_samples) * X.T @ (y_pred - y)
            coef = coef - self.learning_rate * grad
            if np.linalg.norm(grad) < self.tol:
                break

        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Call fit before predict.")
        return X @ self.coef_
