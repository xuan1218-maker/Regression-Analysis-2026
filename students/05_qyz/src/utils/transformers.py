"""
Transformer API (StandardScaler - Custom Implementation)

Fully sklearn-compatible design:
- fit
- transform
- fit_transform
- inverse_transform (added)
"""

import numpy as np


class CustomStandardScaler:
    """
    Standardization:
        z = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    # =========================
    # fit
    # =========================
    def fit(self, X):
        X = np.asarray(X)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        # avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        self._fitted = True
        return self

    # =========================
    # transform
    # =========================
    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    # =========================
    # fit_transform
    # =========================
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # =========================
    # inverse_transform（加分项）
    # =========================
    def inverse_transform(self, X_scaled):
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X_scaled = np.asarray(X_scaled)
        return X_scaled * self.std_ + self.mean_