import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where((self.std_ == 0) | np.isnan(self.std_), 1.0, self.std_)

        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("CustomStandardScaler must be fitted before transform().")

        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_