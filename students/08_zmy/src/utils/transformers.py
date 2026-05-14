import numpy as np

class CustomImputer:
    def __init__(self):
        self.means_ = None

    def fit(self, X: np.ndarray):
        self.means_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("必须先调用 fit()")
        X_filled = X.copy()
        for i, mean_val in enumerate(self.means_):
            mask = np.isnan(X_filled[:, i])
            if np.any(mask):
                X_filled[mask, i] = mean_val
        return X_filled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("必须先调用 fit()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)