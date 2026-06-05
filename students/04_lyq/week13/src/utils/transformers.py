import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y=None) -> "CustomStandardScaler":
        # 👉 关键：加一个 y=None，才能被 sklearn pipeline 调用
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X.astype(float), axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    # 👉 关键：fit_transform 必须支持 y=None
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class IQROutlierProcessor:
    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor
        self.lower_ = None
        self.upper_ = None

    def fit(self, X: np.ndarray, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_factor * iqr
        self.upper_ = q3 + self.iqr_factor * iqr

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower_, self.upper_)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
