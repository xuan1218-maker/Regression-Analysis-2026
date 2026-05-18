import numpy as np

class CustomStandardScaler:
    """
    手写标准化器，遵循 fit/transform/fit_transform 接口
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """计算训练集的均值和标准差"""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 避免除以零
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用已保存的均值和标准差标准化"""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("必须先调用 fit() 或 fit_transform()")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        return self.fit(X).transform(X)