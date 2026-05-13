import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None  # 训练集均值
        self.std_ = None   # 训练集标准差

    def fit(self, X: np.ndarray) -> "CustomStandardScaler":
        """仅学习训练集的统计量，不修改数据"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 避免标准差为0
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用训练集的均值和标准差做标准化"""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """先fit再transform"""
        self.fit(X)
        return self.transform(X)
