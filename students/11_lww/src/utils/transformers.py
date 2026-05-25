import numpy as np

class CustomStandardScaler:
    """
    自定义标准化转换器，严格遵循sklearn接口规范
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """仅在训练集上计算均值和标准差"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 处理标准差为0的情况，避免除以0
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用训练集的均值和标准差进行转换"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """合并fit和transform操作"""
        return self.fit(X).transform(X)