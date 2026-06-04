"""
模块：utils.transformers
用途：手写标准化转换器，遵循 fit/transform/fit_transform 接口
"""

import numpy as np


class CustomStandardScaler:
    """
    自定义标准化器：z = (x - mean) / std
    严格遵循 scikit-learn 的 Transformer API
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        计算训练集的均值和标准差
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 防止除零：如果标准差为 0，则设为 1（该特征不变）
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用已保存的均值和标准差进行标准化
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("必须先调用 fit 或 fit_transform")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        先拟合再转换
        """
        self.fit(X)
        return self.transform(X)