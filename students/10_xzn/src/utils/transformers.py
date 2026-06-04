import numpy as np


class CustomImputer:
    """
    自定义缺失值填补器
    使用列均值填补缺失值，遵循 fit / transform / fit_transform 接口
    """

    def __init__(self):
        self.means_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.means_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X_filled = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = np.isnan(col)
            X_filled[mask, j] = self.means_[j]
        return X_filled

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CustomStandardScaler:
    """
    自定义标准化器
    公式: X_scaled = (X - mean) / std
    遵循大厂 Transformer 接口规范
    """
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """
        计算并保存每列的均值和标准差
        参数 X: array-like, shape (n_samples, n_features)
        返回 self
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        return self
    
    def transform(self, X):
        """
        使用保存的均值和标准差标准化数据
        参数 X: array-like, shape (n_samples, n_features)
        返回标准化后的数据
        """
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """
        先 fit 再 transform（一步完成）
        参数 X: array-like, shape (n_samples, n_features)
        返回标准化后的数据
        """
        return self.fit(X).transform(X)