"""
转换器 API (Transformer Interface)
手写标准化器，遵循 sklearn Transformer 接口规范
"""

import numpy as np


class CustomStandardScaler:
    """
    自定义标准化缩放器 (Custom Standard Scaler)

    遵循 Transformer 接口规范：
    - fit(X): 计算均值和标准差
    - transform(X): 使用保存的统计量转换数据
    - fit_transform(X): 一步完成拟合和转换

    标准化公式: z = (x - μ) / σ
    其中 μ 是均值，σ 是标准差

    用途:
    - 将不同量纲的特征缩放到同一尺度
    - 梯度下降对特征尺度敏感，标准化是必要的预处理

    注意:
    - 只用训练集拟合（fit），再用同样的参数转换验证集和测试集
    - 防止数据泄露
    """

    def __init__(self):
        """
        初始化标准化器

        属性:
        -----------
        mean_ : ndarray or None
            训练集各特征的均值，fit 后才有值
        std_ : ndarray or None
            训练集各特征的标准差，fit 后才有值
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        拟合标准化器：计算训练集 X 的均值和标准差

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据，用于计算统计量

        Returns:
        --------
        self : CustomStandardScaler
            返回自身，支持链式调用

        Note:
        -----
        只使用训练集数据，不涉及验证集/测试集
        """
        X = np.array(X)

        # 计算每个特征的均值
        self.mean_ = np.mean(X, axis=0)

        # 计算每个特征的标准差
        self.std_ = np.std(X, axis=0)

        # 避免除以零：当标准差为 0 时（所有值相同），设为 1
        # 这样该特征会被缩放为 0（因为 X - mean = 0）
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

        return self

    def transform(self, X):
        """
        使用已拟合的均值和标准差转换数据

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            待转换的数据（训练集、验证集或测试集）

        Returns:
        --------
        X_scaled : ndarray, shape (n_samples, n_features)
            标准化后的数据

        Note:
        -----
        - 必须先调用 fit() 拟合，否则会出错
        - 验证集和测试集使用训练集的 mean_ 和 std_，不重新计算
        """
        X = np.array(X)

        # 标准化公式: (X - μ) / σ
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        一步完成拟合和转换

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据

        Returns:
        --------
        X_scaled : ndarray, shape (n_samples, n_features)
            标准化后的训练数据

        等价于: fit(X).transform(X)
        常用于训练集的一步处理
        """
        return self.fit(X).transform(X)
