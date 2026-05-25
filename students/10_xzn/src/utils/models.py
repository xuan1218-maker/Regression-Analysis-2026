import numpy as np


class CustomOLS:
    """
    自定义普通最小二乘线性回归模型。

    Parameters
    ----------
    fit_intercept : bool, default=True
        是否在拟合时添加截距项（常数项列）。
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用正规方程拟合模型参数。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)，不包含常数项列。
        y : np.ndarray
            目标变量，形状为 (n_samples,)。
        """
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        XTX = X.T @ X
        XTy = X.T @ y
        self.beta = np.linalg.solve(XTX, XTy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合好的模型进行预测。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)，不包含常数项列。

        Returns
        -------
        np.ndarray
            预测值，形状为 (n_samples,)。
        """
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        return X @ self.beta


class GradientDescentOLS:
    """
    使用梯度下降求解的线性回归
    用于 Milestone 2 的交叉验证评估
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化梯度下降线性回归
        
        参数:
        - learning_rate: 学习率，控制每次更新的步长
        - n_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        使用梯度下降训练模型
        
        参数:
        - X: array-like, shape (n_samples, n_features)，特征矩阵
        - y: array-like, shape (n_samples,)，目标值
        
        返回:
        - self: 返回自身实例
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # 初始化参数为 0
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # 前向传播：计算预测值
            y_pred = X @ self.weights + self.bias
            
            # 计算损失（均方误差 MSE）
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # 计算梯度
            # dw = (2/n) * X^T * (y_pred - y)
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            # db = (2/n) * sum(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数:
        - X: array-like, shape (n_samples, n_features)，特征矩阵
        
        返回:
        - y_pred: array-like, shape (n_samples,)，预测值
        """
        X = np.array(X)
        return (X @ self.weights + self.bias).flatten()
    
    def get_loss_history(self):
        """返回训练过程中的损失历史记录"""
        return self.loss_history
