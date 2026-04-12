import numpy as np
import time

class AnalyticalSolver:
    """使用正规方程求解析解"""
    
    def __init__(self):
        self.beta = None
        self.fit_time = None
        
    def fit(self, X, y):
        """拟合模型，使用np.linalg.solve而非np.linalg.inv"""
        start_time = time.time()
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # 使用np.linalg.solve求解正规方程，数值更稳定
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y
        self.beta = np.linalg.solve(XTX, XTy)
        
        self.fit_time = time.time() - start_time
        return self
    
    def predict(self, X):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ self.beta


class GradientDescentSolver:
    """使用全批量梯度下降法迭代求解"""
    
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.beta = None
        self.fit_time = None
        
    def fit(self, X, y):
        """
        梯度公式推导：
        L(β) = (1/2N) * Σ(y_i - X_iβ)^2
        ∇L(β) = -(1/N) * X^T (y - Xβ)
        """
        start_time = time.time()
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        N, P = X_with_intercept.shape
        
        # 初始化参数
        self.beta = np.zeros(P)
        
        # 批量梯度下降迭代
        for epoch in range(self.n_epochs):
            # 计算预测值
            y_pred = X_with_intercept @ self.beta
            
            # 计算误差
            error = y_pred - y
            
            # 计算梯度
            gradient = (1/N) * (X_with_intercept.T @ error)
            
            # 更新参数
            self.beta -= self.learning_rate * gradient
        
        self.fit_time = time.time() - start_time
        return self
    
    def predict(self, X):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ self.beta