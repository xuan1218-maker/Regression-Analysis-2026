import numpy as np

class AnalyticalSolver:
    """
    解析解求解器：使用正规方程求解线性回归
    不使用 np.linalg.inv，而是使用更稳定的 np.linalg.solve
    """
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练：计算权重 β
        X: (n_samples, n_features)
        y: (n_samples,)
        返回: β (n_features,)
        """
        # 正规方程：X.T @ X @ β = X.T @ y
        A = X.T @ X
        b = X.T @ y
        
        # 核心：用 solve 而不是逆矩阵（数值更稳定）
        beta = np.linalg.solve(A, b)
        self.beta = beta
        return beta

    def predict(self, X: np.ndarray):
        return X @ self.beta


class GradientDescentSolver:
    """
    全批量梯度下降求解器
    手动推导梯度，纯 Numpy 实现
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        # 初始化权重为 0
        self.beta = np.zeros(n_features)

        # 全批量梯度下降迭代
        for _ in range(self.epochs):
            # 1. 预测值
            y_pred = X @ self.beta
            
            # 2. 手动推导的梯度公式（核心）
            # ∇L(β) = (2 / n_samples) * X.T @ (Xβ - y)
            gradient = (2 / n_samples) * X.T @ (y_pred - y)
            
            # 3. 更新权重
            self.beta -= self.lr * gradient

        return self.beta

    def predict(self, X: np.ndarray):
        return X @ self.beta

if __name__ == "__main__":
    # 构造测试数据
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 3 * X[:,0] + 1.5 * X[:,1] + 0.5 * X[:,2]

    # 1. 测试解析解
    ana = AnalyticalSolver()
    beta_ana = ana.fit(X, y)
    print("解析解权重：", beta_ana.round(4))

    # 2. 测试梯度下降
    gd = GradientDescentSolver(learning_rate=0.1, epochs=5000)
    beta_gd = gd.fit(X, y)
    print("梯度下降权重：", beta_gd.round(4))