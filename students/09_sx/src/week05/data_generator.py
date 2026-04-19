import numpy as np


def generate_design_matrix(n_samples=100, rho=0.0):
    """
    生成固定设计矩阵 X，包含两个特征 X1 和 X2，相关系数为 rho。

    参数:
        n_samples: 样本数量
        rho: X1 与 X2 的相关系数，控制共线性程度

    返回:
        X: 形状为 (n_samples, 2) 的设计矩阵（已包含截距项？不，这里按题目只生成两个特征）
    """
    # 生成 X1 和 X2，使得相关系数约为 rho
    # 方法：先生成独立标准正态，再构造 X2 = rho * X1 + sqrt(1 - rho^2) * noise
    X1 = np.random.randn(n_samples)
    noise = np.random.randn(n_samples)
    X2 = rho * X1 + np.sqrt(1 - rho**2) * noise

    # 将 X1, X2 堆叠成 (n_samples, 2)
    X = np.column_stack((X1, X2))
    return X
