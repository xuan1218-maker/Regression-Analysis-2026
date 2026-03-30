import numpy as np

def generate_data(n: int, beta_0: float, beta_1: float, rng):
    """生成线性回归模拟数据"""
    X = rng.normal(0, 1, n)
    epsilon = rng.normal(0, 1, n)
    y = beta_0 + beta_1 * X + epsilon
    return X, y