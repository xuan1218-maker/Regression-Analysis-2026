"""
架构说明：本模块只负责"上帝视角"的数据生成。
核心要求：必须将"固定设计矩阵 (Fixed X)"与"动态噪音 (Epsilon)"严格分离！
"""

import numpy as np


def generate_fixed_design_matrix(
    n_samples: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """
    任务：生成两列特征 X1 和 X2。要求它们之间的相关系数为rho。
    提示：可以使用正态分布生成 Z1，Z2，然后通过线性组合构造所需的共线性
    """
    # 生成两个独立的标准正态变量
    z1 = rng.normal(loc=0, scale=1, size=n_samples)
    z2 = rng.normal(loc=0, scale=1, size=n_samples)

    # 构造相关变量：X1 = z1, X2 = rho*z1 + sqrt(1-rho²)*z2，保证相关系数严格为rho
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2

    # 构造设计矩阵（两列特征，无截距项，拟合时自动处理）
    X = np.column_stack([x1, x2])
    return X


def generate_dynamic_response(
    X: np.ndarray, true_beta: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """
    任务：基于传入的固定 X，生成一次带随机噪音的 y。
    注意：每次调用此函数，返回的 y 应该都不同，因为噪音不同。
    """
    # 线性模型：y = X @ true_beta + ε，ε ~ N(0, σ²)
    epsilon = rng.normal(loc=0, scale=sigma, size=X.shape[0])
    y = X @ true_beta + epsilon
    return y
