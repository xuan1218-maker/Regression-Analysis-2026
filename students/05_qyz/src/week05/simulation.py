"""
架构说明：这是蒙特卡洛模拟的心脏模块。
核心难点：如何在 1000 次甚至 100000 次循环中，将算法的时间复杂度降到最低？
"""

import numpy as np
from data_generator import generate_dynamic_response


def run_monte_carlo(
    X: np.ndarray,
    true_beta: np.ndarray,
    sigma: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:

    # --- 循环外预计算（仅执行1次，不随循环变化）---
    XTX = X.T @ X  # X^T X
    XT = X.T  # X^T
    # 预计算最小二乘投影矩阵：(X^T X)^{-1} X^T，用solve保证数值稳定性
    proj_matrix = np.linalg.solve(XTX, XT)

    beta_samples = []

    for _ in range(n_simulations):
        # 1. 生成带随机噪音的响应变量y
        y = generate_dynamic_response(X, true_beta, sigma, rng)

        # 2. 利用预计算的投影矩阵，快速计算beta_hat = proj_matrix @ y
        beta_hat = proj_matrix @ y

        # 3. 保存本次估计结果
        beta_samples.append(beta_hat)

    return np.array(beta_samples)
