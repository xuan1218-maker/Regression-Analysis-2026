import numpy as np
from data_generator import generate_design_matrix


def run_simulation(
    rho, n_samples=100, n_simulations=1000, beta_true=np.array([5.0, 3.0]), sigma=2.0
):
    """
    执行蒙特卡洛模拟

    参数:
        rho: 特征相关系数
        n_samples: 样本量
        n_simulations: 模拟次数
        beta_true: 真实系数
        sigma: 真实噪声标准差

    返回:
        betas: 形状 (n_simulations, 2) 的所有估计值
        X: 固定的设计矩阵
    """
    # 固定设计矩阵（只生成一次）
    X = generate_design_matrix(n_samples, rho)

    betas = []
    for _ in range(n_simulations):
        # 生成新的随机噪声
        epsilon = np.random.normal(0, sigma, size=n_samples)
        # 生成因变量 y = X * beta_true + epsilon
        y = X @ beta_true + epsilon
        # 最小二乘估计
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        betas.append(beta_hat)

    return np.array(betas), X


def compute_theoretical_cov_matrix(X, sigma=2.0):
    """
    根据公式 Var(beta_hat) = sigma^2 * (X^T X)^{-1} 计算理论协方差矩阵
    """
    XtX_inv = np.linalg.inv(X.T @ X)
    return sigma**2 * XtX_inv


def compute_empirical_cov_matrix(betas):
    """
    根据模拟得到的 beta_hat 样本计算经验协方差矩阵
    """
    return np.cov(betas.T)
