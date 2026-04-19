import numpy as np
from simulation import (
    run_simulation,
    compute_theoretical_cov_matrix,
    compute_empirical_cov_matrix,
)
from analysis import plot_beta_scatter


def main():
    np.random.seed(42)  # 保证可复现
    n_simulations = 1000
    n_samples = 100
    sigma = 2.0
    beta_true = np.array([5.0, 3.0])

    # 实验 A: 正交 (rho=0)
    print("=== 实验 A: rho = 0 (正交特征) ===")
    betas_orth, X_orth = run_simulation(
        rho=0.0,
        n_samples=n_samples,
        n_simulations=n_simulations,
        beta_true=beta_true,
        sigma=sigma,
    )
    emp_cov_orth = compute_empirical_cov_matrix(betas_orth)
    theo_cov_orth = compute_theoretical_cov_matrix(X_orth, sigma)
    print("经验协方差矩阵:\n", emp_cov_orth)
    print("理论协方差矩阵:\n", theo_cov_orth)
    print()

    # 实验 B: 高度共线 (rho=0.99)
    print("=== 实验 B: rho = 0.99 (高度共线) ===")
    betas_collin, X_collin = run_simulation(
        rho=0.99,
        n_samples=n_samples,
        n_simulations=n_simulations,
        beta_true=beta_true,
        sigma=sigma,
    )
    emp_cov_collin = compute_empirical_cov_matrix(betas_collin)
    theo_cov_collin = compute_theoretical_cov_matrix(X_collin, sigma)
    print("经验协方差矩阵:\n", emp_cov_collin)
    print("理论协方差矩阵:\n", theo_cov_collin)
    print()

    # 绘制对比散点图
    plot_beta_scatter(betas_orth, betas_collin, beta_true)


if __name__ == "__main__":
    main()
