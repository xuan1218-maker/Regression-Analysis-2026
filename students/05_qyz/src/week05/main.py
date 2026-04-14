"""
架构说明：整个工程的调度中心。
所有超参数（如模拟次数、真实参数）都必须作为全局常量在这里定义
"""

import numpy as np

# 导入自定义模块
from data_generator import generate_fixed_design_matrix
from simulation import run_monte_carlo
from analysis import verify_covariance_matrix, plot_covariance_ellipses


def main():
    # --- 全局实验配置 (Configuration) ---
    N_SAMPLES = 100  # 样本量
    N_SIMULATIONS = 1000  # 蒙特卡洛模拟次数
    TRUE_BETA = np.array([5.0, 3.0])  # 真实参数β
    SIGMA = 2.0  # 噪音标准差σ
    RNG = np.random.default_rng(seed=2026)  # 固定随机种子，保证可复现

    # --- 实验 A: 正交特征 rho = 0.0 ---
    print(">>> 启动实验 A (正交特征)...")
    # 1. 生成固定的设计矩阵X（仅生成1次，后续模拟保持不变）
    X_ortho = generate_fixed_design_matrix(N_SAMPLES, rho=0.0, rng=RNG)
    # 2. 执行蒙特卡洛模拟，收集1000次β估计值
    beta_samples_ortho = run_monte_carlo(X_ortho, TRUE_BETA, SIGMA, N_SIMULATIONS, RNG)
    # 3. 验证理论与经验协方差矩阵对齐
    print("\n>>> 实验A 协方差矩阵验证：")
    verify_covariance_matrix(X_ortho, beta_samples_ortho, SIGMA)

    # --- 实验 B: 多重共线性 rho = 0.99 ---
    print("\n>>> 启动实验 B (共线特征)...")
    # 1. 生成高度共线的固定设计矩阵X
    X_coll = generate_fixed_design_matrix(N_SAMPLES, rho=0.99, rng=RNG)
    # 2. 执行蒙特卡洛模拟，收集1000次β估计值
    beta_samples_coll = run_monte_carlo(X_coll, TRUE_BETA, SIGMA, N_SIMULATIONS, RNG)
    # 3. 验证理论与经验协方差矩阵对齐
    print("\n>>> 实验B 协方差矩阵验证：")
    verify_covariance_matrix(X_coll, beta_samples_coll, SIGMA)

    # --- 可视化 ---
    print("\n>>> 绘制协方差矩阵的具象化散点图...")
    # 绘制对比散点图，保存为png
    plot_covariance_ellipses(beta_samples_ortho, beta_samples_coll, TRUE_BETA)
    print(">>> 图表已保存为 covariance_ellipses.png！")


if __name__ == "__main__":
    main()
