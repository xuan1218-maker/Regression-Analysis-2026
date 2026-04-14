"""
架构说明：表现层与验证层。将数值转化为学术洞察。
"""

import numpy as np
import matplotlib.pyplot as plt


def verify_covariance_matrix(X: np.ndarray, beta_samples: np.ndarray, sigma: float):
    """
    任务：将 1000 次模拟产生的"经验协方差"与公式推导出的"理论协方差"进行对齐。
    在控制台打印这两个矩阵。
    """
    # 1. 计算理论协方差矩阵：σ² (X^T X)^{-1}
    XTX = X.T @ X
    theoretical_cov = sigma**2 * np.linalg.inv(XTX)

    # 2. 计算经验协方差矩阵：基于1000次模拟的样本协方差
    # beta_samples shape: (n_simulations, 2)，转置后np.cov按行计算变量
    empirical_cov = np.cov(beta_samples.T)

    # 格式化打印两个矩阵
    print("=" * 70)
    print("理论协方差矩阵 (Theoretical Covariance Matrix):")
    print(np.array2string(theoretical_cov, precision=4, suppress_small=True))
    print("\n经验协方差矩阵 (Empirical Covariance Matrix):")
    print(np.array2string(empirical_cov, precision=4, suppress_small=True))
    print("=" * 70)

    return theoretical_cov, empirical_cov


def plot_covariance_ellipses(
    beta_samples_ortho: np.ndarray, beta_samples_coll: np.ndarray, true_beta: np.ndarray
):
    """
    任务：将正交特征 (rho=0) 和共线特征 (rho=0.99) 的估计结果画在同一张 2D 散点图上。
    """
    # 设置绘图风格
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # 绘制实验A（正交特征，ρ=0.0）：蓝色，透明度0.6
    ax.scatter(
        beta_samples_ortho[:, 0],
        beta_samples_ortho[:, 1],
        color="#1f77b4",
        alpha=0.6,
        label="Experiment A (Orthogonal Features, ρ=0.0)",
        s=15,
    )

    # 绘制实验B（高度共线，ρ=0.99）：红色，透明度0.6
    ax.scatter(
        beta_samples_coll[:, 0],
        beta_samples_coll[:, 1],
        color="#ff4b5c",
        alpha=0.6,
        label="Experiment B (Highly Collinear, ρ=0.99)",
        s=15,
    )

    # 标记真实参数β的靶心位置（黑色星标）
    ax.scatter(
        true_beta[0],
        true_beta[1],
        color="black",
        marker="*",
        s=200,
        label=f"True Parameters β = [{true_beta[0]}, {true_beta[1]}]",
        zorder=5,
    )

    # 坐标轴与标题设置
    ax.set_xlabel(r"$\hat{\beta}_1$", fontsize=12)
    ax.set_ylabel(r"$\hat{\beta}_2$", fontsize=12)
    ax.set_title(
        "Impact of Multicollinearity on Parameter Estimation Variance: Orthogonal vs Collinear",
        fontsize=14,
        pad=15,
    )
    ax.legend(fontsize=10)

    # 强制坐标轴等比例，保证椭圆形状真实反映协方差结构
    ax.axis("equal")

    # 保存图片
    plt.savefig("covariance_ellipses.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
