import matplotlib.pyplot as plt


def plot_beta_scatter(betas_orth, betas_collin, beta_true):
    """
    绘制两种情况下 beta_hat 的散点图
    betas_orth: 正交情况 (rho=0) 下的估计值数组 (n_simulations, 2)
    betas_collin: 共线情况 (rho=0.99) 下的估计值数组
    beta_true: 真实 beta 值
    """
    plt.figure(figsize=(8, 6))

    # 正交情况（蓝色）
    plt.scatter(
        betas_orth[:, 0],
        betas_orth[:, 1],
        alpha=0.5,
        s=10,
        label=r"$\rho=0$ (Orthogonal)",
        color="blue",
    )

    # 共线情况（红色）
    plt.scatter(
        betas_collin[:, 0],
        betas_collin[:, 1],
        alpha=0.5,
        s=10,
        label=r"$\rho=0.99$ (Collinear)",
        color="red",
    )

    # 真实 beta 点
    plt.scatter(
        beta_true[0],
        beta_true[1],
        color="black",
        marker="x",
        s=200,
        linewidths=3,
        label="True $\\beta$",
    )

    plt.xlabel(r"$\hat{\beta}_1$")
    plt.ylabel(r"$\hat{\beta}_2$")
    plt.title("Monte Carlo Estimates of $\\beta$ under Different Multicollinearity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("./src/week05/covariance_scatter.png", dpi=150)
