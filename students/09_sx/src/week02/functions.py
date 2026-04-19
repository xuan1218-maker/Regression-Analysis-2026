import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def generate_data(n=100, beta0=1, beta1=2, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n)
    epsilon = np.random.normal(0, 1, n)
    Y = beta0 + beta1 * X + epsilon
    return X, Y


def formula_estimation(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    beta1_hat = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
    beta0_hat = Y_mean - beta1_hat * X_mean
    n = len(X)
    sigma2_hat = np.sum((Y - (beta0_hat + beta1_hat * X)) ** 2) / (n - 2)
    var_beta1 = sigma2_hat / np.sum((X - X_mean) ** 2)
    return beta0_hat, beta1_hat, var_beta1


def sklearn_estimation(X, Y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    return model.intercept_, model.coef_[0]


def statsmodels_estimation(X, Y):
    model = sm.OLS(Y, sm.add_constant(X)).fit()
    return model.params[0], model.params[1], model


def run():
    X, Y = generate_data()

    # 公式法
    b0_f, b1_f, var_b1 = formula_estimation(X, Y)
    print(f"公式法: β0={b0_f:.4f}, β1={b1_f:.4f}, Var(β1)={var_b1:.6f}")

    # sklearn
    b0_sk, b1_sk = sklearn_estimation(X, Y)
    print(f"sklearn: β0={b0_sk:.4f}, β1={b1_sk:.4f}")

    # statsmodels
    b0_sm, b1_sm, model = statsmodels_estimation(X, Y)
    print(f"statsmodels: β0={b0_sm:.4f}, β1={b1_sm:.4f}")

    # 偏差
    print(f"偏差: β0={b0_f - 1:.4f}, β1={b1_f - 2:.4f}")

    # 假设检验
    print(f"假设检验 p-value: {model.pvalues[1]:.6f}")

    # 方差分析
    print(
        f"R²={model.rsquared:.4f}, F={model.fvalue:.4f}, F p-value={model.f_pvalue:.6f}"
    )

    # 绘图
    plt.scatter(X, Y, alpha=0.6)
    x_line = np.linspace(X.min(), X.max(), 100)
    plt.plot(x_line, b0_f + b1_f * x_line, "r-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("一元线性回归")
    plt.savefig("regression_plot.png")
    plt.show()


if __name__ == "__main__":
    run()
