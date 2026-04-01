import numpy as np
import time
from solvers import AnalyticalSolver, GradientDescentSolver

def generate_data(n_samples, n_features):
    """生成线性数据 X, y"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.0] * n_features)  # 真实权重全1
    y = X @ true_beta + np.random.randn(n_samples) * 0.5  # 加噪声
    return X, y

def mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def run_experiment(name, X, y):
    """运行一个实验：两种求解器 + 计时 + MSE"""
    print(f"\n==================== {name} ====================")

    # ---------- 解析解 ----------
    start = time.time()
    ana = AnalyticalSolver()
    ana.fit(X, y)
    y_pred_ana = ana.predict(X)
    time_ana = time.time() - start
    mse_ana = mse(y, y_pred_ana)

    print(f"[AnalyticalSolver]")
    print(f"  Time: {time_ana:.4f} s")
    print(f"  MSE:  {mse_ana:.4f}")

    # ---------- 梯度下降 ----------
    start = time.time()
    gd = GradientDescentSolver(learning_rate=0.1, epochs=3000)
    gd.fit(X, y)
    y_pred_gd = gd.predict(X)
    time_gd = time.time() - start
    mse_gd = mse(y, y_pred_gd)

    print(f"[GradientDescentSolver]")
    print(f"  Time: {time_gd:.4f} s")
    print(f"  MSE:  {mse_gd:.4f}")

    return {
        "analytical": {"time": time_ana, "mse": mse_ana},
        "gradient": {"time": time_gd, "mse": mse_gd}
    }

def main():
    # ==================== Task 2 实验 ====================
    # 实验A：低维 N=10000, P=10
    X_A, y_A = generate_data(n_samples=10000, n_features=10)
    run_experiment("Experiment A: Low-dim (N=10000, P=10)", X_A, y_A)

    # 实验B：高维 N=10000, P=2000
    X_B, y_B = generate_data(n_samples=10000, n_features=2000)
    run_experiment("Experiment B: High-dim (N=10000, P=2000)", X_B, y_B)

if __name__ == "__main__":
    main()


import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor    
def run_experiment_task3(name, X, y):
    print(f"\n====== {name} (Task3) ======")

    # 1. statsmodels OLS
    start = time.time()
    model_sm = sm.OLS(y, X)
    result_sm = model_sm.fit()
    t_sm = time.time() - start
    print(f"statsmodels.OLS   | Time: {t_sm:.4f}s")

    # 2. sklearn LinearRegression
    start = time.time()
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    t_lr = time.time() - start
    print(f"Sklearn LR        | Time: {t_lr:.4f}s")

    # 3. sklearn SGDRegressor
    start = time.time()
    sgd = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=3000, tol=None)
    sgd.fit(X, y)
    t_sgd = time.time() - start
    print(f"Sklearn SGD       | Time: {t_sgd:.4f}s")

def main():
    # ==================== Task 3 实验 ====================
    # 实验：高维 N=10000, P=2000
    X, y = generate_data(n_samples=10000, n_features=2000)
    run_experiment_task3(f"Statsmodels vs Sklearn", X, y)

if __name__ == "__main__":
    main()
