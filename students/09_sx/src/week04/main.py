import numpy as np
import time
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from solvers import AnalyticalSolver, GradientDescentSolver


def generate_data(N, P, random_seed=42):
    """生成测试数据"""
    np.random.seed(random_seed)
    X = np.random.randn(N, P)
    true_beta = np.random.randn(P + 1)
    X_with_intercept = np.column_stack([np.ones(N), X])
    y = X_with_intercept @ true_beta + 0.1 * np.random.randn(N)
    return X, y


def run_experiment(N, P, description):
    """运行实验"""
    print(f"\n{'='*60}")
    print(f"{description}: N={N}, P={P}")
    print(f"{'='*60}")
    
    # 生成数据
    X, y = generate_data(N, P)
    
    # 分割训练集和测试集
    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    results = {}
    
    # 1. 自定义解析求解器
    print("运行 AnalyticalSolver...")
    solver1 = AnalyticalSolver()
    solver1.fit(X_train, y_train)
    y_pred1 = solver1.predict(X_test)
    mse1 = mean_squared_error(y_test, y_pred1)
    results["AnalyticalSolver (Custom)"] = {"MSE": mse1, "Time": solver1.fit_time}
    
    # 2. 自定义梯度下降求解器
    print("运行 GradientDescentSolver...")
    solver2 = GradientDescentSolver(learning_rate=0.01, n_epochs=1000)
    solver2.fit(X_train, y_train)
    y_pred2 = solver2.predict(X_test)
    mse2 = mean_squared_error(y_test, y_pred2)
    results["GradientDescentSolver (Custom)"] = {"MSE": mse2, "Time": solver2.fit_time}
    
    # 3. statsmodels OLS
    print("运行 Statsmodels OLS...")
    start = time.time()
    X_train_sm = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_sm)
    res = model.fit()
    X_test_sm = sm.add_constant(X_test)
    y_pred3 = res.predict(X_test_sm)
    time3 = time.time() - start
    mse3 = mean_squared_error(y_test, y_pred3)
    results["Statsmodels OLS"] = {"MSE": mse3, "Time": time3}
    
    # 4. sklearn LinearRegression
    print("运行 Sklearn LinearRegression...")
    start = time.time()
    solver4 = LinearRegression()
    solver4.fit(X_train, y_train)
    time4 = time.time() - start
    y_pred4 = solver4.predict(X_test)
    mse4 = mean_squared_error(y_test, y_pred4)
    results["Sklearn LinearRegression"] = {"MSE": mse4, "Time": time4}
    
    # 5. sklearn SGDRegressor
    print("运行 Sklearn SGDRegressor...")
    # SGD需要特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    start = time.time()
    solver5 = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    solver5.fit(X_train_scaled, y_train)
    time5 = time.time() - start
    y_pred5 = solver5.predict(X_test_scaled)
    mse5 = mean_squared_error(y_test, y_pred5)
    results["Sklearn SGDRegressor"] = {"MSE": mse5, "Time": time5}
    
    return results


def main():
    # 实验A：低维场景
    results_low = run_experiment(N=10000, P=10, description="实验A (低维场景)")
    
    # 实验B：高维场景
    results_high = run_experiment(N=10000, P=2000, description="实验B (高维场景)")
    
    # 打印结果表格
    print("\n\n" + "="*80)
    print("耗时对比表格")
    print("="*80)
    
    print("\n【实验A：低维场景 (N=10000, P=10)】")
    print("-" * 55)
    print(f"{'求解器':<35} {'耗时(秒)':<15} {'MSE':<15}")
    print("-" * 55)
    for name, metrics in results_low.items():
        print(f"{name:<35} {metrics['Time']:<15.4f} {metrics['MSE']:<15.6f}")
    
    print("\n【实验B：高维场景 (N=10000, P=2000)】")
    print("-" * 55)
    print(f"{'求解器':<35} {'耗时(秒)':<15} {'MSE':<15}")
    print("-" * 55)
    for name, metrics in results_high.items():
        print(f"{name:<35} {metrics['Time']:<15.4f} {metrics['MSE']:<15.6f}")

if __name__ == "__main__":
    main()