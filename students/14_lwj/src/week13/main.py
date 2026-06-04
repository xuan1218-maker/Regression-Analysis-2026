# main.py —— 完全独立，无需 src，直接运行
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, GridSearchCV, KFold
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ---------------------- 自建指标（不依赖任何文件） ----------------------
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ---------------------- 路径与文件夹 ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------- A1：生成高度共线数据 ----------------------
def generate_correlated_data(n_samples=500, n_features=10, random_state=42):
    np.random.seed(random_state)
    # 3个高度相关特征
    base = np.random.randn(n_samples, 1)
    corr_features = np.hstack([base + 0.1 * np.random.randn(n_samples, 1) for _ in range(3)])
    # 其余独立特征
    other_features = np.random.randn(n_samples, n_features - 3)
    X = np.hstack([corr_features, other_features])
    # 真实系数：前4个有效，后面全0
    beta = np.array([3, 2, 1, 1.5] + [0] * (n_features - 4))
    y = X @ beta + np.random.randn(n_samples) * 0.5
    return X, y, beta

# ---------------------- A3.1：OLS vs Ridge 系数稳定性 ----------------------
def compare_coefficient_stability(X, y, corr_feature_indices=[0,1,2], n_splits=50, ridge_alpha=1.0):
    ols_coeffs = []
    ridge_coeffs = []
    for _ in range(n_splits):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=None)
        ols = LinearRegression().fit(X_train, y_train)
        ridge = Ridge(alpha=ridge_alpha).fit(X_train, y_train)
        ols_coeffs.append(ols.coef_[corr_feature_indices])
        ridge_coeffs.append(ridge.coef_[corr_feature_indices])
    return np.array(ols_coeffs), np.array(ridge_coeffs)

# ---------------------- A4：前向选择（手写，不依赖src） ----------------------
def forward_selection(X, y, k=5):
    selected = []
    remaining = list(range(X.shape[1]))
    while len(selected) < k and remaining:
        best_r2 = -np.inf
        best_feat = None
        for feat in remaining:
            current = selected + [feat]
            X_sub = X[:, current]
            model = LinearRegression().fit(X_sub, y)
            r2 = model.score(X_sub, y)
            if r2 > best_r2:
                best_r2 = r2
                best_feat = feat
        selected.append(best_feat)
        remaining.remove(best_feat)
    return selected

# ---------------------- 主程序 ----------------------
def main():
    print("[1] 生成数据...")
    X, y, true_beta = generate_correlated_data()
    df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]),
                      columns=[f"x{i}" for i in range(X.shape[1])] + ["y"])
    df.to_csv(os.path.join(DATA_DIR, "synthetic_correlated.csv"), index=False)
    print("✅ 数据保存完成")

    print("[2] OLS vs Ridge 稳定性...")
    ols_coeffs, ridge_coeffs = compare_coefficient_stability(X, y)
    plt.figure(figsize=(10,5))
    plt.boxplot([ols_coeffs.flatten(), ridge_coeffs.flatten()],
                labels=["OLS Coefficients", "Ridge Coefficients"])
    plt.title("Coefficient Stability: OLS vs Ridge")
    plt.ylabel("Coefficient Value")
    plt.savefig(os.path.join(RESULTS_DIR, "coef_stability.png"))
    plt.close()
    print("✅ 稳定性图保存完成")

    print("[3] Pipeline + 网格搜索...")
    alphas = np.logspace(-4, 3, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    pipe_lasso = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=5000))])
    pipe_elastic = Pipeline([("scaler", StandardScaler()), ("elastic", ElasticNet(max_iter=5000))])

    ridge_cv = GridSearchCV(pipe_ridge, {"ridge__alpha": alphas},
                             cv=kf, scoring="neg_mean_squared_error").fit(X,y)
    lasso_cv = GridSearchCV(pipe_lasso, {"lasso__alpha": alphas},
                             cv=kf, scoring="neg_mean_squared_error").fit(X,y)
    elastic_cv = GridSearchCV(pipe_elastic,
                               {"elastic__alpha": alphas, "elastic__l1_ratio": [0.2,0.5,0.8]},
                               cv=kf, scoring="neg_mean_squared_error").fit(X,y)

    print(f"Ridge best alpha: {ridge_cv.best_params_}")
    print(f"Lasso best alpha: {lasso_cv.best_params_}")

    # 画CV曲线
    plt.figure(figsize=(10,5))
    plt.plot(alphas, -ridge_cv.cv_results_["mean_test_score"], label="Ridge")
    plt.plot(alphas, -lasso_cv.cv_results_["mean_test_score"], label="Lasso")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("CV MSE")
    plt.legend()
    plt.title("CV Error vs Alpha")
    plt.savefig(os.path.join(RESULTS_DIR, "cv_error_curve.png"))
    plt.close()
    print("✅ CV曲线图保存完成")

    print("[4] 模型评估...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    best_ridge = ridge_cv.best_estimator_
    best_lasso = lasso_cv.best_estimator_
    best_elastic = elastic_cv.best_estimator_

    def score_model(model, name):
        y_pred = model.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)
        print(f"{name} RMSE: {rmse:.4f}")

    score_model(best_ridge, "Ridge")
    score_model(best_lasso, "Lasso")
    score_model(best_elastic, "ElasticNet")

    print("\n=== 系数对比（x0,x1,x2）===")
    print("Ridge:", best_ridge.named_steps["ridge"].coef_[:3])
    print("Lasso:", best_lasso.named_steps["lasso"].coef_[:3])
    print("ElasticNet:", best_elastic.named_steps["elastic"].coef_[:3])

    print("\n[5] 前向选择 vs Lasso 筛选特征...")
    selected_fwd = forward_selection(X, y, k=5)
    selected_lasso = np.where(best_lasso.named_steps["lasso"].coef_ != 0)[0]
    print("前向选择选出:", selected_fwd)
    print("Lasso 非零特征:", selected_lasso.tolist())

    print("\n🎉 全部跑完！结果在 ./results 里")

if __name__ == "__main__":
    main()