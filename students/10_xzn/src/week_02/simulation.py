"""
Week 2 Simulation: OLS Estimation & Inference
Author: 15_lxl
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression


# ==========================================
# 1. Generate Data
# ==========================================
def generate_data(n: int, beta_0: float, beta_1: float, rng):
    X = rng.normal(0, 1, n)
    epsilon = rng.normal(0, 1, n)
    y = beta_0 + beta_1 * X + epsilon
    return X, y


# ==========================================
# 2. Manual OLS
# ==========================================
def manual_ols(X, y):
    X_mean = X.mean()
    y_mean = y.mean()

    beta_1_hat = ((X - X_mean) * (y - y_mean)).sum() / ((X - X_mean) ** 2).sum()
    beta_0_hat = y_mean - beta_1_hat * X_mean

    y_hat = beta_0_hat + beta_1_hat * X
    residuals = y - y_hat

    sigma2 = (residuals**2).sum() / (len(X) - 2)
    var_beta1 = sigma2 / ((X - X_mean) ** 2).sum()

    return beta_0_hat, beta_1_hat, var_beta1


# ==========================================
# 3. Estimate Once
# ==========================================
def estimate_once(X, y):
    b0, b1, var_b1 = manual_ols(X, y)

    # statsmodels（用于检验）
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()

    # sklearn
    model_sk = LinearRegression().fit(X.reshape(-1, 1), y)

    return {
        "manual_beta1": b1,
        "manual_var": var_b1,
        "sm_beta1": model_sm.params[1],
        "sm_pvalue": model_sm.pvalues[1],
        "sk_beta1": model_sk.coef_[0],
    }


# ==========================================
# 4. Loop
# ==========================================
def loop(n_sim=100, n=100):
    rng = np.random.default_rng(42)
    results = []

    for _ in range(n_sim):
        X, y = generate_data(n, 1, 2, rng)
        results.append(estimate_once(X, y))

    return pd.DataFrame(results)


# ==========================================
# 5. Analysis
# ==========================================
def analysis(df):
    print("\n### Bias 对比")
    print("Manual:", df["manual_beta1"].mean() - 2)
    print("Statsmodels:", df["sm_beta1"].mean() - 2)
    print("Sklearn:", df["sk_beta1"].mean() - 2)

    print("\n### 方差")
    print(df["manual_var"].mean())

    print("\n### 假设检验 (H0: beta1=0)")
    print("平均 p-value:", df["sm_pvalue"].mean())

    # ===== ANOVA =====
    X, y = generate_data(100, 1, 2, np.random.default_rng(0))
    df_tmp = pd.DataFrame({"x": X, "y": y})

    model = smf.ols("y ~ x", data=df_tmp).fit()

    print("\n### ANOVA")
    print(sm.stats.anova_lm(model, typ=2))

    print("\n### 回归总结")
    print(model.summary())