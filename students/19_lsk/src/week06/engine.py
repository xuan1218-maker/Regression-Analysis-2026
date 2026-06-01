import numpy as np
import scipy.stats as stats

class CustomOLS:
    def __init__(self):
        self.coef_ = None          # β 系数
        self.cov_matrix_ = None    # 系数协方差矩阵
        self.sigma2_ = None        # 误差方差
        self.df_resid_ = None      # 残差自由度
        self.n = None              # 样本量
        self.k = None              # 参数个数

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 维度
        self.n, self.k = X.shape

        # β = (X'X)⁻¹ X'y
        xtx = X.T @ X
        xty = X.T @ y
        beta_hat = np.linalg.inv(xtx) @ xty

        # 残差 & 方差
        y_hat = X @ beta_hat
        residuals = y - y_hat
        sse = residuals @ residuals
        self.sigma2_ = sse / (self.n - self.k)
        self.df_resid_ = self.n - self.k

        # 系数协方差矩阵
        self.cov_matrix_ = self.sigma2_ * np.linalg.inv(xtx)
        self.coef_ = beta_hat

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        sse = np.sum((y - y_hat) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sse / sst)

    def f_test(self, C: np.ndarray, d: np.ndarray = None) -> dict:
        # 联合显著性检验：Cβ = d
        if d is None:
            d = np.zeros(C.shape[0])

        cbeta = C @ self.coef_
        diff = cbeta - d
        xtx_inv = self.cov_matrix_ / self.sigma2_
        middle = C @ xtx_inv @ C.T
        middle_inv = np.linalg.inv(middle)

        f_num = diff.T @ middle_inv @ diff
        q = len(d)
        f_stat = f_num / (q * self.sigma2_)

        # p值
        p_val = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {"f_stat": f_stat, "p_value": p_val}