import numpy as np
import scipy.stats as stats

class CustomOLS:
    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept
        self.coef_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.X_design_ = None
        
    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def fit(self, X, y):
        if self.add_intercept:
            self.X_design_ = self._add_intercept(X)
        else:
            self.X_design_ = X
        
        n, k = self.X_design_.shape
        self.df_resid_ = n - k
        
        XTX = self.X_design_.T @ self.X_design_
        XTy = self.X_design_.T @ y
        # 添加小正则项防止奇异
        self.coef_ = np.linalg.solve(XTX + 1e-10 * np.eye(k), XTy)
        
        y_pred = self.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        self.sigma2_ = rss / self.df_resid_ if self.df_resid_ > 0 else rss
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        if self.add_intercept:
            X = self._add_intercept(X)
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0
        return 1 - sse / sst
    
    def f_test(self, C, d):
        C = np.asarray(C)
        d = np.asarray(d).flatten()
        q = C.shape[0]
        diff = C @ self.coef_ - d
        XTX = self.X_design_.T @ self.X_design_
        XTX_inv = np.linalg.pinv(XTX)
        denom = C @ XTX_inv @ C.T
        f_stat = diff.T @ np.linalg.pinv(denom) @ diff / (q * self.sigma2_)
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {'f_stat': f_stat, 'p_value': p_value, 'q': q, 'df_resid': self.df_resid_}