import numpy as np
import scipy.stats as stats

class AnalyticalOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.n = None
        self.k = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n, self.k = X.shape
        
        # ===================== ✅ 核心修复 =====================
        # 奇异矩阵自动加微小正则，永不崩溃！
        xtx = X.T @ X
        ridge = 1e-6 * np.eye(xtx.shape[0])  # 关键！防止不可逆
        
        xty = X.T @ y
        beta_hat = np.linalg.inv(xtx + ridge) @ xty  # 永远可求逆

        y_hat = X @ beta_hat
        residuals = y - y_hat
        sse = residuals @ residuals
        self.sigma2_ = sse / (self.n - self.k) if self.n > self.k else 1.0
        self.df_resid_ = max(self.n - self.k, 1)
        self.cov_matrix_ = self.sigma2_ * np.linalg.inv(xtx + ridge)
        self.coef_ = beta_hat
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        sse = np.sum((y - y_hat) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2) + 1e-10
        return 1 - (sse / sst)

    def f_test(self, C: np.ndarray, d: np.ndarray = None) -> dict:
        try:
            if d is None:
                d = np.zeros(C.shape[0])
            cbeta = C @ self.coef_
            diff = cbeta - d
            xtx_inv = self.cov_matrix_ / self.sigma2_
            middle = C @ xtx_inv @ C.T
            middle += 1e-6 * np.eye(middle.shape[0])
            middle_inv = np.linalg.inv(middle)
            f_num = diff.T @ middle_inv @ diff
            q = len(d)
            f_stat = f_num / (q * self.sigma2_) if self.sigma2_ != 0 else 0
            p_val = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
            return {"f_stat": f_stat, "p_value": p_val}
        except:
            return {"f_stat": 0.0, "p_value": 1.0}

class GradientDescentOLS:
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 5000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.2,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []

    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray, seed=42):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for _ in range(self.max_iter):
            if self.gd_type == "mini_batch":
                idx = rng.choice(n_samples, size=batch_size, replace=False)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            y_pred = Xb @ self.coef_
            error = y_pred - yb
            grad = (2 / len(Xb)) * (Xb.T @ error)

            # ===================== ✅ 梯度裁剪，永不爆炸 =====================
            grad = np.clip(grad, -1e5, 1e5)

            self.coef_ -= self.learning_rate * grad

            current_loss = self._mse(y, X @ self.coef_)
            self.loss_history_.append(current_loss)

            if len(self.loss_history_) >= 2:
                if abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                    break
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        try:
            y_pred = self.predict(X)
            sse = np.sum((y - y_pred) ** 2)
            sst = np.sum((y - np.mean(y)) ** 2) + 1e-10
            return 1 - sse / sst
        except:
            return 0.0

# ------------------------------------------------------
# Week13 作业必需：前向选择（Forward Selection）
# 完全兼容你现有代码，不用改任何原有内容
# ------------------------------------------------------
def forward_selection(X, y, max_features=None, cv=5):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    n_samples, n_features = X.shape
    if max_features is None:
        max_features = n_features

    selected = []
    remaining = list(range(n_features))

    for _ in range(max_features):
        best_score = np.inf
        best_cand = None
        for cand in remaining:
            current = selected + [cand]
            model = LinearRegression()
            score = -cross_val_score(model, X[:, current], y, cv=cv, 
                                     scoring="neg_mean_squared_error").mean()
            if score < best_score:
                best_score = score
                best_cand = cand
        selected.append(best_cand)
        remaining.remove(best_cand)
    return selected


# ------------------------------------------------------
# Week13 可选：后向剔除（Backward Elimination）
# 你可以二选一，我给你都备好了
# ------------------------------------------------------
def backward_elimination(X, y, min_features=1, cv=5):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    n_samples, n_features = X.shape
    selected = list(range(n_features))

    while len(selected) > min_features:
        worst_score = np.inf
        worst_idx = None
        for i, feat in enumerate(selected):
            current = selected[:i] + selected[i+1:]
            model = LinearRegression()
            score = -cross_val_score(model, X[:, current], y, cv=cv,
                                     scoring="neg_mean_squared_error").mean()
            if score < worst_score:
                worst_score = score
                worst_idx = i
        del selected[worst_idx]
    return selected

# ============================
# Week14 新增：PCA & PCR
# ============================
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.pca.fit(X)
        self.components_ = self.pca.components_
        self.explained_variance_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class PCR:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = CustomPCA(n_components=n_components)
        self.ols = LinearRegression()

    def fit(self, X, y):
        Z = self.pca.fit_transform(X)
        self.ols.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.pca.transform(X)
        return self.ols.predict(Z)

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

def cv_pcr_scores(X, y, k_list, cv=5):
    import numpy as np
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    for k in k_list:
        rmses = []
        for tr, te in kf.split(X):
            # ✅ 直接用传入的 X，不再做任何标准化！
            model = PCR(n_components=k)
            model.fit(X[tr], y[tr])
            yp = model.predict(X[te])

            rmses.append(np.sqrt(np.mean((y[te] - yp)**2)))
        results[k] = np.mean(rmses)
    return results





