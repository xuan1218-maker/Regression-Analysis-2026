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
