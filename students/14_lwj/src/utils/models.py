import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # 移除全是 NaN 的行
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        # 如果数据无效，防止程序崩溃
        if X.shape[0] == 0 or X.shape[1] == 0:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0
            return

        # 安全求解回归系数
        try:
            X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            beta = np.linalg.inv(X_with_bias.T @ X_with_bias + 1e-6 * np.eye(X_with_bias.shape[1])) @ X_with_bias.T @ y
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        except:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


def forward_selection(X, y, k=5, cv=5):
    """前向选择：按交叉验证R2得分，选出Top k个特征"""
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))
    scores = []

    for _ in range(k):
        best_score = -np.inf
        best_feature = None
        for feature in remaining:
            current_features = selected + [feature]
            X_subset = X[:, current_features]
            model = LinearRegression()
            cv_score = cross_val_score(model, X_subset, y, cv=cv, scoring="r2").mean()
            if cv_score > best_score:
                best_score = cv_score
                best_feature = feature
        selected.append(best_feature)
        remaining.remove(best_feature)
        scores.append(best_score)
    return selected, scores