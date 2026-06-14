import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin

# ====================== CustomOLS ======================
class CustomOLS:
    def __init__(self, fit_intercept=True, alpha=0.01):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        n, p = X.shape
        if self.fit_intercept:
            X = np.hstack([np.ones((n, 1)), X])
        
        if self.alpha > 0:
            # 岭回归正则化
            L = self.alpha * np.eye(X.shape[1])
            beta = np.linalg.pinv(X.T @ X + L) @ X.T @ y
        else:
            # 标准 OLS，使用伪逆
            beta = np.linalg.pinv(X) @ y
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

    def predict(self, X):
        if self.fit_intercept:
            return self.intercept_ + X @ self.coef_
        return X @ self.coef_


# ====================== 前向选择 ======================
class ForwardSelector:
    def __init__(self, cv=5, scoring="neg_root_mean_squared_error"):
        self.cv = cv
        self.scoring = scoring
        self.best_features_ = []
        self.best_scores_ = []

    def fit(self, X, y):
        n_features = X.shape[1]
        remaining = list(range(n_features))

        while remaining:
            best_score = -np.inf
            best_f = None

            for f in remaining:
                current = self.best_features_ + [f]
                
                X_sub = X[:, current].reshape(X.shape[0], -1)
                
                model = LinearRegression()
                score = cross_val_score(model, X_sub, y, cv=self.cv, scoring=self.scoring).mean()

                if score > best_score:
                    best_score = score
                    best_f = f

            if best_f is not None:
                self.best_features_.append(best_f)
                self.best_scores_.append(best_score)
                remaining.remove(best_f)
            else:
                break
        return self

    def transform(self, X):
        return X[:, self.best_features_].reshape(X.shape[0], -1)


# ====================== PCR 类 ======================
class PCR(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int, scaler):
        self.n_components = n_components
        self.scaler = scaler
        self.pca = PCA(n_components=n_components)
        self.lr = LinearRegression()
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        Z = self.pca.fit_transform(X_scaled)
        self.lr.fit(Z, y)
        self.intercept_ = self.lr.intercept_
        self.coef_ = self.lr.coef_
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        Z = self.pca.transform(X_scaled)
        return self.lr.predict(Z)

    def get_cum_variance(self):
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


# ====================== LassoCV 封装 ======================
def train_lasso_cv(X_train, y_train, cv=5):
    lasso = LassoCV(cv=cv, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    return lasso
