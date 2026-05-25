"""
Module: utils.models
Purpose: Regression models
"""
import math
import random


class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
    
    def _matrix_multiply(self, A, B):
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = [[0.0]*cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                s = 0.0
                for k in range(cols_A):
                    s += A[i][k] * B[k][j]
                result[i][j] = s
        return result
    
    def _matrix_inverse(self, M):
        n = len(M)
        aug = [[0.0]*(2*n) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                aug[i][j] = M[i][j]
            aug[i][n+i] = 1.0
        
        for i in range(n):
            pivot = aug[i][i]
            for j in range(2*n):
                aug[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2*n):
                        aug[k][j] -= factor * aug[i][j]
        
        return [[aug[i][n+j] for j in range(n)] for i in range(n)]
    
    def fit(self, X, y):
        n, p = len(X), len(X[0])
        XTX = [[0.0]*p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                s = 0.0
                for k in range(n):
                    s += X[k][i] * X[k][j]
                XTX[i][j] = s
        
        XTy = [0.0]*p
        for i in range(p):
            s = 0.0
            for k in range(n):
                s += X[k][i] * y[k]
            XTy[i] = s
        
        inv = self._matrix_inverse(XTX)
        
        self.coef_ = [0.0]*p
        for i in range(p):
            s = 0.0
            for j in range(p):
                s += inv[i][j] * XTy[j]
            self.coef_[i] = s
        
        y_pred = [sum(self.coef_[j] * X[i][j] for j in range(p)) for i in range(n)]
        ss_res = sum((y[i] - y_pred[i])**2 for i in range(n))
        self.sigma2_ = ss_res / (n - p)
        self.df_resid_ = n - p
        self.cov_matrix_ = [[inv[i][j] * self.sigma2_ for j in range(p)] for i in range(p)]
        return self
    
    def predict(self, X):
        return [sum(self.coef_[j] * row[j] for j in range(len(self.coef_))) for row in X]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y)/len(y)
        ss_res = sum((y[i] - y_pred[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i] - y_mean)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0
    

class GradientDescentOLS:
    def __init__(self, lr=0.01, tol=1e-5, max_iter=1000, gd_type="full_batch", batch_frac=0.2):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_frac = batch_frac
        self.coef_ = None
        self.loss_history_ = []
    
    def fit(self, X, y, seed=42):
        random.seed(seed)
        n, p = len(X), len(X[0])
        self.coef_ = [0.0]*p
        self.loss_history_ = []
        
        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                batch_size = max(1, int(n * self.batch_frac))
                indices = random.sample(range(n), batch_size)
                X_batch = [X[i] for i in indices]
                y_batch = [y[i] for i in indices]
            else:
                X_batch, y_batch = X, y
            
            pred_batch = [sum(self.coef_[j] * row[j] for j in range(p)) for row in X_batch]
            grad = [0.0]*p
            for j in range(p):
                s = 0.0
                for i in range(len(X_batch)):
                    s += (pred_batch[i] - y_batch[i]) * X_batch[i][j]
                grad[j] = 2 * s / len(X_batch)
            
            for j in range(p):
                self.coef_[j] -= self.lr * grad[j]
            
            pred_all = [sum(self.coef_[j] * row[j] for j in range(p)) for row in X]
            mse = sum((pred_all[i] - y[i])**2 for i in range(n)) / n
            self.loss_history_.append(mse)
            
            if epoch > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
        return self
    
    def predict(self, X):
        return [sum(self.coef_[j] * row[j] for j in range(len(self.coef_))) for row in X]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y)/len(y)
        ss_res = sum((y[i] - y_pred[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i] - y_mean)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0
