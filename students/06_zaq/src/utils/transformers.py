"""
Module: utils.transformers
Purpose: Custom transformers
"""
import math


class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        n = len(X)
        p = len(X[0]) if X else 1
        self.mean_ = [0.0]*p
        for j in range(p):
            s = sum(row[j] for row in X)
            self.mean_[j] = s / n
        
        self.std_ = [0.0]*p
        for j in range(p):
            s = sum((row[j] - self.mean_[j])**2 for row in X)
            self.std_[j] = math.sqrt(s / n) if s > 0 else 1
        return self
    
    def transform(self, X):
        return [[(row[j] - self.mean_[j]) / self.std_[j] for j in range(len(self.mean_))] for row in X]
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values_ = None
    
    def fit(self, X):
        p = len(X[0]) if X else 1
        self.fill_values_ = [0.0]*p
        for j in range(p):
            col = [row[j] for row in X if row[j] is not None]
            if self.strategy == 'mean':
                self.fill_values_[j] = sum(col) / len(col) if col else 0
            elif self.strategy == 'median':
                col_sorted = sorted(col)
                mid = len(col_sorted) // 2
                self.fill_values_[j] = col_sorted[mid] if col_sorted else 0
        return self
    
    def transform(self, X):
        return [[row[j] if row[j] is not None else self.fill_values_[j] for j in range(len(row))] for row in X]
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def add_intercept(X):
    return [[1.0] + row for row in X]
