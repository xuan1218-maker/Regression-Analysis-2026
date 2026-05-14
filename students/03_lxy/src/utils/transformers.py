import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self
    
    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError('Scaler has not been fitted yet.')
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
