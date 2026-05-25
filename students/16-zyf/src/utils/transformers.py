import numpy as np


class CustomStandardScaler:

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):

        X = np.array(
            X,
            dtype=float
        )

        self.mean_ = np.nanmean(
            X,
            axis=0
        )

        self.std_ = np.nanstd(
            X,
            axis=0
        )

        self.std_ = np.where(
            self.std_ == 0,
            1,
            self.std_
        )

        return self

    def transform(self, X):

        X = np.array(
            X,
            dtype=float
        )

        return (
            X - self.mean_
        ) / self.std_

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)