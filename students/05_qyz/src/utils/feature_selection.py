import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class ForwardSelector:
    """
    Forward Feature Selection (CV-based)

    Objective:
        Minimize CV RMSE step-by-step
    """

    def __init__(self, k_features=None, cv=5, tol=1e-6):
        self.k_features = k_features
        self.cv = cv
        self.tol = tol

        self.selected_features_ = []
        self.history_ = []

    def fit(self, X, y):

        remaining = list(range(X.shape[1]))
        selected = []

        if self.k_features is None:
            k_features = X.shape[1]
        else:
            k_features = self.k_features

        prev_score = np.inf

        while len(selected) < k_features and len(remaining) > 0:

            best_score = np.inf
            best_feature = None
            best_scores_all = None

            for f in remaining:

                trial = selected + [f]
                model = LinearRegression()

                scores = -cross_val_score(
                    model,
                    X[:, trial],
                    y,
                    scoring="neg_root_mean_squared_error",
                    cv=self.cv
                )

                mean_rmse = scores.mean()

                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_feature = f
                    best_scores_all = scores

            # early stopping
            if prev_score - best_score < self.tol:
                break

            selected.append(best_feature)
            remaining.remove(best_feature)

            self.history_.append({
                "step": len(selected),
                "selected": selected.copy(),
                "feature_added": best_feature,
                "rmse_mean": best_score,
                "rmse_std": best_scores_all.std()
            })

            prev_score = best_score

        self.selected_features_ = selected
        return self

    def transform(self, X):
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)