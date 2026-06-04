import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin

class ForwardSelector(BaseEstimator, TransformerMixin):
    def __init__(self, significance_level=0.05, max_features=None):
        self.significance_level = significance_level
        self.max_features = max_features
        self.selected_features_ = None
        self.support_ = None

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        all_features = X.columns.tolist()
        selected = []
        remaining = all_features.copy()

        while remaining:
            if self.max_features and len(selected) >= self.max_features:
                break
            new_pvals = {}
            for candidate in remaining:
                features = selected + [candidate]
                model = sm.OLS(y, sm.add_constant(X[features])).fit()
                new_pvals[candidate] = model.pvalues[candidate]
            min_pval = min(new_pvals.values())
            if min_pval < self.significance_level:
                best_feature = min(new_pvals, key=new_pvals.get)
                selected.append(best_feature)
                remaining.remove(best_feature)
            else:
                break
        self.selected_features_ = selected
        self.support_ = [True if col in selected else False for col in all_features]
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("必须先调用 fit()")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X[self.selected_features_]

class BackwardSelector(BaseEstimator, TransformerMixin):
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.selected_features_ = None
        self.support_ = None

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        features = X.columns.tolist()
        while len(features) > 0:
            model = sm.OLS(y, sm.add_constant(X[features])).fit()
            pvalues = model.pvalues.iloc[1:]
            max_pval = pvalues.max()
            if max_pval >= self.significance_level:
                worst_feature = pvalues.idxmax()
                features.remove(worst_feature)
            else:
                break
        self.selected_features_ = features
        self.support_ = [True if col in features else False for col in X.columns]
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("必须先调用 fit()")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X[self.selected_features_]

class StepwiseSelector(BaseEstimator, TransformerMixin):
    def __init__(self, significance_entry=0.05, significance_removal=0.05):
        self.significance_entry = significance_entry
        self.significance_removal = significance_removal
        self.selected_features_ = None
        self.support_ = None

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        all_features = X.columns.tolist()
        selected = []
        changed = True
        while changed:
            changed = False
            # 前向添加
            remaining = [f for f in all_features if f not in selected]
            new_pvals = {}
            for candidate in remaining:
                features = selected + [candidate]
                model = sm.OLS(y, sm.add_constant(X[features])).fit()
                new_pvals[candidate] = model.pvalues[candidate]
            if new_pvals:
                min_pval = min(new_pvals.values())
                if min_pval < self.significance_entry:
                    best_feature = min(new_pvals, key=new_pvals.get)
                    selected.append(best_feature)
                    changed = True
                    continue
            # 后向剔除
            if len(selected) > 0:
                model = sm.OLS(y, sm.add_constant(X[selected])).fit()
                pvalues = model.pvalues.iloc[1:]
                max_pval = pvalues.max()
                if max_pval >= self.significance_removal:
                    worst_feature = pvalues.idxmax()
                    selected.remove(worst_feature)
                    changed = True
        self.selected_features_ = selected
        self.support_ = [True if col in selected else False for col in all_features]
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("必须先调用 fit()")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X[self.selected_features_]