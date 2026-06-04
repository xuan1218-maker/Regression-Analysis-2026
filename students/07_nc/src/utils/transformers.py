"""Reusable preprocessing utilities accumulated from Weeks 10-13.

The Week 13 additions keep the Week 10/11 functionality intact while making
CustomStandardScaler compatible with sklearn Pipeline/GridSearchCV.  This lets
regularized models use the student's own scaler instead of sklearn's scaler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # sklearn is an allowed dependency from Week 13 onward.
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:  # pragma: no cover - fallback for environments without sklearn
    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep: bool = True) -> dict:
            return self.__dict__.copy()

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class TransformerMixin:  # type: ignore[no-redef]
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y=y, **fit_params).transform(X)


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """A minimal sklearn-compatible standard scaler.

    Added in Week 10, reused in Weeks 11-13.  In Week 13 it intentionally
    accepts ``y=None`` and inherits BaseEstimator/TransformerMixin so that it
    can be placed inside sklearn Pipeline and cloned by GridSearchCV.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = None) -> "CustomStandardScaler":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D numeric array")
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0)
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CustomStandardScaler must be fitted before transform")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)


class CustomNumericImputer:
    """Median/mean imputer fitted on training data only."""

    def __init__(self, strategy: str = "median") -> None:
        if strategy not in {"median", "mean"}:
            raise ValueError("strategy must be 'median' or 'mean'")
        self.strategy = strategy
        self.statistics_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "CustomNumericImputer":
        numeric = X.apply(pd.to_numeric, errors="coerce")
        if self.strategy == "median":
            stats = numeric.median()
        else:
            stats = numeric.mean()
        self.statistics_ = stats.fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.statistics_ is None:
            raise RuntimeError("CustomNumericImputer must be fitted before transform")
        numeric = X.apply(pd.to_numeric, errors="coerce")
        return numeric.fillna(self.statistics_)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(X, y=y).transform(X)


class CustomCategoricalImputer:
    """Most-frequent categorical imputer with a safe missing token."""

    def __init__(self, missing_token: str = "Missing") -> None:
        self.missing_token = missing_token
        self.statistics_: dict[str, str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "CustomCategoricalImputer":
        stats: dict[str, str] = {}
        for col in X.columns:
            values = X[col].astype("object").where(X[col].notna(), np.nan)
            modes = values.dropna().mode()
            stats[col] = str(modes.iloc[0]) if not modes.empty else self.missing_token
        self.statistics_ = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.statistics_ is None:
            raise RuntimeError("CustomCategoricalImputer must be fitted before transform")
        out = X.copy()
        for col, fill_value in self.statistics_.items():
            out[col] = out[col].astype("object").where(out[col].notna(), fill_value).astype(str)
        return out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(X, y=y).transform(X)


class CustomWinsorizer:
    """Clip numeric columns at training-set quantiles."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "CustomWinsorizer":
        numeric = X.apply(pd.to_numeric, errors="coerce")
        self.lower_ = numeric.quantile(self.lower_quantile)
        self.upper_ = numeric.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("CustomWinsorizer must be fitted before transform")
        numeric = X.apply(pd.to_numeric, errors="coerce")
        return numeric.clip(lower=self.lower_, upper=self.upper_, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(X, y=y).transform(X)


class CustomOneHotEncoder:
    """Simple one-hot encoder with train-time category memory."""

    def __init__(self, drop_first: bool = True, handle_unknown: str = "ignore") -> None:
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories_: dict[str, list[str]] | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "CustomOneHotEncoder":
        categories: dict[str, list[str]] = {}
        feature_names: list[str] = []
        for col in X.columns:
            unique_values = sorted(pd.Series(X[col].astype(str).unique()).dropna().tolist())
            categories[col] = unique_values
            used_values = unique_values[1:] if self.drop_first and len(unique_values) > 0 else unique_values
            feature_names.extend([f"{col}__{value}" for value in used_values])
        self.categories_ = categories
        self.feature_names_ = feature_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.categories_ is None or self.feature_names_ is None:
            raise RuntimeError("CustomOneHotEncoder must be fitted before transform")
        columns: list[np.ndarray] = []
        for col, categories in self.categories_.items():
            values = X[col].astype(str).to_numpy()
            seen = set(categories)
            if self.handle_unknown == "error" and not set(values).issubset(seen):
                unknown = sorted(set(values) - seen)
                raise ValueError(f"unknown categories in {col}: {unknown}")
            used_categories = categories[1:] if self.drop_first and len(categories) > 0 else categories
            for category in used_categories:
                columns.append((values == category).astype(float))
        if not columns:
            return np.empty((len(X), 0))
        return np.column_stack(columns)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)


class RegressionPreprocessor:
    """Leakage-safe preprocessing bundle for mixed tabular regression data.

    It fits imputation, winsorization, scaling and one-hot encoding on a
    training fold only. Validation/test data should call only transform().
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str] | None = None,
        impute_strategy: str = "median",
        winsor_limits: tuple[float, float] = (0.01, 0.99),
        drop_first: bool = True,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features or [])
        self.numeric_imputer = CustomNumericImputer(strategy=impute_strategy)
        self.categorical_imputer = CustomCategoricalImputer()
        self.winsorizer = CustomWinsorizer(*winsor_limits)
        self.scaler = CustomStandardScaler()
        self.encoder = CustomOneHotEncoder(drop_first=drop_first)
        self.feature_names_: list[str] | None = None

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None) -> "RegressionPreprocessor":
        num = df[self.numeric_features].copy()
        num_imputed = self.numeric_imputer.fit_transform(num)
        num_winsorized = self.winsorizer.fit_transform(num_imputed)
        self.scaler.fit(num_winsorized)

        cat_feature_names: list[str] = []
        if self.categorical_features:
            cat = df[self.categorical_features].copy()
            cat_imputed = self.categorical_imputer.fit_transform(cat)
            self.encoder.fit(cat_imputed)
            cat_feature_names = self.encoder.feature_names_ or []

        self.feature_names_ = self.numeric_features + cat_feature_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_names_ is None:
            raise RuntimeError("RegressionPreprocessor must be fitted before transform")
        num = df[self.numeric_features].copy()
        num_imputed = self.numeric_imputer.transform(num)
        num_winsorized = self.winsorizer.transform(num_imputed)
        num_scaled = self.scaler.transform(num_winsorized)

        if self.categorical_features:
            cat = df[self.categorical_features].copy()
            cat_imputed = self.categorical_imputer.transform(cat)
            cat_encoded = self.encoder.transform(cat_imputed)
            return np.column_stack([num_scaled, cat_encoded])
        return num_scaled

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(df, y=y).transform(df)
