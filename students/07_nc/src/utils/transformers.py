"""从 Week10 到 Week13 持续维护的预处理工具。

Week13 在不删减 Week10/Week11 功能的基础上，让 CustomStandardScaler 兼容
sklearn Pipeline/GridSearchCV。这样正则化模型可以继续使用学生自己实现的标准化器。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # 从 Week13 开始，作业允许使用 sklearn 作为辅助依赖。
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:  # pragma: no cover - 兼容没有 sklearn 的环境
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
    """一个简化版、兼容 sklearn 风格的标准化器。

    它在 Week10 添加，并在 Week11-Week13 持续复用。Week13 中增加 ``y=None``，
    并继承 BaseEstimator/TransformerMixin，使它能放入 sklearn Pipeline，
    也能被 GridSearchCV 克隆。
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
    """只在训练数据上拟合的数值缺失值填补器，支持中位数/均值。"""

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
    """类别变量众数填补器，并提供安全的缺失类别标记。"""

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
    """按照训练集分位数对数值列进行缩尾裁剪。"""

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
    """简单独热编码器，会记住训练阶段见过的类别。"""

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
    """适用于混合类型表格回归数据的无泄露预处理组合。

    缺失值填补、缩尾、标准化和独热编码都只在训练折上拟合。
    验证集/测试集只能调用 transform()，不能重新 fit。
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
