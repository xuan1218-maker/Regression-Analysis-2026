"""
Module: utils.diagnostics
Purpose: Statistical diagnostics for regression models.
"""
import numpy as np
import pandas as pd

from .models import AnalyticalOLS


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add a leading intercept column to a numeric design matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    return np.column_stack([np.ones(X.shape[0]), X])


def calculate_vif(X: np.ndarray, feature_names: list[str] | None = None) -> list:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in X.

    VIF_j = 1 / (1 - R_j^2), where R_j^2 is the R-squared from regressing
    feature j on all other features.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        List of VIF values for each feature
    """
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        # Features excluding j
        X_others = np.delete(X, j, axis=1)
        y_j = X[:, j]

        # Fit OLS on other features to predict j
        ols = AnalyticalOLS()
        try:
            ols.fit(X_others, y_j)
            # R-squared
            r_squared = ols.score(X_others, y_j)
            # VIF
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except np.linalg.LinAlgError:
            vif = np.inf
        vif_values.append(vif)

    return vif_values


def correlation_pairs(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Return absolute correlations above a threshold for numeric columns."""
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    rows: list[dict[str, float | str]] = []
    cols = list(corr.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_1": col_a, "feature_2": col_b, "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Basic residual diagnostics for reports."""
    residuals = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_p95_abs": float(np.quantile(np.abs(residuals), 0.95)),
    }