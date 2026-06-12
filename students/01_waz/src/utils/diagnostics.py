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


# ---------------------------------------------------------------------------
# Week 14: matrix diagnostics for high-dimensional regression
# ---------------------------------------------------------------------------

def matrix_rank(X: np.ndarray, tol: float = 1e-10) -> int:
    """Numerical rank of a matrix via SVD singular-value threshold."""
    X_arr = np.asarray(X, dtype=float)
    S = np.linalg.svd(X_arr, compute_uv=False)
    return int(np.sum(S > tol * S[0]))


def condition_number(X: np.ndarray) -> float:
    """Condition number κ(X) = σ_max / σ_min (via SVD)."""
    X_arr = np.asarray(X, dtype=float)
    S = np.linalg.svd(X_arr, compute_uv=False)
    S_pos = S[S > 1e-12]
    if len(S_pos) == 0:
        return np.inf
    return float(S_pos[0] / S_pos[-1])


def coefficient_std(coef_matrix: np.ndarray) -> np.ndarray:
    """Column-wise standard deviation of coefficient estimates across splits."""
    return np.asarray(np.std(coef_matrix, axis=0), dtype=float)