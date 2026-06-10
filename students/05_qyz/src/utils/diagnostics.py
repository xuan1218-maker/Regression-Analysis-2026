"""
妯″瀷璇婃柇宸ュ叿绠?(Model Diagnostics Toolbox)
鍔熻兘锛?
1. 璁＄畻 VIF锛堟柟宸啫鑳€鍥犲瓙锛夋娴嬪閲嶅叡绾挎€?
2. 褰╄壊缁堢杈撳嚭璀﹀憡
3. 娈嬪樊鍥俱€丵Q鍥俱€佺浉鍏崇煩闃电儹鍔涘浘
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.models import AnalyticalOLS

"""Model diagnostics utilities.

This module provides a few lightweight diagnostic helpers used in the
course exercises: VIF calculation, simple plotting helpers for residuals,
QQ-plots and a correlation matrix heatmap. All docstrings and output
are ASCII-only to avoid encoding issues across environments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.models import AnalyticalOLS
from pathlib import Path


def calculate_rank(X: np.ndarray) -> int:
    """Return matrix rank of X."""
    return np.linalg.matrix_rank(X)


def calculate_condition_number(X: np.ndarray) -> float:
    """Return condition number of X (2-norm)."""
    return np.linalg.cond(X)


def calculate_vif(X: np.ndarray) -> list:
    """Compute VIF values for each column in X.

    VIF_j = 1 / (1 - R_j^2) where R_j^2 is the R^2 of regressing column j on
    the remaining columns.
    """
    n_features = X.shape[1]
    vif_values = []
    for j in range(n_features):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)
        X_with_intercept = np.column_stack([np.ones(len(X_j)), X_j])
        try:
            model = AnalyticalOLS(fit_intercept=False)
            model.fit(X_with_intercept, y_j)
            y_pred = model.predict(X_with_intercept)
            sse = np.sum((y_j - y_pred) ** 2)
            sst = np.sum((y_j - np.mean(y_j)) ** 2)
            r_squared = 1 - sse / sst if sst != 0 else 0.0
            vif = 1.0 / (1.0 - r_squared) if r_squared < 0.999 else float("inf")
        except Exception:
            vif = float("inf")
        vif_values.append(vif)
    return vif_values


def calculate_vif_dataframe(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Return a DataFrame with VIF values for the selected columns."""
    X = df[feature_cols].values
    vifs = calculate_vif(X)
    return pd.DataFrame({"feature": feature_cols, "VIF": vifs})


def print_vif_warning(vif_results: pd.DataFrame, threshold: float = 10.0):
    """Print VIF summary to stdout using colored output for warnings.

    The returned value is True if any VIF exceeds the threshold.
    """
    print("\n" + "=" * 60)
    print("Multicollinearity check (VIF)")
    print("=" * 60)
    print(f"VIF threshold: {threshold} (values above indicate concern)")
    print("-" * 60)
    has_severe = False
    for _, row in vif_results.iterrows():
        feature = row["feature"]
        vif = row["VIF"]
        if vif > threshold:
            print(f"\033[91m  {feature}: VIF = {vif:.2f} (severe)\033[0m")
            has_severe = True
        elif vif > 5.0:
            print(f"\033[93m {feature}: VIF = {vif:.2f} (moderate)\033[0m")
        else:
            print(f"   {feature}: VIF = {vif:.2f}")
    if has_severe:
        print("\n" + "=" * 60)
        severe_features = vif_results[vif_results["VIF"] > threshold][
            "feature"
        ].tolist()
        print(f"\033[91m  Severe features: {severe_features}\033[0m")
        print(
            "\033[91m  Suggestion: consider removing highly correlated features or use regularization\033[0m"
        )
        print("=" * 60)
    else:
        print("\nNo severe multicollinearity detected")
    return has_severe


def plot_residuals(y_true, y_pred, save_name="residuals.png"):
    """Plot residuals vs predicted values and save figure to week13 results."""
    res = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, res, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    res_dir = Path(__file__).resolve().parents[1] / "week13" / "results" / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(res_dir / save_name, dpi=150)
    plt.close()


def plot_qq_residuals(y_true, y_pred, save_name="qq_plot.png"):
    """Plot a Q-Q plot of residuals and save figure to week13 results."""
    import scipy.stats as stats

    res = y_true - y_pred
    plt.figure(figsize=(8, 8))
    stats.probplot(res, plot=plt)
    plt.title("Residual Q-Q Plot")
    res_dir = Path(__file__).resolve().parents[1] / "week13" / "results" / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(res_dir / save_name, dpi=150)
    plt.close()


def plot_correlation_matrix(df, save_name="corr_matrix.png"):
    """Plot a correlation matrix heatmap for numeric columns."""
    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    res_dir = Path(__file__).resolve().parents[1] / "week13" / "results" / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(res_dir / save_name, dpi=150)
    plt.close()
    plt.title("Residuals vs Predicted Values")
