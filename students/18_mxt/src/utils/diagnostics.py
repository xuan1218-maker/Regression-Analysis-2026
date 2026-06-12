import numpy as np

def calculate_vif(X: np.ndarray) -> list:
    """
    手动计算 VIF 方差膨胀因子，兼容所有模型
    """
    vif_results = []
    n_features = X.shape[1]

    for i in range(n_features):
        # 构造 y = 第i列，X = 其他列
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)

        # 最小二乘计算 R²
        beta = np.linalg.inv(X_other.T @ X_other) @ X_other.T @ y
        y_hat = X_other @ beta

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # VIF 公式
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_results.append(round(vif, 2))

    return vif_results


# ====================== Week14: matrix and stability diagnostics ======================
def matrix_rank(X: np.ndarray, tol: float | None = None) -> int:
    """Return numerical rank of a design matrix."""
    X_arr = np.asarray(X, dtype=float)
    return int(np.linalg.matrix_rank(X_arr, tol=tol))


def condition_number(X: np.ndarray, eps: float = 1e-12) -> float:
    """Return a stable condition-number diagnostic based on singular values."""
    X_arr = np.asarray(X, dtype=float)
    singular_values = np.linalg.svd(X_arr, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] < eps:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def coefficient_stability(coef_matrix: np.ndarray) -> np.ndarray:
    """Column-wise coefficient standard deviation across repeated fits."""
    coef_arr = np.asarray(coef_matrix, dtype=float)
    return np.std(coef_arr, axis=0, ddof=1)

