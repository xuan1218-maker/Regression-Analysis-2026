import numpy as np

def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子 VIF = 1 / (1 - R²)
    使用 numpy.linalg.lstsq 进行回归，并先检查 NaN/inf
    """
    X = np.asarray(X, dtype=np.float64)
    if np.any(np.isnan(X)):
        raise ValueError("输入特征矩阵包含 NaN")
    if np.any(np.isinf(X)):
        raise ValueError("输入特征矩阵包含 inf")

    n_features = X.shape[1]
    if n_features < 2:
        raise ValueError("至少需要两个特征才能计算 VIF")

    vifs = []
    for i in range(n_features):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        # 添加截距列
        X_others = np.column_stack([np.ones(len(X_others)), X_others])
        # 使用最小二乘求解
        coef, residuals, rank, s = np.linalg.lstsq(X_others, y_i, rcond=None)
        y_pred = X_others @ coef
        sse = np.sum((y_i - y_pred) ** 2)
        sst = np.sum((y_i - np.mean(y_i)) ** 2)
        r2 = 1 - sse / sst if sst > 0 else 0.0
        # 防止数值误差导致 r2 略大于 1
        r2 = np.clip(r2, 0.0, 1.0 - 1e-10)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vifs.append(vif)
    return vifs