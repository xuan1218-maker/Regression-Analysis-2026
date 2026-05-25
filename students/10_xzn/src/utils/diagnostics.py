import numpy as np


def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子（VIF）。

    VIF_j = 1 / (1 - R_j^2)，其中 R_j^2 是将第 j 个特征
    对其他所有特征做 OLS 回归得到的决定系数。

    Parameters
    ----------
    X : np.ndarray
        特征矩阵，形状为 (n_samples, n_features)，不包含常数项列。

    Returns
    -------
    list
        每个特征对应的 VIF 值列表，长度等于 n_features。
    """
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        # 将第 j 列作为目标变量
        y_j = X[:, j]

        # 其余列作为自变量
        X_others = np.delete(X, j, axis=1)

        # 添加常数项
        X_b = np.c_[np.ones((X_others.shape[0], 1)), X_others]

        # OLS 回归: beta = (X^T X)^{-1} X^T y
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y_j
        beta = np.linalg.solve(XTX, XTy)

        # 预测值
        y_pred = X_b @ beta

        # 计算 R²
        ss_res = np.sum((y_j - y_pred) ** 2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
        r_squared = 1.0 - (float(ss_res) / float(ss_tot))

        # VIF = 1 / (1 - R²)，处理 R² = 1 的边界情况
        if r_squared >= 0.999999:
            vif = float('inf')
        else:
            vif = 1.0 / (1.0 - r_squared)

        vif_values.append(vif)

    return vif_values