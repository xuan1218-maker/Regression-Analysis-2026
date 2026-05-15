import numpy as np


def calculate_vif(X: np.ndarray) -> list:
    """
    计算每一列特征的 VIF 值。

    VIF_j = 1 / (1 - R_j^2)
    """

    vif_values = []

    for j in range(X.shape[1]):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)

        # 添加截距项
        X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])

        try:
            beta = np.linalg.inv(X_others.T @ X_others) @ X_others.T @ y
            y_pred = X_others @ beta

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            r_squared = 1 - ss_res / ss_tot

            vif = 1 / (1 - r_squared)

        except np.linalg.LinAlgError:
            vif = float("inf")

        vif_values.append(vif)

    return vif_values