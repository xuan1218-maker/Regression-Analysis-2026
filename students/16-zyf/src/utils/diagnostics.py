import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_vif(X, feature_names):
    vif_results = []

    for i in range(X.shape[1]):

        # 当前特征
        y = X[:, i]

        # 剩余特征
        X_other = np.delete(X, i, axis=1)

        model = LinearRegression()
        model.fit(X_other, y)

        r2 = model.score(X_other, y)

        # 防止除0
        if r2 >= 0.9999:
            vif = np.inf
        else:
            vif = 1 / (1 - r2)

        vif_results.append((feature_names[i], vif))

    return vif_results