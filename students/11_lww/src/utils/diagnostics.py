import numpy as np
from utils.models import AnalyticalOLS

def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子VIF，检测多重共线性
    VIF_j = 1 / (1 - R_j²)
    """
    n_samples, n_features = X.shape
    vif_values = []
    ols_model = AnalyticalOLS()

    for i in range(n_features):
        # 第i个特征作为目标变量，其余作为自变量
        y_target = X[:, i]
        X_features = np.delete(X, i, axis=1)
        # 添加截距项
        X_features = np.column_stack([np.ones(n_samples), X_features])
        
        # 拟合OLS模型计算R²
        ols_model.fit(X_features, y_target)
        r_squared = ols_model.score(X_features, y_target)
        
        # 计算VIF，处理R²=1的极端情况
        if r_squared >= 1.0:
            vif = float('inf')
        else:
            vif = 1.0 / (1.0 - r_squared)
        
        vif_values.append(round(vif, 4))
    
    return vif_values