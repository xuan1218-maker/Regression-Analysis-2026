import numpy as np
import matplotlib.pyplot as plt

def calculate_vif(X: np.ndarray) -> list:
    # 强制转浮点，修复类型错误
    X = X.astype(np.float64)
    
    n_features = X.shape[1]
    vif_values = []
    for i in range(n_features):
        y_temp = X[:, i]
        X_temp = np.delete(X, i, axis=1)

        # 安全求逆
        try:
            beta = np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ y_temp
        except np.linalg.LinAlgError:
            vif_values.append(np.inf)
            continue

        y_hat = X_temp @ beta
        sst = np.sum((y_temp - np.mean(y_temp)) ** 2)
        sse = np.sum((y_temp - y_hat) ** 2)
        
        if sst == 0:
            r2 = 1.0
        else:
            r2 = 1 - (sse / sst)

        if r2 >= 1.0:
            vif = np.inf
        else:
            vif = 1 / (1 - r2)
        vif_values.append(vif)
    return vif_values


def plot_correlation_matrix(X: np.ndarray, feature_names: list, save_path: str = "results/correlation_matrix.png"):
    """
    绘制特征相关系数矩阵热力图
    用于观察多重共线性的直观强度
    """
    X = X.astype(np.float64)
    corr = np.corrcoef(X.T)

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar()

    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return corr


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "results/residual_plot.png"):
    """
    绘制残差诊断图：
    - 残差是否围绕0波动
    - 是否存在异方差
    - 是否满足线性回归假设
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residual vs Fitted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # 返回简单诊断统计量
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    return {
        "residual_mean": round(res_mean, 4),
        "residual_std": round(res_std, 4)
    }
