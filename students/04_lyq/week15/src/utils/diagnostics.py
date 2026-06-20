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

import matplotlib.pyplot as plt

# ====================== 新增分类任务绘图 ======================
def plot_linear_logistic_compare(X_main, y_true, linear_pred, logit_prob, save_path="results/linear_vs_logistic.png"):
    """Task A4 单特征对比图：横轴特征，纵轴模型输出"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X_main, y_true, label="True label (0/1)", alpha=0.6, color="gray")
    sort_idx = np.argsort(X_main)
    plt.plot(X_main[sort_idx], linear_pred[sort_idx], label="LinearRegression output", color="red", lw=2)
    plt.plot(X_main[sort_idx], logit_prob[sort_idx], label="LogisticRegression probability", color="blue", lw=2)
    plt.xlabel("Main predictive feature X1")
    plt.ylabel("Model output value")
    plt.title("Linear Regression vs Logistic Regression Prediction")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_loss_curve(save_path="results/loss_curve_compare.png"):
    """Task B2 损失对比图：p横轴，loss纵轴，y=0/y=1两组MSE+logloss"""
    p_range = np.linspace(0.001, 0.999, 200)
    # y=1
    mse1 = (1 - p_range) ** 2
    nll1 = -np.log(p_range)
    # y=0
    mse0 = (0 - p_range) ** 2
    nll0 = -np.log(1 - p_range)

    plt.figure(figsize=(9, 5))
    plt.plot(p_range, mse1, label="MSE | y=1", c="#ff7777", linestyle="--")
    plt.plot(p_range, nll1, label="LogLoss | y=1", c="#dd2222")
    plt.plot(p_range, mse0, label="MSE | y=0", c="#77aaff", linestyle="--")
    plt.plot(p_range, nll0, label="LogLoss | y=0", c="#2255dd")
    plt.xlabel("Predicted positive probability p")
    plt.ylabel("Single sample loss value")
    plt.title("MSE vs Log Loss under True Label y=0 / y=1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_threshold_tradeoff(threshold_list, metric_records, save_path="results/threshold_metrics.png"):
    """Task C3 阈值-指标曲线"""
    thresholds = [x["threshold"] for x in metric_records]
    acc = [x["accuracy"] for x in metric_records]
    prec = [x["precision"] for x in metric_records]
    rec = [x["recall"] for x in metric_records]
    f1 = [x["F1"] for x in metric_records]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, acc, label="Accuracy", lw=2)
    plt.plot(thresholds, prec, label="Precision", lw=2)
    plt.plot(thresholds, rec, label="Recall", lw=2)
    plt.plot(thresholds, f1, label="F1 Score", lw=2)
    plt.xlabel("Classification threshold")
    plt.ylabel("Metric value (0~1)")
    plt.title("Metrics Trade-off with Changing Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_l1_l2_coeff_compare(l1_coeffs, l2_coeffs, save_path="results/l1_l2_coeff_auc.png"):
    """Task D3 L1/L2系数分布对比"""
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.bar(range(len(l1_coeffs)), l1_coeffs, label="L1 penalty coefficients")
    plt.title("L1 Logistic Coefficients (Sparse)")
    plt.subplot(1,2,2)
    plt.bar(range(len(l2_coeffs)), l2_coeffs, label="L2 penalty coefficients")
    plt.title("L2 Logistic Coefficients (Dense)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

