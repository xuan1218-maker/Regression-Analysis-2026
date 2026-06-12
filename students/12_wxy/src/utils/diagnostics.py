import numpy as np
import matplotlib.pyplot as plt

# ====================== VIF 计算（已修复公式！） ======================
def calculate_vif(X):
    n, p = X.shape
    vif = np.zeros(p)
    
    for i in range(p):
        # 目标变量
        y = X[:, i]
        # 其他变量
        cols = [j for j in range(p) if j != i]
        x = X[:, cols]
        
        # 手动做回归
        x_b = np.hstack([np.ones((x.shape[0], 1)), x])
        beta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
        y_pred = x_b @ beta
        
        # 正确 R² 公式
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.999
        
        # VIF
        vif[i] = 1 / (1 - r2 + 1e-9)
    
    return vif

def print_vif_report(v, names):
    print("\nVIF 报告:")
    for n, vi in zip(names, v):
        print(f"{n:15} {vi:.1f}")

# ====================== 系数稳定性绘图（无中文、无报错） ======================
def plot_coef_stability(ols_list, ridge_list, feat_names, path):
    plt.figure(figsize=(10, 5))
    
    ols_mat = np.array(ols_list)
    ridge_mat = np.array(ridge_list)

    plt.boxplot(ols_mat, positions=[1,2,3], widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.boxplot(ridge_mat, positions=[1.4,2.4,3.4], widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightcoral"))

    plt.xticks([1.2, 2.2, 3.2], feat_names)
    plt.title("OLS vs Ridge Coefficient Stability")
    plt.ylabel("Coefficient Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def calc_coef_std(coef_list):
    return np.round(np.std(coef_list, axis=0), 4)