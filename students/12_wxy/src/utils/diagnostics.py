import numpy as np
import matplotlib.pyplot as plt

# ====================== VIF 计算（已修复公式！） ======================
def calculate_vif(X):
    n, p = X.shape
    vif = np.zeros(p)
    
    for i in range(p):
        y = X[:, i]
        cols = [j for j in range(p) if j != i]
        x = X[:, cols]
        
        x_b = np.hstack([np.ones((x.shape[0], 1)), x])
        beta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
        y_pred = x_b @ beta
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.999
        
        vif[i] = 1 / (1 - r2 + 1e-9)
    
    return vif

def print_vif_report(v, names):
    print("\nVIF 报告:")
    for n, vi in zip(names, v):
        print(f"{n:15} {vi:.1f}")

# ====================== 系数稳定性绘图 ======================
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

# ====================== 矩阵秩、条件数 ======================
def matrix_condition_metrics(X):
    """返回矩阵秩、条件数"""
    rank = np.linalg.matrix_rank(X)
    cond_num = np.linalg.cond(X)
    return {"rank": rank, "condition_number": cond_num}

# ====================== 新增：系数路径图 ======================
def plot_coef_path(coefs_list, feature_names=None, title="Coefficient Path"):
    """
    绘制系数路径图（用于Lasso等正则化方法）
    """
    plt.figure(figsize=(10, 6))
    
    coef_array = np.array(coefs_list)
    
    # 确保形状是 (n_features, n_steps)
    if coef_array.shape[0] > coef_array.shape[1]:
        coef_array = coef_array.T
    
    for i in range(coef_array.shape[0]):
        plt.plot(coef_array[i], linewidth=1, alpha=0.7, 
                label=feature_names[i] if feature_names and i < len(feature_names) else f"Feat_{i}")
    
    plt.xlabel("Regularization Strength (log scale)")
    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.grid(alpha=0.3)
    
    if feature_names and len(feature_names) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()
