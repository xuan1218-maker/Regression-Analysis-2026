"""
Week 13: Regularized Regression and Variable Selection
正则化回归与变量筛选
"""
import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

from utils.metrics import calculate_rmse, calculate_mae


def generate_correlated_data(n_samples=300, random_seed=42):
    """生成带有共线性的模拟数据"""
    np.random.seed(random_seed)
    
    # 真实相关特征
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = np.random.randn(n_samples)
    
    # 构造高度相关特征族
    x4 = 0.8 * x1 + 0.2 * x2 + np.random.randn(n_samples) * 0.2
    x5 = 0.7 * x1 + 0.3 * x3 + np.random.randn(n_samples) * 0.2
    x6 = 0.9 * x2 + 0.1 * x3 + np.random.randn(n_samples) * 0.2
    
    # 纯噪声特征（与 y 无关）
    x7 = np.random.randn(n_samples)
    x8 = np.random.randn(n_samples)
    x9 = np.random.randn(n_samples)
    x10 = np.random.randn(n_samples)
    
    # 真实系数
    true_coef = {'x1': 3.0, 'x2': 2.0, 'x3': 1.5}
    for i in range(4, 11):
        true_coef[f'x{i}'] = 0
    
    # 生成 y
    y = 3.0 * x1 + 2.0 * x2 + 1.5 * x3 + np.random.randn(n_samples) * 1.0
    
    X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    
    return X, y, feature_names, true_coef


def save_synthetic_data(X, y, feature_names, filepath):
    data = np.column_stack([X, y])
    headers = feature_names + ['y']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"✅ 数据已保存: {filepath}")


def stability_comparison(X, y, feature_names, n_splits=50, alpha_ridge=50.0):
    print("\n" + "="*60)
    print("Task 1: 正则化前后的稳定性对比")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ols_coefs = []
    ridge_coefs = []
    
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=i
        )
        
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        ols_coefs.append(ols.coef_)
        
        ridge = Ridge(alpha=alpha_ridge)
        ridge.fit(X_train, y_train)
        ridge_coefs.append(ridge.coef_)
    
    ols_coefs = np.array(ols_coefs)
    ridge_coefs = np.array(ridge_coefs)
    
    print(f"OLS 系数标准差 (前6个): {[f'{s:.4f}' for s in np.std(ols_coefs, axis=0)[:6]]}")
    print(f"Ridge 系数标准差 (前6个): {[f'{s:.4f}' for s in np.std(ridge_coefs, axis=0)[:6]]}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].boxplot(ols_coefs, labels=feature_names)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('OLS 系数分布 (50次随机切分)')
    axes[0].set_ylabel('系数值')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(ridge_coefs, labels=feature_names)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Ridge (alpha={alpha_ridge}) 系数分布')
    axes[1].set_ylabel('系数值')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'coefficient_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 稳定性对比图已保存")


def gridsearch_optimization(X_train, y_train):
    """GridSearchCV 超参数寻优"""
    print("\n" + "="*60)
    print("Task 2: GridSearchCV 超参数寻优")
    print("="*60)
    
    alphas = np.logspace(-2, 3, 30)
    
    # Ridge
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train, y_train)
    best_ridge_alpha = ridge_cv.best_params_['alpha']
    best_ridge = ridge_cv.best_estimator_
    print(f"Ridge 最佳 alpha: {best_ridge_alpha:.6f}")
    
    # Lasso
    lasso = Lasso(max_iter=10000)
    lasso_cv = GridSearchCV(lasso, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train, y_train)
    best_lasso_alpha = lasso_cv.best_params_['alpha']
    best_lasso = lasso_cv.best_estimator_
    print(f"Lasso 最佳 alpha: {best_lasso_alpha:.6f}")
    
    # Elastic Net
    l1_ratios = np.linspace(0.1, 0.9, 5)
    enet = ElasticNet(max_iter=10000)
    param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}
    enet_cv = GridSearchCV(enet, param_grid, cv=5, scoring='neg_mean_squared_error')
    enet_cv.fit(X_train, y_train)
    best_enet_alpha = enet_cv.best_params_['alpha']
    best_enet_l1 = enet_cv.best_params_['l1_ratio']
    best_enet = enet_cv.best_estimator_
    print(f"Elastic Net 最佳 alpha: {best_enet_alpha:.6f}, l1_ratio: {best_enet_l1:.4f}")
    
    # 绘制 CV 曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ridge_scores = [-s for s in ridge_cv.cv_results_['mean_test_score']]
    axes[0].plot(alphas, ridge_scores, 'o-', color='steelblue')
    axes[0].axvline(x=best_ridge_alpha, color='red', linestyle='--')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('alpha')
    axes[0].set_ylabel('CV MSE')
    axes[0].set_title('Ridge CV Error')
    axes[0].grid(True, alpha=0.3)
    
    lasso_scores = [-s for s in lasso_cv.cv_results_['mean_test_score']]
    axes[1].plot(alphas, lasso_scores, 'o-', color='darkorange')
    axes[1].axvline(x=best_lasso_alpha, color='red', linestyle='--')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('alpha')
    axes[1].set_ylabel('CV MSE')
    axes[1].set_title('Lasso CV Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = Path(__file__).parent / "results"
    plt.savefig(results_dir / 'cv_error_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ CV 曲线图已保存")
    
    return (best_ridge, best_lasso, best_enet), (best_ridge_alpha, best_lasso_alpha, best_enet_alpha, best_enet_l1)


def compare_models(models, X_train, y_train, X_test, y_test, feature_names, true_coef):
    """对比三种模型"""
    print("\n" + "="*60)
    print("Task 3: 模型性格大比拼")
    print("="*60)
    
    best_ridge, best_lasso, best_enet = models
    
    print("\n测试集性能:")
    print("-" * 50)
    for name, model in [('Ridge', best_ridge), ('Lasso', best_lasso), ('Elastic Net', best_enet)]:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = calculate_rmse(y_test, y_pred)
        print(f"{name:<15} | R²={r2:.4f} | RMSE={rmse:.4f}")
    
    print("\n系数对比:")
    print("-" * 80)
    print(f"{'Feature':<10} | {'True':<8} | {'Ridge':<10} | {'Lasso':<10} | {'Elastic Net':<12}")
    print("-" * 80)
    
    ridge_coef = best_ridge.coef_
    lasso_coef = best_lasso.coef_
    enet_coef = best_enet.coef_
    
    for i, name in enumerate(feature_names):
        true_val = true_coef.get(name, 0)
        print(f"{name:<10} | {true_val:<8.2f} | {ridge_coef[i]:<10.4f} | {lasso_coef[i]:<10.4f} | {enet_coef[i]:<12.4f}")
    
    # 统计筛选结果
    lasso_selected = [feature_names[i] for i, c in enumerate(lasso_coef) if abs(c) > 0.01]
    print(f"\nLasso 选中特征 ({len(lasso_selected)}个): {lasso_selected}")
    
    return ridge_coef, lasso_coef, enet_coef, lasso_selected


def forward_selection(X, y, feature_names, max_features=5):
    """前向选择"""
    print("\n" + "="*60)
    print("Task 4: 前向选择 (Forward Selection)")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selected = []
    remaining = list(range(X.shape[1]))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for step in range(min(max_features, X.shape[1])):
        best_score = -np.inf
        best_feature = None
        
        for feature in remaining:
            current_set = selected + [feature]
            X_subset = X_scaled[:, current_set]
            
            scores = []
            for train_idx, val_idx in kf.split(X_subset):
                X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                model = LinearRegression()
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                scores.append(-mean_squared_error(y_val, y_pred))
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_feature = feature
        
        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
            print(f"Step {step+1}: 添加 {feature_names[best_feature]}, CV MSE={-best_score:.4f}")
    
    print(f"\n前向选择选中 ({len(selected)}个): {[feature_names[i] for i in selected]}")
    
    return selected


def lasso_with_alpha_selection(X, y, feature_names):
    """用不同 alpha 测试 Lasso，选择能筛选出3-5个特征的 alpha"""
    print("\n" + "="*60)
    print("Task 4 补充: Lasso 特征筛选 (多 alpha 对比)")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"\n{'alpha':<8} | {'非零系数数':<10} | {'选中的特征'}")
    print("-" * 60)
    
    results = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        coefs = lasso.coef_
        non_zero = [i for i, c in enumerate(coefs) if abs(c) > 0.01]
        selected_names = [feature_names[i] for i in non_zero]
        print(f"{alpha:<8.2f} | {len(non_zero):<10} | {selected_names}")
        results.append((alpha, len(non_zero), selected_names, lasso))
    
    # 选择一个合适的 alpha (选中3-5个特征的)
    best_alpha = 1.0
    best_lasso = None
    for alpha, nz, names, lasso_model in results:
        if 3 <= nz <= 5:
            best_alpha = alpha
            best_lasso = lasso_model
            break
    
    if best_lasso is None:
        # 如果没找到，用 alpha=1.0
        best_alpha = 1.0
        lasso = Lasso(alpha=best_alpha, max_iter=10000)
        best_lasso = lasso.fit(X_scaled, y)
    
    lasso_selected = [feature_names[i] for i, c in enumerate(best_lasso.coef_) if abs(c) > 0.01]
    print(f"\n✅ 选择 alpha={best_alpha}, Lasso 选中 {len(lasso_selected)} 个特征: {lasso_selected}")
    
    return best_alpha, lasso_selected, best_lasso


def plot_feature_selection_comparison(feature_names, forward_selected, lasso_selected, results_dir):
    """绘制特征选择对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    forward_present = [1 if i in forward_selected else 0 for i in range(len(feature_names))]
    lasso_present = [1 if feature_names[i] in lasso_selected else 0 for i in range(len(feature_names))]
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, forward_present, width, label='Forward Selection', color='steelblue')
    bars2 = ax.bar(x + width/2, lasso_present, width, label='Lasso', color='darkorange')
    
    ax.set_xlabel('特征')
    ax.set_ylabel('是否选中')
    ax.set_title('变量筛选对比')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_selection_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征选择对比图已保存")


def write_report(results_dir, true_coef, feature_names, tuning_results, forward_selected, lasso_selected):
    """生成报告"""
    report_path = results_dir / 'synthetic_report.md'
    best_ridge_alpha, best_lasso_alpha, best_enet_alpha, best_enet_l1 = tuning_results
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Week 13: 正则化回归与变量筛选报告\n\n")
        
        f.write("## 一、数据生成机制\n\n")
        f.write("### 真实 DGP\n")
        f.write("```\ny = 3*x1 + 2*x2 + 1.5*x3 + ε\n```\n\n")
        
        f.write("### 真实系数\n\n")
        f.write("| 特征 | 真实系数 | 说明 |\n")
        f.write("|------|----------|------|\n")
        for name, coef in true_coef.items():
            f.write(f"| {name} | {coef:.1f} | {'真实相关' if coef != 0 else '噪声特征'} |\n")
        
        f.write("\n### 构造的相关特征\n\n")
        f.write("- x4 = 0.8*x1 + 0.2*x2 + noise\n")
        f.write("- x5 = 0.7*x1 + 0.3*x3 + noise\n")
        f.write("- x6 = 0.9*x2 + 0.1*x3 + noise\n")
        f.write("- x7-x10: 纯噪声\n\n")
        
        f.write("## 二、GridSearchCV 结果\n\n")
        f.write(f"- **Ridge 最优 alpha**: {best_ridge_alpha:.6f}\n")
        f.write(f"- **Lasso 最优 alpha**: {best_lasso_alpha:.6f}\n")
        f.write(f"- **Elastic Net 最优 alpha**: {best_enet_alpha:.6f}, l1_ratio={best_enet_l1:.4f}\n\n")
        
        f.write("## 三、变量筛选结果\n\n")
        f.write(f"- **前向选择选中**: {len(forward_selected)} 个特征\n")
        f.write(f"  - 选中: {[feature_names[i] for i in forward_selected]}\n\n")
        f.write(f"- **Lasso 选中**: {len(lasso_selected)} 个特征\n")
        f.write(f"  - 选中: {lasso_selected}\n\n")
        
        f.write("## 四、结论\n\n")
        f.write("1. 正则化能提高系数稳定性\n")
        f.write("2. Lasso 适用于特征筛选，通过选择合适的 alpha 可以剔除噪声特征\n")
        f.write("3. 前向选择也能选出重要特征，但计算量更大\n")
    
    print(f"✅ 报告已保存: {report_path}")


def main():
    print("="*60)
    print("Week 13: Regularized Regression and Variable Selection")
    print("="*60)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 生成数据
    print("\n生成模拟数据...")
    X, y, feature_names, true_coef = generate_correlated_data(n_samples=300)
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    save_synthetic_data(X, y, feature_names, data_dir / "synthetic_correlated.csv")
    
    # 稳定性对比
    stability_comparison(X, y, feature_names, n_splits=50, alpha_ridge=50.0)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # GridSearchCV
    models, tuning_results = gridsearch_optimization(X_train_scaled, y_train)
    
    # 模型对比
    ridge_coef, lasso_coef, enet_coef, lasso_selected = compare_models(
        models, X_train_scaled, y_train, X_test_scaled, y_test, feature_names, true_coef
    )
    
    # 前向选择
    forward_selected = forward_selection(X, y, feature_names, max_features=5)
    
    # Lasso 多 alpha 分析（用于特征筛选）
    best_lasso_alpha, lasso_selected_from_analysis, best_lasso_model = lasso_with_alpha_selection(X, y, feature_names)
    
    # 使用筛选后的 Lasso 结果
    final_lasso_selected = lasso_selected_from_analysis
    
    # 绘制对比图
    plot_feature_selection_comparison(feature_names, forward_selected, final_lasso_selected, results_dir)
    
    # 报告
    write_report(results_dir, true_coef, feature_names, tuning_results, forward_selected, final_lasso_selected)
    
    print("\n" + "="*60)
    print("✅ Week 13 完成！")
    print(f"📁 结果保存在: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
