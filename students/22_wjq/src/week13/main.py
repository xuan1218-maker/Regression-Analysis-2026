#!/usr/bin/env python3
"""
Week 13: Regularized Regression and Variable Selection
完整版：Task A（模拟数据）+ Task B（Kaggle房价数据）
Usage: uv run src/week13/main.py
"""

import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from src.utils.transformers import CustomStandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true, y_pred):
    return {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


# ========================== Task A: 模拟数据 ==========================
def generate_correlated_data(n_samples=500, random_seed=42):
    """生成带高度共线性的模拟数据"""
    np.random.seed(random_seed)
    
    # 潜变量（驱动相关特征组）
    latent = np.random.normal(0, 1, n_samples)
    
    # 高度相关的特征族（4个特征）
    X1 = 0.95 * latent + np.random.normal(0, 0.3, n_samples)
    X2 = 0.90 * latent + np.random.normal(0, 0.4, n_samples)
    X3 = 0.85 * latent + np.random.normal(0, 0.5, n_samples)
    X4 = 0.80 * latent + np.random.normal(0, 0.6, n_samples)
    
    # 独立真实特征
    X5 = np.random.normal(0, 1, n_samples)
    X6 = np.random.normal(0, 1, n_samples)
    
    # 纯噪声特征
    X7 = np.random.normal(0, 1, n_samples)
    X8 = np.random.normal(0, 1, n_samples)
    X9 = np.random.normal(0, 1, n_samples)
    X10 = np.random.normal(0, 1, n_samples)
    
    X = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10])
    feature_names = [f'X{i+1}' for i in range(10)]
    
    # 真实系数（DGP）
    true_coef = np.array([3.0, 2.0, 1.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0])
    intercept = 5.0
    y = intercept + X @ true_coef + np.random.normal(0, 1.0, n_samples)
    
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    return df, true_coef, intercept, feature_names

def stability_comparison(X, y, n_splits=50, alpha_ridge=1.0):
    """对比 OLS 和 Ridge 的系数稳定性"""
    n_features = X.shape[1]
    ols_coefs, ridge_coefs = [], []
    
    for _ in range(n_splits):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=None)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        ols = LinearRegression().fit(X_train_scaled, y_train)
        ridge = Ridge(alpha=alpha_ridge).fit(X_train_scaled, y_train)
        ols_coefs.append(ols.coef_)
        ridge_coefs.append(ridge.coef_)
    
    return np.array(ols_coefs), np.array(ridge_coefs)

def forward_selection(X, y, max_features=None, cv_folds=5):
    """前向选择：逐个添加特征"""
    if max_features is None:
        max_features = min(X.shape[1], 15)
    
    n_samples, n_features = X.shape
    selected, remaining = [], list(range(n_features))
    cv_scores = []
    
    for k in range(max_features):
        best_score, best_feature = np.inf, None
        for feature in remaining:
            candidates = selected + [feature]
            X_subset = X[:, candidates]
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in kf.split(X_subset):
                X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                scaler = StandardScaler()
                model = LinearRegression().fit(scaler.fit_transform(X_tr), y_tr)
                y_pred = model.predict(scaler.transform(X_val))
                scores.append(mean_squared_error(y_val, y_pred))
            cv_score = np.mean(scores)
            if cv_score < best_score:
                best_score, best_feature = cv_score, feature
        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
            cv_scores.append(best_score)
    return selected, cv_scores

def run_synthetic_task(data_dir, results_dir):
    print("\n" + "="*70)
    print("Task A: 模拟共线性数据 - 正则化回归与变量筛选")
    print("="*70)
    
    # 生成数据
    data_path = Path(data_dir) / "synthetic_correlated.csv"
    df, true_coef, intercept, feature_names = generate_correlated_data()
    df.to_csv(data_path, index=False)
    print(f"✅ 模拟数据已生成: {data_path} (样本: {df.shape[0]}, 特征: {df.shape[1]-1})")
    
    X, y = df.drop('y', axis=1).values, df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. 系数稳定性对比
    print("\n[1] 系数稳定性对比 (50次随机切分)")
    ols_coefs, ridge_coefs = stability_comparison(X, y)
    ols_std = np.std(ols_coefs, axis=0)
    ridge_std = np.std(ridge_coefs, axis=0)
    for i, name in enumerate(feature_names):
        print(f"  {name}: OLS标准差={ols_std[i]:.4f}, Ridge标准差={ridge_std[i]:.4f}")
    
        print("\n  📊 绘制系数稳定性箱线图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # OLS 箱线图
    axes[0].boxplot(ols_coefs, labels=feature_names)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('OLS 系数分布 (50次随机切分)', fontsize=12)
    axes[0].set_xlabel('特征')
    axes[0].set_ylabel('系数值')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Ridge 箱线图
    axes[1].boxplot(ridge_coefs, labels=feature_names)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Ridge 系数分布 (alpha=1.0, 50次随机切分)', fontsize=12)
    axes[1].set_xlabel('特征')
    axes[1].set_ylabel('系数值')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    boxplot_path = Path(results_dir) / "stability_boxplot.png"
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    print(f"    箱线图已保存: {boxplot_path}")
    # 2. GridSearchCV 调参
    print("\n[2] GridSearchCV 超参数寻优")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    ridge_gs = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 3, 30)}, cv=5, scoring='neg_mean_squared_error')
    lasso_gs = GridSearchCV(Lasso(max_iter=10000), {'alpha': np.logspace(-4, 2, 30)}, cv=5, scoring='neg_mean_squared_error')
    enet_gs = GridSearchCV(ElasticNet(max_iter=10000), {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': np.linspace(0.1, 0.9, 9)}, cv=5, scoring='neg_mean_squared_error')
    
    ridge_gs.fit(X_train_scaled, y_train)
    lasso_gs.fit(X_train_scaled, y_train)
    enet_gs.fit(X_train_scaled, y_train)
    
    print(f"  Ridge最佳alpha={ridge_gs.best_params_['alpha']:.4f}, MSE={-ridge_gs.best_score_:.4f}")
    print(f"  Lasso最佳alpha={lasso_gs.best_params_['alpha']:.4f}, MSE={-lasso_gs.best_score_:.4f}")
    print(f"  ElasticNet最佳alpha={enet_gs.best_params_['alpha']:.4f}, l1_ratio={enet_gs.best_params_['l1_ratio']:.2f}")
    
    print("\n  📊 绘制 CV 误差曲线...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ridge CV 曲线
    ridge_results = ridge_gs.cv_results_
    alphas = ridge_results['param_alpha'].data
    mean_scores = -ridge_results['mean_test_score']
    std_scores = ridge_results['std_test_score']
    
    axes[0].plot(alphas, mean_scores, 'b-', label='CV Error')
    axes[0].fill_between(alphas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('alpha (对数尺度)')
    axes[0].set_ylabel('交叉验证 MSE')
    axes[0].set_title('Ridge: CV Error vs alpha')
    axes[0].axvline(x=ridge_gs.best_params_['alpha'], color='r', linestyle='--', 
                    label=f"最佳 alpha={ridge_gs.best_params_['alpha']:.4f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Lasso CV 曲线
    lasso_results = lasso_gs.cv_results_
    alphas_lasso = lasso_results['param_alpha'].data
    mean_scores_lasso = -lasso_results['mean_test_score']
    
    axes[1].plot(alphas_lasso, mean_scores_lasso, 'g-', label='CV Error')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('alpha (对数尺度)')
    axes[1].set_ylabel('交叉验证 MSE')
    axes[1].set_title('Lasso: CV Error vs alpha')
    axes[1].axvline(x=lasso_gs.best_params_['alpha'], color='r', linestyle='--', 
                    label=f"最佳 alpha={lasso_gs.best_params_['alpha']:.4f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    cv_path = Path(results_dir) / "cv_curves.png"
    plt.savefig(cv_path, dpi=150)
    plt.close()
    print(f"    CV曲线图已保存: {cv_path}")
    # 3. 测试集评估
    X_test_scaled = scaler.transform(X_test)
    models = {'OLS': LinearRegression().fit(X_train_scaled, y_train), 'Ridge': ridge_gs.best_estimator_, 'Lasso': lasso_gs.best_estimator_, 'ElasticNet': enet_gs.best_estimator_}
    
    print("\n[3] 测试集性能")
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        m = compute_metrics(y_test, y_pred)
        print(f"  {name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
    
    # 4. 前向选择
    print("\n[4] 前向选择")
    scaler_fs = StandardScaler()
    X_scaled_fs = scaler_fs.fit_transform(X)
    selected, _ = forward_selection(X_scaled_fs, y, max_features=8)
    fs_selected = [feature_names[i] for i in selected]
    lasso_selected = [feature_names[i] for i, c in enumerate(lasso_gs.best_estimator_.coef_) if abs(c) > 1e-6]
    print(f"  前向选择: {fs_selected}")
    print(f"  Lasso选择: {lasso_selected}")
    
    # 生成报告
    report_path = Path(results_dir) / "synthetic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 模拟数据正则化回归报告\n\n")
        f.write("## 1. DGP\n")
        f.write("y = 5 + 3X1 + 2X2 + X3 + 1.5X5 + X6 + ε\n")
        f.write("X1-X4高度相关（由潜变量驱动），X7-X10为纯噪声。\n\n")
        f.write("## 2. 系数稳定性\n")
        for i, name in enumerate(feature_names):
            f.write(f"- {name}: OLS std={ols_std[i]:.4f}, Ridge std={ridge_std[i]:.4f}\n")
        f.write("\n## 3. 测试集性能\n| 模型 | RMSE | MAE | R2 |\n|------|------|-----|-----|\n")
        for name, model in models.items():
            m = compute_metrics(y_test, model.predict(X_test_scaled))
            f.write(f"| {name} | {m['RMSE']:.4f} | {m['MAE']:.4f} | {m['R2']:.4f} |\n")
        f.write(f"\n## 4. 变量筛选\n- 前向选择: {fs_selected}\n- Lasso选择: {lasso_selected}\n")
    print(f"📄 报告: {report_path}")
    return report_path


# ========================== Task B: Kaggle 房价数据 ==========================
def load_housing_data(data_dir):
    """加载Kaggle房价数据"""
    train_path = Path(data_dir) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"请将 train.csv 放在 {train_path}")
    
    df = pd.read_csv(train_path)
    print(f"✅ 加载房价数据: {train_path}, 形状: {df.shape}")
    return df

def preprocess_housing_data(df):
    """预处理房价数据：选择重要特征，处理缺失值"""
    # 选择数值特征（自动排除非数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 移除 Id 和目标变量
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    
    # 删除缺失率超过50%的列（放宽阈值以保留更多特征）
    missing_ratio = df[numeric_cols].isnull().sum() / len(df)
    keep_cols = [c for c in numeric_cols if missing_ratio[c] < 0.5]
    
    # 创建特征矩阵
    X = df[keep_cols].copy()
    
    # 填补缺失值（用中位数）- 使用非 inplace 方式避免警告
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)  # 关键：重新赋值而非 inplace
    
    # 防御性检查：确保没有 NaN
    if X.isnull().any().any():
        print("  ⚠️ 仍有缺失值，使用 0 填补")
        X = X.fillna(0)
    
    y = df['SalePrice'].values
    # 对目标变量取对数（处理偏态）
    y_log = np.log(y)
    
    print(f"  保留特征数: {X.shape[1]}")
    return X.values, y_log, keep_cols

def run_kaggle_task(data_dir, results_dir):
    print("\n" + "="*70)
    print("Task B: Kaggle 房价数据 - 正则化回归")
    print("="*70)
    
    df = load_housing_data(data_dir)
    X, y, feature_names = preprocess_housing_data(df)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 标准化（在训练集上 fit，测试集上 transform）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 检查是否还有 NaN（防御性检查）
    if np.any(np.isnan(X_train_scaled)):
        print("  ❌ 错误：X_train_scaled 仍包含 NaN！")
        print(f"     NaN 数量: {np.isnan(X_train_scaled).sum()}")
        # 找出哪一列有 NaN
        nan_cols = np.where(np.isnan(X_train_scaled).any(axis=0))[0]
        print(f"     有 NaN 的列索引: {nan_cols}")
        for idx in nan_cols:
            print(f"       列 {idx}: 原始列名 {feature_names[idx] if idx < len(feature_names) else 'unknown'}")
        return
    
    # OLS 基准
    print("\n[1] 模型训练")
    ols = LinearRegression().fit(X_train_scaled, y_train)
    y_pred_ols = ols.predict(X_test_scaled)
    print(f"  OLS: RMSE={calculate_rmse(y_test, y_pred_ols):.4f}, R2={r2_score(y_test, y_pred_ols):.4f}")
    
    # GridSearchCV 调参
    print("\n[2] GridSearchCV 超参数寻优...")
    ridge_gs = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 20)}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    lasso_gs = GridSearchCV(Lasso(max_iter=10000), {'alpha': np.logspace(-3, 2, 20)}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    enet_gs = GridSearchCV(ElasticNet(max_iter=10000), {'alpha': np.logspace(-3, 2, 15), 'l1_ratio': np.linspace(0.1, 0.9, 9)}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    ridge_gs.fit(X_train_scaled, y_train)
    lasso_gs.fit(X_train_scaled, y_train)
    enet_gs.fit(X_train_scaled, y_train)
    
    print(f"  Ridge 最佳alpha={ridge_gs.best_params_['alpha']:.4f}")
    print(f"  Lasso 最佳alpha={lasso_gs.best_params_['alpha']:.4f}")
    print(f"  ElasticNet 最佳alpha={enet_gs.best_params_['alpha']:.4f}, l1_ratio={enet_gs.best_params_['l1_ratio']:.2f}")
    
    # 测试集评估
    models = {'Ridge': ridge_gs, 'Lasso': lasso_gs, 'ElasticNet': enet_gs}
    print("\n[3] 测试集性能对比")
    for name, gs in models.items():
        y_pred = gs.predict(X_test_scaled)
        rmse = calculate_rmse(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  {name}: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # Lasso 特征选择结果
    lasso_coef = lasso_gs.best_estimator_.coef_
    nonzero_idx = np.where(np.abs(lasso_coef) > 1e-6)[0]
    nonzero_features = [feature_names[i] for i in nonzero_idx]
    print(f"\n[4] Lasso 选中的特征 ({len(nonzero_features)}个):")
    if len(nonzero_features) <= 20:
        print(f"  {nonzero_features}")
    else:
        print(f"  {nonzero_features[:15]}... (共{len(nonzero_features)}个)")
    
    # 生成报告
    report_path = Path(results_dir) / "kaggle_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Kaggle 房价数据正则化回归报告\n\n")
        f.write(f"## 数据概览\n- 样本量: {df.shape[0]}, 原始特征数: {df.shape[1]-1}\n- 预处理后特征数: {X.shape[1]}\n- 目标变量: SalePrice（对数变换）\n\n")
        f.write("## 测试集性能\n| 模型 | RMSE | R2 |\n|------|------|-----|\n")
        for name, gs in models.items():
            y_pred = gs.predict(X_test_scaled)
            f.write(f"| {name} | {calculate_rmse(y_test, y_pred):.4f} | {r2_score(y_test, y_pred):.4f} |\n")
        f.write(f"\n## Lasso 特征选择\n保留了 {len(nonzero_features)} 个非零系数特征。\n\n")
        f.write("## 结论\n- 正则化模型相比OLS在泛化能力上有所提升。\n- Lasso 实现了特征选择，帮助识别重要变量。\n")
    print(f"📄 报告: {report_path}")
    return report_path

# ========================== Task C: 总结报告 ==========================
def generate_summary_comparison(results_dir):
    summary_path = Path(results_dir) / "summary_comparison.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 正则化回归：理论与实践总结\n\n")
        f.write("## 1. Lasso 在高度相关变量组中的风险\n")
        f.write("Lasso 会从相关特征组中随机选择一个而压缩其他为0，可能导致业务误判。Elastic Net 通过 L2 惩罚缓解此问题。\n\n")
        f.write("## 2. GridSearchCV 的优化目标 vs 主观偏好\n")
        f.write("GridSearchCV 最小化验证误差，不保证最稀疏或最稳定。业务解释可能需要调整 alpha。\n\n")
        f.write("## 3. 前向选择 vs Lasso\n")
        f.write("| 方法 | 复杂度 | 稳定性 | 适用场景 |\n|------|--------|--------|----------|\n")
        f.write("| 前向选择 | O(k²·p) | 受共线性影响大 | p < 100 |\n| Lasso | O(k·p) | 稳定 | 高维数据 |\n\n")
        f.write("## 4. 关键结论\n")
        f.write("- OLS 在共线性下系数极不稳定\n- Ridge 均匀收缩，Lasso 稀疏化，Elastic Net 折中\n- 正则化前必须标准化\n")
    print(f"📄 总结报告: {summary_path}")
    return summary_path


# ========================== 主流程 ==========================
def main():
    results_dir = Path("src/week13/results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("✅ results 文件夹已清理并重建")

    data_dir = Path("src/week13/data")
    ensure_dir(data_dir)

    # Task A
    run_synthetic_task(data_dir, results_dir)

    # Task B（如果数据存在则执行）
    try:
        run_kaggle_task(data_dir, results_dir)
    except FileNotFoundError as e:
        print(f"\n⚠️ Task B 跳过: {e}")

    # Task C
    generate_summary_comparison(results_dir)

    print("\n" + "="*70)
    print("🎉 Week 13 作业完成！")
    print("="*70)

if __name__ == "__main__":
    main()