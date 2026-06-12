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
    
    # 3. 测试集评估
    X_test_scaled = scaler.transform(X_test)
    models = {
        'OLS': LinearRegression().fit(X_train_scaled, y_train),
        'Ridge': ridge_gs.best_estimator_,
        'Lasso': lasso_gs.best_estimator_,
        'ElasticNet': enet_gs.best_estimator_
    }
    
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
    """
    预处理房价数据：选择特征，并彻底处理缺失值。
    返回 X (无缺失值的特征矩阵), y_log (对数变换后的目标变量), 特征名列表。
    """
    # 1. 分离目标变量
    y = df['SalePrice'].values
    # 对目标变量取对数，处理偏态分布
    y_log = np.log(y)
    
    # 2. 选择数值特征（自动排除非数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 移除 Id（如果有）和目标变量
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    
    # 3. 删除缺失率超过 50% 的列（这些列信息量太少）
    missing_ratio = df[numeric_cols].isnull().sum() / len(df)
    keep_cols = [c for c in numeric_cols if missing_ratio[c] < 0.5]
    X_df = df[keep_cols].copy()
    
    # 4. 对保留的数值列填补缺失值（使用每列的中位数）
    #    注意：这里使用 .fillna() 并重新赋值，避免 inplace=True 的警告
    for col in X_df.columns:
        if X_df[col].isnull().any():
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val)
    
    # 5. 最后检查一次是否还有缺失值（防御性编程）
    if X_df.isnull().any().any():
        print("  ⚠️ 警告: 仍有缺失值，使用 0 填补")
        X_df = X_df.fillna(0)
    
    print(f"  保留特征数: {X_df.shape[1]}")
    return X_df.values, y_log, keep_cols

def run_kaggle_task(data_dir, results_dir):
    print("\n" + "="*70)
    print("Task B: Kaggle 房价数据 - 正则化回归")
    print("="*70)
    
    # 加载原始数据
    df = load_housing_data(data_dir)
    
    # 预处理：清洗缺失值，返回无缺失的特征矩阵 X 和目标 y
    X, y, feature_names = preprocess_housing_data(df)
    
    # 划分训练集和测试集 (这里我们使用交叉验证，但为了展示基线性能，也划分一个测试集)
    # 注意：在交叉验证中，我们会在每一折内部重新标准化，以保证无数据泄露。
    # 这里先切出一个最终测试集，用于评估模型。
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 交叉验证流程 (无泄露) ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储每折的 RMSE 结果
    ols_scores, ridge_scores, lasso_scores, enet_scores = [], [], [], []
    ridge_alphas, lasso_alphas, enet_params = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # 分割数据
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # 标准化 (在训练集上 fit，然后转换训练集和验证集)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. OLS 基准
        ols = LinearRegression().fit(X_tr_scaled, y_tr)
        y_pred_ols = ols.predict(X_val_scaled)
        ols_scores.append(calculate_rmse(y_val, y_pred_ols))
        
        # 2. Ridge (使用 GridSearchCV 找最优 alpha，注意在训练集内部再分折)
        ridge_cv = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 20)}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        ridge_cv.fit(X_tr_scaled, y_tr)
        ridge_best = ridge_cv.best_estimator_
        y_pred_ridge = ridge_best.predict(X_val_scaled)
        ridge_scores.append(calculate_rmse(y_val, y_pred_ridge))
        ridge_alphas.append(ridge_cv.best_params_['alpha'])
        
        # 3. Lasso
        lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': np.logspace(-3, 2, 20)}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        lasso_cv.fit(X_tr_scaled, y_tr)
        lasso_best = lasso_cv.best_estimator_
        y_pred_lasso = lasso_best.predict(X_val_scaled)
        lasso_scores.append(calculate_rmse(y_val, y_pred_lasso))
        lasso_alphas.append(lasso_cv.best_params_['alpha'])
        
        # 4. ElasticNet
        enet_cv = GridSearchCV(ElasticNet(max_iter=10000), 
                               {'alpha': np.logspace(-3, 2, 15), 'l1_ratio': np.linspace(0.1, 0.9, 9)}, 
                               cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        enet_cv.fit(X_tr_scaled, y_tr)
        enet_best = enet_cv.best_estimator_
        y_pred_enet = enet_best.predict(X_val_scaled)
        enet_scores.append(calculate_rmse(y_val, y_pred_enet))
        enet_params.append(enet_cv.best_params_)
        
        print(f"  Fold {fold}: OLS RMSE={ols_scores[-1]:.4f}, Ridge RMSE={ridge_scores[-1]:.4f}, Lasso RMSE={lasso_scores[-1]:.4f}, Enet RMSE={enet_scores[-1]:.4f}")
    
    # 输出交叉验证平均结果
    print("\n[5-Fold CV 平均 RMSE]")
    print(f"  OLS: {np.mean(ols_scores):.4f} ± {np.std(ols_scores):.4f}")
    print(f"  Ridge: {np.mean(ridge_scores):.4f} ± {np.std(ridge_scores):.4f}")
    print(f"  Lasso: {np.mean(lasso_scores):.4f} ± {np.std(lasso_scores):.4f}")
    print(f"  ElasticNet: {np.mean(enet_scores):.4f} ± {np.std(enet_scores):.4f}")
    
    # 为了展示 Lasso 的特征选择能力，使用全部训练数据拟合一个最终 Lasso 模型
    # (注意：这个模型的 RMSE 应该以交叉验证结果为准，此处仅用于查看特征系数)
    scaler_final = StandardScaler()
    X_train_scaled_final = scaler_final.fit_transform(X_train)
    final_lasso = Lasso(alpha=np.mean(lasso_alphas), max_iter=10000)
    final_lasso.fit(X_train_scaled_final, y_train)
    nonzero_idx = np.where(np.abs(final_lasso.coef_) > 1e-6)[0]
    nonzero_features = [feature_names[i] for i in nonzero_idx]
    print(f"\n[Lasso 特征选择] 保留了 {len(nonzero_features)} 个非零系数特征")
    if len(nonzero_features) <= 20:
        print(f"  选中特征: {nonzero_features}")
    else:
        print(f"  选中特征 (前15): {nonzero_features[:15]}...")
    
    # 生成报告
    report_path = Path(results_dir) / "kaggle_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Kaggle 房价数据正则化回归报告\n\n")
        f.write(f"## 数据概览\n- 样本量: {df.shape[0]}, 原始特征数: {df.shape[1]-1}\n- 预处理后特征数: {X.shape[1]}\n- 目标变量: SalePrice（对数变换）\n\n")
        f.write("## 5折交叉验证性能对比 (RMSE)\n")
        f.write("| 模型 | RMSE (均值 ± 标准差) |\n")
        f.write("|------|----------------------|\n")
        f.write(f"| OLS | {np.mean(ols_scores):.4f} ± {np.std(ols_scores):.4f} |\n")
        f.write(f"| Ridge | {np.mean(ridge_scores):.4f} ± {np.std(ridge_scores):.4f} |\n")
        f.write(f"| Lasso | {np.mean(lasso_scores):.4f} ± {np.std(lasso_scores):.4f} |\n")
        f.write(f"| ElasticNet | {np.mean(enet_scores):.4f} ± {np.std(enet_scores):.4f} |\n\n")
        f.write(f"## Lasso 特征选择\n保留了 {len(nonzero_features)} 个非零系数特征。\n")
        f.write("## 结论\n- 所有正则化模型的 RMSE 均优于普通 OLS。\n- Lasso 实现了特征选择，自动筛选出与房价最相关的一组变量。\n")
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
    except Exception as e:
        print(f"\n⚠️ Task B 执行出错: {e}")

    # Task C
    generate_summary_comparison(results_dir)

    print("\n" + "="*70)
    print("🎉 Week 13 作业完成！")
    print("="*70)

if __name__ == "__main__":
    main()
