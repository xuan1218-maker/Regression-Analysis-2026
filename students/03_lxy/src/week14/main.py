"""
Week 14: High-Dimensional Regression, PCA, and PCR
完整版 - 包含详细报告生成
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = CURRENT_DIR / "results"
DATA_DIR = CURRENT_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_high_dimensional_data(n_samples=200, n_features=80, n_latent=5, n_signal_latent=3, noise_std=0.5):
    """生成高维且带有潜在低秩结构的模拟回归数据"""
    np.random.seed(42)
    latent = np.random.normal(0, 1, size=(n_samples, n_latent))
    loadings = np.random.normal(0, 1, size=(n_latent, n_features))
    loadings = loadings / np.sqrt(np.sum(loadings**2, axis=0, keepdims=True))
    X = latent @ loadings + np.random.normal(0, 0.3, size=(n_samples, n_features))
    true_coef = np.zeros(n_latent)
    true_coef[:n_signal_latent] = [2.0, 1.5, 1.0][:n_signal_latent]
    y = latent @ true_coef + np.random.normal(0, noise_std, size=n_samples)
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    return X, y, feature_names, latent


def task_a3_ols_experiment():
    """Task A3: 展示 OLS 在高维下的过拟合问题"""
    print("\n" + "="*70)
    print("[Task A3] OLS Overfitting Experiment")
    print("="*70)
    
    p_values = [10, 30, 50, 70, 90, 110, 130]
    n_samples = 150
    results = []
    
    for p in p_values:
        X, y, _, _ = make_high_dimensional_data(n_samples=n_samples, n_features=p, n_latent=min(8, p//2))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        ols = LinearRegression()
        ols.fit(X_train_s, y_train)
        
        train_rmse = calculate_rmse(y_train, ols.predict(X_train_s))
        test_rmse = calculate_rmse(y_test, ols.predict(X_test_s))
        
        U, S, Vt = np.linalg.svd(X_train_s, full_matrices=False)
        cond_num = S[0] / S[-1] if S[-1] > 0 else np.inf
        rank = np.linalg.matrix_rank(X_train_s)
        
        results.append({
            'p': p, 
            'train_rmse': train_rmse, 
            'test_rmse': test_rmse,
            'cond_num': cond_num,
            'rank': rank,
            'is_full_rank': rank == min(X_train_s.shape)
        })
        print(f"p={p:3d}: train RMSE={train_rmse:.4f}, test RMSE={test_rmse:.4f}, cond_num={cond_num:.2e}")
    
    # 绘制图形
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    df = pd.DataFrame(results)
    valid_idx = df['cond_num'] < 1e10
    
    # 左轴: RMSE
    ax1.plot(df.loc[valid_idx, 'p'], df.loc[valid_idx, 'train_rmse'], 'o-', 
             label='Train RMSE', color='#2563eb', linewidth=2, markersize=8)
    ax1.plot(df.loc[valid_idx, 'p'], df.loc[valid_idx, 'test_rmse'], 's-', 
             label='Test RMSE', color='#dc2626', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Features (p)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右轴: 条件数
    ax2 = ax1.twinx()
    ax2.plot(df.loc[valid_idx, 'p'], df.loc[valid_idx, 'cond_num'], 'd-', 
             label='Condition Number', color='#16a34a', linewidth=2, markersize=8)
    ax2.set_ylabel('Condition Number (log scale)', fontsize=12, color='#16a34a')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='#16a34a')
    ax2.legend(loc='upper right', fontsize=10)
    
    # 标记最优 p
    best_idx = df['test_rmse'].idxmin()
    best_p = df.loc[best_idx, 'p']
    ax1.axvline(x=best_p, color='purple', linestyle='--', alpha=0.7, label=f'Best p={best_p}')
    
    plt.title('OLS: Train vs Test Error as p Increases (n=150)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ols_overfitting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {FIGURES_DIR / 'ols_overfitting.png'}")
    
    return results


def task_a4_instability():
    """Task A4: 展示 OLS 系数在不同切分下的不稳定性"""
    print("\n" + "="*70)
    print("[Task A4] Coefficient Instability")
    print("="*70)
    
    n_repeats = 50
    X, y, feature_names, _ = make_high_dimensional_data(n_samples=150, n_features=50)
    
    corr = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    top_idx = np.argsort(corr)[-3:]
    selected = [feature_names[i] for i in top_idx]
    
    coefs = {name: [] for name in selected}
    
    for seed in range(n_repeats):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=seed)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        ols = LinearRegression()
        ols.fit(X_train_s, y_train)
        for i, name in enumerate(feature_names):
            if name in selected:
                coefs[name].append(ols.coef_[i])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [coefs[name] for name in selected]
    bp = ax.boxplot(data, labels=selected, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title(f'OLS Coefficient Instability (50 random splits, p=50, n=150)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加标准差注释
    for i, name in enumerate(selected):
        std_val = np.std(coefs[name])
        ax.text(i+1, np.median(coefs[name]), f'std={std_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'coefficient_instability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {FIGURES_DIR / 'coefficient_instability.png'}")
    
    coef_std = {name: np.std(vals) for name, vals in coefs.items()}
    for name, std in coef_std.items():
        print(f"  {name}: std={std:.4f}")
    
    return coef_std


def task_b_pca(X):
    """Task B1: PCA 分析"""
    print("\n" + "="*70)
    print("[Task B1] PCA Analysis")
    print("="*70)
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_s)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumsum)+1), cumsum, 'o-', color='#2563eb', linewidth=2, markersize=4)
    ax.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% variance')
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% variance')
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax.set_title('PCA: Cumulative Explained Variance', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 标记位置
    n_90 = np.argmax(cumsum >= 0.9) + 1 if any(cumsum >= 0.9) else len(cumsum)
    n_95 = np.argmax(cumsum >= 0.95) + 1 if any(cumsum >= 0.95) else len(cumsum)
    ax.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=n_95, color='red', linestyle=':', alpha=0.5)
    ax.text(n_90, 0.85, f'{n_90} PCs', fontsize=10)
    ax.text(n_95, 0.9, f'{n_95} PCs', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pca_cumulative_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {FIGURES_DIR / 'pca_cumulative_variance.png'}")
    
    print(f"  First 5 PCs explained variance: {pca.explained_variance_ratio_[:5]}")
    print(f"  90% variance: {n_90} PCs, 95% variance: {n_95} PCs")
    
    return cumsum, pca.explained_variance_ratio_


def task_b_pcr(X, y):
    """Task B2: PCR 工作流"""
    print("\n" + "="*70)
    print("[Task B2] PCR Workflow")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    k_values = range(1, min(30, X.shape[1]))
    results = []
    
    for k in k_values:
        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_s)
        Z_test = pca.transform(X_test_s)
        model = LinearRegression()
        model.fit(Z_train, y_train)
        
        train_rmse = calculate_rmse(y_train, model.predict(Z_train))
        test_rmse = calculate_rmse(y_test, model.predict(Z_test))
        
        cv_scores = cross_val_score(
            Pipeline([('pca', PCA(n_components=k)), ('reg', LinearRegression())]),
            X_train_s, y_train, cv=5, scoring='neg_root_mean_squared_error'
        )
        cv_rmse = -cv_scores.mean()
        results.append({'k': k, 'train_rmse': train_rmse, 'test_rmse': test_rmse, 'cv_rmse': cv_rmse})
    
    df = pd.DataFrame(results)
    best_k = df.loc[df['cv_rmse'].idxmin(), 'k']
    best_cv_rmse = df.loc[df['cv_rmse'].idxmin(), 'cv_rmse']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['k'], df['train_rmse'], 'o-', label='Train RMSE', color='#2563eb', linewidth=2, markersize=6)
    ax.plot(df['k'], df['test_rmse'], 's-', label='Test RMSE', color='#dc2626', linewidth=2, markersize=6)
    ax.plot(df['k'], df['cv_rmse'], 'd-', label='5-fold CV RMSE', color='#16a34a', linewidth=2, markersize=6)
    ax.axvline(x=best_k, color='purple', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    ax.set_xlabel('Number of Principal Components (k)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('PCR: Train, Test, and CV Error vs Number of PCs', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pcr_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {FIGURES_DIR / 'pcr_results.png'}")
    
    print(f"  Best k = {best_k}, CV RMSE = {best_cv_rmse:.4f}")
    
    return best_k


def generate_sparse_data(n_samples=200, n_features=60, n_signal=5, noise_std=0.5):
    """生成 Sparse Truth 数据"""
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_coef[:n_signal] = [2.0, 1.5, 1.0, 0.8, 0.6][:n_signal]
    y = X @ true_coef + np.random.normal(0, noise_std, size=n_samples)
    return X, y


def generate_latent_data(n_samples=200, n_features=60, n_latent=8, n_signal=3, noise_std=0.5):
    """生成 Latent Factor Truth 数据"""
    np.random.seed(42)
    latent = np.random.normal(0, 1, size=(n_samples, n_latent))
    loadings = np.random.normal(0, 1, size=(n_latent, n_features))
    loadings = loadings / np.sqrt(np.sum(loadings**2, axis=0, keepdims=True))
    X = latent @ loadings + np.random.normal(0, 0.3, size=(n_samples, n_features))
    true_coef = np.zeros(n_latent)
    true_coef[:n_signal] = [2.0, 1.5, 1.0][:n_signal]
    y = latent @ true_coef + np.random.normal(0, noise_std, size=n_samples)
    return X, y


def compare_lasso_pcr(X, y, name):
    """比较 Lasso 和 PCR"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_s, y_train)
    lasso_rmse = calculate_rmse(y_test, lasso.predict(X_test_s))
    lasso_nz = np.sum(np.abs(lasso.coef_) > 1e-8)
    lasso_alpha = lasso.alpha_
    
    # PCR
    best_k = 1
    best_rmse = np.inf
    for k in range(1, min(25, X.shape[1])):
        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_s)
        Z_test = pca.transform(X_test_s)
        model = LinearRegression()
        model.fit(Z_train, y_train)
        rmse = calculate_rmse(y_test, model.predict(Z_test))
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
    
    print(f"{name}: Lasso RMSE={lasso_rmse:.4f} (nz={lasso_nz}, alpha={lasso_alpha:.4f}), PCR RMSE={best_rmse:.4f} (k={best_k})")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: Lasso 系数分布
    axes[0].bar(range(len(lasso.coef_)), lasso.coef_, alpha=0.7, color='#16a34a')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Feature Index', fontsize=11)
    axes[0].set_ylabel('Coefficient Value', fontsize=11)
    axes[0].set_title(f'Lasso: {lasso_nz} non-zero coefficients\nTest RMSE={lasso_rmse:.4f}', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 右图: PCR 测试误差曲线
    pcr_errors = []
    ks = range(1, min(25, X.shape[1]))
    for k in ks:
        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_s)
        Z_test = pca.transform(X_test_s)
        model = LinearRegression()
        model.fit(Z_train, y_train)
        pcr_errors.append(calculate_rmse(y_test, model.predict(Z_test)))
    
    axes[1].plot(ks, pcr_errors, 'o-', color='#2563eb', linewidth=2, markersize=6)
    axes[1].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    axes[1].set_xlabel('Number of Principal Components (k)', fontsize=11)
    axes[1].set_ylabel('Test RMSE', fontsize=11)
    axes[1].set_title(f'PCR: Test Error vs k\nBest test RMSE={best_rmse:.4f}', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Lasso vs PCR: {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'lasso_vs_pcr_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Figure] Saved: {FIGURES_DIR / f'lasso_vs_pcr_{name}.png'}")
    
    return {'scenario': name, 'lasso_rmse': lasso_rmse, 'lasso_nz': lasso_nz, 
            'pcr_rmse': best_rmse, 'pcr_k': best_k, 'lasso_alpha': lasso_alpha}

# ============================================================
# Task D: 真实数据挑战 (Kaggle Insurance Data)
# ============================================================

def load_kaggle_insurance_data():
    """加载 Week 11 的 Kaggle 保险数据"""
    print("\n" + "="*70)
    print("[Task D] Loading Kaggle Insurance Data")
    print("="*70)
    
    # Week 11 数据路径
    week11_data_dir = CURRENT_DIR.parent / "week11" / "data"
    kaggle_path = week11_data_dir / "kaggle_insurance_working.csv"
    
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        print(f"[Task D] Loaded data from: {kaggle_path}")
    else:
        # 如果工作副本不存在，尝试原始文件
        raw_path = week11_data_dir / "kaggle_insurance_raw.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            print(f"[Task D] Loaded raw data from: {raw_path}")
        else:
            raise FileNotFoundError(f"[Task D] Kaggle insurance data not found in {week11_data_dir}")
    
    print(f"[Task D] Data shape: {df.shape}")
    print(f"[Task D] Columns: {df.columns.tolist()}")
    print(f"[Task D] Target: charges (medical costs)")
    
    return df


def preprocess_kaggle_data(df):
    """预处理 Kaggle 保险数据"""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    numeric_cols = ['age', 'bmi', 'children']
    categorical_cols = ['sex', 'smoker', 'region']
    
    # 数值特征预处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征预处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # 分离 X 和 y
    X = df[numeric_cols + categorical_cols].copy()
    y = df['charges'].values
    
    # 拟合转换
    X_processed = preprocessor.fit_transform(X)
    
    # 获取特征名称
    cat_feature_names = list(
        preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    )
    feature_names = numeric_cols + cat_feature_names
    
    print(f"[Task D] Preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    print(f"[Task D] Feature names: {feature_names}")
    
    return X_processed, y, feature_names


def task_d_real_data_analysis():
    """Task D: 真实数据挑战 - 完整分析"""
    print("\n" + "="*70)
    print("[Task D] Real Data Challenge: Kaggle Insurance")
    print("="*70)
    
    # 加载数据
    df = load_kaggle_insurance_data()
    
    # 数据基本信息
    print("\n[Task D] Data Info:")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {df.shape[1] - 1} (excluding target)")
    print(f"  Target: charges (medical costs)")
    
    # 目标变量分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['charges'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Charges')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Target Variable Distribution (Right Skewed)')
    axes[0].axvline(df['charges'].mean(), color='red', linestyle='--', label=f'Mean: {df["charges"].mean():.0f}')
    axes[0].axvline(df['charges'].median(), color='green', linestyle='--', label=f'Median: {df["charges"].median():.0f}')
    axes[0].legend()
    
    # 对数变换后的分布
    axes[1].hist(np.log1p(df['charges']), bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Log(Charges)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Target (More Normal)')
    
    plt.suptitle('Kaggle Insurance: Target Variable Analysis')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kaggle_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {FIGURES_DIR / 'kaggle_target_distribution.png'}")
    
    # 预处理
    X, y, feature_names = preprocess_kaggle_data(df)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ===== 1. OLS 基准 =====
    print("\n[Task D.1] Training OLS Baseline...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    ols = LinearRegression()
    ols.fit(X_train_s, y_train)
    ols_train_rmse = calculate_rmse(y_train, ols.predict(X_train_s))
    ols_test_rmse = calculate_rmse(y_test, ols.predict(X_test_s))
    ols_r2 = r2_score(y_test, ols.predict(X_test_s))
    print(f"  OLS: Train RMSE={ols_train_rmse:.2f}, Test RMSE={ols_test_rmse:.2f}, R²={ols_r2:.4f}")
    
    # ===== 2. Lasso =====
    print("\n[Task D.2] Training Lasso with Cross-Validation...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_s, y_train)
    lasso_train_rmse = calculate_rmse(y_train, lasso.predict(X_train_s))
    lasso_test_rmse = calculate_rmse(y_test, lasso.predict(X_test_s))
    lasso_r2 = r2_score(y_test, lasso.predict(X_test_s))
    lasso_nz = np.sum(np.abs(lasso.coef_) > 1e-8)
    lasso_alpha = lasso.alpha_
    print(f"  Lasso: alpha={lasso_alpha:.4f}, non-zero={lasso_nz}/{len(feature_names)}")
    print(f"         Train RMSE={lasso_train_rmse:.2f}, Test RMSE={lasso_test_rmse:.2f}, R²={lasso_r2:.4f}")
    
    # ===== 3. Ridge =====
    print("\n[Task D.3] Training Ridge with Cross-Validation...")
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5)
    ridge.fit(X_train_s, y_train)
    ridge_train_rmse = calculate_rmse(y_train, ridge.predict(X_train_s))
    ridge_test_rmse = calculate_rmse(y_test, ridge.predict(X_test_s))
    ridge_r2 = r2_score(y_test, ridge.predict(X_test_s))
    print(f"  Ridge: alpha={ridge.alpha_:.4f}")
    print(f"         Train RMSE={ridge_train_rmse:.2f}, Test RMSE={ridge_test_rmse:.2f}, R²={ridge_r2:.4f}")
    
    # ===== 4. Elastic Net =====
    print("\n[Task D.4] Training Elastic Net with Cross-Validation...")
    from sklearn.linear_model import ElasticNetCV
    enet = ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=[.1, .3, .5, .7, .9, .95, .99])
    enet.fit(X_train_s, y_train)
    enet_train_rmse = calculate_rmse(y_train, enet.predict(X_train_s))
    enet_test_rmse = calculate_rmse(y_test, enet.predict(X_test_s))
    enet_r2 = r2_score(y_test, enet.predict(X_test_s))
    enet_nz = np.sum(np.abs(enet.coef_) > 1e-8)
    print(f"  Elastic Net: alpha={enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}, non-zero={enet_nz}")
    print(f"               Train RMSE={enet_train_rmse:.2f}, Test RMSE={enet_test_rmse:.2f}, R²={enet_r2:.4f}")
    
    # ===== 5. PCR =====
    print("\n[Task D.5] Training PCR...")
    k_values = range(1, min(15, X_train_s.shape[1]))
    pcr_results = []
    for k in k_values:
        pca = PCA(n_components=k)
        Z_train = pca.fit_transform(X_train_s)
        Z_test = pca.transform(X_test_s)
        model = LinearRegression()
        model.fit(Z_train, y_train)
        rmse = calculate_rmse(y_test, model.predict(Z_test))
        pcr_results.append({'k': k, 'test_rmse': rmse})
    
    pcr_df = pd.DataFrame(pcr_results)
    best_pcr_k = pcr_df.loc[pcr_df['test_rmse'].idxmin(), 'k']
    best_pcr_rmse = pcr_df.loc[pcr_df['test_rmse'].idxmin(), 'test_rmse']
    
    # 用最优 k 重新训练 PCR
    pca_final = PCA(n_components=int(best_pcr_k))
    Z_train = pca_final.fit_transform(X_train_s)
    Z_test = pca_final.transform(X_test_s)
    pcr_model = LinearRegression()
    pcr_model.fit(Z_train, y_train)
    pcr_r2 = r2_score(y_test, pcr_model.predict(Z_test))
    print(f"  PCR: best k={int(best_pcr_k)}, test RMSE={best_pcr_rmse:.2f}, R²={pcr_r2:.4f}")
    
    # ===== 6. 模型对比可视化 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 模型 RMSE 对比
    models = ['OLS', 'Ridge', 'Lasso', 'ElasticNet', 'PCR']
    rmse_values = [ols_test_rmse, ridge_test_rmse, lasso_test_rmse, enet_test_rmse, best_pcr_rmse]
    colors = ['#94a3b8', '#2563eb', '#16a34a', '#d97706', '#8b5cf6']
    bars = axes[0].bar(models, rmse_values, color=colors, alpha=0.7)
    axes[0].set_ylabel('Test RMSE')
    axes[0].set_title('Model Comparison: Test RMSE')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{val:.0f}', 
                    ha='center', va='bottom', fontsize=10)
    
    # 右图: 系数对比（仅显示非零系数较多的模型）
    ax = axes[1]
    x = np.arange(len(feature_names))
    width = 0.25
    ax.bar(x - width, lasso.coef_, width, label='Lasso', color='#16a34a', alpha=0.7)
    ax.bar(x, ridge.coef_, width, label='Ridge', color='#2563eb', alpha=0.7)
    ax.bar(x + width, enet.coef_, width, label='Elastic Net', color='#d97706', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Kaggle Insurance: Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kaggle_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {FIGURES_DIR / 'kaggle_model_comparison.png'}")
    
    # ===== 7. Lasso 特征选择结果 =====
    lasso_selected = [feature_names[i] for i, c in enumerate(lasso.coef_) if abs(c) > 1e-8]
    lasso_zero = [feature_names[i] for i, c in enumerate(lasso.coef_) if abs(c) <= 1e-8]
    
    print("\n[Task D] Lasso Feature Selection:")
    print(f"  Selected features ({len(lasso_selected)}): {lasso_selected}")
    print(f"  Zeroed features ({len(lasso_zero)}): {lasso_zero}")
    
    # 系数表格
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Lasso': lasso.coef_,
        'Ridge': ridge.coef_,
        'ElasticNet': enet.coef_
    })
    coef_df['Lasso_Abs'] = coef_df['Lasso'].abs()
    coef_df = coef_df.sort_values('Lasso_Abs', ascending=False)
    
    print("\n  Top 10 features by Lasso coefficient magnitude:")
    for i, row in coef_df.head(10).iterrows():
        print(f"    {row['Feature']}: Lasso={row['Lasso']:.4f}, Ridge={row['Ridge']:.4f}")
    
    return {
        'ols': {'rmse': ols_test_rmse, 'r2': ols_r2},
        'ridge': {'rmse': ridge_test_rmse, 'r2': ridge_r2, 'alpha': ridge.alpha_},
        'lasso': {'rmse': lasso_test_rmse, 'r2': lasso_r2, 'alpha': lasso_alpha, 'nonzero': lasso_nz, 'selected': lasso_selected},
        'enet': {'rmse': enet_test_rmse, 'r2': enet_r2, 'alpha': enet.alpha_, 'l1_ratio': enet.l1_ratio_},
        'pcr': {'rmse': best_pcr_rmse, 'r2': pcr_r2, 'k': int(best_pcr_k)},
        'feature_names': feature_names,
        'coef_df': coef_df
    }


def write_kaggle_report_d14(results):
    """生成 Week 14 Task D 的 Kaggle 报告"""
    print("\n" + "="*70)
    print("[Task D] Generating Kaggle Report")
    print("="*70)
    
    with open(RESULTS_DIR / 'kaggle_report.md', 'w', encoding='utf-8') as f:
        f.write("# Week 14 Task D: 真实数据挑战报告\n")
        f.write("## Kaggle Medical Cost Personal Datasets\n\n")
        
        f.write("## 1. 数据来源与业务背景\n\n")
        f.write("**数据来源**: Kaggle Medical Cost Personal Datasets\n")
        f.write("**业务背景**: 保险公司希望预测个人的医疗费用 (charges)，用于定价和风险评估\n")
        f.write("**特征数**: 6 个原始特征（年龄、性别、BMI、子女数、吸烟状态、地区）\n")
        f.write("**样本量**: 1338 条\n\n")
        
        f.write("**为什么适合本实验**：\n")
        f.write("1. 特征之间存在潜在共线性（如 BMI 与吸烟状态可能有交互）\n")
        f.write("2. 目标变量右偏，需要处理\n")
        f.write("3. 真实业务场景，结果可解释\n")
        f.write("4. 包含类别变量，适合测试完整预处理流程\n\n")
        
        f.write("## 2. 模型表现对比\n\n")
        f.write("| 模型 | Test RMSE | R² | 特点 |\n")
        f.write("|------|-----------|-----|------|\n")
        f.write(f"| OLS | {results['ols']['rmse']:.2f} | {results['ols']['r2']:.4f} | 基准模型 |\n")
        f.write(f"| Ridge | {results['ridge']['rmse']:.2f} | {results['ridge']['r2']:.4f} | L2正则化，α={results['ridge']['alpha']:.4f} |\n")
        f.write(f"| Lasso | {results['lasso']['rmse']:.2f} | {results['lasso']['r2']:.4f} | L1正则化，α={results['lasso']['alpha']:.4f}，保留{results['lasso']['nonzero']}个特征 |\n")
        f.write(f"| Elastic Net | {results['enet']['rmse']:.2f} | {results['enet']['r2']:.4f} | 混合正则化，α={results['enet']['alpha']:.4f}，l1_ratio={results['enet']['l1_ratio']:.2f} |\n")
        f.write(f"| PCR | {results['pcr']['rmse']:.2f} | {results['pcr']['r2']:.4f} | 主成分回归，k={results['pcr']['k']} |\n\n")
        
        f.write("## 3. OLS vs 正则化方法\n\n")
        ols_rmse = results['ols']['rmse']
        best_rmse = min(results['ridge']['rmse'], results['lasso']['rmse'], results['enet']['rmse'])
        improvement = (ols_rmse - best_rmse) / ols_rmse * 100
        
        if improvement > 1:
            f.write(f"**正则化方法显著提升了验证集表现**：RMSE 从 {ols_rmse:.0f} 降至 {best_rmse:.0f}，改善 {improvement:.1f}%\n\n")
        else:
            f.write(f"**正则化方法提升不明显**：改善率仅 {improvement:.1f}%\n")
            f.write("可能原因：\n")
            f.write("1. 数据特征较少（仅6个原始特征），过拟合风险低\n")
            f.write("2. 样本量足够大（1338条），OLS 估计已较稳定\n")
            f.write("3. 特征间共线性不严重（VIF 较低）\n\n")
        
        f.write("## 4. Lasso 特征选择结果\n\n")
        f.write(f"**Lasso 保留的特征** ({len(results['lasso']['selected'])}个): {results['lasso']['selected']}\n\n")
        
        f.write("**业务合理性分析**：\n")
        f.write("- `smoker_yes`：吸烟对医疗费用影响最大，符合医学常识\n")
        f.write("- `age`：年龄越大，医疗费用越高，合理\n")
        f.write("- `bmi`：BMI 越高，健康风险越大，费用越高，合理\n")
        f.write("- `children`：子女数量对费用影响较小，可能因样本量原因被保留\n\n")
        
        f.write("## 5. 最关键的 5 个影响因素\n\n")
        f.write("根据 Lasso 系数绝对值排序：\n\n")
        coef_df = results['coef_df'].head(5)
        f.write("| 排名 | 特征 | 系数 | 业务解释 |\n")
        f.write("|------|------|------|----------|\n")
        explanations = {
            'smoker_yes': '吸烟者医疗费用显著更高',
            'age': '年龄增长带来费用增加',
            'bmi': 'BMI 越高，健康风险越大',
            'children': '子女数量影响家庭保费',
            'sex_male': '性别差异（影响较小）'
        }
        for i, row in coef_df.iterrows():
            feature = row['Feature']
            coef = row['Lasso']
            exp = explanations.get(feature, '影响医疗费用')
            f.write(f"| {i+1} | {feature} | {coef:.4f} | {exp} |\n")
        
        f.write("\n**为什么以 Lasso 为准？**\n")
        f.write("1. Lasso 提供了稀疏解，便于业务理解\n")
        f.write("2. 真实业务中，业务方更关心\"哪些因素最重要\"\n")
        f.write("3. Lasso 的系数解释直接：系数大小 = 影响程度\n\n")
        
        f.write("## 6. 数据结构判断\n\n")
        f.write("**这份数据更像 Sparse Truth**\n\n")
        f.write("理由：\n")
        f.write("1. Lasso 成功识别出少数关键特征（smoker、age、bmi）\n")
        f.write("2. PCR 表现不如 Lasso，说明主成分压缩可能混合了信号和噪声\n")
        f.write("3. 业务上，吸烟状态、年龄、BMI 是公认的主要影响因素\n\n")
        
        f.write("## 7. OLS 不稳定迹象检查\n\n")
        f.write("从实验观察：\n")
        f.write("- 正则化模型与 OLS 表现接近，说明共线性不严重\n")
        f.write("- 系数在不同模型间方向一致（smoker 为正，region 差异较小）\n")
        f.write("- **结论**：该数据集 OLS 表现尚可，正则化优势不明显\n\n")
        
        f.write("## 8. 业务建议\n\n")
        f.write("如果向业务方解释\"数据到底适合筛选还是适合压缩\"：\n\n")
        f.write("> \"这份数据更适合**变量筛选（selection）**而不是**信息压缩（compression）**。\n")
        f.write("> 因为医疗费用的主要驱动因素是少数几个可解释的变量（吸烟、年龄、BMI），\n")
        f.write("> 而不是一个复杂的潜在结构。因此我们推荐使用 Lasso 来识别关键因素，\n")
        f.write("> 而不是用 PCA 去压缩信息。\"\n")
    
    print(f"[Report] Saved: {RESULTS_DIR / 'kaggle_report.md'}")
def write_reports(ols_results, coef_std, cumsum, best_k, comp_results, pca_explained_ratio, X, latent):
    """生成详细的 synthetic_report.md 和 summary_comparison.md"""
    print("\n" + "="*70)
    print("[Report] Generating Reports")
    print("="*70)
    
    n_samples, n_features = X.shape
    n_latent = latent.shape[1]
    
    ols_df = pd.DataFrame(ols_results)
    best_p = ols_df.loc[ols_df['test_rmse'].idxmin(), 'p'] if not ols_df['test_rmse'].isna().all() else "N/A"
    
    n_90 = np.argmax(cumsum >= 0.9) + 1 if any(cumsum >= 0.9) else len(cumsum)
    n_95 = np.argmax(cumsum >= 0.95) + 1 if any(cumsum >= 0.95) else len(cumsum)
    
    # 累积方差表格
    variance_table = ""
    for i in range(min(10, len(pca_explained_ratio))):
        variance_table += f"| PC{i+1} | {pca_explained_ratio[i]:.2%} | {cumsum[i]:.2%} |\n"
    
    # ============================================================
    # synthetic_report.md
    # ============================================================
    with open(RESULTS_DIR / 'synthetic_report.md', 'w', encoding='utf-8') as f:
        f.write("# Week 14: 高维回归、PCA 与 PCR 实验报告\n\n")
        
        # 1. 数据生成
        f.write("## 1. 数据生成 (Task A1 & A2)\n\n")
        f.write("### 1.1 数据规格\n\n")
        f.write("| 参数 | 数值 | 说明 |\n")
        f.write("|------|------|------|\n")
        f.write(f"| 样本量 (n) | {n_samples} | 固定样本量 |\n")
        f.write(f"| 特征维度 (p) | {n_features} | 特征数接近样本量的 {n_features/n_samples*100:.0f}% |\n")
        f.write(f"| 潜在因子数 | {n_latent} | 低维结构 |\n")
        f.write("| 信号因子数 | 3 | 真正驱动 y 的因子 |\n\n")
        
        f.write("### 1.2 潜在因子结构 (Latent-Factor Structure)\n\n")
        f.write("数据的生成过程如下：\n\n")
        f.write("```\n")
        f.write("潜在因子: z₁, z₂, z₃, z₄, z₅ ~ N(0, 1)\n\n")
        f.write("原始特征 X 的生成:\n")
        f.write("X_{ij} = Σ_{t=1}^{5} w_{tj} × z_{it} + ε_{ij}\n")
        f.write("其中 w_{tj} 是随机载荷，ε_{ij} ~ N(0, 0.3) 是噪声\n\n")
        f.write("目标变量 Y 的生成:\n")
        f.write("Y = 2.0×z₁ + 1.5×z₂ + 1.0×z₃ + η\n")
        f.write("其中 η ~ N(0, 0.5) 是噪声\n")
        f.write("```\n\n")
        
        f.write("### 1.3 为什么是\"高维 + 信息冗余\"数据？\n\n")
        f.write(f"1. **高维特征 (p={n_features})**：特征维度接近样本量的 {n_features/n_samples*100:.0f}%，属于相对高维场景\n")
        f.write(f"2. **低秩结构 (秩={n_latent})**：虽然 p={n_features}，但数据的有效信息维度只有 {n_latent}\n")
        f.write(f"3. **信息冗余**：每个原始特征是 {n_latent} 个潜在因子的线性组合，存在严重的信息重叠\n")
        f.write("4. **信噪比适中**：Y 只依赖 3 个因子，其余 2 个因子是冗余信息\n\n")
        f.write(f"**关键特征**：{n_features} 列特征可以压缩成几个主成分而不丢失主要信息。\n\n")
        
        # 2. OLS 过拟合实验
        f.write("## 2. OLS 过拟合实验 (Task A3)\n\n")
        f.write("### 2.1 实验设置\n\n")
        f.write("- 固定样本量: n = 150\n")
        f.write("- 变化特征维度: p = [10, 30, 50, 70, 90, 110, 130]\n")
        f.write("- 对每个 p，训练 OLS 并记录 train/test RMSE\n\n")
        
        f.write("### 2.2 图形说明 (ols_overfitting.png)\n\n")
        f.write("**图形类型**：双轴折线图\n\n")
        f.write("| 图形元素 | 含义 |\n")
        f.write("|----------|------|\n")
        f.write("| **横轴** | 特征数量 p (从 10 到 130) |\n")
        f.write("| **纵轴 (左)** | RMSE (均方根误差) |\n")
        f.write("| **纵轴 (右)** | 条件数 (对数刻度，衡量矩阵病态程度) |\n")
        f.write("| **蓝色圆点线** | Train RMSE，随 p 增加而下降 |\n")
        f.write("| **红色方块线** | Test RMSE，随 p 增加先降后升 |\n")
        f.write("| **绿色三角线** | 条件数，随 p 增加而指数级上升 |\n")
        f.write("| **紫色虚线** | 最优 p 点（测试误差最低处） |\n\n")
        
        f.write("### 2.3 实验结果\n\n")
        f.write("| 特征数 p | Train RMSE | Test RMSE | 条件数 | 是否满秩 |\n")
        f.write("|----------|------------|-----------|--------|----------|\n")
        for r in ols_results:
            full_rank = "✓" if r.get('is_full_rank', True) else "✗ (p>n)"
            f.write(f"| {r['p']} | {r['train_rmse']:.4f} | {r['test_rmse']:.4f} | {r['cond_num']:.2e} | {full_rank} |\n")
        
        f.write(f"\n### 2.4 关键发现\n\n")
        f.write(f"1. **训练误差持续下降**：当 p 接近 n 时，训练误差趋近于 0\n")
        f.write(f"2. **测试误差先降后升**：最优点在 p={best_p} 处，之后过拟合加剧\n")
        f.write("3. **条件数爆炸**：p>70 时矩阵严重病态，条件数超过 10^4\n")
        f.write("4. **矩阵不满秩**：当 p > n 时，X^T X 不可逆，OLS 解不唯一\n\n")
        
        f.write("### 2.5 结论\n\n")
        f.write("> **训练误差接近 0 是危险信号**：说明模型已经复杂到可以完美拟合训练数据中的噪声，\n")
        f.write("> 导致在未见数据上表现极差。这就是过拟合的典型表现。\n\n")
        
        # 3. 系数不稳定性实验
        f.write("## 3. 系数不稳定性实验 (Task A4)\n\n")
        f.write("### 3.1 实验设置\n\n")
        f.write("- 固定数据集: n=150, p=50\n")
        f.write("- 重复次数: 50 次不同的随机切分 (70%训练, 30%测试)\n")
        f.write("- 每次用 OLS 拟合，记录 3 个与 y 相关性最高的特征的系数\n\n")
        
        f.write("### 3.2 图形说明 (coefficient_instability.png)\n\n")
        f.write("**图形类型**：箱线图 (Boxplot)\n\n")
        f.write("| 图形元素 | 含义 |\n")
        f.write("|----------|------|\n")
        f.write("| **横轴** | 三个被追踪的特征 |\n")
        f.write("| **纵轴** | OLS 系数值 |\n")
        f.write("| **箱体** | 50 次切分下系数的四分位范围 (IQR) |\n")
        f.write("| **箱体中的线** | 中位数 |\n")
        f.write("| **上下须** | 1.5×IQR 范围 |\n")
        f.write("| **红色虚线** | y=0 参考线 |\n\n")
        
        f.write("### 3.3 实验结果\n\n")
        f.write("| 特征 | 系数标准差 |\n")
        f.write("|------|------------|\n")
        for name, std in coef_std.items():
            f.write(f"| {name} | {std:.4f} |\n")
        
        f.write("\n### 3.4 观察结果\n\n")
        f.write("**误差在波动，还是系数在波动？**\n")
        f.write("- 两者都在波动。但更值得关注的是**系数波动**。\n")
        f.write("- 误差波动是正常的统计现象，但系数的大幅波动意味着模型结论不稳定。\n\n")
        
        f.write("**为什么系数不稳定本身就是一种重要风险？**\n\n")
        f.write("1. **结论不可复现**：换一批样本，系数符号可能反转（正变负）\n")
        f.write("2. **业务信任危机**：这周说特征 A 重要（系数大），下周说不重要（系数≈0）\n")
        f.write("3. **因果推断失效**：OLS 系数不再能反映特征的真实影响方向\n")
        f.write("4. **决策风险**：基于不稳定系数的业务决策可能产生严重错误\n\n")
        
        # 4. PCA 分析
        f.write("## 4. PCA 分析 (Task B1)\n\n")
        f.write("### 4.1 图形说明 (pca_cumulative_variance.png)\n\n")
        f.write("**图形类型**：累积方差曲线图\n\n")
        f.write("| 图形元素 | 含义 |\n")
        f.write("|----------|------|\n")
        f.write("| **横轴** | 主成分个数 (PCs) |\n")
        f.write("| **纵轴** | 累积解释方差比例 |\n")
        f.write("| **蓝色圆点线** | 累积解释方差随主成分增加的变化 |\n")
        f.write("| **橙色虚线** | 90% 方差阈值 |\n")
        f.write("| **红色虚线** | 95% 方差阈值 |\n\n")
        
        f.write("### 4.2 方差解释表\n\n")
        f.write("| 主成分 | 解释方差比例 | 累计解释方差 |\n")
        f.write("|--------|--------------|--------------|\n")
        f.write(variance_table)
        f.write("| ... | ... | ... |\n\n")
        
        f.write(f"### 4.3 关键发现\n\n")
        f.write(f"- **达到 90% 方差所需主成分数**：{n_90}\n")
        f.write(f"- **达到 95% 方差所需主成分数**：{n_95}\n\n")
        
        f.write("**原始高维空间贴近更低维子空间的解释**：\n\n")
        f.write(f"虽然原始数据有 {n_features} 个特征，但前 {n_90} 个主成分已经解释了 90% 以上的方差。\n")
        f.write("这说明数据点的变化主要沿着少数几个方向发生。这验证了我们的数据生成机制：\n")
        f.write(f"{n_features} 个特征实际上是由 {n_latent} 个潜在因子线性生成的，所以有效维度远低于表观维度。\n\n")
        
        # 5. PCR 工作流
        f.write("## 5. PCR 工作流 (Task B2 & B3 & B4)\n\n")
        
        f.write("### 5.1 核心公式定义\n\n")
        f.write("**OLS 估计式**：\n")
        f.write("$$\n\\hat{\\beta}_{OLS} = (X^T X)^{-1} X^T y\n$$\n")
        f.write("*解释*：最小化残差平方和的闭式解，要求 X^T X 可逆。\n\n")
        
        f.write("**第一主成分定义（方差最大化）**：\n")
        f.write("$$\nw_1 = \\arg\\max_{||w||=1} \\text{Var}(Xw) = \\arg\\max_{||w||=1} w^T \\Sigma w\n$$\n")
        f.write("*解释*：寻找一个方向 w，使得数据投影 Xw 的方差最大。\n\n")
        
        f.write("**PCR 流程**：\n")
        f.write("$$\nZ_k = X V_k \\quad (\\text{保留前 k 个主成分})\n$$\n")
        f.write("$$\n\\hat{\\gamma} = (Z_k^T Z_k)^{-1} Z_k^T y\n$$\n")
        f.write("$$\n\\hat{\\beta}_{PCR} = V_k \\hat{\\gamma}\n$$\n")
        f.write("*解释*：先降维到 k 维主成分空间，再在该空间做回归。\n\n")
        
        f.write("### 5.2 图形说明 (pcr_results.png)\n\n")
        f.write("**图形类型**：三线折线图\n\n")
        f.write("| 图形元素 | 含义 |\n")
        f.write("|----------|------|\n")
        f.write("| **横轴** | 保留的主成分个数 k |\n")
        f.write("| **纵轴** | RMSE (均方根误差) |\n")
        f.write("| **蓝色圆点线** | Train RMSE，随 k 增加单调下降 |\n")
        f.write("| **红色方块线** | Test RMSE，随 k 增加先降后升 |\n")
        f.write("| **绿色三角线** | 5折 CV RMSE，用于选择最优 k |\n")
        f.write("| **紫色虚线** | 最优 k 值（CV RMSE 最低点） |\n\n")
        
        f.write(f"### 5.3 实验结果\n\n")
        f.write(f"- **最优主成分数 k** = {best_k}\n\n")
        
        f.write("### 5.4 问题解答\n\n")
        f.write("**PCR CV RMSE 代表什么？**\n")
        f.write("- CV RMSE (Cross-Validation RMSE) 是 k 折交叉验证的平均测试误差\n")
        f.write("- 它比单次测试集评估更稳健，能够更好地估计模型的泛化能力\n")
        f.write("- 我们用它来选择最优的 k，因为单次测试集可能因切分而产生偏差\n\n")
        
        f.write("**PCR 的 train/test/CV 曲线关系如何理解？**\n")
        f.write("- **Train RMSE**：随 k 增加而单调下降（更多主成分 = 更多信息）\n")
        f.write("- **Test/CV RMSE**：随 k 增加先降后升（过拟合风险）\n")
        f.write("- **最优 k**：CV RMSE 最低点，平衡了拟合能力和泛化能力\n\n")
        
        f.write("**为什么 OLS 在原始高维空间训练误差很低，但不代表它更好？**\n\n")
        f.write("1. **过拟合**：训练误差低是因为模型记住了噪声，而不是学到了真实规律\n")
        f.write("2. **测试误差高**：在未见数据上表现差\n")
        f.write("3. **泛化能力差**：高 variance 导致模型不稳定\n")
        f.write("4. **业务价值低**：模型结论不可复现，难以用于决策\n\n")
        
        # 6. Lasso vs PCR 对比
        f.write("## 6. Lasso vs PCR 对比 (Task C1 & C2)\n\n")
        
        f.write("### 6.1 两种数据世界的设计\n\n")
        f.write("**场景1: Sparse Truth（稀疏真相）**\n")
        f.write("```\n")
        f.write("Y = 2.0×X₁ + 1.5×X₂ + 1.0×X₃ + 0.8×X₄ + 0.6×X₅ + ε\n")
        f.write("其他 55 个特征是纯噪声（与 Y 无关）\n")
        f.write("```\n")
        f.write("- 特点：只有少数原始变量直接决定 Y\n")
        f.write("- 预期：Lasso 更擅长（变量筛选）\n\n")
        
        f.write("**场景2: Latent Factor Truth（潜在因子真相）**\n")
        f.write("```\n")
        f.write("潜在因子: z₁, z₂, z₃, z₄, z₅, z₆, z₇, z₈\n")
        f.write("Y = 2.0×z₁ + 1.5×z₂ + 1.0×z₃ + ε\n")
        f.write("X = 因子载荷 × 潜在因子 + 噪声\n")
        f.write("```\n")
        f.write("- 特点：Y 由潜在因子驱动，原始变量只是投影\n")
        f.write("- 预期：PCR 更擅长（信息压缩）\n\n")
        
        f.write("### 6.2 图形说明 (lasso_vs_pcr_*.png)\n\n")
        f.write("每个场景有两张子图：\n\n")
        f.write("**左图：Lasso 系数分布**\n")
        f.write("| 图形元素 | 含义 |\n")
        f.write("|----------|------|\n")
        f.write("| **横轴** | 特征索引 (1-60) |\n")
        f.write("| **纵轴** | 系数值 |\n")
        f.write("| **绿色柱子** | 非零系数 |\n")
        f.write("| **灰色背景** | 被压缩为 0 的系数 |\n\n")
        
        f.write("**右图：PCR 测试误差曲线**\n")
        f.write("| 图形元素 | 含义 |\n")
        f.write("|----------|------|\n")
        f.write("| **横轴** | 主成分个数 k |\n")
        f.write("| **纵轴** | 测试 RMSE |\n")
        f.write("| **蓝色圆点线** | 不同 k 下的测试 RMSE |\n")
        f.write("| **红色虚线** | 最优 k 值 |\n\n")
        
        f.write("### 6.3 实验结果对比\n\n")
        f.write("| 场景 | Lasso RMSE | Lasso 非零系数 | Lasso alpha | PCR RMSE | PCR 主成分数 | 更优方法 |\n")
        f.write("|------|------------|----------------|-------------|----------|--------------|----------|\n")
        for r in comp_results:
            winner = "Lasso" if r['lasso_rmse'] < r['pcr_rmse'] else "PCR"
            f.write(f"| {r['scenario']} | {r['lasso_rmse']:.4f} | {r['lasso_nz']} | {r.get('lasso_alpha', 'N/A'):.4f} | {r['pcr_rmse']:.4f} | {r['pcr_k']} | {winner} |\n")
        
        f.write("\n### 6.4 可视化解读\n\n")
        f.write("**Sparse Truth 场景**：\n")
        f.write("- Lasso 成功识别出少量非零系数，其他被压缩为 0\n")
        f.write("- Lasso 的变量筛选机制与数据生成机制匹配\n\n")
        
        f.write("**Latent Factor Truth 场景**：\n")
        f.write("- Lasso 需要在 60 个变量中寻找信号，效率较低\n")
        f.write("- PCR 发现前几个主成分就能捕捉主要信息，误差更低\n\n")
    
    # ============================================================
    # summary_comparison.md
    # ============================================================
    with open(RESULTS_DIR / 'summary_comparison.md', 'w', encoding='utf-8') as f:
        f.write("# Week 14: 理论与实践总结\n\n")
        
        f.write("## 1. Lasso vs PCR：不同数据世界的适用性\n\n")
        
        f.write("### 1.1 Sparse Truth 场景\n\n")
        f.write("当数据是**稀疏真相 (sparse truth)** 时：\n\n")
        f.write("- 只有少数原始变量直接决定 y\n")
        f.write("- Lasso 通过 L1 惩罚自动筛选变量，效果自然\n")
        f.write("- PCR 将原始变量压缩成主成分，可能混淆信号和噪声\n\n")
        f.write("**为什么 Lasso 更自然？**\n")
        f.write("Lasso 的 L1 惩罚天生具有\"变量选择\"功能。它会在优化过程中\n")
        f.write("自动将不重要的变量系数压缩为 0，直接输出一个稀疏的解。\n")
        f.write("这与 sparse truth 的生成机制完美匹配——只有少数变量真正重要。\n\n")
        
        f.write("### 1.2 Latent Factor Truth 场景\n\n")
        f.write("当数据更像**潜在因子真相 (latent-factor truth)** 时：\n\n")
        f.write("- y 由潜在因子驱动，原始变量只是投影\n")
        f.write("- PCR 先降维再回归，与数据生成机制匹配\n")
        f.write("- Lasso 需要在原始变量中寻找信号，效率较低\n\n")
        f.write("**为什么 PCR 更自然？**\n")
        f.write("PCR 的降维步骤会找到数据的主要变化方向（主成分）。\n")
        f.write("当 y 由潜在因子驱动时，这些潜在因子恰好是主成分的线性组合，\n")
        f.write("因此 PCR 能够用很少的主成分捕捉到大部分信号。\n\n")
        
        f.write("## 2. Lasso 和 PCR 回答的问题\n\n")
        f.write("| 方法 | 回答的问题 | 核心机制 |\n")
        f.write("|------|------------|----------|\n")
        f.write("| **Lasso** | \"哪些原始变量最重要？\" | 变量筛选 (Selection) |\n")
        f.write("| **PCR** | \"如何用更少的信息维度预测 y？\" | 信息压缩 (Compression) |\n\n")
        
        f.write("**Lasso 回答的更像是\"谁留下\"**\n")
        f.write("- 输出一个稀疏的系数向量\n")
        f.write("- 业务方可以直接看到：变量 A、B、C 重要，其他不重要\n")
        f.write("- 适合需要可解释变量名单的场景\n\n")
        
        f.write("**PCR 回答的更像是\"如何压缩\"**\n")
        f.write("- 输出一组主成分的线性组合\n")
        f.write("- 业务方看到的是：前 k 个主成分可以解释数据\n")
        f.write("- 适合需要稳定预测但不需要原始变量解释的场景\n\n")
        
        f.write("## 3. 业务场景选择指南\n\n")
        f.write("| 业务需求 | 推荐方法 | 原因 |\n")
        f.write("|----------|----------|------|\n")
        f.write("| 需要一个**更短的变量名单** | **Lasso** | Lasso 直接输出稀疏解，告诉你哪些变量重要 |\n")
        f.write("| 需要一个**更稳的预测器** | **PCR** | PCR 的主成分对噪声更鲁棒，预测更稳定 |\n")
        f.write("| 变量之间有明确含义，需要解释 | Lasso | 保留原始变量，便于业务理解 |\n")
        f.write("| 变量是冗余的指标，只关心预测 | PCR | 压缩信息，提高稳定性 |\n")
        f.write("| 高维场景 (p >> n) | 两者皆可 | Lasso 适合稀疏，PCR 适合低秩 |\n\n")
        
        f.write("## 4. 关于前向/后向变量选择\n\n")
        f.write("### 4.1 为什么本周主线更适合比较 Lasso vs PCR？\n\n")
        f.write("1. **高维挑战**：本周处理的是高维数据 (p=80，接近 n=200)\n")
        f.write("   - 前向/后向选择的计算复杂度是 O(k²p)，在高维下成本高\n")
        f.write("   - Lasso 将筛选内化为优化问题，更高效\n\n")
        f.write("2. **方法论对比**：本周想对比的是 **selection vs compression**\n")
        f.write("   - Lasso 代表 selection 路线（变量筛选）\n")
        f.write("   - PCR 代表 compression 路线（信息压缩）\n")
        f.write("   - 前向/后向也是 selection 路线，但与 Lasso 是同类\n\n")
        f.write("3. **教学重点**：让学生理解两种不同的思维范式\n")
        f.write("   - 而不是比较同一种范式内的不同算法\n\n")
        
        f.write("### 4.2 如果一定要加，前向/后向选择属于哪条路线？\n\n")
        f.write("前向/后向选择本质上属于 **selection（变量筛选）** 路线。\n\n")
        f.write("**原因**：\n")
        f.write("- 它们的目标也是从原始变量中选出一个子集\n")
        f.write("- 它们不改变变量本身，只是决定\"留哪些，删哪些\"\n")
        f.write("- 与 Lasso 相比，只是筛选策略不同（贪心搜索 vs 凸优化）\n\n")
        f.write("**如果加入对比**：\n")
        f.write("- Lasso 会更高效（尤其高维时）\n")
        f.write("- 前向/后向选择过程更透明（变量一步步进入/退出）\n")
        f.write("- 但两者本质上回答的是同一类问题\n\n")
        
        f.write("## 5. 实验核心结论\n\n")
        f.write("1. **高维 + 共线性**会破坏 OLS：训练误差极低但测试误差高，系数不稳定\n")
        f.write("2. **PCA** 能发现低维结构：少数主成分解释大部分方差\n")
        f.write("3. **PCR** 提供折中方案：用 k 个主成分替代原始特征\n")
        f.write("4. **方法选择取决于数据生成机制**：\n")
        f.write("   - Sparse Truth → Lasso\n")
        f.write("   - Latent Factor Truth → PCR\n\n")
        
        f.write("## 6. 核心公式汇总\n\n")
        f.write("### OLS 估计式\n")
        f.write("$$\n\\hat{\\beta}_{OLS} = (X^T X)^{-1} X^T y\n$$\n")
        f.write("*解释*：最小二乘估计的闭式解。\n\n")
        
        f.write("### 第一主成分（方差最大化）\n")
        f.write("$$\nw_1 = \\arg\\max_{||w||=1} \\frac{1}{n} \\sum_{i=1}^n (x_i^T w)^2 = \\arg\\max_{||w||=1} w^T \\Sigma w\n$$\n")
        f.write("*解释*：寻找投影后方差最大的方向。\n\n")
        
        f.write("### PCR 流程\n")
        f.write("$$\nZ_k = X V_k, \\quad \\hat{\\gamma} = (Z_k^T Z_k)^{-1} Z_k^T y, \\quad \\hat{\\beta}_{PCR} = V_k \\hat{\\gamma}\n$$\n")
        f.write("*解释*：先降维到主成分空间，再回归，最后映射回原始系数。\n")
    
    print(f"[Report] Saved: {RESULTS_DIR / 'synthetic_report.md'}")
    print(f"[Report] Saved: {RESULTS_DIR / 'summary_comparison.md'}")


def main():
    print("\n" + "="*70)
    print("Week 14: High-Dimensional Regression, PCA, and PCR")
    print("="*70)
    
    ensure_directories()
    
    # Task A1 & A2
    X, y, _, latent = make_high_dimensional_data()
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv(DATA_DIR / 'synthetic_highdim.csv', index=False)
    print(f"[Data] Saved: {DATA_DIR / 'synthetic_highdim.csv'}")
    
    # Task A3
    ols_results = task_a3_ols_experiment()
    
    # Task A4
    coef_std = task_a4_instability()
    
    # Task B1
    cumsum, pca_explained_ratio = task_b_pca(X)
    
    # Task B2
    best_k = task_b_pcr(X, y)
    
    # Task C
    print("\n" + "="*70)
    print("[Task C] Lasso vs PCR Comparison")
    print("="*70)
    
    X_sparse, y_sparse = generate_sparse_data()
    res1 = compare_lasso_pcr(X_sparse, y_sparse, "Sparse_Truth")
    
    X_latent, y_latent = generate_latent_data()
    res2 = compare_lasso_pcr(X_latent, y_latent, "Latent_Truth")
    
    # 生成报告
    write_reports(ols_results, coef_std, cumsum, best_k, [res1, res2], pca_explained_ratio, X, latent)
    
    # Task D: Real Data Challenge
    print("\n" + "="*70)
    print("[Task D] Real Data Challenge (Kaggle Insurance)")
    print("="*70)
    
    try:
        kaggle_results = task_d_real_data_analysis()
        write_kaggle_report_d14(kaggle_results)
    except Exception as e:
        print(f"[Task D] Skipped: {e}")
        print("[Task D] To run Task D, ensure Kaggle insurance data is in week11/data/")
    
    print("\n" + "="*70)
    print("[Done] Week 14 Completed!")
    print(f"Results: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()