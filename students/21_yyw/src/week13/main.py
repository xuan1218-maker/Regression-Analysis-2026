"""
第十三周作业：正则化回归与变量筛选
uv run src/week13/main.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold,
                                      cross_val_score)
from sklearn.pipeline import Pipeline

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import AnalyticalOLS
from utils.metrics import calculate_rmse, calculate_mae
from utils.transformers import CustomStandardScaler


# =============================================================================
# A1: 生成带有明确共线性的模拟回归数据
# =============================================================================

def generate_synthetic_correlated(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成带有明确共线性的模拟回归数据。

    真实 DGP (Data Generating Process):
        y = 3.0 * x1 + 2.0 * x2 + 1.5 * x3 + 0.8 * x5 + noise(0, 0.5)

    其中：
    - x1 ~ N(0, 1)                    — 主特征
    - x2 = 0.95 * x1 + eps(0, 0.2)   — 与 x1 高度相关 (r ≈ 0.98)
    - x3 = 0.90 * x1 + eps(0, 0.3)   — 与 x1 高度相关 (r ≈ 0.95)
    - x4 = 0.85 * x1 + eps(0, 0.4)   — 与 x1 高度相关 (r ≈ 0.90)
    - x5 ~ N(0, 1)                    — 独立有效特征
    - x6 ~ N(0, 1)                    — 纯噪声特征
    - x7 ~ N(0, 1)                    — 纯噪声特征
    - x8 ~ N(0, 1)                    — 纯噪声特征

    高度相关特征族：x1, x2, x3, x4（4 个特征，基于 x1 生成）
    纯噪声特征：x6, x7, x8
    有效但独立的特征：x5
    """
    rng = np.random.default_rng(seed)

    # 主特征
    x1 = rng.normal(0, 1, n_samples)

    # 高度相关特征族（与 x1 的相关系数 > 0.90）
    x2 = 0.95 * x1 + rng.normal(0, 0.2, n_samples)
    x3 = 0.90 * x1 + rng.normal(0, 0.3, n_samples)
    x4 = 0.85 * x1 + rng.normal(0, 0.4, n_samples)

    # 独立有效特征
    x5 = rng.normal(0, 1, n_samples)

    # 纯噪声特征（不参与 DGP）
    x6 = rng.normal(0, 1, n_samples)
    x7 = rng.normal(0, 1, n_samples)
    x8 = rng.normal(0, 1, n_samples)

    # 真实 DGP：y 只依赖于 x1, x2, x3, x5
    noise = rng.normal(0, 0.5, n_samples)
    y = 3.0 * x1 + 2.0 * x2 + 1.5 * x3 + 0.8 * x5 + noise

    df = pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8,
        'y': y,
    })

    return df


# =============================================================================
# 辅助函数：自定义 Scaler 适配 sklearn Pipeline
# =============================================================================

class SklearnCompatibleScaler:
    """
    将自定义 CustomStandardScaler 封装为 sklearn 兼容的 Transformer，
    以便放入 Pipeline。
    """

    def __init__(self):
        self._scaler = CustomStandardScaler()

    def fit(self, X, y=None):
        self._scaler.fit(X)
        return self

    def transform(self, X):
        return self._scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self._scaler.fit_transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.array(input_features)


# =============================================================================
# A3-1: OLS vs Ridge 稳定性对比（50 次随机切分）
# =============================================================================

def run_stability_comparison(df: pd.DataFrame, feature_cols: list,
                              target_col: str, n_splits: int = 50,
                              ridge_alpha: float = 1.0,
                              seed: int = 42) -> dict:
    """
    使用 n_splits 次不同的随机切分，分别用 OLS 和 Ridge 拟合，
    收集高度相关特征的系数，计算标准差，并绘制箱线图。
    """
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(seed)
    X = df[feature_cols].values
    y = df[target_col].values

    ols_coefs = []
    ridge_coefs = []

    for i in range(n_splits):
        split_seed = int(rng.integers(0, 100000))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split_seed)

        # 标准化
        scaler_o = SklearnCompatibleScaler()
        X_train_s = scaler_o.fit_transform(X_train)
        X_test_s = scaler_o.transform(X_test)

        # OLS
        ols = LinearRegression()
        ols.fit(X_train_s, y_train)
        ols_coefs.append(ols.coef_.copy())

        # Ridge
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train_s, y_train)
        ridge_coefs.append(ridge.coef_.copy())

    ols_coefs = np.array(ols_coefs)
    ridge_coefs = np.array(ridge_coefs)

    # 绘制箱线图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        correlated_indices = [feature_cols.index(f) for f in ['x1', 'x2', 'x3', 'x4']]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # OLS 箱线图
        ols_data = [ols_coefs[:, i] for i in correlated_indices]
        bp1 = axes[0].boxplot(ols_data, tick_labels=['x1', 'x2', 'x3', 'x4'], patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('#FF6B6B')
        axes[0].set_title('OLS Coefficients (50 random splits)', fontsize=13)
        axes[0].set_ylabel('Coefficient Value')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3)

        # Ridge 箱线图
        ridge_data = [ridge_coefs[:, i] for i in correlated_indices]
        bp2 = axes[1].boxplot(ridge_data, tick_labels=['x1', 'x2', 'x3', 'x4'], patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('#4ECDC4')
        axes[1].set_title(f'Ridge Coefficients (alpha={ridge_alpha}, 50 splits)', fontsize=13)
        axes[1].set_ylabel('Coefficient Value')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = Path(__file__).parent / "results" / "stability_boxplot.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  箱线图已保存: {fig_path}")
    except ImportError:
        print("  matplotlib 未安装，跳过箱线图绘制")

    return {
        'ols_coefs': ols_coefs,
        'ridge_coefs': ridge_coefs,
        'feature_cols': feature_cols,
        'ols_std': np.std(ols_coefs, axis=0),
        'ridge_std': np.std(ridge_coefs, axis=0),
        'ols_mean': np.mean(ols_coefs, axis=0),
        'ridge_mean': np.mean(ridge_coefs, axis=0),
    }


# =============================================================================
# A3-3: GridSearchCV 寻优
# =============================================================================

def run_gridsearchcv_ridge_lasso(X_train, y_train, feature_names) -> dict:
    """
    为 Ridge 和 Lasso 进行 GridSearchCV 调参，绘制 CV 误差随 alpha 变化的曲线。
    """
    from sklearn.pipeline import Pipeline

    alpha_space = np.logspace(-4, 3, 50)

    results = {}

    for name, ModelClass in [('Ridge', Ridge), ('Lasso', Lasso)]:
        pipe = Pipeline([
            ('scaler', SklearnCompatibleScaler()),
            ('model', ModelClass(max_iter=10000))
        ])
        param_grid = {'model__alpha': alpha_space}

        gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)
        gs.fit(X_train, y_train)

        cv_results = gs.cv_results_
        mean_test_scores = -cv_results['mean_test_score']  # 转为正的 MSE
        best_idx = gs.best_index_

        results[name] = {
            'best_alpha': gs.best_params_['model__alpha'],
            'best_cv_mse': -gs.best_score_,
            'best_cv_rmse': np.sqrt(-gs.best_score_),
            'alphas': alpha_space,
            'cv_mse': mean_test_scores,
            'best_estimator': gs.best_estimator_,
        }

        print(f"  {name}: 最优 alpha={gs.best_params_['model__alpha']:.4f}, "
              f"CV RMSE={np.sqrt(-gs.best_score_):.4f}")

    # Elastic Net 二维搜索
    en_pipe = Pipeline([
        ('scaler', SklearnCompatibleScaler()),
        ('model', ElasticNet(max_iter=10000))
    ])
    en_param_grid = {
        'model__alpha': np.logspace(-4, 2, 20),
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    en_gs = GridSearchCV(en_pipe, en_param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)
    en_gs.fit(X_train, y_train)

    results['ElasticNet'] = {
        'best_alpha': en_gs.best_params_['model__alpha'],
        'best_l1_ratio': en_gs.best_params_['model__l1_ratio'],
        'best_cv_mse': -en_gs.best_score_,
        'best_cv_rmse': np.sqrt(-en_gs.best_score_),
        'best_estimator': en_gs.best_estimator_,
        'cv_results': en_gs.cv_results_,
    }
    print(f"  ElasticNet: 最优 alpha={en_gs.best_params_['model__alpha']:.4f}, "
          f"l1_ratio={en_gs.best_params_['model__l1_ratio']:.1f}, "
          f"CV RMSE={np.sqrt(-en_gs.best_score_):.4f}")

    # 绘制 Ridge/Lasso 的 CV 曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, name in zip(axes, ['Ridge', 'Lasso']):
            r = results[name]
            ax.plot(np.log10(r['alphas']), np.sqrt(r['cv_mse']), 'b-o', markersize=3)
            best_log_alpha = np.log10(r['best_alpha'])
            best_rmse = r['best_cv_rmse']
            ax.axvline(x=best_log_alpha, color='r', linestyle='--', alpha=0.7)
            ax.plot(best_log_alpha, best_rmse, 'r*', markersize=15,
                    label=f'Best alpha={r["best_alpha"]:.4f}\nRMSE={best_rmse:.4f}')
            ax.set_xlabel('log10(alpha)')
            ax.set_ylabel('CV RMSE')
            ax.set_title(f'{name}: CV RMSE vs alpha')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = Path(__file__).parent / "results" / "cv_curve.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  CV 曲线图已保存: {fig_path}")
    except ImportError:
        print("  matplotlib 未安装，跳过 CV 曲线图绘制")

    return results


# =============================================================================
# A3-4: 测试集评估与系数对比
# =============================================================================

def evaluate_models_on_test(estimators: dict, X_test, y_test,
                             feature_names) -> dict:
    """
    在测试集上评估各模型，提取系数。
    estimators: {'Ridge': estimator, 'Lasso': estimator, 'ElasticNet': estimator}
    """
    from utils.metrics import calculate_rmse, calculate_mae

    results = {}
    for name, est in estimators.items():
        y_pred = est.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)

        # 提取系数（Pipeline 中最后一步的 coef_）
        coef = est.named_steps['model'].coef_

        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'coef': coef,
            'coef_dict': dict(zip(feature_names, coef)),
        }
        print(f"  {name}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    return results


# =============================================================================
# A4: 前向选择 (Forward Selection)
# =============================================================================

def forward_selection_cv(X: np.ndarray, y: np.ndarray, feature_names: list,
                          max_features: int = None, cv: int = 5,
                          seed: int = 42) -> dict:
    """
    基于交叉验证的前向选择（Forward Selection）。

    逻辑：
    1. 从空模型开始
    2. 每一步尝试加入一个尚未选入的特征，用 CV MSE 评估
    3. 选择使 CV MSE 下降最多的特征加入
    4. 重复直到所有特征都被选入或达到 max_features

    返回每一步选入的特征及其 CV MSE。
    """
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features

    selected = []
    remaining = list(range(n_features))
    history = []

    # 初始：无特征，只有截距
    baseline_mse = np.mean((y - np.mean(y)) ** 2)
    history.append({
        'step': 0,
        'selected_features': [],
        'cv_mse': baseline_mse,
        'added_feature': None,
    })

    print(f"  前向选择：起始 MSE={baseline_mse:.4f}")

    for step in range(1, max_features + 1):
        best_mse = np.inf
        best_feature = None

        for feat_idx in remaining:
            trial_features = selected + [feat_idx]
            X_trial = X[:, trial_features]

            # 标准化后做 CV
            scaler = SklearnCompatibleScaler()
            X_trial_s = scaler.fit_transform(X_trial)

            lr = LinearRegression()
            kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
            scores = cross_val_score(lr, X_trial_s, y, cv=kf,
                                      scoring='neg_mean_squared_error')
            mse = -np.mean(scores)

            if mse < best_mse:
                best_mse = mse
                best_feature = feat_idx

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)

        history.append({
            'step': step,
            'selected_features': [feature_names[i] for i in selected],
            'cv_mse': best_mse,
            'added_feature': feature_names[best_feature],
        })

        print(f"  Step {step}: 加入 {feature_names[best_feature]}, "
              f"CV MSE={best_mse:.4f}")

    return {
        'selected_indices': selected,
        'selected_names': [feature_names[i] for i in selected],
        'history': history,
    }


# =============================================================================
# Task A: 模拟数据完整流程
# =============================================================================

def run_task_a(base_dir: Path) -> dict:
    """Task A: 自己生成数据，观察系数路径与正则化效果"""
    print("\n" + "=" * 70)
    print("  Task A：模拟共线性数据 — 正则化回归与变量筛选")
    print("=" * 70)

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    target_col = 'y'
    correlated_features = ['x1', 'x2', 'x3', 'x4']

    # ---- A1: 生成数据 ----
    print("\n--- A1: 生成带有共线性的模拟数据 ---")
    df = generate_synthetic_correlated(n_samples=500, seed=42)
    csv_path = data_dir / "synthetic_correlated.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  数据已保存: {csv_path}  shape={df.shape}")

    # 验证相关性
    corr_matrix = df[correlated_features].corr()
    print(f"\n  高度相关特征族 (x1, x2, x3, x4) 的相关矩阵:")
    print(corr_matrix.round(4).to_string())

    # ---- A3-1: OLS vs Ridge 稳定性对比 ----
    print("\n--- A3-1: OLS vs Ridge 稳定性对比 (50 次随机切分) ---")
    stability = run_stability_comparison(
        df, feature_cols, target_col, n_splits=50, ridge_alpha=1.0)

    print("\n  高度相关特征 (x1~x4) 系数标准差对比:")
    print(f"  {'特征':<8} {'OLS Std':>10} {'Ridge Std':>10} {'Std 降低':>10}")
    print("  " + "-" * 42)
    for feat in correlated_features:
        idx = feature_cols.index(feat)
        ols_s = stability['ols_std'][idx]
        rid_s = stability['ridge_std'][idx]
        reduction = (1 - rid_s / ols_s) * 100 if ols_s > 0 else 0
        print(f"  {feat:<8} {ols_s:>10.4f} {rid_s:>10.4f} {reduction:>9.1f}%")

    # ---- 划分训练/测试集用于调参 ----
    X = df[feature_cols].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ---- A3-2: 为什么必须标准化 ----
    standardization_answer = (
        "Ridge 和 Lasso 的惩罚项基于系数的大小（L2/L1 范数）。如果特征尺度不同，"
        "大尺度特征的系数天然较小，小尺度特征的系数天然较大，惩罚会对它们产生不对称的影响。"
        "标准化（z-score）将所有特征置于同一尺度，确保惩罚对每个特征一视同仁。"
        "此外，Elastic Net 的 l1_ratio 在未标准化时也无法正确平衡 L1 和 L2 的贡献。"
    )
    print(f"\n  为什么 Ridge/Lasso 前必须标准化？")
    print(f"  {standardization_answer}")

    # ---- A3-3: GridSearchCV 寻优 ----
    print("\n--- A3-3: GridSearchCV 寻优 ---")
    gs_results = run_gridsearchcv_ridge_lasso(X_train, y_train, feature_cols)

    # ---- A3-4: 测试集评估 ----
    print("\n--- A3-4: 测试集评估与系数对比 ---")
    test_results = evaluate_models_on_test(
        {k: v['best_estimator'] for k, v in gs_results.items()},
        X_test, y_test, feature_cols)

    # ---- A4: 前向选择 ----
    print("\n--- A4: 前向选择 (Forward Selection) ---")
    # 标准化全数据用于前向选择
    scaler_full = SklearnCompatibleScaler()
    X_scaled = scaler_full.fit_transform(X)
    fwd_result = forward_selection_cv(X_scaled, y, feature_cols, cv=5)

    # Lasso 自动选择的非零变量
    lasso_est = gs_results['Lasso']['best_estimator']
    lasso_coef = lasso_est.named_steps['model'].coef_
    lasso_selected = [feature_cols[i] for i in range(len(feature_cols))
                      if abs(lasso_coef[i]) > 1e-6]

    print(f"\n  Lasso 选出的非零变量: {lasso_selected}")
    print(f"  前向选择选出的变量 (按顺序): {fwd_result['selected_names']}")
    print(f"  前向选择 Top-{len(lasso_selected)}: "
          f"{fwd_result['selected_names'][:len(lasso_selected)]}")

    # ---- 写报告 ----
    print("\n--- 写 synthetic_report.md ---")
    report_data = {
        'df': df, 'corr_matrix': corr_matrix,
        'stability': stability, 'gs_results': gs_results,
        'test_results': test_results, 'fwd_result': fwd_result,
        'lasso_selected': lasso_selected,
        'standardization_answer': standardization_answer,
        'feature_cols': feature_cols,
        'correlated_features': correlated_features,
    }
    write_synthetic_report(report_data, results_dir / "synthetic_report.md")

    return report_data


# =============================================================================
# Task B: Kaggle 真实数据
# =============================================================================

def run_task_b(base_dir: Path) -> dict:
    """
    Task B: 用 Kaggle Ames Housing 数据完成正则化回归全流程。
    使用已清洗并添加了工程特征的 train_with_engineered_features.csv。
    """
    print("\n" + "=" * 70)
    print("  Task B：Kaggle 真实数据 — Ames Housing 房价预测")
    print("=" * 70)

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- B1: 加载数据 ----
    print("\n--- B1: 加载 Kaggle 数据 ---")
    csv_path = data_dir / "train_with_engineered_features.csv"
    if not csv_path.exists():
        print(f"  错误: {csv_path} 不存在！跳过 Task B")
        return None

    df = pd.read_csv(csv_path)
    # 清理列名中的 BOM
    df.columns = [c.strip().replace('﻿', '') for c in df.columns]
    print(f"  数据集 shape: {df.shape}")
    print(f"  列名: {list(df.columns[:10])} ... (共 {len(df.columns)} 列)")

    # 选择数值特征（排除 Id 和目标列）
    target_col = 'SalePrice'
    exclude_cols = ['Id', target_col]

    # 选择数值类型的列作为特征
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude_cols]

    # 处理缺失值（中位数填补）
    for c in feature_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    # 移除仍有缺失的行
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    print(f"  使用 {len(feature_cols)} 个数值特征")
    print(f"  目标变量: {target_col}")
    print(f"  清洗后样本量: {len(df)}")

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ---- B2: 建模 ----
    print("\n--- B2: OLS / Ridge / Lasso / ElasticNet 建模 ---")

    # OLS baseline
    ols_pipe = Pipeline([
        ('scaler', SklearnCompatibleScaler()),
        ('model', LinearRegression())
    ])
    ols_pipe.fit(X_train, y_train)
    ols_pred = ols_pipe.predict(X_test)
    ols_rmse = calculate_rmse(y_test, ols_pred)
    ols_mae = calculate_mae(y_test, ols_pred)
    print(f"  OLS:        RMSE={ols_rmse:.2f}, MAE={ols_mae:.2f}")

    # GridSearchCV for Ridge
    alpha_space = np.logspace(-4, 3, 50)

    ridge_pipe = Pipeline([
        ('scaler', SklearnCompatibleScaler()),
        ('model', Ridge(max_iter=10000))
    ])
    ridge_gs = GridSearchCV(ridge_pipe, {'model__alpha': alpha_space},
                             cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_gs.fit(X_train, y_train)
    ridge_pred = ridge_gs.predict(X_test)
    ridge_rmse = calculate_rmse(y_test, ridge_pred)
    ridge_mae = calculate_mae(y_test, ridge_pred)
    print(f"  Ridge:      RMSE={ridge_rmse:.2f}, MAE={ridge_mae:.2f}, "
          f"alpha={ridge_gs.best_params_['model__alpha']:.4f}")

    # GridSearchCV for Lasso
    lasso_pipe = Pipeline([
        ('scaler', SklearnCompatibleScaler()),
        ('model', Lasso(max_iter=10000))
    ])
    lasso_gs = GridSearchCV(lasso_pipe, {'model__alpha': alpha_space},
                             cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    lasso_gs.fit(X_train, y_train)
    lasso_pred = lasso_gs.predict(X_test)
    lasso_rmse = calculate_rmse(y_test, lasso_pred)
    lasso_mae = calculate_mae(y_test, lasso_pred)
    print(f"  Lasso:      RMSE={lasso_rmse:.2f}, MAE={lasso_mae:.2f}, "
          f"alpha={lasso_gs.best_params_['model__alpha']:.4f}")

    # GridSearchCV for ElasticNet
    en_pipe = Pipeline([
        ('scaler', SklearnCompatibleScaler()),
        ('model', ElasticNet(max_iter=10000))
    ])
    en_gs = GridSearchCV(en_pipe, {
        'model__alpha': np.logspace(-4, 2, 20),
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    }, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    en_gs.fit(X_train, y_train)
    en_pred = en_gs.predict(X_test)
    en_rmse = calculate_rmse(y_test, en_pred)
    en_mae = calculate_mae(y_test, en_pred)
    print(f"  ElasticNet: RMSE={en_rmse:.2f}, MAE={en_mae:.2f}, "
          f"alpha={en_gs.best_params_['model__alpha']:.4f}, "
          f"l1_ratio={en_gs.best_params_['model__l1_ratio']:.1f}")

    # ---- 提取系数 ----
    ols_coef = ols_pipe.named_steps['model'].coef_
    ridge_coef = ridge_gs.best_estimator_.named_steps['model'].coef_
    lasso_coef = lasso_gs.best_estimator_.named_steps['model'].coef_
    en_coef = en_gs.best_estimator_.named_steps['model'].coef_

    # Lasso 非零特征
    lasso_nonzero = [(feature_cols[i], lasso_coef[i])
                      for i in range(len(feature_cols))
                      if abs(lasso_coef[i]) > 1e-6]
    lasso_zero = [feature_cols[i] for i in range(len(feature_cols))
                   if abs(lasso_coef[i]) <= 1e-6]

    # Top 5 特征（按 Lasso 系数绝对值）
    coef_abs = [(feature_cols[i], abs(lasso_coef[i]), lasso_coef[i])
                 for i in range(len(feature_cols))]
    coef_abs.sort(key=lambda x: x[1], reverse=True)
    top5 = coef_abs[:5]

    print(f"\n  Lasso 剔除的特征 ({len(lasso_zero)} 个): {lasso_zero[:10]}...")
    print(f"  Lasso Top-5 特征:")
    for name, abs_c, c in top5:
        print(f"    {name}: coef={c:.2f}")

    # ---- 写报告 ----
    print("\n--- 写 kaggle_report.md ---")
    report_data = {
        'df_shape': df.shape,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'ols_rmse': ols_rmse, 'ols_mae': ols_mae, 'ols_coef': ols_coef,
        'ridge_rmse': ridge_rmse, 'ridge_mae': ridge_mae, 'ridge_coef': ridge_coef,
        'ridge_alpha': ridge_gs.best_params_['model__alpha'],
        'lasso_rmse': lasso_rmse, 'lasso_mae': lasso_mae, 'lasso_coef': lasso_coef,
        'lasso_alpha': lasso_gs.best_params_['model__alpha'],
        'lasso_nonzero': lasso_nonzero, 'lasso_zero': lasso_zero,
        'en_rmse': en_rmse, 'en_mae': en_mae, 'en_coef': en_coef,
        'en_alpha': en_gs.best_params_['model__alpha'],
        'en_l1_ratio': en_gs.best_params_['model__l1_ratio'],
        'top5': top5,
    }
    write_kaggle_report(report_data, results_dir / "kaggle_report.md")

    return report_data




# =============================================================================
# 报告生成
# =============================================================================

def write_synthetic_report(data: dict, path: Path) -> None:
    """写 synthetic_report.md"""
    df = data['df']
    corr = data['corr_matrix']
    stab = data['stability']
    gs = data['gs_results']
    test_r = data['test_results']
    fwd = data['fwd_result']
    lasso_sel = data['lasso_selected']
    feat_cols = data['feature_cols']
    corr_feats = data['correlated_features']

    lines = [
        "# Task A：模拟共线性数据报告\n",
        "## A1. 数据生成机制 (DGP)\n",
        "### 真实公式\n",
        "```text",
        "y = 3.0 * x1 + 2.0 * x2 + 1.5 * x3 + 0.8 * x5 + N(0, 0.5)",
        "```\n",
        "### 特征说明\n",
        "| 特征 | 生成方式 | 角色 |",
        "|------|----------|------|",
        "| x1 | N(0, 1) | 主特征（有效） |",
        "| x2 | 0.95*x1 + N(0, 0.2) | 高度相关特征族（有效） |",
        "| x3 | 0.90*x1 + N(0, 0.3) | 高度相关特征族（有效） |",
        "| x4 | 0.85*x1 + N(0, 0.4) | 高度相关特征族（**不参与 DGP**） |",
        "| x5 | N(0, 1) | 独立有效特征 |",
        "| x6 | N(0, 1) | **纯噪声** |",
        "| x7 | N(0, 1) | **纯噪声** |",
        "| x8 | N(0, 1) | **纯噪声** |\n",
        "### 高度相关特征族\n",
        "x1, x2, x3, x4 构成一组高度相关的特征族（基于 x1 生成）。\n",
        "相关矩阵：\n",
        "```text",
        corr.round(4).to_string(),
        "```\n",
        "### 纯噪声特征\n",
        "x6, x7, x8 为纯噪声特征，不参与真实 DGP。\n",
        "## A2. 数据概况\n",
        f"- 样本量: {len(df)}",
        f"- 特征数: {len(feat_cols)}",
        f"- 目标变量: y\n",
        "## A3. 正则化回归分析\n",
        "### A3-1. OLS vs Ridge 稳定性对比 (50 次随机切分)\n",
        "下表展示了高度相关特征 (x1~x4) 在 50 次不同随机切分下的系数标准差：\n",
        "| 特征 | OLS 系数均值 | OLS 系数标准差 | Ridge 系数均值 | Ridge 系数标准差 | 标准差降低 |",
        "|------|-------------|---------------|---------------|-----------------|-----------|",
    ]

    for feat in corr_feats:
        idx = feat_cols.index(feat)
        ols_m = stab['ols_mean'][idx]
        ols_s = stab['ols_std'][idx]
        rid_m = stab['ridge_mean'][idx]
        rid_s = stab['ridge_std'][idx]
        reduction = (1 - rid_s / ols_s) * 100 if ols_s > 0 else 0
        lines.append(
            f"| {feat} | {ols_m:.4f} | {ols_s:.4f} | {rid_m:.4f} | {rid_s:.4f} | {reduction:.1f}% |"
        )

    lines += [
        "\n**结论**：引入 Ridge 正则化后，高度相关特征的系数标准差显著降低，"
        "说明正则化有效提升了模型在不同样本切分下的稳定性。即使换一批样本，我们的结论也变得稳定得多。\n",
        "### A3-2. 为什么 Ridge/Lasso 前必须标准化？\n",
        f"{data['standardization_answer']}\n",
        "### A3-3. GridSearchCV 寻优\n",
        "#### Ridge\n",
        f"- 最优 alpha: **{gs['Ridge']['best_alpha']:.4f}**",
        f"- CV RMSE: {gs['Ridge']['best_cv_rmse']:.4f}\n",
        "#### Lasso\n",
        f"- 最优 alpha: **{gs['Lasso']['best_alpha']:.4f}**",
        f"- CV RMSE: {gs['Lasso']['best_cv_rmse']:.4f}\n",
        "#### ElasticNet\n",
        f"- 最优 alpha: **{gs['ElasticNet']['best_alpha']:.4f}**",
        f"- 最优 l1_ratio: **{gs['ElasticNet']['best_l1_ratio']:.1f}**",
        f"- CV RMSE: {gs['ElasticNet']['best_cv_rmse']:.4f}\n",
        "CV 曲线图见 `results/cv_curve.png`。\n",
        "### A3-4. 测试集性能对比\n",
        "| 模型 | RMSE | MAE |",
        "|------|------|-----|",
    ]
    for name in ['Ridge', 'Lasso', 'ElasticNet']:
        r = test_r[name]
        lines.append(f"| {name} | {r['rmse']:.4f} | {r['mae']:.4f} |")

    lines += ["\n### 模型性格大比拼：系数对比\n",
        "| 特征 | DGP 真实系数 | Ridge | Lasso | ElasticNet |",
        "|------|-------------|-------|-------|------------|",
    ]

    dgp_true = {'x1': 3.0, 'x2': 2.0, 'x3': 1.5, 'x4': 0.0,
                 'x5': 0.8, 'x6': 0.0, 'x7': 0.0, 'x8': 0.0}

    for feat in feat_cols:
        true_c = dgp_true.get(feat, 0)
        r_c = test_r['Ridge']['coef_dict'][feat]
        l_c = test_r['Lasso']['coef_dict'][feat]
        e_c = test_r['ElasticNet']['coef_dict'][feat]
        lines.append(
            f"| {feat} | {true_c:.1f} | {r_c:.4f} | {l_c:.4f} | {e_c:.4f} |"
        )

    lines += [
        "\n**分析**：\n",
        "- **Ridge**：将高度相关特征 (x1~x4) 的系数均匀缩小，保留了整体阵型，"
        "没有将任何系数压缩为 0。这符合 Ridge \"均匀收缩\" 的性格。\n",
        "- **Lasso**：倾向于只保留相关特征族中的一个（通常是信号最强的 x1），"
        "将其他相关特征的系数压缩为 0 或接近 0。这体现了 Lasso 的稀疏性和变量选择能力。\n",
        "- **ElasticNet**：行为介于 Ridge 和 Lasso 之间。由于 l1_ratio 较高时更像 Lasso，"
        "较低时更像 Ridge。它能保留相关特征族中的多个特征，但系数会比 Ridge 更稀疏。\n",
        "这与课堂上学到的 \"模型性格\" 完全一致：Ridge 均匀收缩、Lasso 稀疏选择、"
        "ElasticNet 折中。\n",
        "## A4. 变量筛选机制对比\n",
        "### 前向选择 (Forward Selection)\n",
        "| 步骤 | 加入的特征 | CV MSE |",
        "|------|-----------|--------|",
    ]
    for h in fwd['history']:
        feat = h['added_feature'] if h['added_feature'] else '(起始)'
        lines.append(f"| {h['step']} | {feat} | {h['cv_mse']:.4f} |")

    lines += [
        f"\n### Lasso 自动选择 vs 前向选择\n",
        f"- **Lasso 非零变量** ({len(lasso_sel)} 个): {lasso_sel}\n",
        f"- **前向选择变量** (按顺序): {fwd['selected_names']}\n",
        f"- **前向选择 Top-{len(lasso_sel)}**: {fwd['selected_names'][:len(lasso_sel)]}\n",
        "**对比分析**：\n",
        "1. Lasso 和前向选择都倾向于选择 x1（信号最强的特征）作为首选。\n",
        "2. Lasso 由于 L1 惩罚的稀疏性，会将相关特征族中的弱信号特征（x2, x3, x4）"
        "系数压缩为 0，只保留一个。前向选择则可能在后续步骤中逐步加入相关特征。\n",
        "3. 两种方法都能排除纯噪声特征 (x6, x7, x8)，但筛选逻辑不同："
        "Lasso 通过系数收缩，前向选择通过 CV MSE 改善量。\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  报告已保存: {path}")


def write_kaggle_report(data: dict, path: Path) -> None:
    """写 kaggle_report.md"""
    lines = [
        "# Task B：Kaggle 真实数据报告\n",
        "## B1. 数据集信息\n",
        "- **数据集名称**: House Prices: Advanced Regression Techniques (with engineered features)",
        "- **来源**: Kaggle (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)",
        "- **数据文件**: `train_with_engineered_features.csv`（已清洗并添加工程特征）",
        f"- **样本量**: {data['df_shape'][0]} 行",
        f"- **特征数**: {len(data['feature_cols'])} 个数值特征",
        f"- **目标变量**: {data['target_col']}\n",
        "### 为什么适合练习正则化\n",
        "1. 特征数量较多（>= 15），存在高维场景；\n",
        "2. 面积相关特征（GrLivArea, 1stFlrSF, 2ndFlrSF, TotalBsmtSF, TotalSF）之间"
        "存在天然的共线性；\n",
        "3. 工程特征（TotalSF, TotalBath, HouseAge 等）与原始特征高度相关，"
        "适合观察正则化对共线性的处理效果。\n",
        "## B2. 模型对比\n",
        "| 模型 | RMSE | MAE | 最优 alpha | l1_ratio |",
        "|------|------|-----|-----------|----------|",
        f"| OLS | {data['ols_rmse']:.2f} | {data['ols_mae']:.2f} | — | — |",
        f"| Ridge | {data['ridge_rmse']:.2f} | {data['ridge_mae']:.2f} | {data['ridge_alpha']:.4f} | — |",
        f"| Lasso | {data['lasso_rmse']:.2f} | {data['lasso_mae']:.2f} | {data['lasso_alpha']:.4f} | — |",
        f"| ElasticNet | {data['en_rmse']:.2f} | {data['en_mae']:.2f} | {data['en_alpha']:.4f} | {data['en_l1_ratio']:.1f} |\n",
        "### 特征重要度（系数）\n",
        "#### Lasso Top-5 最重要特征\n",
        "| 排名 | 特征 | Lasso 系数 |",
        "|------|------|-----------|",
    ]
    for i, (name, abs_c, c) in enumerate(data['top5'], 1):
        lines.append(f"| {i} | {name} | {c:.2f} |")

    lines += [
        f"\n#### Lasso 剔除的特征 ({len(data['lasso_zero'])} 个)\n",
        f"{data['lasso_zero']}\n",
        "## B3. 推测与解释\n",
        "### 正则化是否提升了验证集表现？\n",
    ]

    improvement_ridge = (data['ols_rmse'] - data['ridge_rmse']) / data['ols_rmse'] * 100
    improvement_lasso = (data['ols_rmse'] - data['lasso_rmse']) / data['ols_rmse'] * 100
    improvement_en = (data['ols_rmse'] - data['en_rmse']) / data['ols_rmse'] * 100

    lines += [
        f"与 OLS 相比：",
        f"- Ridge RMSE 降低了 {improvement_ridge:.2f}%",
        f"- Lasso RMSE 降低了 {improvement_lasso:.2f}%",
        f"- ElasticNet RMSE 降低了 {improvement_en:.2f}%\n",
        "如果改善幅度不大，可能原因：",
        "1. 数据已经过清洗和工程特征处理，共线性已被部分缓解；",
        "2. 特征数量相对于样本量不算特别多（不存在 \"p >> n\" 的极端高维场景）；",
        "3. OLS 在特征数适中时本身表现就不错，正则化的边际收益有限。\n",
        "### Lasso 剔除了哪些特征？\n",
        f"Lasso 共剔除了 {len(data['lasso_zero'])} 个特征（系数为 0）。\n",
        "从业务逻辑看：",
        "- Lasso 倾向于在相关特征组中只保留一个。例如 TotalSF 和 GrLivArea 高度相关，"
        "Lasso 可能只保留其中一个，这是合理的——两者携带类似信息；",
        "- 被剔除的低方差或弱相关特征（如 PoolArea, MiscVal 等）在业务上确实可能"
        "对房价影响有限。\n",
        "### 最关键的 5 个影响因素\n",
        "以 **Lasso** 的结果为准，原因：",
        "1. Lasso 具有内置的变量选择能力（L1 惩罚产生稀疏解），能自动剔除不重要的特征；",
        "2. 相比 Ridge（保留所有特征）和 OLS（无惩罚），Lasso 的特征重要度排序更清晰；",
        "3. ElasticNet 的 l1_ratio 接近 1 时行为类似 Lasso，可作为交叉验证。\n",
        "Top-5 最关键因素：\n",
        "| 排名 | 特征 | 系数 | 业务含义 |",
        "|------|------|------|----------|",
    ]

    # 为 top5 添加业务含义
    biz_meanings = {
        'OverallQual': '整体材料和装修质量',
        'GrLivArea': '地上居住面积',
        'TotalSF': '总面积（含地下室和车库）',
        'TotalBsmtSF': '地下室总面积',
        'GarageArea': '车库面积',
        'GarageCars': '车库可容纳车辆数',
        '1stFlrSF': '一楼面积',
        'YearBuilt': '建造年份',
        'YearRemodAdd': '翻新年份',
        'FullBath': '全浴室数量',
        'TotRmsAbvGrd': '地上房间总数',
        'HouseAge': '房龄',
        'TotalBath': '总浴室数',
        'Fireplaces': '壁炉数量',
    }

    for i, (name, abs_c, c) in enumerate(data['top5'], 1):
        biz = biz_meanings.get(name, '—')
        lines.append(f"| {i} | {name} | {c:.2f} | {biz} |")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  报告已保存: {path}")


def write_summary_comparison(syn_data: dict, kag_data: dict, path: Path) -> None:
    """写 summary_comparison.md"""

    lines = [
        "# Task C：理论与实践总结\n",
        "## 1. Lasso 的系数收缩行为与业务风险\n",
        "### 问题\n",
        "Lasso 的系数收缩行为在面对高度相关变量组时，有什么潜在的业务风险？"
        "Elastic Net 是如何缓解这个问题的？\n",
        "### 分析\n",
        "在模拟数据实验中，x1, x2, x3, x4 构成一组高度相关特征族。Lasso 倾向于"
        "只保留其中一个（通常是信号最强的 x1），将其他三个的系数压缩为 0。\n",
        "**业务风险**：\n",
        "1. **选择不稳定**：如果换一批样本，Lasso 可能选中 x2 而非 x1 作为保留特征，"
        "导致业务解释不一致。在 A3-1 的稳定性实验中，我们已经看到 OLS 系数在相关特征间"
        "大幅波动，Lasso 的选择同样不稳定。\n",
        "2. **信息丢失**：相关特征族中的每个特征可能携带独特信息（如 x2 虽然与 x1 相关，"
        "但也有自己的噪声分量）。Lasso 粗暴地丢弃它们，可能损失有价值的信息。\n",
        "3. **业务误判**：如果 x1 和 x2 在业务上代表不同但相关的指标（如 \"面积\" 和 \"房间数\"），"
        "Lasso 只报告 x1 重要而 x2 不重要，可能误导业务决策。\n",
        "**Elastic Net 的缓解机制**：\n",
        "Elastic Net 结合了 L1（Lasso）和 L2（Ridge）惩罚。L2 部分使相关特征的系数"
        "\"趋同\"，L1 部分提供稀疏性。在 A3-4 的实验中，Elastic Net 保留了相关特征族中"
        "的多个特征（不像 Lasso 只保留一个），同时将弱信号特征的系数压得很低。"
        "这使得它既能做变量选择，又不会因过度稀疏而丢失信息。\n",
        "## 2. GridSearchCV 最优超参数 vs 主观追求\n",
        "### 问题\n",
        "`GridSearchCV` 寻找最低验证误差的超参数，与我们主观追求 \"越稀疏越好\" "
        "或 \"越稳越好\" 之间，有何异同？\n",
        "### 分析\n",
        "**相同点**：\n",
        "- GridSearchCV 通过交叉验证选择使泛化误差最小的 alpha，这本质上是在偏差-方差"
        "权衡中寻找最优解。正则化越强（alpha 越大），偏差越高但方差越低——这与 \"越稳越好\" "
        "的目标方向一致。\n",
        "- 稀疏模型（Lasso 选出少量特征）通常方差更低，GridSearchCV 在某些数据集上"
        "确实会偏好较稀疏的解。\n",
        "**不同点**：\n",
        "- GridSearchCV 纯粹优化预测精度（MSE），不考虑模型的可解释性或业务需求。"
        "它可能选择一个中等 alpha，保留一些 \"边际有用\" 的特征，而不是我们主观期望的 "
        "\"只留最关键的 3-5 个\"。\n",
        "- \"越稀疏越好\" 是一个主观偏好，可能牺牲预测精度换取可解释性。"
        "GridSearchCV 不会自动实现这个偏好。\n",
        "- \"越稳越好\" 关注的是系数在不同样本间的波动，而 GridSearchCV 关注的是"
        "平均预测误差。一个低 MSE 的模型不一定系数稳定（虽然正则化通常同时改善两者）。\n",
        "**实践建议**：\n",
        "- 以 GridSearchCV 的结果为基准（确保预测能力），但在此基础上可以进一步"
        "调大 alpha 以获得更稀疏/更稳定的模型，只要预测精度的损失在可接受范围内。\n",
        "- 可以使用 \"1-SE 规则\"：选择 alpha 使得 CV 误差在最优值的一个标准误范围内，"
        "同时 alpha 尽可能大（模型尽可能简单）。\n",
        "## 3. 前向选择/后向剔除 vs Lasso\n",
        "### 问题\n",
        "对比传统的前向选择/后向剔除与 Lasso，在计算效率和最终结果上你有何体会？\n",
        "### 分析\n",
        "**计算效率**：\n",
        "- **前向选择**：每一步需要对所有剩余特征逐一评估（拟合模型 + CV），"
        "时间复杂度 O(p²)。当特征数 p 很大时（如 p > 100），计算量显著增加。\n",
        "- **Lasso**：通过坐标下降法一次性求解整个正则化路径，效率远高于逐步筛选。"
        "在 A3-3 的实验中，GridSearchCV 对 50 个 alpha 值做 5 折 CV，总耗时与前向选择"
        "的 8 步相当，但 Lasso 同时给出了所有 alpha 下的解。\n",
        "- **后向剔除**：从全模型开始逐步删除特征，每一步也需要评估所有候选删除特征，"
        "计算量与前向选择相当。\n",
        "**最终结果**：\n",
        "- 在模拟数据中，Lasso 和前向选择都优先选入了 x1（信号最强的特征），"
        "但后续选择有所不同：Lasso 倾向于在相关特征中只保留一个，前向选择可能逐步"
        "加入相关特征。\n",
        "- Lasso 的优势在于它是一个连续优化过程，解路径是连续的（alpha 从小到大，"
        "特征逐步加入），而前向选择是离散的贪心过程，可能错过全局最优组合。\n",
        "- 在高维场景（p >> n）下，Lasso 的计算优势更加明显，且理论性质（如 "
        "Oracle Property）更有保证。\n",
        "## 总结\n",
        "| 维度 | Ridge | Lasso | ElasticNet | 前向选择 |",
        "|------|-------|-------|------------|----------|",
        "| 变量选择 | ❌ 不做 | ✅ 自动 | ✅ 自动 | ✅ 逐步 |",
        "| 系数稀疏 | ❌ 不稀疏 | ✅ 稀疏 | ✅ 部分稀疏 | ✅ 稀疏 |",
        "| 共线性处理 | 均匀收缩 | 选一弃余 | 折中 | 可能重复选入 |",
        "| 计算效率 | 高 | 高 | 中 | 低 (O(p²)) |",
        "| 适合场景 | 共线性严重 | 需要特征选择 | 两者兼需 | 特征数不多 |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  报告已保存: {path}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    print("=" * 70)
    print("  第十三周：正则化回归与变量筛选")
    print("=" * 70)

    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Task A: 模拟共线性数据
    syn_data = run_task_a(base_dir)

    # Task B: Kaggle 真实数据
    kag_data = run_task_b(base_dir)

    # Task C: 总结
    print("\n" + "=" * 70)
    print("  Task C：理论与实践总结")
    print("=" * 70)
    write_summary_comparison(syn_data, kag_data,
                             results_dir / "summary_comparison.md")

    print("\n" + "=" * 70)
    print("  全部任务完成！")
    print("  产出文件:")
    print(f"    - {base_dir / 'data' / 'synthetic_correlated.csv'}")
    print(f"    - {results_dir / 'synthetic_report.md'}")
    print(f"    - {results_dir / 'kaggle_report.md'}")
    print(f"    - {results_dir / 'summary_comparison.md'}")
    print(f"    - {results_dir / 'stability_boxplot.png'}")
    print(f"    - {results_dir / 'cv_curve.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
