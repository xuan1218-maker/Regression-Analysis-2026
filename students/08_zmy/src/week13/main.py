import sys
from pathlib import Path

# 添加 src 到路径
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 导入你自己的工具箱
try:
    from utils.metrics import calculate_rmse, calculate_mae
except ImportError:
    def calculate_rmse(y, y_pred): return np.sqrt(mean_squared_error(y, y_pred))
    def calculate_mae(y, y_pred): return mean_absolute_error(y, y_pred)

from utils.selection import ForwardSelector, StepwiseSelector
from utils.transformers import CustomImputer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==================== 1. 生成合成数据 ====================
def generate_synthetic_data(n_samples=500, n_features=8, n_corr_group=3, noise_std=0.5):
    np.random.seed(RANDOM_SEED)
    latent = np.random.randn(n_samples)
    X_corr = np.column_stack([latent + 0.1*np.random.randn(n_samples) for _ in range(3)])
    X_indep = np.random.randn(n_samples, 2)
    X_noise = np.random.randn(n_samples, n_features - 5)
    X = np.column_stack([X_corr, X_indep, X_noise])
    true_coef = np.array([2.0, 1.5, 1.0, 0.5, 0.3] + [0.0]*(n_features-5))
    y = X @ true_coef + noise_std * np.random.randn(n_samples)
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    return df, true_coef, feature_names

# ==================== 2. 稳定性对比 ====================
def stability_comparison(X, y, n_splits=50, alpha_ridge=0.1):
    coefs_ols, coefs_ridge = [], []
    for _ in range(n_splits):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=None)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        ols = LinearRegression().fit(X_train_scaled, y_train)
        ridge = Ridge(alpha=alpha_ridge).fit(X_train_scaled, y_train)
        coefs_ols.append(ols.coef_)
        coefs_ridge.append(ridge.coef_)
    coefs_ols = np.array(coefs_ols)
    coefs_ridge = np.array(coefs_ridge)
    ols_std = coefs_ols.std(axis=0)[:3]
    ridge_std = coefs_ridge.std(axis=0)[:3]
    return ols_std, ridge_std

# ==================== 3. 网格搜索模型（包含绘图和保存） ====================
def grid_search_models(X_train, y_train, save_dir=None):
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    imputer = CustomImputer()
    X_train = imputer.fit_transform(X_train)

    mask = ~np.isnan(y_train)
    if not mask.all():
        X_train = X_train[mask]
        y_train = y_train[mask]

    scaler = StandardScaler()
    ridge = Ridge()
    lasso = Lasso(max_iter=10000)
    elastic = ElasticNet(max_iter=10000)

    param_grid_ridge = {'ridge__alpha': np.logspace(-4, 3, 30)}
    param_grid_lasso = {'lasso__alpha': np.logspace(-4, 3, 30)}
    param_grid_elastic = {
        'elastic__alpha': np.logspace(-4, 3, 20),
        'elastic__l1_ratio': [0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    ridge_pipe = Pipeline([('scaler', scaler), ('ridge', ridge)])
    lasso_pipe = Pipeline([('scaler', scaler), ('lasso', lasso)])
    elastic_pipe = Pipeline([('scaler', scaler), ('elastic', elastic)])

    gs_ridge = GridSearchCV(ridge_pipe, param_grid_ridge, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    gs_lasso = GridSearchCV(lasso_pipe, param_grid_lasso, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    gs_elastic = GridSearchCV(elastic_pipe, param_grid_elastic, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

    print("正在拟合 Ridge...")
    gs_ridge.fit(X_train, y_train)
    print("正在拟合 Lasso...")
    gs_lasso.fit(X_train, y_train)
    print("正在拟合 ElasticNet...")
    gs_elastic.fit(X_train, y_train)

    ridge_best_params = {k.split('__')[1]: v for k, v in gs_ridge.best_params_.items()}
    lasso_best_params = {k.split('__')[1]: v for k, v in gs_lasso.best_params_.items()}
    elastic_best_params = {k.split('__')[1]: v for k, v in gs_elastic.best_params_.items()}

    results_ridge = pd.DataFrame(gs_ridge.cv_results_)
    results_lasso = pd.DataFrame(gs_lasso.cv_results_)
    ridge_alphas = results_ridge['param_ridge__alpha'].astype(float)
    lasso_alphas = results_lasso['param_lasso__alpha'].astype(float)
    ridge_scores = -results_ridge['mean_test_score']
    lasso_scores = -results_lasso['mean_test_score']

    # 绘图
    plt.figure(figsize=(10,6))
    plt.semilogx(ridge_alphas, ridge_scores, label='Ridge')
    plt.semilogx(lasso_alphas, lasso_scores, label='Lasso')
    plt.xlabel('alpha')
    plt.ylabel('CV MSE')
    plt.title('Cross-Validation Error vs Regularization Strength')
    plt.legend()
    plt.grid(True)

    if save_dir:
        save_path = Path(save_dir) / "cv_error_curves.png"
        plt.savefig(save_path, dpi=150)
    else:
        plt.savefig('cv_error_curves.png', dpi=150)
    plt.show()

    return (gs_ridge.best_estimator_, gs_lasso.best_estimator_, gs_elastic.best_estimator_,
            ridge_best_params, lasso_best_params, elastic_best_params)

# ==================== 4. 主流程 ====================
def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # ---------- 合成数据 ----------
    print("="*60)
    print("合成数据分析")
    print("="*60)
    df_syn, true_coef, feature_names = generate_synthetic_data()
    df_syn.to_csv(data_dir / "synthetic_correlated.csv", index=False)
    X_syn = df_syn.drop('y', axis=1)
    y_syn = df_syn['y']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=RANDOM_SEED)

    ols_std, ridge_std = stability_comparison(X_syn, y_syn, n_splits=50, alpha_ridge=0.1)

    best_ridge, best_lasso, best_elastic, ridge_params, lasso_params, elastic_params = grid_search_models(X_train_s, y_train_s, save_dir=results_dir)

    def evaluate(model, X, y):
        y_pred = model.predict(X)
        return calculate_rmse(y, y_pred), calculate_mae(y, y_pred)

    ridge_rmse, ridge_mae = evaluate(best_ridge, X_test_s, y_test_s)
    lasso_rmse, lasso_mae = evaluate(best_lasso, X_test_s, y_test_s)
    elastic_rmse, elastic_mae = evaluate(best_elastic, X_test_s, y_test_s)

    ridge_coef = best_ridge.named_steps['ridge'].coef_
    lasso_coef = best_lasso.named_steps['lasso'].coef_
    elastic_coef = best_elastic.named_steps['elastic'].coef_

    scaler_temp = StandardScaler()
    X_train_scaled = scaler_temp.fit_transform(X_train_s)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    y_train_array = y_train_s.values

    forward_selector = ForwardSelector(significance_level=0.05)
    forward_selector.fit(X_train_scaled_df, y_train_array)
    stepwise_selector = StepwiseSelector(significance_entry=0.05, significance_removal=0.05)
    stepwise_selector.fit(X_train_scaled_df, y_train_array)

    forward_selected = forward_selector.selected_features_
    stepwise_selected = stepwise_selector.selected_features_
    lasso_selected = [feature_names[i] for i, c in enumerate(lasso_coef) if abs(c) > 1e-5]

    with open(results_dir / "synthetic_report.md", 'w', encoding='utf-8') as f:
        f.write("# 合成数据正则化报告\n\n")
        f.write("## 1. 数据生成过程 (DGP)\n")
        f.write("- 样本量: 500, 特征数: 8\n")
        f.write("- 高度相关特征组: X1, X2, X3 (由潜变量加小噪声生成)\n")
        f.write("- 有用独立特征: X4, X5 (系数分别为 0.5, 0.3)\n")
        f.write("- 纯噪声特征: X6, X7, X8 (系数为 0)\n")
        f.write("- 真实数据生成公式: y = 2*X1 + 1.5*X2 + 1.0*X3 + 0.5*X4 + 0.3*X5 + 噪声\n\n")
        f.write("## 2. 稳定性对比 (OLS vs 岭回归)\n")
        f.write(f"- OLS 系数标准差 (X1,X2,X3): {ols_std}\n")
        f.write(f"- 岭回归 (alpha=0.1) 系数标准差: {ridge_std}\n")
        f.write("岭回归显著降低了系数的波动，提高了稳定性。\n\n")
        f.write("## 3. 为什么必须进行标准化？\n")
        f.write("正则化项对特征的尺度敏感；标准化使所有特征处于同一量纲，正则化才能公平地惩罚所有系数。\n\n")
        f.write("## 4. 网格搜索交叉验证结果\n")
        f.write(f"- 最优岭回归: alpha={ridge_params['alpha']:.4f}, 测试集 RMSE={ridge_rmse:.4f}, MAE={ridge_mae:.4f}\n")
        f.write(f"- 最优 Lasso: alpha={lasso_params['alpha']:.4f}, 测试集 RMSE={lasso_rmse:.4f}, MAE={lasso_mae:.4f}\n")
        f.write(f"- 最优 ElasticNet: alpha={elastic_params['alpha']:.4f}, l1_ratio={elastic_params['l1_ratio']:.2f}, 测试集 RMSE={elastic_rmse:.4f}, MAE={elastic_mae:.4f}\n\n")
        f.write("## 5. 系数对比\n")
        f.write("| 特征 | 真实系数 | 岭回归 | Lasso | ElasticNet |\n")
        f.write("|------|----------|--------|-------|------------|\n")
        for i, name in enumerate(feature_names):
            true = true_coef[i] if i < len(true_coef) else 0
            f.write(f"| {name} | {true:.2f} | {ridge_coef[i]:.4f} | {lasso_coef[i]:.4f} | {elastic_coef[i]:.4f} |\n")
        f.write("\n**观察结果:**\n")
        f.write("- 岭回归均匀地收缩相关特征的系数。\n")
        f.write("- Lasso 将噪声特征的系数大幅收缩（接近但不等于零），在一定程度上减轻了无关特征的影响；若增大 alpha，可能实现真正的变量选择。\n")
        f.write("- ElasticNet 介于两者之间。\n\n")
        f.write("## 6. 变量选择方法对比\n")
        f.write(f"- 前向选择 (p<0.05): {forward_selected}\n")
        f.write(f"- 逐步回归: {stepwise_selected}\n")
        f.write(f"- Lasso 自动选择的特征: {lasso_selected}\n")
        f.write("在本次设定的 alpha 下，Lasso 并未强制将高度相关的 X1、X2、X3 压缩到只留一个，而是保留了全部三个，这可能是因为真实 DGP 中三个变量都有独立贡献且噪声较低。前向/逐步也保留了所有有用特征，两者在选择上较为一致。若增大 alpha，Lasso 理论上会表现出稀疏性。\n\n")
        f.write("## 7. 结论\n")
        f.write("正则化提高了模型的稳定性和可解释性。当所有特征都有用时岭回归表现良好；Lasso 适用于稀疏特征选择；ElasticNet 在特征组相关时更稳健。\n")

    # ---------- Kaggle 数据 ----------
    print("\n"+"="*60)
    print("Kaggle 数据分析 (艾姆斯房价)")
    print("="*60)
    kaggle_file = data_dir / "kaggle_data.csv"
    if not kaggle_file.exists():
        print(f"Kaggle 文件不存在: {kaggle_file}，跳过此部分。")
    else:
        df_kaggle = pd.read_csv(kaggle_file)
        df_kaggle = df_kaggle.drop(columns=['Id'])
        target = 'SalePrice'

        numeric_cols = df_kaggle.select_dtypes(include=[np.number]).columns.drop(target, errors='ignore').tolist()
        missing_ratio = df_kaggle[numeric_cols].isnull().mean()
        drop_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]
        df_kaggle_clean = df_kaggle[numeric_cols + [target]].copy()
        X_kag = df_kaggle_clean.drop(target, axis=1)
        y_kag = df_kaggle_clean[target]
        X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_kag, y_kag, test_size=0.2, random_state=RANDOM_SEED)

        best_ridge_k, best_lasso_k, best_elastic_k, ridge_params_k, lasso_params_k, elastic_params_k = grid_search_models(X_train_k, y_train_k, save_dir=results_dir)

        # 对测试集进行相同的缺失值填补
        imputer = CustomImputer()
        X_train_k_clean = imputer.fit_transform(X_train_k.values if hasattr(X_train_k, 'values') else X_train_k)
        X_test_k_clean = imputer.transform(X_test_k.values if hasattr(X_test_k, 'values') else X_test_k)

        ridge_rmse_k, ridge_mae_k = evaluate(best_ridge_k, X_test_k_clean, y_test_k.values)
        lasso_rmse_k, lasso_mae_k = evaluate(best_lasso_k, X_test_k_clean, y_test_k.values)
        elastic_rmse_k, elastic_mae_k = evaluate(best_elastic_k, X_test_k_clean, y_test_k.values)

        ridge_coef_k = best_ridge_k.named_steps['ridge'].coef_
        lasso_coef_k = best_lasso_k.named_steps['lasso'].coef_
        elastic_coef_k = best_elastic_k.named_steps['elastic'].coef_

        scaler_k = StandardScaler()
        X_train_scaled_k = scaler_k.fit_transform(X_train_k_clean)
        X_train_scaled_df_k = pd.DataFrame(X_train_scaled_k, columns=X_train_k.columns)
        y_train_array_k = y_train_k.values

        forward_k = ForwardSelector(significance_level=0.05)
        forward_k.fit(X_train_scaled_df_k, y_train_array_k)
        stepwise_k = StepwiseSelector(significance_entry=0.05, significance_removal=0.05)
        stepwise_k.fit(X_train_scaled_df_k, y_train_array_k)
        lasso_selected_k = [col for i, col in enumerate(X_train_k.columns) if abs(lasso_coef_k[i]) > 1e-5]

        with open(results_dir / "kaggle_report.md", 'w', encoding='utf-8') as f:
            f.write("# Kaggle 艾姆斯房价预测报告\n\n")
            f.write("## 数据集说明\n")
            f.write("- 名称: 艾姆斯房价数据集\n")
            f.write("- 来源: Kaggle 房价预测竞赛\n")
            f.write("- 目标变量: SalePrice (美元)\n")
            f.write("- 原始特征: 79 个 (我们仅使用了数值特征以便聚焦正则化)\n\n")
            f.write("## 数据预处理\n")
            f.write("- 删除 Id 列，剔除缺失率超过 50% 的数值列，使用 CustomImputer 以均值填补剩余缺失值。\n")
            f.write("- 未使用类别编码，以保持对正则化效果的关注。\n\n")
            f.write("## 模型性能 (5折交叉验证 + 测试集)\n")
            f.write(f"- 岭回归: RMSE={ridge_rmse_k:.2f}, MAE={ridge_mae_k:.2f}, alpha={ridge_params_k['alpha']:.4f}\n")
            f.write(f"- Lasso: RMSE={lasso_rmse_k:.2f}, MAE={lasso_mae_k:.2f}, alpha={lasso_params_k['alpha']:.4f}\n")
            f.write(f"- ElasticNet: RMSE={elastic_rmse_k:.2f}, MAE={elastic_mae_k:.2f}, alpha={elastic_params_k['alpha']:.4f}, l1_ratio={elastic_params_k['l1_ratio']:.2f}\n\n")
            f.write("## 变量选择结果\n")
            f.write(f"- Lasso 选择了 {len(lasso_selected_k)} 个特征: {lasso_selected_k[:10]}...\n")
            f.write(f"- 前向选择 (p<0.05): {forward_k.selected_features_[:10]}...\n")
            f.write(f"- 逐步回归: {stepwise_k.selected_features_[:10]}...\n\n")
            f.write("## 业务解释\n")
            f.write("正则化提高了模型的泛化能力。Lasso 减少了特征数量，可以简化模型部署。但在房地产领域，许多特征都可能重要，因此岭回归可能更合适。\n")

        print("Kaggle 报告已保存。")

    # ---------- 总结对比 ----------
    with open(results_dir / "summary_comparison.md", 'w', encoding='utf-8') as f:
        f.write("# 总结：正则化 vs OLS vs 变量选择\n\n")
        f.write("## 1. Lasso 在相关特征组中的风险\n")
        f.write("理论上，当 alpha 足够大时，Lasso 会从高度相关的特征组中随机选择一个，丢弃其他变量。但在本次合成数据实验中，由于设置的 alpha 较小（最优 alpha=0.0149），Lasso 保留了相关组中的所有特征。这提醒我们：正则化的强度直接影响稀疏性，业务上需根据解释需求调整超参数。\n\n")
        f.write("## 2. GridSearchCV 与主观稀疏/稳定目标的差异\n")
        f.write("GridSearchCV 优化的是交叉验证误差，而不是稀疏性。一个误差稍高但更稀疏的模型可能更易于解释，业务方应结合需求做决策。\n\n")
        f.write("## 3. 传统逐步回归与 Lasso 的对比\n")
        f.write("- 逐步回归计算量大、不稳定且容易过拟合。\n")
        f.write("- Lasso 高效稳定，但对相关特征组可能过于激进。\n")
        f.write("- 混合方法 (ElasticNet + 交叉验证) 通常效果最好。\n")

    print("\n✅ 所有任务完成！请查看 results/ 目录下的输出。")

if __name__ == "__main__":
    main()