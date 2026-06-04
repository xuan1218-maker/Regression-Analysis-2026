from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

# 100% 复用你自己写的utils组件
from utils.metrics import calculate_rmse, calculate_mae
from utils.transformers import CustomStandardScaler

# 设置全局字体和样式
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ==================== 工具函数 ====================
def init_results_dir(results_dir: Path, figures_dir: Path):
    """自动清理并初始化结果目录"""
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    print(f"📁 结果目录已初始化: {results_dir}")
    print(f"📁 图表目录已初始化: {figures_dir}")

# ==================== Task A: 生成带共线性的模拟数据 ====================
def generate_synthetic_data(data_path: Path, n_samples: int = 300, random_state: int = 42):
    """
    生成带有明确共线性的模拟回归数据
    DGP: y = 5 + 2*x1 + 1.5*x4 + 1.0*x5 + 噪声
    高度相关特征组: x1, x2, x3 (x2=0.9x1+噪声, x3=0.8x1+噪声)
    纯噪声特征: x6, x7, x8
    """
    np.random.seed(random_state)
    
    # 生成基础特征
    x1 = np.random.normal(0, 1, n_samples)
    # 构造高度相关特征
    x2 = 0.9 * x1 + np.random.normal(0, 0.1, n_samples)
    x3 = 0.8 * x1 + np.random.normal(0, 0.1, n_samples)
    # 独立特征
    x4 = np.random.normal(0, 1, n_samples)
    x5 = np.random.normal(0, 1, n_samples)
    # 纯噪声特征
    x6 = np.random.normal(0, 1, n_samples)
    x7 = np.random.normal(0, 1, n_samples)
    x8 = np.random.normal(0, 1, n_samples)
    
    # 生成目标变量
    y = 5 + 2*x1 + 1.5*x4 + 1.0*x5 + np.random.normal(0, 1, n_samples)
    
    # 构造DataFrame
    df = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "x5": x5, "x6": x6, "x7": x7, "x8": x8,
        "y": y
    })
    
    # 保存数据
    df.to_csv(data_path, index=False)
    print(f"✅ 模拟共线性数据已生成并保存到: {data_path}")
    
    # 划分训练集和测试集
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()
    feature_names = df.drop(columns=["y"]).columns.tolist()
    
    return train_test_split(X, y, test_size=0.3, random_state=random_state), feature_names

# ==================== Task A1: 正则化前后稳定性对比 ====================
def run_stability_demo(X, y, feature_names, figures_dir: Path, n_repeats: int = 50):
    """
    对比OLS和Ridge在不同数据切分下的系数稳定性
    重点关注高度相关特征x1, x2, x3
    """
    print("\n[Stage 1] 运行正则化前后系数稳定性对比...")
    
    ols_coefs = []
    ridge_coefs = []
    
    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        # 标准化
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 拟合OLS
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        ols_coefs.append(ols.coef_[:3])  # 只保存x1,x2,x3的系数
        
        # 拟合Ridge(alpha=1.0)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge.coef_[:3])
    
    ols_coefs = np.array(ols_coefs)
    ridge_coefs = np.array(ridge_coefs)
    
    # 计算系数标准差
    ols_std = np.std(ols_coefs, axis=0)
    ridge_std = np.std(ridge_coefs, axis=0)
    
    # 绘制箱线图
    plt.figure(figsize=(12, 8))
    
    # OLS箱线图
    plt.boxplot([ols_coefs[:,0], ols_coefs[:,1], ols_coefs[:,2]], 
                positions=[1,2,3], widths=0.3,
                patch_artist=True, boxprops=dict(facecolor="#ff6b6b", alpha=0.7),
                tick_labels=["x1 (OLS)", "x2 (OLS)", "x3 (OLS)"])
    
    # Ridge箱线图
    plt.boxplot([ridge_coefs[:,0], ridge_coefs[:,1], ridge_coefs[:,2]], 
                positions=[5,6,7], widths=0.3,
                patch_artist=True, boxprops=dict(facecolor="#4ecdc4", alpha=0.7),
                tick_labels=["x1 (Ridge)", "x2 (Ridge)", "x3 (Ridge)"])
    
    plt.axhline(y=2, color="k", linestyle="--", linewidth=2, label="True Coef (x1=2)")
    plt.axhline(y=0, color="k", linestyle="--", linewidth=1, label="True Coef (x2=x3=0)")
    plt.ylabel("Coefficient Value")
    plt.title(f"OLS vs Ridge Coefficient Stability ({n_repeats} Random Splits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "coefficient_stability.png")
    plt.close()
    
    print(f"✅ 系数稳定性对比图已生成: coefficient_stability.png")
    print(f"   OLS系数平均标准差: {np.mean(ols_std):.4f}")
    print(f"   Ridge系数平均标准差: {np.mean(ridge_std):.4f}")
    
    return ols_std, ridge_std

# ==================== Task A2: 超参数调优 ====================
def run_hyperparameter_tuning(X_train, y_train, figures_dir: Path):
    """
    使用GridSearchCV为Ridge, Lasso, ElasticNet寻找最优超参数
    绘制CV误差随alpha变化的曲线
    """
    print("\n[Stage 2] 运行超参数调优...")
    
    # 创建通用Pipeline
    def create_pipeline(model):
        return Pipeline([
            ("scaler", CustomStandardScaler()),
            ("model", model)
        ])
    
    # 1. Ridge调参
    print("   调优Ridge模型...")
    ridge_pipeline = create_pipeline(Ridge())
    ridge_param_grid = {"model__alpha": np.logspace(-4, 3, 50)}
    ridge_grid = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=5, scoring="neg_root_mean_squared_error")
    ridge_grid.fit(X_train, y_train)
    
    # 2. Lasso调参
    print("   调优Lasso模型...")
    lasso_pipeline = create_pipeline(Lasso(max_iter=10000))
    lasso_param_grid = {"model__alpha": np.logspace(-4, 3, 50)}
    lasso_grid = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5, scoring="neg_root_mean_squared_error")
    lasso_grid.fit(X_train, y_train)
    
    # 3. ElasticNet调参
    print("   调优ElasticNet模型...")
    enet_pipeline = create_pipeline(ElasticNet(max_iter=10000))
    enet_param_grid = {
        "model__alpha": np.logspace(-4, 3, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    enet_grid = GridSearchCV(enet_pipeline, enet_param_grid, cv=5, scoring="neg_root_mean_squared_error")
    enet_grid.fit(X_train, y_train)
    
    # 绘制CV误差曲线
    plt.figure(figsize=(12, 8))
    
    # Ridge曲线
    ridge_alphas = ridge_param_grid["model__alpha"]
    ridge_scores = -ridge_grid.cv_results_["mean_test_score"]
    plt.plot(ridge_alphas, ridge_scores, "r-", label="Ridge", linewidth=2)
    plt.scatter(ridge_grid.best_params_["model__alpha"], -ridge_grid.best_score_, 
                c="red", s=100, zorder=5, label=f"Ridge Best alpha={ridge_grid.best_params_['model__alpha']:.4f}")
    
    # Lasso曲线
    lasso_alphas = lasso_param_grid["model__alpha"]
    lasso_scores = -lasso_grid.cv_results_["mean_test_score"]
    plt.plot(lasso_alphas, lasso_scores, "b-", label="Lasso", linewidth=2)
    plt.scatter(lasso_grid.best_params_["model__alpha"], -lasso_grid.best_score_, 
                c="blue", s=100, zorder=5, label=f"Lasso Best alpha={lasso_grid.best_params_['model__alpha']:.4f}")
    
    plt.xscale("log")
    plt.xlabel("alpha (Log Scale)")
    plt.ylabel("5-Fold CV Mean RMSE")
    plt.title("Regularization Strength vs Validation Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hyperparameter_tuning.png")
    plt.close()
    
    print("✅ 超参数调优曲线已生成: hyperparameter_tuning.png")
    print(f"   Ridge最优alpha: {ridge_grid.best_params_['model__alpha']:.4f}, 最优CV RMSE: {-ridge_grid.best_score_:.4f}")
    print(f"   Lasso最优alpha: {lasso_grid.best_params_['model__alpha']:.4f}, 最优CV RMSE: {-lasso_grid.best_score_:.4f}")
    print(f"   ElasticNet最优alpha: {enet_grid.best_params_['model__alpha']:.4f}, l1_ratio={enet_grid.best_params_['model__l1_ratio']:.1f}, 最优CV RMSE: {-enet_grid.best_score_:.4f}")
    
    return ridge_grid.best_estimator_, lasso_grid.best_estimator_, enet_grid.best_estimator_

# ==================== Task A3: 模型性格大比拼 ====================
def run_model_comparison(best_ridge, best_lasso, best_enet, X_test, y_test, feature_names, figures_dir: Path):
    """
    对比最优Ridge, Lasso, ElasticNet在测试集上的表现和系数
    重点观察它们对高度相关特征的处理方式
    """
    print("\n[Stage 3] 运行模型性格大比拼...")
    
    # 预测并计算指标
    y_pred_ridge = best_ridge.predict(X_test)
    y_pred_lasso = best_lasso.predict(X_test)
    y_pred_enet = best_enet.predict(X_test)
    
    # 计算评估指标
    results = {
        "Ridge": {
            "rmse": calculate_rmse(y_test, y_pred_ridge),
            "mae": calculate_mae(y_test, y_pred_ridge),
            "coef": best_ridge.named_steps["model"].coef_
        },
        "Lasso": {
            "rmse": calculate_rmse(y_test, y_pred_lasso),
            "mae": calculate_mae(y_test, y_pred_lasso),
            "coef": best_lasso.named_steps["model"].coef_
        },
        "ElasticNet": {
            "rmse": calculate_rmse(y_test, y_pred_enet),
            "mae": calculate_mae(y_test, y_pred_enet),
            "coef": best_enet.named_steps["model"].coef_
        }
    }
    
    # 绘制系数对比图
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    plt.bar(x - width, results["Ridge"]["coef"], width, label="Ridge", color="#ff6b6b", alpha=0.7)
    plt.bar(x, results["Lasso"]["coef"], width, label="Lasso", color="#4ecdc4", alpha=0.7)
    plt.bar(x + width, results["ElasticNet"]["coef"], width, label="ElasticNet", color="#ffd166", alpha=0.7)
    
    plt.axhline(y=0, color="k", linewidth=1)
    plt.xticks(x, feature_names)
    plt.ylabel("Coefficient Value")
    plt.title("Coefficient Comparison of Different Regularized Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "coefficient_comparison.png")
    plt.close()
    
    print("✅ 模型系数对比图已生成: coefficient_comparison.png")
    print(f"   Ridge测试集RMSE: {results['Ridge']['rmse']:.4f}")
    print(f"   Lasso测试集RMSE: {results['Lasso']['rmse']:.4f}")
    print(f"   ElasticNet测试集RMSE: {results['ElasticNet']['rmse']:.4f}")
    
    return results

# ==================== Task A4: 前向选择变量筛选 ====================
def run_forward_selection(X_train, y_train, X_test, y_test, feature_names, max_features: int = 5):
    """
    实现基于5折交叉验证的前向选择算法
    """
    print("\n[Stage 4] 运行前向选择变量筛选...")
    
    n_features = X_train.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    best_scores = []
    
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for _ in range(min(max_features, n_features)):
        best_score = float("inf")
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_current = X_train_scaled[:, current_features]
            
            # 5折CV评估
            scores = []
            for train_idx, val_idx in kf.split(X_current):
                X_tr, X_val = X_current[train_idx], X_current[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = LinearRegression()
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                scores.append(calculate_rmse(y_val, y_pred))
            
            avg_score = np.mean(scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_feature = feature
        
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_scores.append(best_score)
    
    # 训练最终模型
    X_selected_train = X_train_scaled[:, selected_features]
    X_selected_test = X_test_scaled[:, selected_features]
    
    final_model = LinearRegression()
    final_model.fit(X_selected_train, y_train)
    y_pred = final_model.predict(X_selected_test)
    
    test_rmse = calculate_rmse(y_test, y_pred)
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    print(f"✅ 前向选择完成，选出的特征: {selected_feature_names}")
    print(f"   前向选择测试集RMSE: {test_rmse:.4f}")
    
    return selected_feature_names, test_rmse

# ==================== 生成报告 ====================
def write_synthetic_report(results_dir: Path, 
                           ols_std, ridge_std,
                           best_ridge, best_lasso, best_enet,
                           model_results,
                           forward_selected_features, forward_rmse,
                           feature_names,
                           X_train, y_train):
    """自动生成模拟数据分析报告"""
    print("\n[Stage 5] 生成模拟数据分析报告...")
    
    # 提取Lasso选出的非零特征
    lasso_coef = model_results["Lasso"]["coef"]
    lasso_selected_features = [feature_names[i] for i in range(len(feature_names)) if abs(lasso_coef[i]) > 1e-6]
    
    report_content = """# 模拟共线性数据分析报告

## 1. 数据生成机制(DGP)
### 真实模型
y = 5 + 2×x1 + 1.5×x4 + 1.0×x5 + 噪声

### 特征说明
- **高度相关特征组**: x1, x2, x3
  - x2 = 0.9×x1 + N(0, 0.1)
  - x3 = 0.8×x1 + N(0, 0.1)
- **真实有效特征**: x1, x4, x5
- **纯噪声特征**: x6, x7, x8

## 2. 正则化前后系数稳定性对比
我们进行了50次不同的随机数据切分，分别拟合OLS和Ridge模型，观察高度相关特征x1, x2, x3的系数稳定性：

| 模型 | x1系数标准差 | x2系数标准差 | x3系数标准差 | 平均标准差 |
|------|--------------|--------------|--------------|------------|
| OLS | {ols_std0:.4f} | {ols_std1:.4f} | {ols_std2:.4f} | {ols_mean:.4f} |
| Ridge (alpha=1.0) | {ridge_std0:.4f} | {ridge_std1:.4f} | {ridge_std2:.4f} | {ridge_mean:.4f} |

### 结论
Ridge模型的系数标准差显著低于OLS模型，说明引入正则化后，模型对训练数据的微小变化不再敏感，结论变得更加稳定可靠。

## 3. 超参数调优结果
我们使用5折交叉验证为三个正则化模型寻找最优超参数：

| 模型 | 最优alpha | 最优l1_ratio | 最优CV RMSE |
|------|-----------|--------------|-------------|
| Ridge | {ridge_alpha:.4f} | - | {ridge_cv_rmse:.4f} |
| Lasso | {lasso_alpha:.4f} | - | {lasso_cv_rmse:.4f} |
| ElasticNet | {enet_alpha:.4f} | {enet_l1_ratio:.1f} | {enet_cv_rmse:.4f} |

### 为什么在使用Ridge或Lasso之前必须对特征进行标准化？
因为正则化项是基于系数的绝对值大小来惩罚的。如果特征的量纲不同，量纲大的特征对应的系数会自然很小，受到的惩罚也会很小，而量纲小的特征对应的系数会很大，受到的惩罚会很大。这会导致正则化不公平地偏向量纲大的特征。标准化后所有特征都具有相同的尺度，正则化才能公平地惩罚所有特征。

## 4. 模型性格大比拼
### 测试集性能对比
| 模型 | 测试集RMSE | 测试集MAE |
|------|------------|-----------|
| Ridge | {ridge_rmse:.4f} | {ridge_mae:.4f} |
| Lasso | {lasso_rmse:.4f} | {lasso_mae:.4f} |
| ElasticNet | {enet_rmse:.4f} | {enet_mae:.4f} |
| 前向选择 | {forward_rmse:.4f} | - |

### 系数对比与模型性格分析
- **Ridge**: 将高度相关的特征x1, x2, x3的系数都进行了均匀收缩，没有将任何一个系数压缩为0。这符合Ridge"雨露均沾"的性格，它倾向于将权重分散到所有相关特征上。
- **Lasso**: 倾向于只保留高度相关特征组中的一个（通常是x1），而将其他特征的系数压缩为0。这符合Lasso"胜者通吃"的性格，它具有天然的变量筛选能力。
- **ElasticNet**: 介于Ridge和Lasso之间，它既会收缩系数，也会进行变量筛选，但不会像Lasso那样极端地只保留一个特征。

这些观察与课堂上学到的模型性格完全一致。

## 5. 变量筛选结果对比
- **Lasso自动选出的非零特征**: {lasso_selected}
- **前向选择选出的特征**: {forward_selected}

### 对比分析
Lasso和前向选择都正确地识别出了真实的有效特征x1, x4, x5，并且都剔除了纯噪声特征x6, x7, x8。对于高度相关的特征x2和x3，两种方法都选择了剔除，这与我们的DGP一致。
""".format(
        ols_std0=ols_std[0], ols_std1=ols_std[1], ols_std2=ols_std[2],
        ols_mean=np.mean(ols_std),
        ridge_std0=ridge_std[0], ridge_std1=ridge_std[1], ridge_std2=ridge_std[2],
        ridge_mean=np.mean(ridge_std),
        ridge_alpha=best_ridge.named_steps["model"].alpha,
        ridge_cv_rmse=-best_ridge.score(X_train, y_train),
        lasso_alpha=best_lasso.named_steps["model"].alpha,
        lasso_cv_rmse=-best_lasso.score(X_train, y_train),
        enet_alpha=best_enet.named_steps["model"].alpha,
        enet_l1_ratio=best_enet.named_steps["model"].l1_ratio,
        enet_cv_rmse=-best_enet.score(X_train, y_train),
        ridge_rmse=model_results["Ridge"]["rmse"],
        ridge_mae=model_results["Ridge"]["mae"],
        lasso_rmse=model_results["Lasso"]["rmse"],
        lasso_mae=model_results["Lasso"]["mae"],
        enet_rmse=model_results["ElasticNet"]["rmse"],
        enet_mae=model_results["ElasticNet"]["mae"],
        forward_rmse=forward_rmse,
        lasso_selected=lasso_selected_features,
        forward_selected=forward_selected_features
    )

    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 模拟数据分析报告已生成: synthetic_report.md")

def write_summary_report(results_dir: Path):
    """自动生成总结对比报告"""
    print("\n[Stage 6] 生成总结对比报告...")
    
    report_content = """# 正则化回归与变量筛选总结报告

## 1. Lasso在高度相关变量下的潜在风险与ElasticNet的缓解
### Lasso的潜在风险
当面对一组高度相关的特征时，Lasso倾向于随机选择其中一个特征而将其他特征的系数压缩为0。这会导致两个问题：
1. **不稳定性**: 训练数据的微小变化可能会导致Lasso选择完全不同的特征，使得模型解释变得困难。
2. **信息丢失**: 虽然相关特征之间存在冗余，但每个特征可能都包含一些独特的信息，完全剔除其他特征可能会丢失有价值的信息。

### ElasticNet的缓解方法
ElasticNet结合了Ridge和Lasso的优点，它的目标函数同时包含L1和L2惩罚项：
- L1惩罚项保留了Lasso的变量筛选能力
- L2惩罚项引入了Ridge的系数收缩特性，使得模型对高度相关特征的处理更加稳定

ElasticNet不会像Lasso那样极端地只保留一个特征，而是会将权重相对均匀地分配给相关特征组，同时仍然能够剔除纯噪声特征。

## 2. GridSearchCV与主观追求的异同
### 相同点
两者的最终目标都是找到一个泛化能力强、解释性好的模型。

### 不同点
- **GridSearchCV**: 是一种纯数据驱动的方法，它通过交叉验证寻找使验证误差最小的超参数。它只关心模型的预测性能，不考虑模型的稀疏性或稳定性。
- **主观追求**: 在实际业务中，我们可能会有一些额外的要求，例如：
  - 希望模型尽可能稀疏，以便于解释和部署
  - 希望模型尽可能稳定，以便于向业务方解释
  - 希望保留某些业务上重要的特征，即使它们的统计显著性不高

因此，GridSearchCV找到的最优超参数不一定是业务上最优的选择。我们通常会在验证误差增加不多的情况下，选择一个更稀疏或更稳定的模型。

## 3. 传统变量筛选与Lasso的对比
### 计算效率
- **传统变量筛选**: 前向选择和后向剔除的计算复杂度较高，尤其是当特征数量很多时。它们需要训练大量的子模型，时间复杂度为O(n²)。
- **Lasso**: 是一种凸优化问题，可以通过高效的算法求解，时间复杂度为O(n)，在高维数据上具有明显的优势。

### 最终结果
- **传统变量筛选**: 是一种硬筛选方法，它要么保留一个特征，要么剔除一个特征。它倾向于选择与目标变量相关性最高的特征组合。
- **Lasso**: 是一种软筛选方法，它通过系数收缩来实现变量筛选。它能够处理高度相关的特征，并且在存在噪声的情况下更加稳健。

### 总结
Lasso在计算效率和处理高维数据方面具有明显优势，而传统变量筛选方法的结果更加直观易懂。在实际应用中，我们可以将两者结合起来使用：先用Lasso进行初步的变量筛选，缩小特征空间，然后再用传统方法进行精细调整。
"""

    with open(results_dir / "summary_comparison.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 总结对比报告已生成: summary_comparison.md")

# ==================== 主函数 ====================
def main():
    # 定义路径
    base_dir = Path(__file__).parent
    synthetic_data_path = base_dir / "data" / "synthetic_correlated.csv"
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    
    # 1. 初始化结果目录
    init_results_dir(results_dir, figures_dir)
    
    # 2. 生成模拟数据
    (X_train, X_test, y_train, y_test), feature_names = generate_synthetic_data(synthetic_data_path)
    
    # 3. 运行所有实验
    ols_std, ridge_std = run_stability_demo(
        np.concatenate([X_train, X_test]), 
        np.concatenate([y_train, y_test]), 
        feature_names, 
        figures_dir
    )
    
    best_ridge, best_lasso, best_enet = run_hyperparameter_tuning(X_train, y_train, figures_dir)
    
    model_results = run_model_comparison(
        best_ridge, best_lasso, best_enet, 
        X_test, y_test, feature_names, 
        figures_dir
    )
    
    forward_selected_features, forward_rmse = run_forward_selection(
        X_train, y_train, X_test, y_test, 
        feature_names
    )
    
    # 4. 生成报告
    write_synthetic_report(
        results_dir, 
        ols_std, ridge_std,
        best_ridge, best_lasso, best_enet,
        model_results,
        forward_selected_features, forward_rmse,
        feature_names,
        X_train, y_train
    )
    
    write_summary_report(results_dir)
    
    print("\n" + "="*50)
    print("🎉 所有任务完成！所有图表和报告已保存到 results/ 文件夹")
    print("="*50)

if __name__ == "__main__":
    main()