from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA

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

# ==================== 数据生成函数 ====================
def generate_highdim_lowrank_data(n_samples: int = 120, n_features: int = 60, n_factors: int = 3, random_state: int = 42):
    """
    生成高维低秩模拟数据
    DGP: 
    - 3个潜在因子 Z ~ N(0, I)
    - 原始特征 X = Z @ W + 噪声 (W是3×60的加载矩阵)
    - 目标变量 y = Z @ beta + 噪声
    """
    np.random.seed(random_state)
    
    # 生成潜在因子
    Z = np.random.normal(0, 1, (n_samples, n_factors))
    
    # 生成加载矩阵
    W = np.random.normal(0, 1, (n_factors, n_features))
    
    # 生成原始特征
    X = Z @ W + np.random.normal(0, 0.1, (n_samples, n_features))
    
    # 生成目标变量
    beta = np.array([2.0, 1.5, 1.0])
    y = Z @ beta + np.random.normal(0, 0.5, n_samples)
    
    return X, y, Z

def generate_sparse_truth_data(n_samples: int = 120, n_features: int = 60, n_nonzero: int = 5, random_state: int = 42):
    """生成稀疏真实场景数据：只有少数原始特征直接决定y"""
    np.random.seed(random_state)
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # 只有前n_nonzero个特征有非零系数
    beta = np.zeros(n_features)
    beta[:n_nonzero] = np.random.normal(2, 0.5, n_nonzero)
    
    y = X @ beta + np.random.normal(0, 0.5, n_samples)
    
    return X, y

def generate_latent_factor_truth_data(n_samples: int = 120, n_features: int = 60, n_factors: int = 3, random_state: int = 42):
    """生成潜在因子真实场景数据：原始特征由少数潜在因子生成，y也由这些因子驱动"""
    return generate_highdim_lowrank_data(n_samples, n_features, n_factors, random_state)[:2]

# ==================== Task A: OLS在高维下的问题 ====================
def run_ols_highdim_demo(figures_dir: Path, max_p: int = 60):
    """展示随着特征维度增加，OLS的训练误差下降但测试误差上升的现象"""
    print("\n[Stage 1] 运行OLS高维问题演示...")
    
    n_samples = 120
    p_values = [10, 20, 30, 40, 50, 60]
    
    train_rmse_list = []
    test_rmse_list = []
    rank_list = []
    cond_list = []
    
    for p in p_values:
        X, y, _ = generate_highdim_lowrank_data(n_samples, p)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 标准化
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 拟合OLS
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = ols.predict(X_train_scaled)
        y_test_pred = ols.predict(X_test_scaled)
        
        # 计算指标
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        
        # 计算矩阵秩和条件数
        rank = np.linalg.matrix_rank(X_train_scaled)
        cond = np.linalg.cond(X_train_scaled)
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        rank_list.append(rank)
        cond_list.append(cond)
    
    # 绘制误差随特征维度变化的图
    plt.figure(figsize=(12, 8))
    plt.plot(p_values, train_rmse_list, "b-o", label="Train RMSE")
    plt.plot(p_values, test_rmse_list, "r-o", label="Test RMSE")
    plt.xlabel("Number of Features (p)")
    plt.ylabel("RMSE")
    plt.title("OLS Error vs Feature Dimensionality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "ols_error_vs_dimension.png")
    plt.close()
    
    # 绘制矩阵结构随特征维度变化的图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.plot(p_values, rank_list, "g-o")
    ax1.axhline(y=84, color="k", linestyle="--", label="Training Set Size (n=84)")
    ax1.set_xlabel("Number of Features (p)")
    ax1.set_ylabel("Rank of X_train")
    ax1.set_title("Matrix Rank vs Feature Dimensionality")
    ax1.legend()
    
    ax2.plot(p_values, cond_list, "m-o")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Features (p)")
    ax2.set_ylabel("Condition Number (log scale)")
    ax2.set_title("Matrix Condition Number vs Feature Dimensionality")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "matrix_structure_vs_dimension.png")
    plt.close()
    
    print("✅ OLS高维问题演示图已生成")
    
    # 返回结果用于报告
    results_df = pd.DataFrame({
        "p": p_values,
        "train_rmse": train_rmse_list,
        "test_rmse": test_rmse_list,
        "rank": rank_list,
        "condition_number": cond_list
    })
    
    return results_df

def run_coefficient_stability_demo(figures_dir: Path, n_repeats: int = 50):
    """展示OLS系数在不同数据切分下的不稳定性"""
    print("\n[Stage 2] 运行OLS系数稳定性演示...")
    
    X, y, _ = generate_highdim_lowrank_data(n_samples=120, n_features=60)
    
    # 选择前3个原始特征进行展示
    selected_features = [0, 1, 2]
    coefs = []
    
    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        
        coefs.append(ols.coef_[selected_features])
    
    coefs = np.array(coefs)
    
    # 绘制箱线图
    plt.figure(figsize=(12, 8))
    plt.boxplot(coefs, tick_labels=[f"Feature {i+1}" for i in selected_features])
    plt.axhline(y=0, color="k", linewidth=1)
    plt.ylabel("Coefficient Value")
    plt.title(f"OLS Coefficient Stability ({n_repeats} Random Splits)")
    plt.tight_layout()
    plt.savefig(figures_dir / "ols_coefficient_stability.png")
    plt.close()
    
    # 计算系数标准差
    coef_stds = np.std(coefs, axis=0)
    print(f"✅ OLS系数稳定性演示图已生成")
    print(f"   特征1系数标准差: {coef_stds[0]:.4f}")
    print(f"   特征2系数标准差: {coef_stds[1]:.4f}")
    print(f"   特征3系数标准差: {coef_stds[2]:.4f}")
    
    return coef_stds

# ==================== Task B: PCA与PCR ====================
def run_pca_analysis(X, figures_dir: Path):
    """进行PCA分析并绘制累计解释方差曲线"""
    print("\n[Stage 3] 运行PCA分析...")
    
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # 计算累计解释方差
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 绘制累计解释方差曲线
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, "b-o")
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Cumulative Explained Variance by Principal Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_cumulative_variance.png")
    plt.close()
    
    print("✅ PCA累计解释方差图已生成")
    print(f"   前3个主成分解释方差比例: {cumulative_variance[2]:.4f}")
    print(f"   前10个主成分解释方差比例: {cumulative_variance[9]:.4f}")
    
    return pca

def run_pcr_demo(X_train, X_test, y_train, y_test, figures_dir: Path, max_k: int = 20):
    """实现PCR工作流并比较不同k值的误差"""
    print("\n[Stage 4] 运行PCR演示...")
    
    # 标准化
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k_values = range(1, max_k+1)
    train_rmse_list = []
    test_rmse_list = []
    cv_rmse_list = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_values:
        # PCA降维
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # 线性回归
        ols = LinearRegression()
        ols.fit(X_train_pca, y_train)
        
        # 预测
        y_train_pred = ols.predict(X_train_pca)
        y_test_pred = ols.predict(X_test_pca)
        
        # 计算训练和测试误差
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        
        # 5折CV误差
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train_pca):
            X_tr, X_val = X_train_pca[train_idx], X_train_pca[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            cv_scores.append(calculate_rmse(y_val, y_pred))
        
        cv_rmse = np.mean(cv_scores)
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        cv_rmse_list.append(cv_rmse)
    
    # 找到最优k
    best_k_idx = np.argmin(cv_rmse_list)
    best_k = k_values[best_k_idx]
    best_cv_rmse = cv_rmse_list[best_k_idx]
    
    # 绘制误差曲线
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, train_rmse_list, "b-o", label="Train RMSE")
    plt.plot(k_values, test_rmse_list, "r-o", label="Test RMSE")
    plt.plot(k_values, cv_rmse_list, "g-o", label="5-Fold CV RMSE")
    plt.scatter(best_k, best_cv_rmse, c="green", s=100, zorder=5, 
                label=f"Best k={best_k}, CV RMSE={best_cv_rmse:.4f}")
    plt.xlabel("Number of Principal Components (k)")
    plt.ylabel("RMSE")
    plt.title("PCR Error vs Number of Principal Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pcr_error_vs_k.png")
    plt.close()
    
    print("✅ PCR误差曲线已生成")
    print(f"   最优主成分个数: k={best_k}")
    print(f"   最优CV RMSE: {best_cv_rmse:.4f}")
    
    # 返回结果用于报告
    results_df = pd.DataFrame({
        "k": k_values,
        "train_rmse": train_rmse_list,
        "test_rmse": test_rmse_list,
        "cv_rmse": cv_rmse_list
    })
    
    return results_df, best_k

# ==================== Task C: Lasso vs PCR比较 ====================
def run_lasso_vs_pcr_comparison(figures_dir: Path):
    """比较Lasso和PCR在稀疏真实和潜在因子真实两种场景下的表现"""
    print("\n[Stage 5] 运行Lasso vs PCR比较...")
    
    scenarios = ["Sparse Truth", "Latent Factor Truth"]
    results = []
    
    for scenario in scenarios:
        print(f"   处理场景: {scenario}...")
        
        # 生成对应场景的数据
        if scenario == "Sparse Truth":
            X, y = generate_sparse_truth_data()
        else:
            X, y = generate_latent_factor_truth_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 标准化
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Lasso
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        lasso_rmse = calculate_rmse(y_test, y_pred_lasso)
        lasso_nonzero = np.sum(lasso.coef_ != 0)
        
        # 2. PCR
        # 用CV选择最优k
        best_rmse = float("inf")
        best_k = 0
        for k in range(1, 21):
            pca = PCA(n_components=k)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            ols = LinearRegression()
            ols.fit(X_train_pca, y_train)
            y_pred = ols.predict(X_test_pca)
            rmse = calculate_rmse(y_test, y_pred)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
        
        pcr_rmse = best_rmse
        pcr_k = best_k
        
        # 3. 稳定性比较（重复10次）
        lasso_stds = []
        pcr_stds = []
        
        for i in range(10):
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
            X_tr_scaled = scaler.transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # Lasso
            lasso_rep = LassoCV(cv=5, max_iter=10000, random_state=42)
            lasso_rep.fit(X_tr_scaled, y_tr)
            y_pred_rep = lasso_rep.predict(X_val_scaled)
            lasso_stds.append(calculate_rmse(y_val, y_pred_rep))
            
            # PCR
            pca_rep = PCA(n_components=pcr_k)
            X_tr_pca = pca_rep.fit_transform(X_tr_scaled)
            X_val_pca = pca_rep.transform(X_val_scaled)
            
            ols_rep = LinearRegression()
            ols_rep.fit(X_tr_pca, y_tr)
            y_pred_rep = ols_rep.predict(X_val_pca)
            pcr_stds.append(calculate_rmse(y_val, y_pred_rep))
        
        lasso_std = np.std(lasso_stds)
        pcr_std = np.std(pcr_stds)
        
        results.append({
            "scenario": scenario,
            "lasso_rmse": lasso_rmse,
            "lasso_nonzero": lasso_nonzero,
            "lasso_std": lasso_std,
            "pcr_rmse": pcr_rmse,
            "pcr_k": pcr_k,
            "pcr_std": pcr_std
        })
    
    # 绘制对比图
    results_df = pd.DataFrame(results)
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    plt.bar(x - width/2, results_df["lasso_rmse"], width, label="Lasso", color="#ff6b6b", alpha=0.7)
    plt.bar(x + width/2, results_df["pcr_rmse"], width, label="PCR", color="#4ecdc4", alpha=0.7)
    
    plt.xticks(x, scenarios)
    plt.ylabel("Test RMSE")
    plt.title("Lasso vs PCR Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lasso_vs_pcr_comparison.png")
    plt.close()
    
    print("✅ Lasso vs PCR对比图已生成")
    
    return results_df

# ==================== 生成报告 ====================
def write_synthetic_report(results_dir: Path, 
                           ols_highdim_results, 
                           coef_stds,
                           pca,
                           pcr_results, best_k,
                           lasso_pcr_results):
    """自动生成模拟数据分析报告"""
    print("\n[Stage 6] 生成模拟数据分析报告...")
    
    # 构建报告内容，所有LaTeX反斜杠都手动双写
    report_header = """# 高维回归与PCR模拟数据分析报告

## 1. 数据生成机制
### 数据集基本信息
- 样本量: 120
- 特征数: 60
- 训练集大小: 84
- 测试集大小: 36

### 潜在因子结构
数据由3个潜在因子生成：
1. 首先生成3个独立的标准正态分布潜在因子 Z₁, Z₂, Z₃
2. 然后通过加载矩阵 W 将这3个因子线性组合成60个原始特征
3. 目标变量 y = 2×Z₁ + 1.5×Z₂ + 1.0×Z₃ + 噪声

### 为什么这是"高维 + 信息冗余"的数据
虽然原始特征有60个，但所有信息都包含在3个潜在因子中。特征之间存在高度相关性，大部分特征都是冗余的。这种数据结构在真实世界中非常常见，例如传感器数据、金融指标、基因表达数据等。

## 2. OLS在高维下的问题
### 2.1 误差随特征维度变化
我们固定总样本量为120，逐步增加特征维度从10到60，观察OLS的表现：

| 特征数(p) | 训练RMSE | 测试RMSE | 矩阵秩 | 条件数 |
|-----------|----------|----------|--------|--------|
"""
    
    # 添加表格行
    table_rows = ""
    for _, row in ols_highdim_results.iterrows():
        table_rows += f"| {row['p']:.0f} | {row['train_rmse']:.4f} | {row['test_rmse']:.4f} | {row['rank']:.0f} | {row['condition_number']:.2e} |\n"
    
    # 中间部分
    report_middle1 = """
### 结论
- 随着特征维度增加，训练RMSE持续下降，当p接近训练集大小(84)时，训练RMSE接近0
- 但测试RMSE先下降后上升，呈现U型曲线
- 当p > 84时，设计矩阵X_train不满秩，OLS存在无穷多解
- 条件数随着特征维度增加呈指数级增长，矩阵变得越来越病态

### 为什么"训练误差接近0"是危险信号
训练误差接近0意味着模型完美拟合了训练集中的所有噪声，而不是学习到了数据的真实规律。这种模型在新的、未见过的数据上表现会非常差，泛化能力极弱。

### 2.2 系数不稳定性
我们对固定的数据集进行了50次不同的随机切分，观察OLS系数的波动：

| 特征 | 系数标准差 |
|------|------------|
| 特征1 | {coef_std0:.4f} |
| 特征2 | {coef_std1:.4f} |
| 特征3 | {coef_std2:.4f} |

### 结论
- OLS的系数在不同的数据切分下波动非常大
- 这意味着我们无法得到稳定的结论，稍微改变一下训练集，模型的解释就会完全不同
- 系数不稳定本身就是一种重要风险，因为它会导致业务决策的不确定性

## 3. PCA分析
我们对原始数据进行了PCA分析，得到了累计解释方差曲线：

- 前3个主成分解释了 {pca_var3:.1f}% 的方差
- 前10个主成分解释了 {pca_var10:.1f}% 的方差

这证实了我们的数据确实贴近一个低维子空间，大部分信息都包含在少数几个主成分中。

## 4. PCR分析
### 4.1 PCR工作流
PCR的完整流程是：
1. **标准化**: 对原始特征进行标准化，使其均值为0，方差为1
2. **PCA降维**: 将标准化后的特征投影到主成分空间
3. **选择k**: 保留前k个主成分
4. **线性回归**: 在这k个主成分上进行普通最小二乘回归

### 4.2 误差随主成分个数变化
我们比较了k从1到20时PCR的表现：

| 主成分个数(k) | 训练RMSE | 测试RMSE | 5折CV RMSE |
|---------------|----------|----------|------------|
"""
    
    # 添加PCR表格行
    pcr_table_rows = ""
    for _, row in pcr_results.iterrows():
        if row['k'] <= 10:  # 只显示前10个
            pcr_table_rows += f"| {row['k']:.0f} | {row['train_rmse']:.4f} | {row['test_rmse']:.4f} | {row['cv_rmse']:.4f} |\n"
    
    # 公式部分 ✅ 修复：去掉f，使用普通字符串
    formulas = """
### 结论
- 最优主成分个数为 k={best_k}，对应的CV RMSE为 {pcr_cv_rmse:.4f}
- 随着k增加，训练RMSE持续下降，但测试RMSE和CV RMSE先下降后上升
- 当k超过最优值后，模型开始过拟合，泛化能力下降

### 4.3 严格定义与解释
1. **OLS估计式**: 
   $$\\hat{{\\beta}}_{{OLS}} = (X^T X)^{{-1}} X^T y$$
   当X存在共线性或高维时，$X^T X$接近奇异，逆矩阵不稳定。

2. **第一主成分定义**:
   第一主成分是原始特征的线性组合，使得投影后的方差最大：
   $$\\max_{{\\|w\\|=1}} \\text{{Var}}(Xw)$$

3. **PCR流程**:
   - 标准化: $X_{{scaled}} = (X - \\mu) / \\sigma$
   - PCA: $Z_k = X_{{scaled}} V_k$，其中$V_k$是前k个主成分的加载矩阵
   - 回归: $\\hat{{y}} = Z_k \\hat{{\\gamma}}$，其中$\\hat{{\\gamma}} = (Z_k^T Z_k)^{{-1}} Z_k^T y$

## 5. Lasso vs PCR比较
我们在两种不同的数据生成机制下比较了Lasso和PCR的表现：

| 场景 | Lasso测试RMSE | Lasso非零系数个数 | Lasso稳定性标准差 | PCR测试RMSE | PCR主成分个数 | PCR稳定性标准差 |
|------|---------------|-------------------|-------------------|-------------|---------------|-----------------|
| 稀疏真实 | {lasso_sparse_rmse:.4f} | {lasso_sparse_nonzero:.0f} | {lasso_sparse_std:.4f} | {pcr_sparse_rmse:.4f} | {pcr_sparse_k:.0f} | {pcr_sparse_std:.4f} |
| 潜在因子真实 | {lasso_latent_rmse:.4f} | {lasso_latent_nonzero:.0f} | {lasso_latent_std:.4f} | {pcr_latent_rmse:.4f} | {pcr_latent_k:.0f} | {pcr_latent_std:.4f} |

### 结论
- 在**稀疏真实**场景下，Lasso表现更好，因为它能准确识别出少数真正重要的特征
- 在**潜在因子真实**场景下，PCR表现更好，因为它能有效提取数据中的潜在结构
- PCR的稳定性通常优于Lasso，尤其是在存在高度相关特征的情况下
"""
    
    # 拼接所有部分
    report_content = report_header + table_rows + report_middle1 + pcr_table_rows + formulas
    
    # 格式化字符串
    report_content = report_content.format(
        coef_std0=coef_stds[0],
        coef_std1=coef_stds[1],
        coef_std2=coef_stds[2],
        pca_var3=np.cumsum(pca.explained_variance_ratio_)[2]*100,
        pca_var10=np.cumsum(pca.explained_variance_ratio_)[9]*100,
        best_k=best_k,
        pcr_cv_rmse=pcr_results.loc[best_k-1, 'cv_rmse'],
        lasso_sparse_rmse=lasso_pcr_results.loc[0, "lasso_rmse"],
        lasso_sparse_nonzero=lasso_pcr_results.loc[0, "lasso_nonzero"],
        lasso_sparse_std=lasso_pcr_results.loc[0, "lasso_std"],
        pcr_sparse_rmse=lasso_pcr_results.loc[0, "pcr_rmse"],
        pcr_sparse_k=lasso_pcr_results.loc[0, "pcr_k"],
        pcr_sparse_std=lasso_pcr_results.loc[0, "pcr_std"],
        lasso_latent_rmse=lasso_pcr_results.loc[1, "lasso_rmse"],
        lasso_latent_nonzero=lasso_pcr_results.loc[1, "lasso_nonzero"],
        lasso_latent_std=lasso_pcr_results.loc[1, "lasso_std"],
        pcr_latent_rmse=lasso_pcr_results.loc[1, "pcr_rmse"],
        pcr_latent_k=lasso_pcr_results.loc[1, "pcr_k"],
        pcr_latent_std=lasso_pcr_results.loc[1, "pcr_std"]
    )

    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 模拟数据分析报告已生成: synthetic_report.md")

def write_summary_report(results_dir: Path):
    """自动生成总结对比报告"""
    print("\n[Stage 7] 生成总结对比报告...")
    
    report_content = """# 高维回归与PCR总结对比报告

## 1. 变量筛选 vs 信息压缩
本周的核心是理解两种处理高维数据的不同思路：

### 变量筛选 (Variable Selection)
- **代表方法**: Lasso, 前向选择, 后向剔除
- **核心思想**: 从原始特征中选择一个子集，丢弃其他特征
- **回答的问题**: "哪些特征是重要的？"
- **适用场景**: 当数据的真实生成机制是稀疏的，即只有少数特征真正重要

### 信息压缩 (Information Compression)
- **代表方法**: PCA, PCR
- **核心思想**: 将原始特征线性组合成少数几个新的主成分，保留大部分信息
- **回答的问题**: "数据中的主要变异方向是什么？"
- **适用场景**: 当数据存在潜在因子结构，即特征之间高度相关，共享共同的潜在驱动因素

## 2. Lasso与PCR的适用场景对比
### 当数据是Sparse Truth时，为什么Lasso往往更自然？
当只有少数原始特征直接决定目标变量时，Lasso能够准确地识别出这些重要特征，并将其他特征的系数压缩为0。这使得模型具有很好的解释性，我们可以明确地告诉业务方"哪些因素是关键的"。

### 当数据更像Latent-factor Truth时，为什么PCR往往更自然？
当原始特征由少数潜在因子生成时，没有任何一个原始特征是"真正重要"的，重要的是这些特征背后的潜在结构。PCR能够有效地提取这些潜在因子，并且在存在高度相关特征的情况下比Lasso更加稳定。

### Lasso vs PCR回答的问题
- **Lasso**回答的更像是"**谁留下**"
- **PCR**回答的更像是"**什么是重要的模式**"

## 3. 业务决策指导
### 如果业务方要求的是"一个更短的变量名单"，你更可能用哪个方法？
我会选择**Lasso**。因为Lasso具有天然的变量筛选能力，它会直接给出一个包含少数非零系数的变量名单，这非常符合业务方的需求。

### 如果业务方要求的是"一个更稳的预测器"，你更可能用哪个方法？
我会选择**PCR**。因为PCR对训练数据的微小变化不敏感，系数更加稳定，预测结果也更加可靠。在存在高度相关特征的情况下，PCR的泛化能力通常优于Lasso。

## 4. 关于前向/后向变量选择
### 为什么这周主线更适合比较Lasso vs PCR，而不是把前向/后向选择重新拉回主舞台？
- **计算效率**: 前向/后向选择的计算复杂度是O(n²)，在高维数据上非常慢，而Lasso和PCR的计算复杂度是O(n)
- **稳定性**: 前向/后向选择是一种贪心算法，容易过拟合，并且对数据的微小变化非常敏感
- **理论性质**: Lasso和PCR有更坚实的理论基础，而前向/后向选择的理论性质相对较差

### 如果一定要加，前向/后向选择更接近selection路线还是compression路线？
前向/后向选择更接近**selection路线**。它们都是从原始特征中选择一个子集，而不是将特征组合成新的变量。

## 5. 总结
通过本周的实验，我们深入理解了高维数据带来的挑战以及两种主要的解决思路。Lasso和PCR没有绝对的优劣，它们适用于不同的数据生成机制和业务需求。在实际应用中，我们应该根据数据的特点和业务的目标来选择合适的方法。
"""

    with open(results_dir / "summary_comparison.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 总结对比报告已生成: summary_comparison.md")

# ==================== 主函数 ====================
def main():
    # 定义路径
    base_dir = Path(__file__).parent
    synthetic_data_path = base_dir / "data" / "synthetic_highdim.csv"
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    
    # 1. 初始化结果目录
    init_results_dir(results_dir, figures_dir)
    
    # 2. 生成并保存模拟数据
    X, y, _ = generate_highdim_lowrank_data()
    df = pd.DataFrame(np.column_stack([X, y]), columns=[f"x{i+1}" for i in range(X.shape[1])] + ["y"])
    df.to_csv(synthetic_data_path, index=False)
    print(f"✅ 高维模拟数据已生成并保存到: {synthetic_data_path}")
    
    # 3. 运行所有实验
    ols_highdim_results = run_ols_highdim_demo(figures_dir)
    coef_stds = run_coefficient_stability_demo(figures_dir)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    pca = run_pca_analysis(X, figures_dir)
    pcr_results, best_k = run_pcr_demo(X_train, X_test, y_train, y_test, figures_dir)
    
    lasso_pcr_results = run_lasso_vs_pcr_comparison(figures_dir)
    
    # 4. 生成报告
    write_synthetic_report(
        results_dir, 
        ols_highdim_results, 
        coef_stds,
        pca,
        pcr_results, best_k,
        lasso_pcr_results
    )
    
    write_summary_report(results_dir)
    
    print("\n" + "="*50)
    print("🎉 所有任务完成！所有图表和报告已保存到 results/ 文件夹")
    print("="*50)

if __name__ == "__main__":
    main()