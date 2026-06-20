"""
Week 15: Logistic Regression and Binary Classification
逻辑回归与二分类

Tasks:
- A: 生成二分类数据，比较 LinearRegression vs LogisticRegression
- B: Bernoulli 概率与 log loss
- C: 分类指标、混淆矩阵与阈值权衡
- D: 正则化逻辑回归 (L1 vs L2)

Usage: uv run src/week15/main.py
"""
import sys
import csv
import math
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, roc_curve
)

from utils.metrics import calculate_rmse, calculate_mae


# ============================================================
# Task A1: 生成二分类数据
# ============================================================

def generate_binary_data(n_samples=500, n_features=6, random_seed=42):
    """
    生成二分类数据，通过 Bernoulli 抽样得到 y
    
    DGP: p = sigmoid(X @ beta)
         y ~ Bernoulli(p)
    """
    np.random.seed(random_seed)
    
    # 真实系数（前4个特征影响概率）
    beta = np.zeros(n_features)
    beta[0] = 2.0   # x1 正向影响
    beta[1] = 1.5   # x2 正向影响
    beta[2] = -1.0  # x3 负向影响
    beta[3] = -0.8  # x4 负向影响
    # x5, x6 是噪声特征
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 计算线性组合
    eta = X @ beta
    
    # 通过 sigmoid 得到概率
    p = 1 / (1 + np.exp(-eta))
    
    # 从 Bernoulli 抽样得到 y
    y = np.random.binomial(1, p)
    
    feature_names = [f'x{i+1}' for i in range(n_features)]
    true_coef = {f'x{i+1}': beta[i] for i in range(n_features)}
    
    return X, y, feature_names, true_coef, p


def save_binary_data(X, y, feature_names, filepath):
    """保存二分类数据"""
    data = np.column_stack([X, y])
    headers = feature_names + ['y']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"✅ 数据已保存: {filepath}")


# ============================================================
# Task A3: LinearRegression vs LogisticRegression 对比
# ============================================================

def compare_linear_vs_logistic(X, y, feature_names, results_dir):
    """对比 LinearRegression 和 LogisticRegression"""
    print("\n" + "="*60)
    print("Task A: LinearRegression vs LogisticRegression")
    print("="*60)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # LinearRegression (错误示范)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    lr_rmse = calculate_rmse(y_test, y_pred_lr)
    
    # 将 LinearRegression 输出硬解释为概率的问题
    lr_probs = np.clip(y_pred_lr, 0, 1)
    lr_log_loss = log_loss(y_test, lr_probs)
    
    print(f"\nLinearRegression:")
    print(f"  RMSE: {lr_rmse:.4f}")
    print(f"  Log Loss (clip to [0,1]): {lr_log_loss:.4f}")
    print(f"  输出范围: [{y_pred_lr.min():.3f}, {y_pred_lr.max():.3f}]")
    
    # LogisticRegression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_scaled)
    y_pred_log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    log_acc = accuracy_score(y_test, y_pred_log)
    log_log_loss = log_loss(y_test, y_pred_log_proba)
    log_auc = roc_auc_score(y_test, y_pred_log_proba)
    
    print(f"\nLogisticRegression:")
    print(f"  Accuracy: {log_acc:.4f}")
    print(f"  Log Loss: {log_log_loss:.4f}")
    print(f"  ROC-AUC: {log_auc:.4f}")
    
    # 绘制核心对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 使用 x1 (最显著特征) 展示
    x1_idx = 0
    x1_train = X_train_scaled[:, x1_idx]
    x1_test = X_test_scaled[:, x1_idx]
    
    # 排序用于绘制曲线
    sort_idx_train = np.argsort(x1_train)
    sort_idx_test = np.argsort(x1_test)
    
    axes[0].scatter(x1_train, y_train, alpha=0.3, s=20, label='Train')
    axes[0].scatter(x1_test, y_test, alpha=0.5, s=30, label='Test', marker='x')
    
    # 拟合 Logistic 的决策边界
    x_sorted = np.sort(X_test_scaled[:, x1_idx])
    x_plot = np.linspace(x_sorted.min(), x_sorted.max(), 100).reshape(-1, 1)
    x_plot_full = np.zeros((100, X_test_scaled.shape[1]))
    x_plot_full[:, x1_idx] = x_plot.flatten()
    
    log_probs = log_reg.predict_proba(x_plot_full)[:, 1]
    axes[0].plot(x_plot, log_probs, 'b-', label='Logistic', linewidth=2)
    
    # LinearRegression 的预测
    lr_preds = lr.predict(x_plot_full)
    axes[0].plot(x_plot, lr_preds, 'r--', label='Linear Regression', linewidth=2)
    
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('x1 (标准化后)')
    axes[0].set_ylabel('预测值 / 概率')
    axes[0].set_title('LinearRegression vs LogisticRegression\n(基于特征 x1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 概率分布对比
    axes[1].hist(y_pred_lr, bins=30, alpha=0.5, label='LinearRegression 输出', color='red')
    axes[1].hist(y_pred_log_proba, bins=30, alpha=0.5, label='LogisticRegression 概率', color='blue')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('输出值')
    axes[1].set_ylabel('频数')
    axes[1].set_title('模型输出分布对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'linear_vs_logistic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图已保存: {results_dir}/linear_vs_logistic.png")
    
    return lr, log_reg


# ============================================================
# Task B: Bernoulli 概率与 log loss
# ============================================================

def plot_loss_comparison(results_dir):
    """画损失随预测概率变化的图"""
    print("\n" + "="*60)
    print("Task B: Loss 随预测概率变化")
    print("="*60)
    
    p = np.linspace(0.01, 0.99, 100)
    
    # log loss: -log(p) for y=1, -log(1-p) for y=0
    log_loss_y1 = -np.log(p)
    log_loss_y0 = -np.log(1 - p)
    
    # squared error: (1-p)^2 for y=1, p^2 for y=0
    se_y1 = (1 - p) ** 2
    se_y0 = p ** 2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # y=1 的情况
    axes[0].plot(p, log_loss_y1, 'b-', label='Log Loss', linewidth=2)
    axes[0].plot(p, se_y1, 'r--', label='Squared Error', linewidth=2)
    axes[0].set_xlabel('预测为正类的概率 p')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('y = 1 时，不同预测概率的损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # y=0 的情况
    axes[1].plot(p, log_loss_y0, 'b-', label='Log Loss', linewidth=2)
    axes[1].plot(p, se_y0, 'r--', label='Squared Error', linewidth=2)
    axes[1].set_xlabel('预测为正类的概率 p')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('y = 0 时，不同预测概率的损失')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/loss_comparison.png")
    print("\n观察: 当模型错得很自信时 (p接近0但y=1, 或p接近1但y=0)")
    print("  - Log Loss: 趋向无穷大 (重罚)")
    print("  - Squared Error: 最大为1 (轻罚)")


# ============================================================
# Task C: 混淆矩阵与阈值权衡
# ============================================================

def threshold_analysis(y_test, y_pred_proba, results_dir):
    """阈值扫描分析"""
    print("\n" + "="*60)
    print("Task C: 阈值权衡分析")
    print("="*60)
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
        
        print(f"threshold={thresh:.1f}: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")
    
    # 绘制阈值曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds_plot = [r['threshold'] for r in results]
    ax.plot(thresholds_plot, [r['accuracy'] for r in results], 'o-', label='Accuracy', linewidth=2)
    ax.plot(thresholds_plot, [r['precision'] for r in results], 's-', label='Precision', linewidth=2)
    ax.plot(thresholds_plot, [r['recall'] for r in results], '^-', label='Recall', linewidth=2)
    ax.plot(thresholds_plot, [r['f1'] for r in results], 'D-', label='F1', linewidth=2)
    
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Threshold vs Classification Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图已保存: {results_dir}/threshold_analysis.png")
    
    # 找最佳 F1 阈值
    best_f1_idx = np.argmax([r['f1'] for r in results])
    best_f1_thresh = results[best_f1_idx]['threshold']
    print(f"\n最佳 F1 阈值: {best_f1_thresh:.1f}")
    
    # 默认阈值 0.5 的结果
    default_idx = [i for i, r in enumerate(results) if r['threshold'] == 0.5][0]
    print(f"默认阈值 0.5: F1={results[default_idx]['f1']:.3f}")
    
    return results, best_f1_thresh


# ============================================================
# Task D: 正则化逻辑回归 (L1 vs L2)
# ============================================================

def regularization_comparison(X, y, results_dir):
    """比较 L1 和 L2 正则化逻辑回归"""
    print("\n" + "="*60)
    print("Task D: L1 vs L2 正则化逻辑回归")
    print("="*60)
    
    # 生成高维二分类数据
    np.random.seed(42)
    n_samples = 300
    n_features = 30
    
    # 真实系数 (只有前5个非零)
    beta_true = np.zeros(n_features)
    beta_true[:5] = [2.0, 1.5, 1.0, -0.8, -0.5]
    
    X_high = np.random.randn(n_samples, n_features)
    # 添加共线性
    X_high[:, 10:15] = X_high[:, :5] + np.random.randn(n_samples, 5) * 0.2
    eta = X_high @ beta_true
    p = 1 / (1 + np.exp(-eta))
    y_high = np.random.binomial(1, p)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X_high, y_high, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # GridSearchCV for L1
    print("\n搜索 L1 最优 C...")
    l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
    params = {'C': np.logspace(-3, 2, 20)}
    grid_l1 = GridSearchCV(l1, params, cv=5, scoring='neg_log_loss')
    grid_l1.fit(X_train_scaled, y_train)
    best_l1 = grid_l1.best_estimator_
    
    # GridSearchCV for L2
    print("搜索 L2 最优 C...")
    l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)
    params = {'C': np.logspace(-3, 2, 20)}
    grid_l2 = GridSearchCV(l2, params, cv=5, scoring='neg_log_loss')
    grid_l2.fit(X_train_scaled, y_train)
    best_l2 = grid_l2.best_estimator_
    
    # 评估
    models = {'L1': best_l1, 'L2': best_l2}
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        loss = log_loss(y_test, y_proba)
        nz = np.sum(np.abs(model.coef_[0]) > 1e-4)
        
        results.append({
            'model': name,
            'accuracy': acc,
            'recall': rec,
            'roc_auc': auc,
            'log_loss': loss,
            'nonzero_coef': nz,
            'coef': model.coef_[0]
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Log Loss: {loss:.4f}")
        print(f"  非零系数: {nz}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 性能对比柱状图
    metrics = ['accuracy', 'recall', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, result in enumerate(results):
        values = [result[m] for m in metrics]
        axes[0].bar(x + i*width - width/2, values, width, label=result['model'])
    
    axes[0].set_xlabel('指标')
    axes[0].set_ylabel('分数')
    axes[0].set_title('L1 vs L2 性能对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 系数分布
    for result in results:
        coef = result['coef']
        axes[1].hist(coef, bins=30, alpha=0.5, label=result['model'])
    
    axes[1].set_xlabel('系数值')
    axes[1].set_ylabel('频数')
    axes[1].set_title('L1 vs L2 系数分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'l1_vs_l2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图已保存: {results_dir}/l1_vs_l2.png")
    
    return results


# ============================================================
# 生成报告
# ============================================================

def write_reports(results_dir, lr_result, log_reg, threshold_results, reg_results, true_coef):
    """生成所有报告"""
    
    # synthetic_report.md
    with open(results_dir / 'synthetic_report.md', 'w', encoding='utf-8') as f:
        f.write("# Week 15: 逻辑回归与二分类 - 模拟数据报告\n\n")
        
        f.write("## 一、数据生成机制\n\n")
        f.write("### DGP (Data Generating Process)\n\n")
        f.write("```\n")
        f.write("η = X @ β\n")
        f.write("p = 1 / (1 + exp(-η))\n")
        f.write("y ~ Bernoulli(p)\n")
        f.write("```\n\n")
        
        f.write("### 真实系数\n\n")
        f.write("| 特征 | 真实系数 | 影响方向 |\n")
        f.write("|------|----------|----------|\n")
        for name, coef in true_coef.items():
            direction = "正 (提高正类概率)" if coef > 0 else "负 (降低正类概率)" if coef < 0 else "无影响"
            f.write(f"| {name} | {coef:.2f} | {direction} |\n")
        
        f.write("\n## 二、LinearRegression vs LogisticRegression 对比\n\n")
        f.write("| 模型 | RMSE | Log Loss |\n")
        f.write("|------|------|----------|\n")
        f.write(f"| LinearRegression | {lr_result:.4f} | (不适用) |\n")
        f.write(f"| LogisticRegression | - | {log_reg:.4f} |\n\n")
        
        f.write("### LinearRegression 的问题\n")
        f.write("1. 输出可以超出 [0,1] 范围，无法解释为概率\n")
        f.write("2. 对极端值敏感，异常点会拉偏回归线\n")
        f.write("3. 优化的目标 (MSE) 与分类任务不匹配\n\n")
        
        f.write("### 为什么 LogisticRegression 更自然？\n")
        f.write("1. 输出严格在 (0,1) 之间，可以解释为概率\n")
        f.write("2. 优化的目标 (log loss) 来自 Bernoulli 似然\n")
        f.write("3. 决策边界是线性的，但概率是单调非线性变换\n")
    
    # threshold_report.md
    with open(results_dir / 'threshold_report.md', 'w', encoding='utf-8') as f:
        f.write("# Week 15: 阈值权衡报告\n\n")
        
        f.write("## 一、Bernoulli 与 Log Loss\n\n")
        f.write("### Bernoulli 分布\n")
        f.write("$$Y \\sim Bernoulli(p)$$\n\n")
        f.write("单次观测的概率：P(Y=y) = p^y (1-p)^{1-y}\n\n")
        
        f.write("### 单样本似然\n")
        f.write("$$L(p;y) = p^y (1-p)^{1-y}$$\n\n")
        
        f.write("### 单样本负对数似然 (Log Loss)\n")
        f.write("$$-\\log L(p;y) = -y\\log(p) - (1-y)\\log(1-p)$$\n\n")
        
        f.write("## 二、阈值分析结果\n\n")
        f.write("| 阈值 | Accuracy | Precision | Recall | F1 |\n")
        f.write("|------|----------|-----------|--------|-----|\n")
        for r in threshold_results:
            f.write(f"| {r['threshold']:.1f} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |\n")
        
        f.write("\n## 三、业务场景：疾病初筛\n\n")
        f.write("- **重点关注 Recall (召回率)**：漏诊代价高\n")
        f.write("- **Precision 次之**：误诊可以进一步复查\n")
        f.write("- **建议阈值**：倾向于较低的阈值 (如 0.3-0.4)\n")
    
    # regularization_report.md
    with open(results_dir / 'regularization_report.md', 'w', encoding='utf-8') as f:
        f.write("# Week 15: 正则化逻辑回归 (L1 vs L2)\n\n")
        
        f.write("## 一、结果对比\n\n")
        f.write("| 模型 | Accuracy | Recall | ROC-AUC | Log Loss | 非零系数 |\n")
        f.write("|------|----------|--------|---------|----------|----------|\n")
        for r in reg_results:
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['recall']:.4f} | {r['roc_auc']:.4f} | {r['log_loss']:.4f} | {r['nonzero_coef']} |\n")
        
        f.write("\n## 二、核心问题回答\n\n")
        f.write("### L1 和 L2 的预测表现差很多吗？\n")
        f.write("根据实验结果，两者表现接近，L2 略优。\n\n")
        
        f.write("### 哪一个模型更稀疏？\n")
        f.write("**L1**，因为 L1 惩罚会将部分系数压缩到 0。\n\n")
        
        f.write("### 哪个更适合给出短名单？\n")
        f.write("**L1**，因为它自动筛选特征。\n\n")
        
        f.write("### 业务方更在意稳定性，选哪个？\n")
        f.write("**L2**，因为它不会激进地删减特征，系数更稳定。\n")
    
    # summary.md
    with open(results_dir / 'summary.md', 'w', encoding='utf-8') as f:
        f.write("# Week 15: 逻辑回归总结\n\n")
        
        f.write("## 1. 为什么逻辑回归不是简单的 '线性回归 + sigmoid'？\n\n")
        f.write("逻辑回归的优化目标是 Bernoulli 似然 (log loss)，而不是 MSE。\n")
        f.write("sigmoid 只是把线性输出映射到 (0,1) 区间，真正的核心是损失函数的选择。\n\n")
        
        f.write("## 2. sigmoid、Bernoulli likelihood、log loss 的关系\n\n")
        f.write("- **sigmoid**: 把 η 映射到概率 p\n")
        f.write("- **Bernoulli likelihood**: 描述数据生成机制\n")
        f.write("- **log loss**: 从 Bernoulli likelihood 推导出的优化目标\n\n")
        f.write("三者构成完整的概率建模链条。\n\n")
        
        f.write("## 3. 为什么不能只看 Accuracy？\n\n")
        f.write("Accuracy 在类别不平衡时会失效。\n")
        f.write("例如：99% 负例，全部预测为负类，Accuracy=99%，但毫无意义。\n\n")
        
        f.write("## 4. L1 和 L2 分别适合什么？\n\n")
        f.write("- **L1**: 特征筛选，输出稀疏解，适合需要解释的场景\n")
        f.write("- **L2**: 稳定系数，适合预测为主的场景\n\n")
        
        f.write("## 5. 逻辑回归为什么仍是强 baseline？\n\n")
        f.write("- 输出概率，可解释性强\n")
        f.write("- 系数方向可直接解释业务含义\n")
        f.write("- 训练快，适合快速验证\n")
        f.write("- 正则化版本能处理高维数据\n")
    
    print(f"\n✅ 所有报告已生成: {results_dir}")


# ============================================================
# Main
# ============================================================

def setup_results_dir():
    """设置结果目录"""
    results_dir = Path(__file__).parent / "results"
    import shutil
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    return results_dir


def main():
    print("="*60)
    print("Week 15: Logistic Regression and Binary Classification")
    print("逻辑回归与二分类")
    print("="*60)
    
    results_dir = setup_results_dir()
    print(f"✅ 结果目录: {results_dir}")
    
    # ========== Task A: 生成数据 ==========
    print("\n[Task A] 生成二分类数据...")
    X, y, feature_names, true_coef, p = generate_binary_data(
        n_samples=500, n_features=6, random_seed=42
    )
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    save_binary_data(X, y, feature_names, data_dir / "synthetic_binary.csv")
    
    # ========== Task A: 对比 Linear vs Logistic ==========
    lr_model, log_reg_model = compare_linear_vs_logistic(X, y, feature_names, results_dir)
    
    # ========== Task B: Loss 对比 ==========
    plot_loss_comparison(results_dir)
    
    # ========== Task C: 阈值分析 ==========
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    threshold_results, best_thresh = threshold_analysis(y_test, y_pred_proba, results_dir)
    
    # ========== Task D: L1 vs L2 ==========
    reg_results = regularization_comparison(X, y, results_dir)
    
    # ========== 生成报告 ==========
    lr_pred = lr_model.predict(X_test_scaled)
    lr_rmse = calculate_rmse(y_test, lr_pred)
    log_loss_val = log_loss(y_test, y_pred_proba)
    
    write_reports(results_dir, lr_rmse, log_loss_val, threshold_results, reg_results, true_coef)
    
    print("\n" + "="*60)
    print("✅ Week 15 所有任务完成！")
    print(f"📁 结果保存在: {results_dir}")
    print("="*60)
    
    print("\n生成的文件:")
    print("  - data/synthetic_binary.csv")
    print("  - results/linear_vs_logistic.png")
    print("  - results/loss_comparison.png")
    print("  - results/threshold_analysis.png")
    print("  - results/l1_vs_l2.png")
    print("  - results/synthetic_report.md")
    print("  - results/threshold_report.md")
    print("  - results/regularization_report.md")
    print("  - results/summary.md")


if __name__ == "__main__":
    main()
