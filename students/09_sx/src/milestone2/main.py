#!/usr/bin/env python3
"""
第二阶段里程碑大作业：工业流水线与无泄漏的泛化评估
"""

import sys
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler, SimpleImputer


def find_data_file(filename: str) -> str:
    """查找数据文件"""
    possible_paths = [
        f"homework/week09/data/{filename}",
        f"data/{filename}",
        f"students/09_sx/data/{filename}",
        f"../data/{filename}",
        f"../../data/{filename}",
        f"../../../data/{filename}",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"找不到数据文件: {filename}")


def setup_results_dir() -> str:
    """设置结果目录（在 main.py 同目录下）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    print(f"✅ 已创建结果目录: {results_dir}/")
    return results_dir


def load_and_prepare_data(file_path: str, target_col: str = "Sales"):
    """加载数据"""
    df = pd.read_csv(file_path)
    print(f"\n📂 加载数据: {file_path}")
    print(f"   数据形状: {df.shape}")
    
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    if target_col not in df.columns:
        raise ValueError(f"目标变量 '{target_col}' 不在数据中")
    
    feature_names = [col for col in df.columns if col != target_col]
    X = df[feature_names].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    
    print(f"   特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    return X, y, feature_names


def bad_cross_validation(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """有数据泄露的交叉验证"""
    print("\n" + "=" * 70)
    print("🔥 Task 3: 危险的诱惑 —— 制造数据泄露")
    print("=" * 70)
    print("⚠️  警告：以下操作存在严重的数据泄露！")
    print("     - 对全量数据进行标准化（验证集信息泄露到训练集）")
    print("     - 用全局均值填补缺失值（验证集信息泄露到训练集）")
    print("=" * 70)
    
    global_imputer = SimpleImputer(strategy='mean')
    X_cleaned = global_imputer.fit_transform(X)
    
    global_scaler = CustomStandardScaler()
    X_scaled = global_scaler.fit_transform(X_cleaned)
    
    n_samples = X_scaled.shape[0]
    fold_size = n_samples // n_folds
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    rmse_scores, mae_scores, mape_scores = [], [], []
    
    print(f"\n{'折数':<10}{'训练集大小':<15}{'验证集大小':<15}{'RMSE':<12}{'MAE':<12}{'MAPE(%)':<12}")
    print("-" * 75)
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train = X_scaled[train_indices]
        y_train = y[train_indices]
        X_val = X_scaled[val_indices]
        y_val = y[val_indices]
        
        model = GradientDescentOLS(learning_rate=0.01, max_iter=500, gd_type="full_batch")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        rmse_scores.append(calculate_rmse(y_val, y_pred))
        mae_scores.append(calculate_mae(y_val, y_pred))
        mape_scores.append(calculate_mape(y_val, y_pred))
        
        print(f"{fold+1:<10}{len(train_indices):<15}{len(val_indices):<15}{rmse_scores[-1]:<12.4f}{mae_scores[-1]:<12.4f}{mape_scores[-1]:<12.2f}")
    
    print("-" * 75)
    print(f"\n📊 平均指标（存在数据泄露）:")
    print(f"   RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"   MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"   MAPE: {np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")
    
    return {
        'rmse_mean': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores),
        'mape_mean': np.mean(mape_scores), 'mape_std': np.std(mape_scores),
        'all_rmse': rmse_scores, 'all_mae': mae_scores, 'all_mape': mape_scores
    }


def good_cross_validation(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """无数据泄露的交叉验证"""
    print("\n" + "=" * 70)
    print("🛡️ Task 4: 坚不可摧的护城河 —— 无泄露交叉验证")
    print("=" * 70)
    print("✅ 正确操作：")
    print("     - 每折独立使用训练集填补缺失值")
    print("     - 每折独立使用训练集标准化")
    print("     - 用训练集参数转换验证集")
    print("=" * 70)
    
    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    rmse_scores, mae_scores, mape_scores = [], [], []
    
    print(f"\n{'折数':<10}{'训练集大小':<15}{'验证集大小':<15}{'RMSE':<12}{'MAE':<12}{'MAPE(%)':<12}")
    print("-" * 75)
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train_raw = X[train_indices].copy()
        y_train = y[train_indices]
        X_val_raw = X[val_indices].copy()
        y_val = y[val_indices]
        
        imputer = SimpleImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train_raw)
        X_val_filled = imputer.transform(X_val_raw)
        
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        
        model = GradientDescentOLS(learning_rate=0.01, max_iter=500, gd_type="full_batch")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        rmse_scores.append(calculate_rmse(y_val, y_pred))
        mae_scores.append(calculate_mae(y_val, y_pred))
        mape_scores.append(calculate_mape(y_val, y_pred))
        
        print(f"{fold+1:<10}{len(train_indices):<15}{len(val_indices):<15}{rmse_scores[-1]:<12.4f}{mae_scores[-1]:<12.4f}{mape_scores[-1]:<12.2f}")
    
    print("-" * 75)
    print(f"\n📊 平均指标（无数据泄露）:")
    print(f"   RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"   MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"   MAPE: {np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")
    
    return {
        'rmse_mean': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores),
        'mape_mean': np.mean(mape_scores), 'mape_std': np.std(mape_scores),
        'all_rmse': rmse_scores, 'all_mae': mae_scores, 'all_mape': mape_scores
    }


def generate_comparison_chart(results_dir: str, bad_results: dict, good_results: dict):
    """生成三图合一的对比图"""
    chart_path = os.path.join(results_dir, "leakage_analysis.png")
    
    # 创建一个 1x3 的子图布局
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['RMSE', 'MAE', 'MAPE (%)']
    bad_values = [bad_results['rmse_mean'], bad_results['mae_mean'], bad_results['mape_mean']]
    good_values = [good_results['rmse_mean'], good_results['mae_mean'], good_results['mape_mean']]
    bad_stds = [bad_results['rmse_std'], bad_results['mae_std'], bad_results['mape_std']]
    good_stds = [good_results['rmse_std'], good_results['mae_std'], good_results['mape_std']]
    
    categories = ['With Leakage', 'No Leakage']
    colors = ['#FF6B6B', '#4ECDC4']
    
    for i, ax in enumerate(axes):
        bars = ax.bar(categories, [bad_values[i], good_values[i]], 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # 添加误差线
        ax.errorbar(categories, [bad_values[i], good_values[i]], 
                   yerr=[bad_stds[i], good_stds[i]], 
                   fmt='none', capsize=5, color='gray', linewidth=1.5)
        
        # 在柱子上方显示数值
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points", 
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(metrics[i], fontsize=12, fontweight='bold')
        ax.set_title(f'{metrics[i]} Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(bad_values[i], good_values[i]) * 1.15)
    
    # 整体标题
    fig.suptitle('Impact of Data Leakage on Model Evaluation Metrics', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 三合一对比图已保存: {chart_path}")


def save_report(results_dir: str, bad_results: dict, good_results: dict):
    """保存报告（不包含时间戳）"""
    report_path = os.path.join(results_dir, "evaluation_comparison.md")
    
    rmse_diff = good_results['rmse_mean'] - bad_results['rmse_mean']
    mae_diff = good_results['mae_mean'] - bad_results['mae_mean']
    mape_diff = good_results['mape_mean'] - bad_results['mape_mean']
    
    report = f"""# 模型评估对比报告

## 评估指标对比

| 指标 | 有数据泄露 | 无数据泄露 | 差异 |
|------|-----------|-----------|------|
| RMSE | {bad_results['rmse_mean']:.4f} ± {bad_results['rmse_std']:.4f} | {good_results['rmse_mean']:.4f} ± {good_results['rmse_std']:.4f} | {rmse_diff:+.4f} |
| MAE | {bad_results['mae_mean']:.4f} ± {bad_results['mae_std']:.4f} | {good_results['mae_mean']:.4f} ± {good_results['mae_std']:.4f} | {mae_diff:+.4f} |
| MAPE | {bad_results['mape_mean']:.2f}% ± {bad_results['mape_std']:.2f}% | {good_results['mape_mean']:.2f}% ± {good_results['mape_std']:.2f}% | {mape_diff:+.2f}% |

## 结论

有数据泄露的版本误差更小（看起来更好），但这是虚假的！无数据泄露的版本才能反映模型的真实泛化能力。

---

## 🤖 AI 辅助学习问答

### Q1: 什么是数据泄露？为什么致命？

**A:** 数据泄露是指验证集的信息在训练阶段被"偷看"。这会导致模型评估结果过于乐观，上线后实际效果远差于预期。就像考试前让学生看到答案，考出来的分数不能代表真实水平。

### Q2: 标准化应该在交叉验证的哪个环节做？

**A:** 必须在每一折内部单独做。先用训练集 fit 标准化器（计算均值和标准差），再用这个标准化器 transform 验证集。绝对不能用全量数据先标准化再做交叉验证。

### Q3: 为什么有数据泄露的版本误差更小？

**A:** 因为验证集的统计信息（均值、标准差）被用在了训练集中，模型"提前看到"了验证集的特征分布。这就像用今天的数据预测今天，准确率当然高，但毫无意义。

### Q4: 如何向业务人员解释数据泄露的危害？

**A:** 用具体数字说明。如果泄露版本 MAE=62万，真实 MAE=75万，意味着每天低估13万风险，每年可能造成数千万的预算失误。用业务损失来说服他们。

### Q5: 业务要求用全局预处理怎么办？

**A:** 解释这是"用今天预测今天"的虚假评估。承诺提供封装好的 Pipeline，使用简单但结果可信。强调：多写10分钟代码，避免数百万的决策失误。

---

## 📊 可视化图表

![Error Comparison Chart](./leakage_analysis.png)

*图1：有数据泄露 vs 无数据泄露的误差对比（三指标合一）*
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📄 报告已保存: {report_path}")


def main():
    print("=" * 70)
    print("🏆 第二阶段里程碑大作业：工业流水线与无泄漏的泛化评估")
    print("=" * 70)
    
    results_dir = setup_results_dir()
    
    try:
        data_path = find_data_file("dirty_q4_marketing.csv")
    except FileNotFoundError:
        data_path = find_data_file("dirty_marketing.csv")
    
    X, y, _ = load_and_prepare_data(data_path)
    
    bad_results = bad_cross_validation(X, y, n_folds=5)
    good_results = good_cross_validation(X, y, n_folds=5)
    
    save_report(results_dir, bad_results, good_results)
    generate_comparison_chart(results_dir, bad_results, good_results)
    
    print("\n" + "=" * 70)
    print("✅ 实验完成！")
    print(f"📁 结果保存在: {results_dir}/")
    print("   - evaluation_comparison.md: 对比报告（含5个AI问答）")
    print("   - leakage_analysis.png: 三合一对比柱状图")
    print("=" * 70)


if __name__ == "__main__":
    main()