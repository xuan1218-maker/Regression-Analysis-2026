#!/usr/bin/env python3
"""
模型评估脚本
功能：读取清洗后的数据，进行多重共线性诊断和交叉验证评估
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif


def print_red(text: str):
    """打印红色字体"""
    print(f'\033[91m{text}\033[0m')


def print_yellow(text: str):
    """打印黄色字体"""
    print(f'\033[93m{text}\033[0m')


def main():
    parser = argparse.ArgumentParser(description='模型诊断与交叉验证评估')
    parser.add_argument('--data', '-d', type=str, required=True, help='清洗后的数据文件路径')
    parser.add_argument('--target', '-t', type=str, default='Sales', help='目标变量列名（默认: Sales）')
    parser.add_argument('--folds', '-k', type=int, default=5, help='交叉验证折数，默认5')
    
    args = parser.parse_args()
    
    data_path = args.data
    target_col = args.target
    n_folds = args.folds
    
    print("=" * 60)
    print("模型诊断与交叉验证评估")
    print("=" * 60)
    print(f"读取数据: {data_path}")
    print(f"目标变量: {target_col}")
    
    # 1. 读取清洗后的数据
    df = pd.read_csv(data_path)
    print(f"数据加载成功，形状: {df.shape}")
    
    # 检查目标变量是否存在
    if target_col not in df.columns:
        print(f"\n错误：目标变量 '{target_col}' 不在数据中！")
        print(f"可用的列名: {list(df.columns)}")
        sys.exit(1)
    
    # 分离特征和目标
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"\n特征数: {X.shape[1]}")
    print(f"样本数: {X.shape[0]}")
    print(f"特征列: {feature_cols}")
    
    # 2. 多重共线性体检 - 计算 VIF
    print("\n" + "=" * 60)
    print("多重共线性诊断（VIF - 方差膨胀因子）")
    print("=" * 60)
    
    vif_values = calculate_vif(X)
    
    print("\n特征                               VIF值        状态")
    print("-" * 60)
    
    has_high_vif = False
    high_vif_features = []
    
    for i in range(len(feature_cols)):
        name = feature_cols[i]
        vif = vif_values[i]
        
        if vif == float('inf'):
            status = "🔴 严重共线性！"
            has_high_vif = True
            high_vif_features.append(name)
            print_red(f"{name:<30} {'∞':<12} {status}")
        elif vif > 10:
            status = f"🔴 VIF > 10"
            has_high_vif = True
            high_vif_features.append(name)
            print_red(f"{name:<30} {vif:.2f}{'':<6} {status}")
        elif vif > 5:
            status = "🟡 较强共线性"
            print_yellow(f"{name:<30} {vif:.2f}{'':<6} {status}")
        else:
            status = "🟢 正常"
            print(f"{name:<30} {vif:.2f}{'':<6} {status}")
    
    if has_high_vif:
        print_red("\n" + "=" * 60)
        print_red(f"🚨 警告：检测到严重多重共线性！")
        print_red(f"   特征 {high_vif_features} 的 VIF > 10")
        print_red("   建议：删除其中一个高度相关的特征")
        print_red("=" * 60)
    
    # 3. 交叉验证
    print("\n" + "=" * 60)
    print(f"{n_folds}折交叉验证 - AnalyticalOLS 模型评估")
    print("=" * 60)
    
    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    
    # 随机打乱索引
    indices = np.random.permutation(n_samples)
    
    r_squared_scores = []
    
    print(f"\n{'折数':<10}{'训练集大小':<15}{'验证集大小':<15}{'R²':<15}")
    print("-" * 55)
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        try:
            model = AnalyticalOLS(add_intercept=True)
            model.fit(X_train, y_train)
            
            r2 = model.score(X_val, y_val)
            r_squared_scores.append(r2)
            
            status = "✅" if r2 > 0.8 else "⚠️"
            print(f"{fold+1:<10}{len(train_indices):<15}{len(val_indices):<15}{r2:.4f} {status}")
            
        except np.linalg.LinAlgError as e:
            print(f"{fold+1:<10}{len(train_indices):<15}{len(val_indices):<15}{'奇异矩阵!':<15} ❌")
            print_red(f"\n错误：第 {fold+1} 折出现奇异矩阵！")
            print_red("请检查是否存在虚拟变量陷阱或多重共线性问题。")
            sys.exit(1)
    
    print("-" * 55)
    
    mean_r2 = np.mean(r_squared_scores)
    std_r2 = np.std(r_squared_scores)
    
    print(f"\n📈 交叉验证结果汇总:")
    print(f"   平均 R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   各折 R²: {[f'{r:.4f}' for r in r_squared_scores]}")
    
    # 4. 课堂讨论问题
    print("\n" + "=" * 60)
    print("💭 课堂讨论思考题")
    print("=" * 60)
    print_yellow("问题：既然在 data_prep.py 里用全量数据的均值填补了所有缺失值，")
    print_yellow("      那么在 5 折交叉验证时，验证集数据真的算是")
    print_yellow("      ‘完全未见过的陌生数据’吗？")
    print("\n答案：不算！")
    print("      原因：均值填补时使用的是整个数据集的均值（包含验证集），")
    print("      这导致了数据泄露（Data Leakage），使模型评估结果过于乐观。")
    print("      正确做法：在每折训练集中独立计算均值进行填补。")
    print("=" * 60)
    
    if mean_r2 > 0.9:
        print_yellow(f"\n⚠️ 注意：R² = {mean_r2:.4f} > 0.9，分数异常偏高！")
        print_yellow("   这可能是因为数据泄露导致的乐观估计。")


if __name__ == "__main__":
    main()