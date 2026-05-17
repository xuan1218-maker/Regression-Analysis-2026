#!/usr/bin/env python3
"""
数据预处理命令行脚本
功能：从命令行读取输入输出路径，进行数据清洗
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    
    parser.add_argument('--input', '-i', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出清洗后数据的文件路径')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    print("=" * 60)
    print("数据急救员 - 数据预处理脚本")
    print("=" * 60)
    print(f"读取数据: {input_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60)
    
    # 1. 读取数据
    df = pd.read_csv(input_path)
    print(f"\n原始数据加载成功，形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 2. 统计缺失值
    print(f"\n缺失值统计:")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"   {col}: {missing_count} 个缺失值")
    
    # 3. 处理分类变量（One-Hot 编码，drop_first=True 避免虚拟变量陷阱）
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n处理分类变量: {categorical_cols}")
    # dtype=int 确保输出 0 和 1，而不是 True/False
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"One-Hot 编码完成（已删除第一列，避免虚拟变量陷阱）")
    print(f"编码后列数: {len(df.columns)}")
    
    # 4. 处理异常值（Winsorization：将超过99%分位数的值缩尾）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n处理异常值（Winsorization - 缩尾到99%分位数）:")
    for col in numeric_cols:
        percentile_99 = df[col].quantile(0.99)
        outlier_mask = df[col] > percentile_99
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            old_max = df[col].max()
            df.loc[outlier_mask, col] = percentile_99
            print(f"   {col}: 发现 {outlier_count} 个异常值，原最大值={old_max:.2f}，缩尾到99%分位数={percentile_99:.2f}")
    
    # 5. 处理缺失值（使用均值填补）
    print(f"\n处理缺失值（使用均值填补）:")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"   {col}: 使用均值 {mean_val:.2f} 填补 {missing_count} 个缺失值")
    
    # 6. 保存清洗后的数据
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n清洗后的数据已保存到: {output_path}")
    print(f"最终数据形状: {df.shape}")
    print(f"最终列名: {list(df.columns)}")
    
    # 7. 验证 One-Hot 编码结果是 0/1
    dummy_cols = [col for col in df.columns if col.startswith('Region_')]
    if dummy_cols:
        print(f"\n验证 One-Hot 编码结果（应为 0 和 1）:")
        for col in dummy_cols:
            unique_vals = df[col].unique()
            print(f"   {col}: 取值 {sorted(unique_vals)}")
    
    print("\n数据预处理完成！")


if __name__ == "__main__":
    main()