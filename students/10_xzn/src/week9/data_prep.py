#!/usr/bin/env python3
"""
数据预处理命令行脚本。

对原始营销数据进行清洗：
  1. 分类变量 One-Hot 编码（drop_first=True，防止虚拟变量陷阱）
  2. 数值列 Winsorization（大于 99 分位数的值缩尾到 99 分位数）
  3. 缺失值填补（数值列用均值，分类列用众数）

用法示例：
    python students/10_xzn/src/week9/data_prep.py \
        --input homework/week09/data/dirty_marketing.csv \
        --output students/10_xzn/data/clean_marketing.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="数据清洗脚本：One-Hot 编码、Winsorization、缺失值填补"
    )
    parser.add_argument(
        "--input", required=True,
        help="输入 CSV 数据的路径"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出清洗后 CSV 数据的路径"
    )
    args = parser.parse_args()

    # 读取数据
    print(f"[INFO] 读取数据: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[INFO] 原始数据形状: {df.shape}")

    # ------- 步骤 1：处理分类变量（One-Hot 编码）-------
    # 自动识别文本/category 类型的列
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"[INFO] 识别到分类列: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"[INFO] One-Hot 编码后数据形状: {df.shape}")
    else:
        print("[INFO] 未发现分类列，跳过 One-Hot 编码")

    # ------- 步骤 2：处理异常值（Winsorization）-------
    # 对数值列中大于 99 分位数的值进行缩尾
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        p99 = df[col].quantile(0.99)
        exceed_count = (df[col] > p99).sum()
        if exceed_count > 0:
            print(f"[INFO] 列 '{col}': {exceed_count} 个值 > 99 分位数({p99:.4f})，进行缩尾")
            df[col] = df[col].clip(upper=p99)
        else:
            print(f"[INFO] 列 '{col}': 无异常值超出 99 分位数")

    # ------- 步骤 3：处理缺失值 -------
    # 数值列用均值填补
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            nan_count = df[col].isna().sum()
            print(f"[INFO] 列 '{col}': {nan_count} 个缺失值，用均值 ({mean_val:.4f}) 填补")
            df[col] = df[col].fillna(mean_val)

    # 分类列（One-Hot 编码后理论上没有，但保留处理逻辑）
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            nan_count = df[col].isna().sum()
            print(f"[INFO] 列 '{col}': {nan_count} 个缺失值，用众数 ({mode_val}) 填补")
            df[col] = df[col].fillna(mode_val)

    # 最终检查：确保没有剩余缺失值
    if df.isna().any().any():
        remaining = df.isna().sum()
        remaining = remaining[remaining > 0]
        print(f"[WARNING] 仍存在缺失值的列:\n{remaining}")
        # 兜底：用 0 填补
        df = df.fillna(0)
    else:
        print("[INFO] 所有缺失值已处理完毕")

    # 保存清洗后的数据
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[INFO] 清洗后数据已保存至: {args.output}")
    print(f"[INFO] 最终数据形状: {df.shape}")
    print(f"[INFO] 列名: {list(df.columns)}")


if __name__ == "__main__":
    main()