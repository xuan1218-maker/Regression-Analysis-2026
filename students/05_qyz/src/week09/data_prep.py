"""
Week 9 数据预处理 CLI 脚本
=============================
功能：
    1. 将分类变量进行 One-Hot 编码（并丢弃第一列，防止虚拟变量陷阱）
    2. 对数值型特征进行 Winsorization（缩尾处理），将极端值限制在 99 分位数
    3. 处理缺失值，使用列级均值进行填充
    4. 保存清洁后的数据

用法:
    python data_prep.py --input path/to/dirty_data.csv --output path/to/clean_data.csv

示例:
    python data_prep.py --input dirty_marketing.csv --output clean_marketing.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os


def main():
    """
    主函数：CLI 入口，负责数据预处理流水线
    """
    # ========== 第一步：解析命令行参数 ==========
    parser = argparse.ArgumentParser(
        description="数据清洗脚本：处理分类变量、异常值和缺失值"
    )
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径（脏数据）")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径（干净数据）")
    args = parser.parse_args()

    # ========== 验证输入文件 ==========
    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        sys.exit(1)

    # ========== 第二步：加载数据 ==========
    df = pd.read_csv(args.input)
    print(f"✓ 已加载数据，形状: {df.shape}")

    # ========== 第三步：识别列类型 ==========
    # 识别分类列（object 类型）和数值列（numeric 类型）
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"  - 分类列: {categorical_cols}")
    print(f"  - 数值列: {numerical_cols}")

    # ========== 第四步：One-Hot 编码（处理分类变量） ==========
    # 防雷：必须使用 drop_first=True 来丢弃第一列分类
    # 原因：如果 k 个分类有 k 列虚拟变量，会导致完全多重共线性，X'X 不可逆
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"✓ 已进行 One-Hot 编码（drop_first=True），新形状: {df.shape}")

    # ========== 第五步：Winsorization（缩尾处理异常值） ==========
    # 将超过 99 分位数的极端值限制在 99 分位数的水平
    # 作用：防止异常值对回归系数的过度影响
    for col in numerical_cols:
        if col in df.columns:  # 列可能在 One-Hot 后被移除
            percentile_99 = df[col].quantile(0.99)
            count_before = (df[col] > percentile_99).sum()
            df[col] = np.where(df[col] > percentile_99, percentile_99, df[col])
            if count_before > 0:
                print(
                    f"  - {col}: 缩尾处理 {count_before} 个异常值到 {percentile_99:.2f}"
                )

    # ========== 第六步：填补缺失值 ==========
    # 使用列级均值进行填充（本周临时方案，下周会改进）
    # 注意：这会导致数据泄露，因为我们用全局均值填补缺失值
    initial_na_count = df.isna().sum().sum()
    df = df.fillna(df.mean())
    print(f"✓ 已填补缺失值: {initial_na_count} 个 NaN 值已用列均值替换")

    # ========== 第七步：保存清洁数据 ==========
    df.to_csv(args.output, index=False)
    print(f"✓ 清洁数据已保存到: {args.output}")
    print(f"  最终形状: {df.shape}")


if __name__ == "__main__":
    main()
