import argparse
import pandas as pd
import numpy as np

def winsorize_column(series: pd.Series, quantile: float = 0.99) -> pd.Series:
    """对数值列进行99分位数缩尾处理，处理异常值"""
    upper_threshold = series.quantile(quantile)
    return series.clip(upper=upper_threshold)

def main():
    # 严格使用argparse解析命令行参数，不硬编码任何路径
    parser = argparse.ArgumentParser(description="第九周数据清洗CLI工具")
    parser.add_argument("--input", required=True, help="输入脏数据的CSV文件路径")
    parser.add_argument("--output", required=True, help="输出清洗后数据的CSV文件路径")
    args = parser.parse_args()

    # 1. 读取原始数据
    df = pd.read_csv(args.input)

    # 2. 处理缺失值：数值型用中位数填充，分类型用众数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. 处理异常值：对所有预算列进行99分位数缩尾
    for col in numeric_cols:
        if "Budget" in col or "budget" in col:
            df[col] = winsorize_column(df[col])

    # 4. One-Hot编码 + 丢弃第一列，彻底避免虚拟变量陷阱
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 5. 保存清洗后的数据
    df.to_csv(args.output, index=False)
    print(f"✅ 数据清洗完成，已保存至: {args.output}")

if __name__ == "__main__":
    main()