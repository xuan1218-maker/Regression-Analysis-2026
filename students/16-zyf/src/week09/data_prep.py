import argparse
import pandas as pd
import numpy as np


def winsorize_column(series, percentile=0.99):
    upper_limit = series.quantile(percentile)
    return np.where(series > upper_limit, upper_limit, series)


def main():
    # =========================
    # 读取命令行参数
    # =========================
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    # =========================
    # 读取数据
    # =========================
    df = pd.read_csv(args.input)

    print("原始数据：")
    print(df.head())

    # =========================
    # 处理缺失值
    # 数值列用中位数填补
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # =========================
    # 处理异常值（Winsorization）
    # 对预算列进行99分位缩尾
    # =========================
    budget_cols = [
        "TV_Budget",
        "Online_Video_Budget",
        "Radio_Budget"
    ]

    for col in budget_cols:
        if col in df.columns:
            df[col] = winsorize_column(df[col])

    # =========================
    # One-Hot编码
    # 防止 Dummy Variable Trap
    # =========================
    categorical_cols = ["Region"]

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    # =========================
    # 保存数据
    # =========================
    df.to_csv(args.output, index=False)

    print("\n清洗完成！")
    print(f"文件已保存到: {args.output}")


if __name__ == "__main__":
    main()