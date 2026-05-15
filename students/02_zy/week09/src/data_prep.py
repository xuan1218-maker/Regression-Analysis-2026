import argparse
import pandas as pd


def clean_data(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # 1. 处理异常值：预算列超过 99 分位数的值，缩到 99 分位数
    possible_budget_cols = ["budget", "Budget", "marketing_budget", "ad_budget"]

    for col in possible_budget_cols:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=q99)

    # 2. 处理缺失值
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. One-Hot 编码，drop_first=True 防止虚拟变量陷阱
    df = pd.get_dummies(df, drop_first=True)

    # 4. 保存清洗后的数据
    df.to_csv(output_path, index=False)

    print(f"Clean data saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 9 Data Preparation Script")

    parser.add_argument(
        "--input",
        required=True,
        help="原始数据路径，例如 data/dirty_marketing.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="清洗后数据保存路径，例如 data/clean_marketing.csv",
    )

    args = parser.parse_args()

    clean_data(args.input, args.output)


if __name__ == "__main__":
    main()