
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ 加载数据: {path}, 形状: {df.shape}")
    return df

def handle_missing_all_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    对所有数值列进行均值填补（防止任何 NaN 残留）
    """
    print("\n🔧 处理缺失值 (均值填补所有数值列)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            missing_cnt = df[col].isnull().sum()
            df[col] = df[col].fillna(mean_val)   # 避免链式赋值警告
            print(f"   {col}: 填补 {missing_cnt} 个缺失值，均值 = {mean_val:.2f}")
    return df

def winsorize(df: pd.DataFrame, num_cols: list, pct: float = 99) -> pd.DataFrame:
    print(f"\n🔧 异常值处理 (Winsorization 到 {pct}% 分位数)...")
    for col in num_cols:
        if col not in df.columns:
            continue
        cap = df[col].quantile(pct / 100)
        before_max = df[col].max()
        if before_max > cap:
            outliers = (df[col] > cap).sum()
            df[col] = np.where(df[col] > cap, cap, df[col])
            print(f"   {col}: 缩尾阈值 = {cap:.2f}, 处理 {outliers} 个异常值")
    return df

def encode_categorical(df: pd.DataFrame, cat_cols: list, drop_first: bool = True) -> pd.DataFrame:
    print(f"\n🔧 分类变量 One-Hot 编码 (drop_first={drop_first})...")
    for col in cat_cols:
        if col in df.columns:
            print(f"   {col}: 类别 = {df[col].unique().tolist()}")
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    return df

def save_data(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n✅ 清洗后数据保存至: {path}")
    print(f"   最终形状: {df.shape}")
    # 确认无缺失值
    if df.isnull().sum().sum() > 0:
        print("⚠️ 警告：保存的数据中仍有缺失值！")
        print(df.isnull().sum())
    else:
        print("✅ 数据中无缺失值")

def main():
    parser = argparse.ArgumentParser(description="营销数据清洗工具")
    parser.add_argument("--input", "-i", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--winsorize-pct", type=float, default=99, help="缩尾百分位数，默认 99")
    parser.add_argument("--cat-cols", nargs="+", default=["Region"], help="分类变量列名")
    parser.add_argument("--num-cols", nargs="+", 
                        default=["TV_Budget", "Online_Video_Budget", "Radio_Budget"],
                        help="需要缩尾的数值变量列名")
    args = parser.parse_args()

    print("=" * 60)
    print("数据急救员 - 营销数据清洗 (修复版)")
    print("=" * 60)

    df = load_data(args.input)

    # 1. 先处理所有数值列的缺失值
    df = handle_missing_all_numeric(df)

    # 2. 对指定数值列进行缩尾
    df = winsorize(df, args.num_cols, args.winsorize_pct)

    # 3. 分类变量编码
    df = encode_categorical(df, args.cat_cols, drop_first=True)

    # 4. 编码后再次检查并填补缺失值（理论上不会产生新缺失，但安全起见）
    df = handle_missing_all_numeric(df)

    # 5. 保存
    save_data(df, args.output)

if __name__ == "__main__":
    main()