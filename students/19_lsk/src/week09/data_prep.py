import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_arguments():
    """
    解析命令行参数
    
    支持两种模式：
    1. argparse 模式（推荐）：--input xxx --output xxx
    2. sys.argv 遗留模式：script.py input.csv output.csv

    # 1. 进入工作目录
    cd ~/Regression-Analysis-2026/students/19_lsk

    # 2. 清洗数据
    uv run src/week09/data_prep.py \
    --input ../../src/week09/dirty_marketing.csv \
    --output week09/clean_marketing.csv

    # 3. 运行评估（确保 evaluate.py 中的数据路径已修正）
    uv run src/week09/evaluate.py
    """
    # 检查是否使用 --input/--output 格式
    if len(sys.argv) > 1 and (sys.argv[1] == '--input' or sys.argv[1] == '-i'):
        parser = argparse.ArgumentParser(
            description='数据清洗脚本：处理缺失值、异常值和分类变量编码',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  %(prog)s --input raw.csv --output clean.csv
  %(prog)s -i raw.csv -o clean.csv
            """
        )
        parser.add_argument('--input', '-i', type=str, required=True,
                           help='输入数据文件路径 (CSV格式)')
        parser.add_argument('--output', '-o', type=str, required=True,
                           help='输出清洗后数据的路径 (CSV格式)')
        parser.add_argument('--fill-method', '-f', type=str, default='median',
                           choices=['mean', 'median'],
                           help='缺失值填补方法 (默认: median)')
        parser.add_argument('--winsorize-quantile', '-w', type=float, default=0.99,
                           help='异常值缩尾分位数 (默认: 0.99)')
        
        return parser.parse_args()
    else:
        # 兼容旧格式：python script.py input.csv output.csv
        class Args:
            pass
        args = Args()
        
        if len(sys.argv) >= 3:
            args.input = sys.argv[1]
            args.output = sys.argv[2]
        else:
            print("错误：请指定输入和输出路径")
            print("用法: uv run src/week09/data_prep.py --input <输入文件> --output <输出文件>")
            sys.exit(1)
        
        args.fill_method = 'median'
        args.winsorize_quantile = 0.99
        return args


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载 CSV 数据文件
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_csv(path)
    print(f"✅ 数据加载成功: {file_path}")
    print(f"   原始形状: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")
    
    return df


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    保存清洗后的数据到 CSV 文件
    """
    path = Path(file_path)
    # 确保输出目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False)
    print(f"✅ 清洗后数据已保存: {file_path}")
    print(f"   清洗后形状: {df.shape}")


def handle_missing_values(df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
    """
    处理缺失值：使用均值或中位数填补
    
    ⚠️ 注意：这是临时方案（暴力填补），下周将会被更高级的方法替代
    """
    before_missing = df.isnull().sum().sum()
    
    if before_missing == 0:
        print(f"   缺失值: 无缺失值")
        return df
    
    print(f"   缺失值处理前: {before_missing} 个缺失值")
    
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if method == 'mean':
                    fill_value = df[col].mean()
                else:  # median
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                print(f"      - {col}: 使用 {method} ({fill_value:.2f}) 填补 {df[col].isnull().sum()} 个缺失值")
            else:
                # 非数值列使用众数填补
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(mode_value)
                print(f"      - {col}: 使用众数 '{mode_value}' 填补")
    
    after_missing = df.isnull().sum().sum()
    print(f"   缺失值处理后: {after_missing} 个缺失值")
    
    return df


def handle_outliers_winsorize(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """
    处理异常值：Winsorization（缩尾）
    
    将超过指定分位数的值替换为该分位数的值
    例如：99%分位数 = 100，那么大于 100 的值都会被替换为 100
    """
    outlier_count = 0
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # 计算上下分位数（只处理上尾，因为有极端大值的场景）
            upper_bound = df[col].quantile(quantile)
            # 计算超出边界的数量
            above = (df[col] > upper_bound).sum()
            if above > 0:
                df.loc[df[col] > upper_bound, col] = upper_bound
                outlier_count += above
                print(f"      - {col}: 将 {above} 个超过 {quantile*100:.0f}% 分位数 ({upper_bound:.2f}) 的值缩尾")
    
    print(f"   异常值处理: 共处理了 {outlier_count} 个异常值（缩尾到 {quantile*100:.0f}% 分位数）")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot 编码分类变量
    
    ⚠️ 关键：使用 drop_first=True 避免虚拟变量陷阱！
    如果不丢弃一列，特征列会线性相关，导致 (XᵀX) 不可逆，模型崩溃。
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("   分类变量: 无分类变量需要编码")
        return df
    
    print(f"   分类变量编码前: {categorical_cols}")
    
    # 对每一列进行 One-Hot 编码，drop_first=True 是关键！
    for col in categorical_cols:
        # 使用 pandas 的 get_dummies，drop_first=True 丢弃第一列
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        # 删除原列，添加哑变量列
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        print(f"      - {col}: One-Hot 编码为 {dummies.columns.tolist()} (drop_first=True)")
    
    print(f"   分类变量编码后: 新增 {sum([len(dummies.columns) for _ in categorical_cols])} 列")
    
    return df


def main():
    """主函数：执行完整的数据清洗流程"""
    print("=" * 70)
    print("数据清洗脚本 (Data Preparation CLI)")
    print("=" * 70)
    
    # 1. 解析命令行参数
    args = parse_arguments()
    print(f"\n输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"缺失值填补方法: {args.fill_method}")
    print(f"缩尾分位数: {args.winsorize_quantile}")
    
    # 2. 加载数据
    df = load_data(args.input)
    
    # 3. 处理缺失值（第一步：先填补缺失值，以便后续计算分位数）
    print("\n--- 第1步：处理缺失值 ---")
    df = handle_missing_values(df, method=args.fill_method)
    
    # 4. 处理异常值（Winsorization：缩尾）
    print("\n--- 第2步：处理异常值 (Winsorization) ---")
    df = handle_outliers_winsorize(df, quantile=args.winsorize_quantile)
    
    # 5. 编码分类变量（One-Hot，注意 drop_first）
    print("\n--- 第3步：One-Hot 编码分类变量 ---")
    df = encode_categorical(df)
    
    # 6. 保存清洗后的数据
    print("\n--- 第4步：保存清洗后数据 ---")
    save_data(df, args.output)
    
    # 7. 输出数据摘要
    print("\n--- 数据摘要 ---")
    print(f"最终特征数: {df.shape[1]} (不含目标变量)")
    print(f"最终样本数: {df.shape[0]}")
    print(f"数据类型分布:")
    for dtype in df.dtypes.value_counts().items():
        print(f"  - {dtype[0]}: {dtype[1]} 列")
    
    print("\n" + "=" * 70)
    print("✅ 数据清洗完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()