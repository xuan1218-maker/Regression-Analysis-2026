"""
用途：读取清洗后的数据，进行多重共线性诊断和交叉验证评估
用法：
    uv run src/week09/evaluate.py --data week09/clean_marketing.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif, print_vif_report


def load_clean_data(file_path: str) -> pd.DataFrame:
    """加载清洗后的数据"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}\n请先运行 data_prep.py 清洗数据")
    
    df = pd.read_csv(path)
    print(f"✅ 加载数据成功: {file_path}")
    print(f"   数据形状: {df.shape}")
    return df


def separate_features_target(df: pd.DataFrame, target_col: str = 'Sales') -> tuple:
    """
    分离特征矩阵 X 和目标变量 y
    """
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据中，可用的列: {df.columns.tolist()}")
    
    feature_names = [col for col in df.columns if col != target_col]
    
    # 关键修复：强制转换为 float64，解决 dtype('O') 问题
    X = df[feature_names].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    
    print(f"\n特征-目标分离:")
    print(f"   特征矩阵形状: {X.shape}")
    print(f"   特征矩阵数据类型: {X.dtype}")
    print(f"   特征列数: {len(feature_names)}")
    print(f"   特征名称（前10个）: {feature_names[:10]}")
    if len(feature_names) > 10:
        print(f"   ... 还有 {len(feature_names) - 10} 个特征")
    
    return X, y, feature_names


def add_intercept(X: np.ndarray) -> np.ndarray:
    """为特征矩阵添加截距列（全1列）"""
    return np.column_stack([np.ones(X.shape[0]), X])


def cross_validate_analytical_ols(X: np.ndarray, y: np.ndarray, n_folds: int = 5, random_seed: int = 42) -> dict:
    """
    使用 CustomOLS (AnalyticalOLS) 进行 K 折交叉验证
    """
    # 防御性检查：确保 X 是数值类型
    if X.dtype == 'object':
        print("⚠️ 警告：X 包含 object 类型，尝试转换为 float64...")
        X = X.astype(np.float64)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    r2_scores = []
    rmse_scores = []
    
    print(f"\n开始 {n_folds} 折交叉验证...")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # 划分数据
        X_train_raw = X[train_idx]
        y_train = y[train_idx]
        X_val_raw = X[val_idx]
        y_val = y[val_idx]
        
        # 添加截距列
        X_train = add_intercept(X_train_raw)
        X_val = add_intercept(X_val_raw)
        
        # 训练模型
        model = AnalyticalOLS()
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_val)
        
        sse = np.sum((y_val - y_pred) ** 2)
        sst = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - sse / sst if sst != 0 else 0.0
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        
        print(f"  Fold {fold}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    
    print("-" * 50)
    print(f"平均 R²:   {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"平均 RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    
    return {
        'r2_scores': r2_scores,
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'rmse_scores': rmse_scores,
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores)
    }


def main():
    """主函数：执行完整的诊断和评估流程"""
    parser = argparse.ArgumentParser(description='模型诊断与交叉验证评估')
    parser.add_argument('--data', type=str, default='week09/clean_marketing.csv',
                       help='清洗后的数据文件路径（默认: week09/clean_marketing.csv）')
    parser.add_argument('--target', type=str, default='Sales',
                       help='目标变量列名（默认: Sales）')
    parser.add_argument('--folds', type=int, default=5,
                       help='交叉验证折数（默认: 5）')
    args = parser.parse_args()
    
    print("=" * 70)
    print("模型诊断与交叉验证评估")
    print("=" * 70)
    
    # 处理数据路径
    data_path = Path(__file__).parent / "clean_marketing.csv"
    
    # 如果不是绝对路径，尝试相对于项目根目录
    if not data_path.is_absolute():
        # 获取项目根目录（假设当前文件在 src/week09/evaluate.py）
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_path = project_root / args.data
    
    print(f"\n数据路径: {data_path.absolute()}")
    
    # 检查文件是否存在
    if not data_path.exists():
        print(f"\n❌ 错误: 找不到数据文件")
        print(f"   尝试的路径: {data_path}")
        print(f"\n请先运行数据清洗脚本:")
        print(f"   python src/week09/data_prep.py --input week09/dirty_marketing.csv --output week09/clean_marketing.csv")
        return
    
    try:
        # 1. 加载数据
        df = load_clean_data(str(data_path))
        
        # 2. 分离特征和目标
        X, y, feature_names = separate_features_target(df, target_col=args.target)
        
        # 3. 数据诊断
        print(f"\n--- 数据诊断 ---")
        print(f"X 数据类型: {X.dtype}")
        print(f"X 形状: {X.shape}")
        print(f"X 中是否有 NaN: {np.any(np.isnan(X))}")
        print(f"X 中是否有 Inf: {np.any(np.isinf(X))}")
        print(f"y 数据类型: {y.dtype}")
        print(f"y 中是否有 NaN: {np.any(np.isnan(y))}")
        print("-" * 50)
        
        # 4. 多重共线性诊断
        if X.shape[1] > 0 and X.shape[1] < 100:  # VIF 计算可能很慢，限制特征数量
            print("\n" + "=" * 70)
            print("任务 1: 多重共线性诊断 (VIF)")
            print("=" * 70)
            print("注意: VIF 计算可能需要几秒钟...")
            
            vif_values = calculate_vif(X)
            print_vif_report(feature_names, vif_values)
        else:
            print(f"\n⚠️ 跳过 VIF 诊断 (特征数量: {X.shape[1]})")
        
        # 5. 交叉验证评估
        print("\n" + "=" * 70)
        print(f"任务 2: 基线交叉验证 ({args.folds}-Fold CV)")
        print("=" * 70)
        
        cv_results = cross_validate_analytical_ols(X, y, n_folds=args.folds)
        
        # 6. 思考题提醒
        print("\n" + "=" * 70)
        print("课堂讨论准备")
        print("=" * 70)
        print(f"\n📌 本次交叉验证得到的平均 R² = {cv_results['mean_r2']:.4f}")
        print("   问题：在 data_prep.py 中使用全量数据填补缺失值，")
        print("         5 折交叉验证中的验证集还算'完全未见过的陌生数据'吗？")
        print("\n💡 提示：这叫做'预处理泄露'，是一种更隐蔽的数据泄露！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()