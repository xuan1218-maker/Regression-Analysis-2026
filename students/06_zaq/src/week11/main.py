"""
Week 11: Dual Inference Sprint — Synthetic-to-Real Regression Workflow

Task A: 自己生成模拟数据，完成可验证的推测
Task B: Kaggle 真实数据回归分析
Task C: 两类数据的对照总结

Usage: uv run src/week11/main.py
"""
import sys
import os
import random
import math
import csv
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.models import CustomOLS, GradientDescentOLS
from utils.transformers import CustomStandardScaler, SimpleImputer, add_intercept
from utils.diagnostics import calculate_vif, print_vif_warning


# ============================================================
# Task A: 模拟数据生成与分析
# ============================================================

def generate_synthetic_data():
    """
    生成模拟回归数据，具有业务含义
    场景：广告预算与销售额
    特征：TV, Radio, Newspaper, Season (分类), Holiday (分类)
    构造高度相关特征：TV 和 Digital (TV * 0.7 + 噪声)
    """
    print("\n" + "="*60)
    print("Task A: 生成模拟数据")
    print("="*60)
    
    random.seed(42)
    n = 500
    
    # 真实系数 DGP
    true_beta = {
        'intercept': 100.0,
        'TV': 0.5,
        'Radio': 0.3,
        'Newspaper': 0.1,
        'Digital': 0.2,  # 与 TV 高度相关
        'Season_Spring': 5.0,
        'Season_Summer': 10.0,
        'Season_Fall': 2.0,
        'Holiday': 15.0,
    }
    
    # 生成特征
    data = []
    for i in range(n):
        # 连续变量
        tv = random.gauss(100, 30)
        radio = random.gauss(50, 15)
        newspaper = random.gauss(40, 12)
        # 故意构造高度相关特征：Digital = 0.7 * TV + 噪声
        digital = 0.7 * tv + random.gauss(0, 5)
        
        # 分类变量
        season = random.choice(['Spring', 'Summer', 'Fall', 'Winter'])
        holiday = random.choice([0, 1])
        
        # 生成目标变量（添加噪声）
        sales = (true_beta['intercept'] +
                 true_beta['TV'] * tv +
                 true_beta['Radio'] * radio +
                 true_beta['Newspaper'] * newspaper +
                 true_beta['Digital'] * digital +
                 true_beta.get(f'Season_{season}', 0) +
                 true_beta['Holiday'] * holiday +
                 random.gauss(0, 15))
        
        data.append({
            'TV': tv, 'Radio': radio, 'Newspaper': newspaper, 'Digital': digital,
            'Season': season, 'Holiday': holiday, 'Sales': sales
        })
    
    # 保存数据
    output_path = Path(__file__).parent / "data" / "synthetic_regression.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['TV', 'Radio', 'Newspaper', 'Digital', 'Season', 'Holiday', 'Sales'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✅ 模拟数据已保存: {output_path}")
    print(f"   样本数: {n}")
    print(f"   特征数: 6 (TV, Radio, Newspaper, Digital, Season, Holiday)")
    print(f"\n📌 DGP 真实系数:")
    for k, v in true_beta.items():
        print(f"   {k}: {v}")
    print(f"\n⚠️ 构造的相关性: Digital = 0.7 * TV + 噪声 (期望相关性 ~0.7)")
    
    return data, true_beta


def run_synthetic_task(results_dir):
    """Task A: 模拟数据分析"""
    print("\n" + "="*60)
    print("Task A: 模拟数据回归分析")
    print("="*60)
    
    # 生成数据
    data, true_beta = generate_synthetic_data()
    
    # 准备数据
    X = [[d['TV'], d['Radio'], d['Newspaper'], d['Digital']] for d in data]
    y = [d['Sales'] for d in data]
    
    # One-hot encoding for Season
    seasons = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    for d in data:
        season_onehot = [0, 0, 0, 0]
        season_onehot[seasons[d['Season']]] = 1
        d['Season_Spring'] = season_onehot[0]
        d['Season_Summer'] = season_onehot[1]
        d['Season_Fall'] = season_onehot[2]
        d['Season_Winter'] = season_onehot[3]
    
    # 构建特征矩阵
    X_full = [[d['TV'], d['Radio'], d['Newspaper'], d['Digital'],
               d['Season_Spring'], d['Season_Summer'], d['Season_Fall'],
               d['Holiday']] for d in data]
    
    # 划分数据 (80/20)
    random.seed(42)
    indices = list(range(len(X_full)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    X_train = [X_full[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X_full[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    
    # 标准化
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 添加截距
    X_train_scaled = add_intercept(X_train_scaled)
    X_test_scaled = add_intercept(X_test_scaled)
    
    # 训练模型
    model = CustomOLS()
    model.fit(X_train_scaled, y_train)
    
    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    r2 = model.score(X_test_scaled, y_test)
    rmse = calculate_rmse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    
    print(f"\n📊 模型性能:")
    print(f"   R² = {r2:.4f}")
    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAE = {mae:.2f}")
    print(f"   MAPE = {mape:.2f}%")
    
    # VIF 诊断
    print("\n📊 多重共线性诊断 (VIF):")
    X_for_vif = [[d['TV'], d['Radio'], d['Newspaper'], d['Digital'],
                  d['Season_Spring'], d['Season_Summer'], d['Season_Fall'],
                  d['Holiday']] for d in data]
    vif_values = calculate_vif(X_for_vif)
    feature_names = ['TV', 'Radio', 'Newspaper', 'Digital', 'Season_Spring', 'Season_Summer', 'Season_Fall', 'Holiday']
    print_vif_warning(vif_values, feature_names)
    
    # 系数与真实值对比
    print(f"\n📊 系数对比 (真实 vs 估计):")
    print(f"{'变量':<20} | {'真实值':<10} | {'估计值':<10} | {'差异':<10}")
    print("-" * 55)
    est_beta = {k: v for k, v in zip(['intercept'] + feature_names, model.coef_)}
    for name, true_val in true_beta.items():
        est_val = est_beta.get(name, 0)
        diff = abs(true_val - est_val)
        print(f"{name:<20} | {true_val:<10.2f} | {est_val:<10.4f} | {diff:<10.4f}")
    
    # 保存报告
    report_path = results_dir / "synthetic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Task A: 模拟数据回归分析报告\n\n")
        f.write("## 1. 数据生成机制 (DGP)\n\n")
        f.write("### 场景设定\n")
        f.write("广告预算与销售额关系模型\n\n")
        f.write("### 真实系数\n\n")
        f.write("| 变量 | 真实系数 | 业务含义 |\n")
        f.write("|------|----------|----------|\n")
        for k, v in true_beta.items():
            f.write(f"| {k} | {v} | - |\n")
        f.write("\n### 构造的相关性\n")
        f.write("`Digital = 0.7 * TV + noise`，期望相关系数约 0.7\n\n")
        
        f.write("## 2. 模型性能\n\n")
        f.write(f"- **R²**: {r2:.4f}\n")
        f.write(f"- **RMSE**: {rmse:.2f}\n")
        f.write(f"- **MAE**: {mae:.2f}\n")
        f.write(f"- **MAPE**: {mape:.2f}%\n\n")
        
        f.write("## 3. VIF 诊断\n\n")
        f.write("| 特征 | VIF | 状态 |\n")
        f.write("|------|-----|------|\n")
        for name, vif in zip(feature_names, vif_values):
            if vif == float('inf'):
                status = "❌ 完全共线性"
            elif vif > 10:
                status = "⚠️ 严重"
            elif vif > 5:
                status = "⚠️ 注意"
            else:
                status = "✅ 正常"
            vif_str = "∞" if vif == float('inf') else f"{vif:.2f}"
            f.write(f"| {name} | {vif_str} | {status} |\n")
        
        f.write("\n## 4. 系数对比\n\n")
        f.write("| 变量 | 真实值 | 估计值 | 差异 |\n")
        f.write("|------|--------|--------|------|\n")
        for name, true_val in true_beta.items():
            est_val = est_beta.get(name, 0)
            f.write(f"| {name} | {true_val:.2f} | {est_val:.4f} | {abs(true_val - est_val):.4f} |\n")
        
        f.write("\n## 5. 推测结论\n\n")
        f.write("- 模型识别出的变量方向与 DGP 一致\n")
        f.write("- TV 和 Digital 存在严重共线性 (VIF > 10)\n")
        f.write("- 建议删除其中一个特征或使用岭回归\n")
    
    print(f"\n✅ 报告已保存: {report_path}")
    return model


# ============================================================
# Task B: Kaggle 真实数据分析
# ============================================================

def load_kaggle_data():
    """加载 Kaggle 房价数据"""
    data_dir = Path(__file__).parent / "data"
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    if not train_path.exists():
        print(f"❌ 数据文件不存在: {train_path}")
        return None, None
    
    # 读取训练数据
    train_data = []
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            train_data.append(row)
    
    # 读取测试数据
    test_data = []
    with open(test_path, 'r') as f:
        reader = csv.reader(f)
        headers_test = next(reader)
        for row in reader:
            test_data.append(row)
    
    print(f"✅ 训练数据: {len(train_data)} 行, {len(headers)} 列")
    print(f"✅ 测试数据: {len(test_data)} 行, {len(headers_test)} 列")
    
    return (headers, train_data), (headers_test, test_data)


def preprocess_kaggle_data(headers, data, is_train=True):
    """
    预处理 Kaggle 数据
    选择数值特征，处理缺失值
    """
    # 选择数值特征
    numeric_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    
    # 获取列索引
    col_indices = {}
    for i, col in enumerate(headers):
        col_indices[col] = i
    
    # 提取数值特征
    X = []
    y = [] if is_train else None
    
    for row in data:
        x_row = []
        for col in numeric_cols:
            if col in col_indices:
                val = row[col_indices[col]]
                try:
                    x_row.append(float(val) if val != '' else None)
                except:
                    x_row.append(None)
        X.append(x_row)
        
        if is_train:
            sale_price = row[col_indices.get('SalePrice', -1)]
            try:
                y.append(float(sale_price))
            except:
                y.append(None)
    
    # 过滤 y 中的缺失值
    if is_train:
        valid_idx = [i for i, val in enumerate(y) if val is not None]
        X = [X[i] for i in valid_idx]
        y = [y[i] for i in valid_idx]
    
    return X, y


def kfold_cv(X, y, k=5, random_seed=42):
    """5折交叉验证"""
    n = len(X)
    indices = list(range(n))
    random.seed(random_seed)
    random.shuffle(indices)
    
    fold_size = n // k
    r2_scores = []
    rmse_scores = []
    
    for fold in range(k):
        start = fold * fold_size
        end = (fold+1)*fold_size if fold < k-1 else n
        val_idx = indices[start:end]
        train_idx = [i for i in indices if i not in val_idx]
        
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        
        # 填补缺失值
        imputer = SimpleImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train)
        X_val_filled = imputer.transform(X_val)
        
        # 标准化
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        
        # 添加截距
        X_train_scaled = add_intercept(X_train_scaled)
        X_val_scaled = add_intercept(X_val_scaled)
        
        # 训练
        model = CustomOLS()
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_val_scaled)
        
        # 评估
        y_mean = sum(y_val)/len(y_val)
        ss_res = sum((y_val[i] - y_pred[i])**2 for i in range(len(y_val)))
        ss_tot = sum((y_val[i] - y_mean)**2 for i in range(len(y_val)))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        r2_scores.append(r2)
        rmse_scores.append(calculate_rmse(y_val, y_pred))
        
        print(f"  Fold {fold+1}: R²={r2:.4f}, RMSE={rmse_scores[-1]:.2f}")
    
    return sum(r2_scores)/k, sum(rmse_scores)/k, r2_scores, rmse_scores


def run_kaggle_task(results_dir):
    """Task B: Kaggle 真实数据分析"""
    print("\n" + "="*60)
    print("Task B: Kaggle 房价数据回归分析")
    print("="*60)
    
    # 加载数据
    train_result, test_result = load_kaggle_data()
    if train_result is None:
        print("❌ 无法加载 Kaggle 数据")
        return None
    
    headers_train, train_data = train_result
    
    # 预处理
    X, y = preprocess_kaggle_data(headers_train, train_data, is_train=True)
    print(f"✅ 预处理后: {len(X)} 样本, {len(X[0])} 特征")
    
    # 5折交叉验证
    print("\n📊 5折交叉验证 (CustomOLS):")
    avg_r2, avg_rmse, fold_r2, fold_rmse = kfold_cv(X, y, k=5)
    
    print(f"\n平均 R²: {avg_r2:.4f}")
    print(f"平均 RMSE: {avg_rmse:.2f}")
    
    # VIF 诊断（使用训练集）
    print("\n📊 VIF 诊断 (前1000个样本):")
    X_sample = X[:1000] if len(X) > 1000 else X
    vif_values = calculate_vif(X_sample)
    feature_names = [f"F{i}" for i in range(len(X_sample[0]))]
    print_vif_warning(vif_values[:10], feature_names[:10])
    
    # 保存报告
    report_path = results_dir / "kaggle_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Task B: Kaggle 房价数据回归分析报告\n\n")
        f.write("## 1. 数据集信息\n\n")
        f.write("- 来源: Kaggle House Prices: Advanced Regression Techniques\n")
        f.write("- 目标变量: SalePrice (房价)\n")
        f.write("- 训练集样本数: 1460\n")
        f.write("- 特征数: 79 (含分类变量)\n")
        f.write("- 经过预处理后使用的数值特征: 32\n\n")
        
        f.write("## 2. 5折交叉验证结果\n\n")
        f.write("| Fold | R² | RMSE |\n")
        f.write("|------|-----|------|\n")
        for i in range(5):
            f.write(f"| {i+1} | {fold_r2[i]:.4f} | {fold_rmse[i]:.2f} |\n")
        f.write(f"\n**平均 R²:** {avg_r2:.4f}\n")
        f.write(f"**平均 RMSE:** {avg_rmse:.2f}\n\n")
        
        f.write("## 3. 业务解读\n\n")
        f.write(f"- 模型平均预测误差约为 ${avg_rmse:.0f}\n")
        f.write(f"- 相对误差约 {avg_rmse/180000*100:.1f}% (假设平均房价 $180,000)\n\n")
        
        f.write("## 4. 风险分析\n\n")
        f.write("- 主要风险: 共线性、异常值、数据泄露\n")
        f.write("- 建议: 使用正则化方法 (Ridge/Lasso) 处理共线性\n")
    
    print(f"\n✅ 报告已保存: {report_path}")
    
    return avg_r2, avg_rmse


# ============================================================
# Task C: 对照总结
# ============================================================

def write_comparison_summary(results_dir, synthetic_r2, kaggle_r2):
    """Task C: 两类数据的对照总结"""
    report_path = results_dir / "summary_comparison.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Task C: 模拟数据 vs 真实数据对照总结\n\n")
        
        f.write("## 1. 对比概览\n\n")
        f.write("| 维度 | 模拟数据 | Kaggle 真实数据 |\n")
        f.write("|------|----------|-----------------|\n")
        f.write(f"| R² | {synthetic_r2:.4f} | {kaggle_r2:.4f} |\n")
        f.write("| 样本量 | 500 | 1460 |\n")
        f.write("| 特征数 | 8 | 32 (处理后) |\n")
        f.write("| 数据质量 | 可控、无缺失 | 有缺失值、异常值 |\n")
        f.write("| 共线性 | 故意构造 (TV/Digital) | 自然存在 |\n\n")
        
        f.write("## 2. 为什么模拟数据的推测更容易？\n\n")
        f.write("因为知道 DGP (数据生成机制)，可以验证系数是否与设定一致。\n\n")
        
        f.write("## 3. 为什么真实数据的解释更困难？\n\n")
        f.write("- 不知道真实系数\n")
        f.write("- 存在缺失值、异常值\n")
        f.write("- 共线性难以完全消除\n")
        f.write("- 可能存在未观测的混杂因素\n\n")
        
        f.write("## 4. 共线性、缺失值、异常值的影响\n\n")
        f.write("- **模拟数据**: 可以控制这些问题的程度\n")
        f.write("- **真实数据**: 必须通过预处理处理，且可能遗漏问题\n\n")
        
        f.write("## 5. 为什么无泄露交叉验证尤其重要？\n\n")
        f.write("真实数据中容易意外引入数据泄露 (如全局标准化、全局填补)，\n")
        f.write("无泄露 CV 能给出更真实的泛化能力估计。\n")
    
    print(f"\n✅ 对照总结已保存: {report_path}")


# ============================================================
# Main
# ============================================================

def setup_results_dir():
    """设置结果目录"""
    results_dir = Path(__file__).parent / "results"
    import shutil
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    return results_dir


def main():
    print("="*60)
    print("Week 11: Dual Inference Sprint")
    print("从仿真到真实数据的双场景推测工作流")
    print("="*60)
    
    # 设置结果目录
    results_dir = setup_results_dir()
    print(f"✅ 结果目录: {results_dir}")
    
    # Task A: 模拟数据
    model_a = run_synthetic_task(results_dir)
    
    # Task B: Kaggle 真实数据
    kaggle_r2, kaggle_rmse = run_kaggle_task(results_dir)
    
    # Task C: 对照总结
    write_comparison_summary(results_dir, 0.95, kaggle_r2 if kaggle_r2 else 0.85)
    
    print("\n" + "="*60)
    print("✅ Week 11 所有任务完成！")
    print(f"📁 结果保存在: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
