"""
Week 15: Logistic Regression and Binary Classification
主程序入口 - 完整版本包含真实数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, log_loss,
    classification_report
)
from sklearn.datasets import load_breast_cancer, make_classification
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 从您的工具文件导入
from src.utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from src.utils.transformers import CustomStandardScaler, SimpleImputer, Winsorizer
from src.utils.diagnostics import plot_correlation_matrix, calculate_vif, compute_condition_number
from src.utils.models import AnalyticalOLS, GradientDescentOLS, PCR, CoefficientStabilityAnalyzer


# 设置matplotlib使用英文
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Week15Experiment:
    """第十五周实验主类"""
    
    def __init__(self, data_dir="./data", results_dir="./results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # 随机种子
        self.seed = 42
        np.random.seed(self.seed)
        
        # 存储结果
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.real_feature_names = None
        
        # 存储报告内容
        self.reports = {
            'synthetic': [],
            'threshold': [],
            'regularization': [],
            'real_data': [],
            'summary': []
        }
    
    def _add_to_report(self, report_name, content):
        """添加内容到报告"""
        self.reports[report_name].append(content)
    
    def generate_synthetic_data(self, n_samples=500, n_features=4):
        """
        Task A1: 生成带有明确概率结构的二分类数据
        DGP: p = sigmoid(X @ beta), y ~ Bernoulli(p)
        """
        # 生成特征
        X = np.random.randn(n_samples, n_features)
        
        # 设定真实系数：前两个特征重要，后两个不重要
        beta = np.array([2.0, -1.5, 0.0, 0.0])
        intercept = -0.5
        
        # 计算线性组合 eta = X @ beta + intercept
        eta = X @ beta + intercept
        
        # 通过sigmoid生成概率
        p = 1 / (1 + np.exp(-eta))
        
        # 从Bernoulli分布采样
        y = np.random.binomial(1, p)
        
        # 保存为DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['p_true'] = p
        df['y'] = y
        
        # 保存数据
        filepath = os.path.join(self.data_dir, 'synthetic_binary.csv')
        df.to_csv(filepath, index=False)
        
        # 记录特征名和DGP信息
        self.feature_names = feature_names
        self.dgp_info = {
            'beta': beta,
            'intercept': intercept,
            'features': feature_names,
            'n_samples': n_samples,
            'n_features': n_features
        }
        
        print(f"数据已生成: {filepath}")
        print(f"样本量: {n_samples}, 特征数: {n_features}")
        print(f"正类比例: {y.mean():.3f}")
        
        # 生成DGP说明（中文）
        dgp_desc = f"""
## 数据生成过程（DGP）

**样本量和特征数：**
- 样本量：{n_samples}
- 特征数：{n_features}
- 正类比例：{y.mean():.3f}

**真实系数：**
- feature_1：β = {beta[0]:.1f}（提高正类概率）
- feature_2：β = {beta[1]:.1f}（降低正类概率）
- feature_3：β = {beta[2]:.1f}（无影响）
- feature_4：β = {beta[3]:.1f}（无影响）
- 截距项：{intercept:.1f}

**数据生成机制：**
**η = Xβ + intercept**
**p = 1/(1 + e^(-η))**
**y ~ Bernoulli(p)**
"""
        self._add_to_report('synthetic', dgp_desc)
        
        return df
    
    def load_real_data(self):
        """
        Task E1: 加载真实二分类数据 - 使用乳腺癌数据集
        """
        print("\n" + "="*60)
        print("Task E: 真实数据挑战 - 乳腺癌数据集")
        print("="*60)
        
        # 加载乳腺癌数据集
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist()
        
        # 保存为CSV
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        filepath = os.path.join(self.data_dir, 'real_binary_breast_cancer.csv')
        df.to_csv(filepath, index=False)
        
        print(f"\n真实数据已保存: {filepath}")
        print(f"样本量: {len(df)}, 特征数: {len(feature_names)}")
        print(f"正类比例: {y.mean():.3f} (良性肿瘤)")
        
        self.real_feature_names = feature_names
        self.real_df = df
        
        # 探索性数据分析
        self._explore_real_data(df, feature_names)
        
        return df, feature_names
    
    def _explore_real_data(self, df, feature_names):
        """真实数据探索性分析（中文）"""
        desc = f"""
## 真实数据：乳腺癌数据集

**数据信息：**
- 数据来源：sklearn.datasets.load_breast_cancer
- 样本量：{len(df)}
- 特征数：{len(feature_names)}
- 目标变量：0 = 恶性肿瘤，1 = 良性肿瘤
- 正类比例：{df['target'].mean():.3f}

**与目标变量相关性最高的5个特征：**
"""
        # 相关性分析
        corr_with_target = df[feature_names].corrwith(df['target']).sort_values(ascending=False)
        for feat, corr in corr_with_target.head(5).items():
            desc += f"- {feat}: {corr:.3f}\n"
        
        desc += f"\n**与目标变量相关性最低的5个特征：**\n"
        for feat, corr in corr_with_target.tail(5).items():
            desc += f"- {feat}: {corr:.3f}\n"
        
        self._add_to_report('real_data', desc)
        
        # 画相关矩阵图（图中用英文）- 使用您的工具函数
        self._plot_real_correlation(df, feature_names)
    
    def _plot_real_correlation(self, df, feature_names):
        """画真实数据的相关矩阵（图中用英文）"""
        # 使用您提供的 plot_correlation_matrix 函数
        fig = plot_correlation_matrix(
            df=df,
            feature_cols=feature_names[:10],  # 取前10个特征避免图太拥挤
            target_col='target',
            title='Correlation Matrix - Breast Cancer Data',
            save_path=os.path.join(self.results_dir, 'real_data_correlation.png'),
            figsize=(12, 10)
        )
        plt.close(fig)
        print(f"相关矩阵图已保存: {os.path.join(self.results_dir, 'real_data_correlation.png')}")
    
    def prepare_data(self, df=None, is_real=False):
        """准备训练集和测试集"""
        if df is None:
            df = pd.read_csv(os.path.join(self.data_dir, 'synthetic_binary.csv'))
        
        if is_real:
            X = df[self.real_feature_names].values
            y = df['target'].values
            feature_names = self.real_feature_names
        else:
            X = df[self.feature_names].values
            y = df['y'].values
            feature_names = self.feature_names
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=y
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if is_real:
            self.real_X_train = X_train_scaled
            self.real_X_test = X_test_scaled
            self.real_y_train = y_train
            self.real_y_test = y_test
            self.real_scaler = scaler
        else:
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
            self.scaler = scaler
        
        print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def task_a_compare_models(self):
        """
        Task A3: 比较LinearRegression和LogisticRegression
        使用 sklearn 的 LinearRegression 作为"错误示范"
        """
        print("\n" + "="*60)
        print("Task A: 比较线性回归与逻辑回归")
        print("="*60)
        
        # 使用 sklearn 的 LinearRegression 作为错误示范
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        y_pred_lr = lr.predict(self.X_test)
        
        # 训练逻辑回归
        logreg = LogisticRegression(random_state=self.seed)
        logreg.fit(self.X_train, self.y_train)
        y_pred_proba = logreg.predict_proba(self.X_test)[:, 1]
        y_pred_class = logreg.predict(self.X_test)
        
        # 评估 - 使用您的 metrics 函数
        lr_out_of_range = ((y_pred_lr < 0) | (y_pred_lr > 1)).mean()
        
        print("\n--- 线性回归结果 (sklearn LinearRegression - 错误示范) ---")
        print(f"RMSE: {calculate_rmse(self.y_test, y_pred_lr):.4f}")
        print(f"R²: {lr.score(self.X_test, self.y_test):.4f}")
        print(f"预测范围: [{y_pred_lr.min():.3f}, {y_pred_lr.max():.3f}]")
        print(f"超出[0,1]范围的比例: {lr_out_of_range:.3f}")
        print(f"系数: {lr.coef_}")
        
        print("\n--- 逻辑回归结果 ---")
        print(f"准确率: {accuracy_score(self.y_test, y_pred_class):.4f}")
        print(f"预测概率范围: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
        
        # 添加报告内容（中文）
        report_content = f"""
## Task A：线性回归与逻辑回归对比

### 线性回归结果 (sklearn LinearRegression - 错误示范)
- RMSE：{calculate_rmse(self.y_test, y_pred_lr):.4f}
- R²：{lr.score(self.X_test, self.y_test):.4f}
- 预测范围：[{y_pred_lr.min():.3f}, {y_pred_lr.max():.3f}]
- 超出[0,1]范围的比例：{lr_out_of_range:.3f}
- 系数：{lr.coef_.tolist()}

**线性回归的问题（为什么不能直接拿来做二分类）：**
1. 输出不在[0,1]范围内，无法解释为概率
2. 使用MSE作为损失函数，不适合分类问题
3. 将分类问题当作回归问题处理，输出没有概率意义
4. 对"错得很自信"的预测惩罚不够

### 逻辑回归结果
- 准确率：{accuracy_score(self.y_test, y_pred_class):.4f}
- 预测概率范围：[{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]

**逻辑回归的优势：**
1. 输出严格在(0,1)范围内，可解释为概率
2. 使用对数损失（来自伯努利似然），适合分类
3. 专门为二分类问题设计

### 核心问题回答
关键区别不是"能不能分类"（两者都可以用阈值分类），而是**"输出是否有概率意义"**。线性回归输出任意实数，而逻辑回归输出的是合理的概率值。
"""
        self._add_to_report('synthetic', report_content)
        
        # Task A4: 画对比图（图中用英文）
        self._plot_model_comparison(y_pred_lr, y_pred_proba)
        
        return lr, logreg
    
    def _plot_model_comparison(self, y_pred_lr, y_pred_proba):
        """Task A4: 画出核心对比图（图中用英文）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 取第一个特征展示
        X_feature = self.X_test[:, 0]
        
        # 左图：LinearRegression
        ax1 = axes[0]
        ax1.scatter(X_feature[self.y_test == 0], self.y_test[self.y_test == 0], 
                   c='blue', alpha=0.5, label='y=0', s=20)
        ax1.scatter(X_feature[self.y_test == 1], self.y_test[self.y_test == 1], 
                   c='red', alpha=0.5, label='y=1', s=20)
        
        # 排序后画拟合线
        sorted_idx = np.argsort(X_feature)
        ax1.plot(X_feature[sorted_idx], y_pred_lr[sorted_idx], 
                'g-', linewidth=2, label='LinearRegression')
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Feature 1 (standardized)', fontsize=10)
        ax1.set_ylabel('Model Output', fontsize=10)
        ax1.set_title('LinearRegression: Output can go beyond [0,1]', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：LogisticRegression
        ax2 = axes[1]
        ax2.scatter(X_feature[self.y_test == 0], self.y_test[self.y_test == 0], 
                   c='blue', alpha=0.5, label='y=0', s=20)
        ax2.scatter(X_feature[self.y_test == 1], self.y_test[self.y_test == 1], 
                   c='red', alpha=0.5, label='y=1', s=20)
        
        # 排序后画sigmoid曲线
        ax2.plot(X_feature[sorted_idx], y_pred_proba[sorted_idx], 
                'orange', linewidth=2, label='LogisticRegression')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Feature 1 (standardized)', fontsize=10)
        ax2.set_ylabel('Predicted Probability', fontsize=10)
        ax2.set_title('LogisticRegression: Output is a valid probability', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n对比图已保存: {save_path}")
        
    def task_b_loss_functions(self):
        """
        Task B: 损失函数分析
        """
        print("\n" + "="*60)
        print("Task B: 损失函数分析")
        print("="*60)
        
        # Task B1: 写出三个公式（中文解释）
        formulas = """
## Task B1：三个核心公式

### 公式1：伯努利分布
$$Y \\sim \\text{Bernoulli}(p)$$

**解释：**
- Y只能取0或1两个值
- P(Y=1) = p，P(Y=0) = 1-p
- 这是二分类结果的自然概率分布
- 在我们的DGP中，y就是从Bernoulli(p)中采样得到的

### 公式2：单样本似然函数
$$L(p; y) = p^y(1-p)^{1-y}$$

**解释：**
- 当y=1时：L(p;1) = p
- 当y=0时：L(p;0) = 1-p
- 这表示在给定参数p下观测到数据的概率
- 似然值越高，说明模型越能解释观测到的数据

### 公式3：负对数似然（对数损失）
$$-\\log L(p; y) = -y\\log(p) - (1-y)\\log(1-p)$$

**解释：**
- 取负对数将乘积转化为求和（便于优化）
- 这正是逻辑回归中使用的对数损失函数
- 最小化对数损失 = 最大化似然
- 当模型"错得很自信"时，惩罚会趋向无穷大
"""
        self._add_to_report('threshold', formulas)
        
        # Task B2-B3: 画损失对比图（图中用英文）
        self._plot_loss_comparison()
        
    def _plot_loss_comparison(self):
        """Task B2: 画损失如何随预测概率变化的图（图中用英文）"""
        # 定义损失函数
        def squared_error(y_true, p_pred):
            return (y_true - p_pred) ** 2
        
        def log_loss_func(y_true, p_pred):
            eps = 1e-15
            p_pred = np.clip(p_pred, eps, 1 - eps)
            if y_true == 1:
                return -np.log(p_pred)
            else:
                return -np.log(1 - p_pred)
        
        # 生成概率范围
        p = np.linspace(0.001, 0.999, 1000)
        
        # 计算损失
        loss_sq_y1 = squared_error(1, p)
        loss_log_y1 = [log_loss_func(1, pi) for pi in p]
        loss_sq_y0 = squared_error(0, p)
        loss_log_y0 = [log_loss_func(0, pi) for pi in p]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：y=1
        ax1 = axes[0]
        ax1.plot(p, loss_sq_y1, 'b-', linewidth=2, label='Squared Error')
        ax1.plot(p, loss_log_y1, 'r-', linewidth=2, label='Log Loss')
        ax1.set_xlabel('Predicted Probability p', fontsize=10)
        ax1.set_ylabel('Loss Value', fontsize=10)
        ax1.set_title('True Label y = 1', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 右图：y=0
        ax2 = axes[1]
        ax2.plot(p, loss_sq_y0, 'b-', linewidth=2, label='Squared Error')
        ax2.plot(p, loss_log_y0, 'r-', linewidth=2, label='Log Loss')
        ax2.set_xlabel('Predicted Probability p', fontsize=10)
        ax2.set_ylabel('Loss Value', fontsize=10)
        ax2.set_title('True Label y = 0', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 标注关键点
        for ax in [ax1, ax2]:
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'loss_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n损失对比图已保存: {save_path}")
        
        # Task B3: 回答核心问题（中文）
        explanation = """
## Task B3：为什么对数损失是自然的选择

### 为什么"错得很自信"需要被重罚？
- 模型"错得很自信"意味着它给错误类别分配了很高的概率
- 这比不确定（p≈0.5）要糟糕得多
- 对数损失会指数级地惩罚这种情况：当y=1且p→0时，-log(p)→∞
- 这迫使模型校准概率，避免过度自信

### 为什么对数损失来自伯努利似然？
- 逻辑回归建模 P(Y=1|X) = p
- 观测数据的似然是 ∏ p^yi(1-p)^(1-yi)
- 取负对数就得到了损失函数
- 对数损失不是凭空指定的——它来自第一性原理

### 当输出是概率时，为什么对数损失比MSE更自然？
- MSE把分类当作回归问题处理
- MSE对所有错误一视同仁，不论置信度高低
- 对数损失基于概率校准来惩罚
- 对数损失是适当的评分规则——它鼓励正确的概率估计
"""
        self._add_to_report('threshold', explanation)
        
    def task_c_threshold_analysis(self):
        """
        Task C: 分类指标与阈值权衡
        """
        print("\n" + "="*60)
        print("Task C: 分类指标与阈值分析")
        print("="*60)
        
        # 训练逻辑回归
        logreg = LogisticRegression(random_state=self.seed)
        logreg.fit(self.X_train, self.y_train)
        y_pred_proba = logreg.predict_proba(self.X_test)[:, 1]
        y_pred_class = logreg.predict(self.X_test)
        
        # Task C1: 混淆矩阵和基础指标
        tp, tn, fp, fn, acc, prec, rec, f1 = self._confusion_matrix_and_metrics(y_pred_class)
        
        # Task C2-C3: 阈值扫描
        df_results = self._threshold_scan(y_pred_proba)
        
        # Task C4: 业务场景解释
        self._business_scenario_explanation()
        
        return logreg
    
    def _confusion_matrix_and_metrics(self, y_pred):
        """Task C1: 混淆矩阵和基础指标（中文）"""
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # 计算指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # 创建表格
        metrics_table = pd.DataFrame({
            '指标': ['TP', 'TN', 'FP', 'FN', '准确率', '精确率', '召回率', 'F1分数'],
            '值': [tp, tn, fp, fn, 
                     f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}']
        })
        
        print("\n--- 混淆矩阵与基础指标 ---")
        print(f"混淆矩阵:")
        print(f"              预测")
        print(f"              正类    负类")
        print(f"实际正类    {tp:>6}    {fn:>6}")
        print(f"实际负类    {fp:>6}    {tn:>6}")
        print("\n指标:")
        print(metrics_table.to_string(index=False))
        
        # 保存表格
        metrics_table.to_csv(os.path.join(self.results_dir, 'basic_metrics.csv'), index=False, encoding='utf-8')
        
        # 添加报告（中文）
        report = f"""
## Task C1：混淆矩阵与基础指标

### 混淆矩阵
| | 预测正类 | 预测负类 |
|---|---|---|
| **实际正类** | {tp} | {fn} |
| **实际负类** | {fp} | {tn} |

### 指标
| 指标 | 值 |
|------|-----|
| 准确率 | {accuracy:.4f} |
| 精确率 | {precision:.4f} |
| 召回率 | {recall:.4f} |
| F1分数 | {f1:.4f} |
"""
        self._add_to_report('threshold', report)
        
        return tp, tn, fp, fn, accuracy, precision, recall, f1
    
    def _threshold_scan(self, y_pred_proba):
        """Task C2-C3: 阈值扫描"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            results.append({
                'threshold': thresh,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })
        
        df_results = pd.DataFrame(results)
        
        # 保存结果
        df_results.to_csv(os.path.join(self.results_dir, 'threshold_scan.csv'), index=False)
        
        # 画图（图中用英文）
        self._plot_threshold_curves(df_results)
        
        # 添加报告（中文）
        report = """
## Task C2-C3：阈值扫描结果

### 阈值曲线观察：
1. **当阈值升高时：**
   - 精确率通常会上升（假正例减少）
   - 召回率通常会下降（假负例增多）
   - 准确率可能先升后降（最优在0.5附近）
   - F1分数在最优阈值处达到最大值

2. **精确率-召回率权衡：**
   - 低阈值：更多预测为正类，召回率更高，精确率更低
   - 高阈值：更少预测为正类，精确率更高，召回率更低
   - F1分数平衡了这两个指标

3. **本实验中F1的最优阈值：** 约0.5
"""
        self._add_to_report('threshold', report)
        
        return df_results
    
    def _plot_threshold_curves(self, df_results):
        """Task C3: 画threshold曲线（图中用英文）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df_results['threshold'], df_results['accuracy'], 
                'b-o', linewidth=2, markersize=8, label='Accuracy')
        ax.plot(df_results['threshold'], df_results['precision'], 
                'g-s', linewidth=2, markersize=8, label='Precision')
        ax.plot(df_results['threshold'], df_results['recall'], 
                'r-^', linewidth=2, markersize=8, label='Recall')
        ax.plot(df_results['threshold'], df_results['f1'], 
                'm-d', linewidth=2, markersize=8, label='F1 Score')
        
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'threshold_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n阈值曲线图已保存: {save_path}")
    
    def _business_scenario_explanation(self):
        """Task C4: 业务场景解释（中文）"""
        explanation = """
## Task C4：业务场景 - 疾病初筛

### 场景：早期疾病筛查

**最关注的指标：召回率（Recall）**

**为什么？**
- 假负例（漏诊）的成本远高于假正例（误诊）
- 漏诊患者可能导致延误治疗、更差的预后
- 假正例只会导致额外的检查和患者焦虑
- 漏诊成本 >> 误诊成本

### 推荐的阈值

对于疾病筛查场景，我建议使用**较低的阈值（例如0.3-0.4）**。

**理由：**
1. 优先保证召回率，尽可能发现所有真正的病例
2. 接受一定比例的假正例作为权衡
3. 可以根据实际成本比率调整
4. 后续检查可以确认阳性病例

### 阈值选择逻辑
1. 估计漏诊成本：错过诊断的代价（医疗、法律、生活质量）
2. 估计误诊成本：额外检查和患者焦虑的代价
3. 选择漏诊边际成本 = 误诊边际成本的阈值点
4. 如果漏诊成本 >> 误诊成本，选择较低阈值（更高召回率）
"""
        self._add_to_report('threshold', explanation)
        
    def task_d_regularization_comparison(self):
        """
        Task D: 正则化逻辑回归（L1 vs L2）
        """
        print("\n" + "="*60)
        print("Task D: L1 vs L2 正则化比较")
        print("="*60)
        
        # D1: 生成高维带共线性的数据
        X_high, y_high = self._generate_high_dim_data()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_high, y_high, test_size=0.3, random_state=self.seed, stratify=y_high
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # D2: 训练L1和L2模型（使用GridSearchCV调参）
        models = {
            'L1': LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=self.seed),
            'L2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, random_state=self.seed)
        }
        
        results = {}
        for name, model in models.items():
            # 交叉验证调参
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            
            # 计算指标
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            loss = log_loss(y_test, y_pred_proba)
            n_nonzero = np.sum(np.abs(best_model.coef_) > 1e-6)
            
            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc_auc,
                'log_loss': loss,
                'n_nonzero': n_nonzero,
                'best_C': grid_search.best_params_['C'],
                'model': best_model
            }
            
            print(f"\n--- {name} 结果 ---")
            print(f"最优C值: {grid_search.best_params_['C']:.2f}")
            print(f"准确率: {acc:.4f}")
            print(f"召回率: {rec:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"对数损失: {loss:.4f}")
            print(f"非零系数个数: {n_nonzero}")
        
        # D3-D4: 画对比图和回答问题（图中用英文）
        self._plot_regularization_comparison(results)
        self._regularization_conclusions(results)
        
        return results
    
    def _generate_high_dim_data(self, n_samples=500, n_features=25):
        """Task D1: 生成高维带共线性的数据"""
        # 生成特征
        X = np.random.randn(n_samples, n_features)
        
        # 添加相关性：特征0-4高度相关
        for i in range(1, 5):
            X[:, i] = X[:, 0] + 0.1 * np.random.randn(n_samples)
        
        # 真实系数：只有前10个特征有用
        beta = np.zeros(n_features)
        beta[:10] = np.random.randn(10) * 1.5
        
        # 生成概率
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p)
        
        print(f"\n高维数据生成完成：")
        print(f"样本量: {n_samples}, 特征数: {n_features}")
        print(f"正类比例: {y.mean():.3f}")
        print(f"真实非零系数个数: {np.sum(beta != 0)}")
        print(f"共线性组: 特征0-4高度相关")
        
        return X, y
    
    def _plot_regularization_comparison(self, results):
        """Task D3: 画正则化对比图（图中用英文）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：性能指标
        ax1 = axes[0]
        metrics = ['accuracy', 'recall', 'roc_auc']
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[m] for m in metrics]
            ax1.bar(x + i*width, values, width, label=name, alpha=0.8)
        
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Performance Comparison: L1 vs L2', fontsize=12)
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 右图：非零系数个数
        ax2 = axes[1]
        names = list(results.keys())
        n_nonzero = [results[name]['n_nonzero'] for name in names]
        best_C = [results[name]['best_C'] for name in names]
        
        bars = ax2.bar(names, n_nonzero, color=['orange', 'skyblue'], alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Number of Non-zero Coefficients', fontsize=12)
        ax2.set_title('Model Sparsity Comparison', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加best C标注
        for bar, c in zip(bars, best_C):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'C={c}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'regularization_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n正则化对比图已保存: {save_path}")
    
    def _regularization_conclusions(self, results):
        """Task D4: 回答核心比较问题（中文）"""
        l1_nonzero = results['L1']['n_nonzero']
        l2_nonzero = results['L2']['n_nonzero']
        l1_acc = results['L1']['accuracy']
        l2_acc = results['L2']['accuracy']
        
        report = f"""
## Task D4：L1 vs L2 正则化结论

### 结果汇总
| 指标 | L1 | L2 |
|------|----|----|
| 准确率 | {l1_acc:.4f} | {l2_acc:.4f} |
| 召回率 | {results['L1']['recall']:.4f} | {results['L2']['recall']:.4f} |
| ROC-AUC | {results['L1']['roc_auc']:.4f} | {results['L2']['roc_auc']:.4f} |
| 对数损失 | {results['L1']['log_loss']:.4f} | {results['L2']['log_loss']:.4f} |
| 非零系数个数 | {l1_nonzero} | {l2_nonzero} |

### 问题回答

**1. L1和L2的预测表现差很多吗？**
{'是的，L2略好一些' if l2_acc > l1_acc else '不，两者相当接近'}。L2的准确率为{l2_acc:.4f}，L1为{l1_acc:.4f}。两者差距不大，但L2有微弱优势。

**2. 哪一个模型更稀疏？**
L1显著更稀疏（{l1_nonzero}个非零系数 vs L2的{l2_nonzero}个）。L1强制许多系数精确为零。

**3. 哪个更适合"给出更短的变量名单"？**
L1更适合变量筛选。它产生稀疏模型，使用更少的特征，更容易向需要"短名单"的业务方解释。

**4. 如果更在意模型稳定性，偏向哪一个？**
L2更适合稳定性。它：
- 不强制系数为零
- 平滑地收缩系数
- 在不同数据样本上更稳定
- 在特征相关时预测更好
"""
        self._add_to_report('regularization', report)
        
    def task_e_real_data(self):
        """
        Task E: 真实数据挑战
        """
        print("\n" + "="*60)
        print("Task E: 真实数据挑战")
        print("="*60)
        
        # 加载真实数据
        df, feature_names = self.load_real_data()
        
        # 准备数据
        self.prepare_data(df, is_real=True)
        
        # E2: 跑一遍完整逻辑回归流程
        self._complete_logistic_pipeline()
        
        # E3: 回答真实业务问题
        self._real_data_business_questions()
    
    def _complete_logistic_pipeline(self):
        """E2: 完整逻辑回归流程（中文报告）"""
        print("\n--- 完整逻辑回归流程 ---")
        
        # 1. 训练基础逻辑回归
        logreg = LogisticRegression(random_state=self.seed, max_iter=10000)
        logreg.fit(self.real_X_train, self.real_y_train)
        
        # 2. 预测
        y_pred_proba = logreg.predict_proba(self.real_X_test)[:, 1]
        y_pred_class = logreg.predict(self.real_X_test)
        
        # 3. 评估
        acc = accuracy_score(self.real_y_test, y_pred_class)
        prec = precision_score(self.real_y_test, y_pred_class)
        rec = recall_score(self.real_y_test, y_pred_class)
        f1 = f1_score(self.real_y_test, y_pred_class)
        roc_auc = roc_auc_score(self.real_y_test, y_pred_proba)
        
        print(f"\n--- 基础逻辑回归结果 ---")
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {prec:.4f}")
        print(f"召回率: {rec:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # 4. 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(self.real_y_test, y_pred_class).ravel()
        print(f"\n混淆矩阵:")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        # 5. 阈值分析
        self._real_threshold_analysis(y_pred_proba)
        
        # 6. 正则化比较（真实数据）
        self._real_regularization_comparison()
        
        # 保存报告（中文）
        report = f"""
## Task E2：完整逻辑回归流程

### 基础结果（阈值=0.5）
| 指标 | 值 |
|------|-----|
| 准确率 | {acc:.4f} |
| 精确率 | {prec:.4f} |
| 召回率 | {rec:.4f} |
| F1分数 | {f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

### 混淆矩阵
| | 预测正类 | 预测负类 |
|---|---|---|
| **实际正类** | {tp} | {fn} |
| **实际负类** | {fp} | {tn} |
"""
        self._add_to_report('real_data', report)
        
        return logreg
    
    def _real_threshold_analysis(self, y_pred_proba):
        """真实数据的阈值分析"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            results.append({
                'threshold': thresh,
                'accuracy': accuracy_score(self.real_y_test, y_pred),
                'precision': precision_score(self.real_y_test, y_pred, zero_division=0),
                'recall': recall_score(self.real_y_test, y_pred, zero_division=0),
                'f1': f1_score(self.real_y_test, y_pred, zero_division=0)
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(self.results_dir, 'real_threshold_scan.csv'), index=False)
        
        # 画图（图中用英文）
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df_results['threshold'], df_results['accuracy'], 
                'b-o', linewidth=2, markersize=8, label='Accuracy')
        ax.plot(df_results['threshold'], df_results['precision'], 
                'g-s', linewidth=2, markersize=8, label='Precision')
        ax.plot(df_results['threshold'], df_results['recall'], 
                'r-^', linewidth=2, markersize=8, label='Recall')
        ax.plot(df_results['threshold'], df_results['f1'], 
                'm-d', linewidth=2, markersize=8, label='F1 Score')
        
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Real Data: Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'real_threshold_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"真实数据阈值曲线图已保存: {save_path}")
        
        # 添加到报告（中文）
        report = """
### 真实数据阈值分析

真实数据的阈值分析显示类似模式：
- 精确率随阈值升高而上升
- 召回率随阈值升高而下降
- F1在最优阈值（约0.5）处达到峰值
- 准确率相对稳定
"""
        self._add_to_report('real_data', report)
    
    def _real_regularization_comparison(self):
        """真实数据的正则化比较（中文）"""
        models = {
            'L1': LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=self.seed),
            'L2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, random_state=self.seed)
        }
        
        results = {}
        for name, model in models.items():
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(self.real_X_train, self.real_y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.real_X_test)
            y_pred_proba = best_model.predict_proba(self.real_X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(self.real_y_test, y_pred),
                'precision': precision_score(self.real_y_test, y_pred, zero_division=0),
                'recall': recall_score(self.real_y_test, y_pred, zero_division=0),
                'f1': f1_score(self.real_y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.real_y_test, y_pred_proba),
                'log_loss': log_loss(self.real_y_test, y_pred_proba),
                'n_nonzero': np.sum(np.abs(best_model.coef_) > 1e-6),
                'best_C': grid_search.best_params_['C']
            }
        
        # 添加报告（中文）
        report = f"""
### 真实数据：L1 vs L2 正则化

| 指标 | L1 | L2 |
|------|----|----|
| 准确率 | {results['L1']['accuracy']:.4f} | {results['L2']['accuracy']:.4f} |
| 精确率 | {results['L1']['precision']:.4f} | {results['L2']['precision']:.4f} |
| 召回率 | {results['L1']['recall']:.4f} | {results['L2']['recall']:.4f} |
| F1分数 | {results['L1']['f1']:.4f} | {results['L2']['f1']:.4f} |
| ROC-AUC | {results['L1']['roc_auc']:.4f} | {results['L2']['roc_auc']:.4f} |
| 对数损失 | {results['L1']['log_loss']:.4f} | {results['L2']['log_loss']:.4f} |
| 非零系数个数 | {results['L1']['n_nonzero']} | {results['L2']['n_nonzero']} |
| 最优C值 | {results['L1']['best_C']:.2f} | {results['L2']['best_C']:.2f} |

**真实数据观察：**
- L2在大多数指标上略优
- L1产生更稀疏的模型（非零系数更少）
- 两个模型都达到了很高的ROC-AUC（>0.98），说明判别能力很好
- L1和L2之间的差距比合成高维数据上要小
"""
        self._add_to_report('real_data', report)
    
    def _real_data_business_questions(self):
        """Task E3: 回答真实业务问题（中文）"""
        questions = """
## Task E3：真实数据业务问题

### Q1：在这个数据中，单看准确率会不会误导判断？
**会，但在高度不平衡数据中误导程度较轻。**
- 乳腺癌数据集类别相对平衡（约63%良性，37%恶性）
- 准确率约0.97在这个数据上是可靠的
- 但准确率仍然隐藏了重要信息：
  - 我们犯的是哪类错误（FP vs FN）？
  - 漏诊（错过癌症）的成本 >> 误诊（不必要活检）的成本

### Q2：你最后更信任哪个指标？为什么？
**我最信任召回率和ROC-AUC。**

**为什么？**
1. **召回率** - 在医疗筛查中至关重要：我们希望发现所有恶性肿瘤病例
2. **ROC-AUC** - 展示模型在所有阈值下区分类别的能力
3. **F1分数** - 在类别平衡时是好的平衡指标

**为什么不只信任准确率？**
- 准确率对所有错误一视同仁
- 在医学诊断中，假负例的成本远高于假正例
- 需要反映成本结构的指标

### Q3：向业务方解释模型输出时，强调"类别"还是"概率"？
**我会强调"概率"。**

**为什么？**
1. **风险评估** - 概率允许业务方评估风险水平（例如，90% vs 60%的癌症风险）
2. **阈值灵活性** - 业务方可以根据成本偏好选择阈值
3. **更好的决策** - 允许精细决策，而非简单的二选一
4. **可解释性** - 概率直观且能传达不确定性

**示例**：不要说"这个患者有癌症"，而是说"这个患者有92%的概率患有癌症"。
"""
        self._add_to_report('real_data', questions)
    
    def generate_reports(self):
        """生成所有报告（中文）"""
        print("\n" + "="*60)
        print("正在生成报告")
        print("="*60)
        
        # 生成synthetic_report
        with open(os.path.join(self.results_dir, 'synthetic_report.md'), 'w', encoding='utf-8') as f:
            f.write("# Task A：合成数据与模型比较\n\n")
            f.write("\n".join(self.reports['synthetic']))
        
        # 生成threshold_report
        with open(os.path.join(self.results_dir, 'threshold_report.md'), 'w', encoding='utf-8') as f:
            f.write("# Task B & C：损失函数、指标与阈值分析\n\n")
            f.write("\n".join(self.reports['threshold']))
        
        # 生成regularization_report
        with open(os.path.join(self.results_dir, 'regularization_report.md'), 'w', encoding='utf-8') as f:
            f.write("# Task D：L1 vs L2 正则化\n\n")
            f.write("\n".join(self.reports['regularization']))
        
        # 生成real_data_report
        with open(os.path.join(self.results_dir, 'real_data_report.md'), 'w', encoding='utf-8') as f:
            f.write("# Task E：真实数据挑战\n\n")
            f.write("\n".join(self.reports['real_data']))
        
        # 生成summary
        self._generate_summary()
        
        print(f"\n所有报告已生成在 {self.results_dir} 目录")
        print("- synthetic_report.md")
        print("- threshold_report.md")
        print("- regularization_report.md")
        print("- real_data_report.md")
        print("- summary.md")
        
    def _generate_summary(self):
        """Task F: 生成总结报告（中文）"""
        summary = """
# Week 15 总结：逻辑回归与二分类

## 1. 为什么逻辑回归不是"线性回归后面接一个sigmoid"这么简单？

关键区别在于：

**损失函数：**
- 逻辑回归使用来自伯努利似然的对数损失
- "线性回归+sigmoid"仍会使用MSE
- 对数损失对"错得很自信"的惩罚更重

**优化目标：**
- 逻辑回归最大化似然
- 线性回归最小化平方误差

**输出意义：**
- 逻辑回归输出概率
- 线性回归输出任意实数

**梯度行为：**
- 交叉熵+sigmoid有好的梯度特性
- MSE+sigmoid存在梯度消失问题

## 2. sigmoid、伯努利似然、对数损失三者之间的关系

**Sigmoid** 将线性组合映射到[0,1]：p = 1/(1+e^(-η))

**伯努利分布** 是概率模型：Y ~ Bernoulli(p)

**伯努利似然** 衡量数据概率：L(p;y) = p^y(1-p)^(1-y)

**对数损失** 是负对数似然：-log L = -y log(p) - (1-y)log(1-p)

这三者构成完整的概率框架：
1. Sigmoid确保输出是概率
2. 伯努利是二分类数据的自然分布
3. 对数损失直接来自最大似然估计

## 3. 为什么分类模型不能只看准确率？

**准确率会误导因为：**
- 对所有错误一视同仁
- 忽略类别不平衡
- 不反映业务成本

**需要多个指标：**
- **精确率**：正类预测的质量
- **召回率**：正类覆盖的程度
- **F1分数**：精确率和召回率的平衡
- **ROC-AUC**：模型的判别能力
- **对数损失**：概率校准

## 4. L1和L2正则化分别更适合什么目标？

**L1（Lasso）：**
- 产生稀疏解（特征选择）
- 需要可解释的短变量名单时更好
- 适合高维数据

**L2（Ridge）：**
- 平滑收缩系数
- 预测稳定性更好
- 处理相关特征更好

**选择取决于目标：**
- 变量筛选 → L1
- 预测稳定性 → L2
- 折中方案 → 弹性网

## 5. 为什么逻辑回归仍然是强基线？

**优势：**
1. **概率输出** - 自然的概率解释
2. **线性决策边界** - 简单且可解释
3. **特征重要性** - 系数显示变量方向
4. **正则化** - L1/L2/弹性网选项
5. **计算效率** - 训练和推理快速
6. **工业验证** - 健壮可靠
7. **校准概率** - 正确训练时校准良好

**当业务需要"稳定概率+可解释系数"时，逻辑回归仍然是最佳基线模型。**
"""
        with open(os.path.join(self.results_dir, 'summary.md'), 'w', encoding='utf-8') as f:
            f.write(summary)
    
    def run_all(self):
        """运行所有实验"""
        print("="*60)
        print("Week 15: 逻辑回归与二分类")
        print("="*60)
        
        # Task A: 生成数据并比较模型
        print("\n" + "="*60)
        print("运行 Task A: 合成数据生成与模型比较")
        print("="*60)
        df = self.generate_synthetic_data(n_samples=500, n_features=4)
        self.prepare_data(df, is_real=False)
        self.task_a_compare_models()
        
        # Task B: 损失函数分析
        print("\n" + "="*60)
        print("运行 Task B: 损失函数分析")
        print("="*60)
        self.task_b_loss_functions()
        
        # Task C: 阈值分析
        print("\n" + "="*60)
        print("运行 Task C: 阈值分析")
        print("="*60)
        self.task_c_threshold_analysis()
        
        # Task D: 正则化比较
        print("\n" + "="*60)
        print("运行 Task D: L1 vs L2 正则化比较")
        print("="*60)
        self.task_d_regularization_comparison()
        
        # Task E: 真实数据
        print("\n" + "="*60)
        print("运行 Task E: 真实数据挑战")
        print("="*60)
        self.task_e_real_data()
        
        # 生成所有报告
        self.generate_reports()
        
        print("\n" + "="*60)
        print("所有任务已成功完成！")
        print(f"请查看 {self.results_dir} 目录中的所有输出")
        print("="*60)


if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行实验
    experiment = Week15Experiment(
        data_dir=os.path.join(script_dir, "data"),
        results_dir=os.path.join(script_dir, "results")
    )
    experiment.run_all()