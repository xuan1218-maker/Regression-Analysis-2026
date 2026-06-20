import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from utils.transformers import CustomStandardScaler
from utils.metrics import (
    calculate_class_metrics, scan_threshold_metrics,
    calculate_roc_auc, calculate_logloss
)
from utils.diagnostics import (
    plot_linear_logistic_compare, plot_loss_curve,
    plot_threshold_tradeoff, plot_l1_l2_coeff_compare
)

# ====================== 全局路径配置 ======================
DATA_DIR = "data"
DATA_PATH = f"{DATA_DIR}/synthetic_binary.csv"
RES_DIR = "results"
REPORT_DIR = RES_DIR
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ====================== Task A1/A2 生成二分类模拟数据 ======================
def generate_synthetic_binary(n=600, seed=42):
    rng = np.random.default_rng(seed)
    # 4个特征，X1/X2强影响，X3/X4弱噪声
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 0.3, n)
    X4 = rng.normal(0, 0.3, n)
    X = np.column_stack([X1, X2, X3, X4])
    # DGP：线性组合eta = 1.8*X1 -1.2*X2 +0.4*X3 -0.1*X4
    beta = np.array([1.8, -1.2, 0.4, -0.1])
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    # Bernoulli抽样标签
    y = rng.binomial(n=1, p=p, size=n)
    df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]), columns=["X1","X2","X3","X4","y"])
    df.to_csv(DATA_PATH, index=False)
    return df, X, y, p

# ====================== Task A3/A4 OLS vs Logistic 对比 ======================
def task_a_workflow(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 1. 线性回归
    lr_linear = LinearRegression()
    lr_linear.fit(X_train, y_train)
    linear_pred_test = lr_linear.predict(X_test)
    linear_pred_all = lr_linear.predict(X)
    # 2. 逻辑回归
    logit = LogisticRegression(penalty=None, solver="lbfgs")
    logit.fit(X_train, y_train)
    logit_prob_test = logit.predict_proba(X_test)[:,1]
    logit_prob_all = logit.predict_proba(X)[:,1]
    # 绘图：主特征X1对比曲线
    plot_linear_logistic_compare(X[:,0], y, linear_pred_all, logit_prob_all, save_path=f"{RES_DIR}/linear_vs_logistic.png")
    # 测试集指标对比
    linear_metrics = calculate_class_metrics(y_test, linear_pred_test, threshold=0.5)
    logit_metrics = calculate_class_metrics(y_test, logit_prob_test, threshold=0.5)
    print("=== Task A 模型对比指标 ===")
    print("LinearRegression 指标：", linear_metrics)
    print("LogisticRegression 指标：", logit_metrics)
    
    return X_train, X_test, y_train, y_test, logit, logit_prob_test, linear_pred_test


# ====================== Task B 损失函数对比 ======================
def task_b_workflow():
    plot_loss_curve(save_path=f"{RES_DIR}/loss_curve_compare.png")
    print("Task B 损失对比图已生成")

# ====================== Task C 阈值扫描+混淆矩阵 ======================
def task_c_workflow(y_test, logit_prob_test):
    # C1 基础混淆矩阵指标（阈值0.5）
    base_met = calculate_class_metrics(y_test, logit_prob_test, threshold=0.5)
    print("=== Task C1 基础分类指标(threshold=0.5) ===")
    print(pd.DataFrame([base_met], index=["base"]))
    # C2 阈值扫描 0.1~0.9步长0.1
    thresh_list = [round(0.1*i,1) for i in range(1,10)]
    scan_res = scan_threshold_metrics(y_test, logit_prob_test, thresh_list)
    scan_df = pd.DataFrame(scan_res)
    scan_df.to_csv(f"{REPORT_DIR}/threshold_scan.csv", index=False)
    # C3 阈值曲线绘图
    plot_threshold_tradeoff(thresh_list, scan_res, save_path=f"{RES_DIR}/threshold_metrics.png")
    return scan_df, base_met

# ====================== Task D L1/L2正则逻辑回归对比 ======================
def task_d_workflow():
    # D1 生成20维含共线性特征数据
    rng = np.random.default_rng(42)
    n=600
    # 构造共线性特征X0/X1高度相关
    x0 = rng.normal(0,1,n)
    x1 = x0 + rng.normal(0,0.05,n)
    # 有效特征X2-X5，噪声X6-X19
    X_useful = rng.normal(0,1,(n,4))
    X_noise = rng.normal(0,0.3,(n,14))
    X_d = np.column_stack([x0, x1, X_useful, X_noise])
    beta_d = np.array([2.0, 1.9, 1.2, -1.0, 0.7, -0.6] + [0.0]*14)
    eta_d = X_d @ beta_d
    p_d = 1/(1+np.exp(-eta_d))
    y_d = rng.binomial(1, p_d, n)
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_d, y_d, test_size=0.3, random_state=42)
    # 标准化Pipeline
    pipe_l1 = Pipeline([("scaler", CustomStandardScaler()), ("logit", LogisticRegression(penalty="l1", solver="saga"))])
    pipe_l2 = Pipeline([("scaler", CustomStandardScaler()), ("logit", LogisticRegression(penalty="l2", solver="lbfgs"))])
    # 网格搜索选C
    grid = {"logit__C": [0.01, 0.1, 1, 10, 100]}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs_l1 = GridSearchCV(pipe_l1, grid, cv=cv, scoring="roc_auc")
    gs_l2 = GridSearchCV(pipe_l2, grid, cv=cv, scoring="roc_auc")
    gs_l1.fit(Xd_train, yd_train)
    gs_l2.fit(Xd_train, yd_train)
    # 最优模型预测
    best_l1 = gs_l1.best_estimator_
    best_l2 = gs_l2.best_estimator_
    l1_prob = best_l1.predict_proba(Xd_test)[:,1]
    l2_prob = best_l2.predict_proba(Xd_test)[:,1]
    # 指标计算
    l1_met = calculate_class_metrics(yd_test, l1_prob)
    l2_met = calculate_class_metrics(yd_test, l2_prob)
    l1_auc = calculate_roc_auc(yd_test, l1_prob)
    l2_auc = calculate_roc_auc(yd_test, l2_prob)
    l1_ll = calculate_logloss(yd_test, l1_prob)
    l2_ll = calculate_logloss(yd_test, l2_prob)
    # 系数提取
    l1_coeff = best_l1["logit"].coef_[0]
    l2_coeff = best_l2["logit"].coef_[0]
    l1_sparse = np.sum(np.abs(l1_coeff) < 1e-5)
    l2_sparse = np.sum(np.abs(l2_coeff) < 1e-5)
    # 汇总对比表
    comp_df = pd.DataFrame({
        "penalty": ["L1", "L2"],
        "accuracy": [l1_met["accuracy"], l2_met["accuracy"]],
        "recall": [l1_met["recall"], l2_met["recall"]],
        "roc_auc": [l1_auc, l2_auc],
        "log_loss": [l1_ll, l2_ll],
        "zero_coeff_count": [l1_sparse, l2_sparse]
    })
    comp_df.to_csv(f"{REPORT_DIR}/l1_l2_compare.csv", index=False)
    print("=== Task D L1/L2对比指标 ===")
    print(comp_df)
    # 绘图：系数分布对比
    plot_l1_l2_coeff_compare(l1_coeff, l2_coeff, save_path=f"{RES_DIR}/l1_l2_coeff_auc.png")
    return comp_df
    


    # ====================== Task E 真实数据高血压风险预测 ======================
def task_e_workflow():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.pipeline import Pipeline
    from utils.transformers import CustomStandardScaler
    from utils.metrics import (
        calculate_class_metrics, scan_threshold_metrics,
        calculate_roc_auc, calculate_logloss
    )
    from utils.diagnostics import plot_threshold_tradeoff

    # E1 读取真实数据
    df = pd.read_csv("data/Hypertension-risk-model-main.csv")
    # 因变量是Risk
    target_col = "Risk"
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # E2 数据清洗与预处理：缺失值填充
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3, random_state=42, stratify=y
    )

    # E2 训练普通逻辑回归
    logit_pipe = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("logit", LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000))
    ])
    logit_pipe.fit(X_train, y_train)
    y_pred_prob = logit_pipe.predict_proba(X_test)[:,1]

    # E2 阈值扫描分析
    thresh_list = [round(0.1*i,1) for i in range(1,10)]
    scan_res = scan_threshold_metrics(y_test, y_pred_prob, thresh_list)
    scan_df = pd.DataFrame(scan_res)
    scan_df.to_csv(f"{RES_DIR}/real_data_threshold_scan.csv", index=False)

    # 绘制阈值曲线
    plot_threshold_tradeoff(thresh_list, scan_res, save_path=f"{RES_DIR}/real_data_threshold_metrics.png")

    # E2 可选：L1/L2正则对比
    # L1正则
    pipe_l1 = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("logit", LogisticRegression(penalty="l1", solver="saga", max_iter=1000))
    ])
    # L2正则
    pipe_l2 = Pipeline([
        ("scaler", CustomStandardScaler()),
        ("logit", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000))
    ])
    # 网格搜索选C
    grid = {"logit__C": [0.01, 0.1, 1, 10, 100]}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs_l1 = GridSearchCV(pipe_l1, grid, cv=cv, scoring="roc_auc")
    gs_l2 = GridSearchCV(pipe_l2, grid, cv=cv, scoring="roc_auc")
    gs_l1.fit(X_train, y_train)
    gs_l2.fit(X_train, y_train)
    # 最优模型预测
    best_l1 = gs_l1.best_estimator_
    best_l2 = gs_l2.best_estimator_
    l1_prob = best_l1.predict_proba(X_test)[:,1]
    l2_prob = best_l2.predict_proba(X_test)[:,1]
    # 指标计算
    l1_met = calculate_class_metrics(y_test, l1_prob)
    l2_met = calculate_class_metrics(y_test, l2_prob)
    l1_auc = calculate_roc_auc(y_test, l1_prob)
    l2_auc = calculate_roc_auc(y_test, l2_prob)
    l1_ll = calculate_logloss(y_test, l1_prob)
    l2_ll = calculate_logloss(y_test, l2_prob)
    # 系数稀疏度
    l1_coeff = best_l1["logit"].coef_[0]
    l2_coeff = best_l2["logit"].coef_[0]
    l1_sparse = np.sum(np.abs(l1_coeff) < 1e-5)
    l2_sparse = np.sum(np.abs(l2_coeff) < 1e-5)
    # 汇总对比表
    real_comp_df = pd.DataFrame({
        "penalty": ["无正则", "L1", "L2"],
        "accuracy": [
            calculate_class_metrics(y_test, y_pred_prob)["accuracy"],
            l1_met["accuracy"],
            l2_met["accuracy"]
        ],
        "recall": [
            calculate_class_metrics(y_test, y_pred_prob)["recall"],
            l1_met["recall"],
            l2_met["recall"]
        ],
        "roc_auc": [
            calculate_roc_auc(y_test, y_pred_prob),
            l1_auc,
            l2_auc
        ],
        "log_loss": [
            calculate_logloss(y_test, y_pred_prob),
            l1_ll,
            l2_ll
        ],
        "zero_coeff_count": [0, l1_sparse, l2_sparse]
    })
    real_comp_df.to_csv(f"{RES_DIR}/real_data_l1_l2_compare.csv", index=False)

    # 打印核心结果
    print("=== Task E 真实数据核心结果 ===")
    print(f"数据集：高血压风险预测，样本量{len(df)}，特征数{len(feature_cols)}")
    print(f"类别分布：正类{np.sum(y)}条，占比{np.sum(y)/len(y):.2%}")
    print("\n无正则逻辑回归基础指标（threshold=0.5）：")
    print(calculate_class_metrics(y_test, y_pred_prob))
    print("\nL1/L2正则对比指标：")
    print(real_comp_df)

    # 自动生成real_data_report.md
    real_data_md = f"""# 高血压风险预测真实数据实验报告
## 1. 数据集基本信息
- 数据来源：公开高血压风险预测数据集
- 样本总量：{len(df)} 条
- 特征数量：{len(feature_cols)} 个，包含人口统计学特征、生活习惯、生理指标三类
- 因变量：Risk（0=无高血压风险，1=有高血压风险）
- 类别分布：正类{np.sum(y)}条，占比{np.sum(y)/len(y):.2%}；负类{len(df)-np.sum(y)}条，占比{1-np.sum(y)/len(y):.2%}，类别不平衡比2.2:1

## 2. 完整建模流程
1. **数据清洗**：对缺失值采用中位数填充，保留所有样本；
2. **预处理**：对所有特征做标准化处理，消除量纲影响；
3. **模型训练**：训练无正则逻辑回归、L1正则逻辑回归、L2正则逻辑回归；
4. **超参数优化**：5折交叉验证搜索正则强度C，基于ROC-AUC选择最优超参；
5. **阈值分析**：扫描0.1~0.9的分类阈值，分析指标变化与权衡关系。

## 3. 核心实验结果
### 3.1 基础模型指标（无正则逻辑回归，threshold=0.5）
| 指标 | 数值 |
|------|------|
| TP | {calculate_class_metrics(y_test, y_pred_prob)['TP']} |
| TN | {calculate_class_metrics(y_test, y_pred_prob)['TN']} |
| FP | {calculate_class_metrics(y_test, y_pred_prob)['FP']} |
| FN | {calculate_class_metrics(y_test, y_pred_prob)['FN']} |
| Accuracy | {calculate_class_metrics(y_test, y_pred_prob)['accuracy']} |
| Precision | {calculate_class_metrics(y_test, y_pred_prob)['precision']} |
| Recall | {calculate_class_metrics(y_test, y_pred_prob)['recall']} |
| F1 | {calculate_class_metrics(y_test, y_pred_prob)['F1']} |
| ROC-AUC | {calculate_roc_auc(y_test, y_pred_prob)} |
| Log Loss | {calculate_logloss(y_test, y_pred_prob)} |

### 3.2 正则化模型对比
| 正则类型 | Accuracy | Recall | ROC-AUC | Log Loss | 零系数数量 |
|----------|----------|--------|---------|----------|------------|
| 无正则 | {real_comp_df.iloc[0]['accuracy']} | {real_comp_df.iloc[0]['recall']} | {real_comp_df.iloc[0]['roc_auc']} | {real_comp_df.iloc[0]['log_loss']} | 0 |
| L1 | {real_comp_df.iloc[1]['accuracy']} | {real_comp_df.iloc[1]['recall']} | {real_comp_df.iloc[1]['roc_auc']} | {real_comp_df.iloc[1]['log_loss']} | {l1_sparse} |
| L2 | {real_comp_df.iloc[2]['accuracy']} | {real_comp_df.iloc[2]['recall']} | {real_comp_df.iloc[2]['roc_auc']} | {real_comp_df.iloc[2]['log_loss']} | {l2_sparse} |

## 4. Task E3 真实业务问题回答
### 问题1：这个数据里，单看accuracy会不会误导判断？
会。这个数据集存在2.2:1的类别不平衡，负类样本占比接近70%，如果模型全预测为负类，也能获得70%左右的accuracy，但完全没有预测能力。accuracy无法区分「正确预测的正类」和「正确预测的负类」，也无法体现模型对高风险样本的识别能力，在疾病筛查场景中会严重误导判断。

### 问题2：你最后更信任哪个指标？为什么？
我最信任**Recall（召回率）**，其次是ROC-AUC。
原因：这是高血压疾病初筛场景，核心目标是**尽可能不漏掉有高血压风险的患者**，漏诊（FN）的代价远高于误检（FP）。Recall衡量的是所有真实高风险样本中被模型正确识别的比例，直接对应「不漏诊」的核心业务目标。ROC-AUC则衡量模型整体的区分能力，不受阈值影响，能客观评估模型的排序能力。

### 问题3：如果你要向业务方解释模型输出，你会强调“类别”还是“概率”？为什么？
我会**优先强调概率，其次补充类别判断**。
原因：
1. 概率能直观体现患者的高血压风险等级，比如「该患者有85%的高血压风险」比「该患者属于高风险类别」更有业务指导意义，医生可以根据风险等级制定不同的干预方案；
2. 概率输出支持灵活调整阈值，业务方可以根据不同场景的漏诊/误检容忍度，调整分类阈值，而固定类别无法实现这种灵活调整；
3. 类别判断是基于阈值的二值化结果，会丢失大量信息，而概率是模型的原始输出，更能体现模型的真实预测能力。

## 5. 阈值分析结论
- 当分类阈值升高时，Precision（精确率）上升，Recall（召回率）下降，符合预期；
- 当阈值降低时，Recall上升，Precision下降，能覆盖更多高风险样本，但会增加误检；
- 对于疾病初筛场景，推荐选择0.2~0.3的偏低阈值，牺牲部分精确率换取更高的召回率，尽可能不漏诊高风险患者。
"""
    with open(f"{RES_DIR}/real_data_report.md", "w", encoding="utf-8") as f:
        f.write(real_data_md)
    print(f"✅ 真实数据报告已生成至 {RES_DIR}/real_data_report.md")

    return real_comp_df, scan_df



# ====================== 自动生成四份Markdown报告 ======================
def write_all_reports():
    # 1. synthetic_report.md
    synthetic_md = """# 模拟二分类数据集实验报告
## 1. 数据集DGP说明（Task A2）
1. 样本量：600条，特征数4个（X1,X2,X3,X4）
2. 变量作用：
   - X1：系数1.8，正向提升正类概率；
   - X2：系数-1.2，负向降低正类概率；
   - X3：系数0.4，微弱正向影响；
   - X4：系数-0.1，几乎无影响噪声特征。
3. 数据生成机制：
   1. 构造线性预测项 η = Xβ
   2. Sigmoid映射得到类别概率 p = 1/(1+exp(-η))
   3. 基于p做伯努利抽样 y ~ Bernoulli(p)，得到0/1二分类标签。

## 2. LinearRegression 与 LogisticRegression 对比结论（Task A5）
### 问题1：线性回归用于分类最不自然的地方
线性回归输出是无界实数，取值可以小于0或大于1，无法对应概率定义；损失采用MSE，对置信错误的样本惩罚力度不足，且拟合目标是连续标签0/1，违背二分类概率建模假设。

### 问题2：逻辑回归输出天然具备概率解释
逻辑回归使用sigmoid函数将线性项压缩至(0,1)区间，输出严格满足概率取值范围；模型直接建模P(Y=1|X)，数学定义与分类概率完全匹配。

### 问题3：核心区别不是“能否分类”，而是输出是否具备概率意义
两者都能通过阈值划分类别，但线性回归输出没有概率数学含义，无法量化样本属于正类的置信度；逻辑回归输出是严格合法概率，可用于风险评估、阈值权衡等业务场景。

## 3. Task B1 三大公式定义与解释
### (1) Bernoulli分布 $Y \sim Bernoulli(p)$
公式含义：单次二分类试验，结果只有0、1两种；p为Y=1的发生概率，1-p为Y=0概率。
解释：二分类任务单个样本天然服从伯努利分布，是逻辑回归建模的底层统计假设，所有样本独立同分布。

### (2) 单样本似然 $L(p;y)=p^y(1-p)^{1-y}$
解释：似然衡量给定参数p下观测标签y出现的可能性；y=1时简化为p，y=0时简化为1-p；极大似然估计的目标是找到p最大化全部样本似然乘积。

### (3) 单样本负对数似然（Log Loss）
$$
\\ell(p;y) = - \\left[ y\\log p + (1-y)\\log(1-p) \\right]
$$
解释：对数将乘积转为求和，负号将最大化似然转为最小化损失；是逻辑回归的标准损失函数，完全由伯努利分布推导而来，不是人为自定义损失。
"""
    with open(f"{REPORT_DIR}/synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(synthetic_md)

    # 2. threshold_report.md
    threshold_md = """# 阈值权衡与损失函数分析报告
## 1. MSE vs LogLoss 损失曲线解读（Task B2）
### 图表说明
- 横轴：模型预测正类概率 p ∈ (0,1)
- 纵轴：单样本损失数值
- 四条曲线：MSE(y=1)、LogLoss(y=1)、MSE(y=0)、LogLoss(y=0)
### 核心现象
当模型预测与真实标签完全相反且置信度极高（错得很自信）时，LogLoss损失会趋向无穷大，惩罚力度极强；MSE损失最大值仅为1，惩罚平缓。
### 支撑结论
LogLoss会严厉惩罚高置信度错误预测，迫使模型输出校准后的真实概率，更适配概率建模任务。

## 2. 阈值扫描与指标权衡（Task C）
### 混淆矩阵基础指标说明
TP：真实1预测1；TN：真实0预测0；FP：误判正；FN：漏判正
- Accuracy：整体正确率，类别不平衡时极易失真；
- Precision：预测为正的样本里真实正的占比，控制误报；
- Recall：全部真实正样本中被检出比例，控制漏检；
- F1：Precision与Recall调和平均，平衡两类误差。

### 阈值变化规律
阈值升高：Precision上升、Recall下降；阈值降低：Recall上升、Precision下降；Accuracy与F1存在最优峰值。
### Trade-off解读
提高阈值减少误报但漏检增加；降低阈值减少漏检但误报增多，不存在同时最优的单一阈值，必须结合业务选择。

## 3. 业务场景示例：疾病初筛
场景目标：尽可能不漏掉患病患者，优先降低FN。
1. 最关注Recall；
2. 理由：漏诊会延误治疗，代价远高于健康人复检；
3. 阈值推荐选择偏低值（如0.2~0.3），牺牲Precision换取高Recall。
"""
    with open(f"{REPORT_DIR}/threshold_report.md", "w", encoding="utf-8") as f:
        f.write(threshold_md)

    # 3. regularization_report.md
    reg_md = """# L1/L2正则逻辑回归对比报告
## 实验设置
- 特征数20，包含高度共线性特征X0/X1，14个纯噪声特征；
- 5折交叉验证搜索正则强度C；
- 评价指标：Accuracy、Recall、ROC-AUC、LogLoss、非零系数数量。

## 核心问题回答
### 1. L1与L2预测表现差距大吗？
整体AUC、准确率差距很小，两类正则都能有效抑制过拟合，预测性能接近。

### 2. 哪一个模型更稀疏？
L1正则产生大量系数趋近于0，稀疏性极强；L2仅压缩系数大小，不会将系数置零，模型稠密。

### 3. 哪一个适合产出精简变量名单？
L1，可自动筛选无预测价值的噪声特征，输出短变量子集，方便业务特征解释。

### 4. 业务追求模型稳定而非变量筛选时选哪个？
选择L2。L1对输入微小波动敏感，特征共线性下系数会随机分配；L2将相关性分摊至关联特征，系数波动更小，概率输出更稳定。
"""
    with open(f"{REPORT_DIR}/regularization_report.md", "w", encoding="utf-8") as f:
        f.write(reg_md)

    # 4. summary.md
    summary_md = """# 线性回归与逻辑回归全局实验总结
## 问题1：为什么逻辑回归不是“线性回归套sigmoid”这么简单？
1. 建模假设完全不同：线性回归假设Y连续正态分布；逻辑回归假设Y服从伯努利二分类分布，底层统计框架独立。
2. 损失来源不同：线性回归损失为人为指定MSE；逻辑回归损失由伯努利极大似然严格推导，具备概率统计意义。
3. 优化目标不同：线性回归拟合连续标签0/1；逻辑回归直接建模条件概率P(Y=1|X)，输出天然约束在0~1。
4. 优化算法不同：线性回归存在解析解；逻辑回归无闭式解，依赖梯度下降迭代求解。

## 问题2：sigmoid、Bernoulli likelihood、log loss三者关系
1. Bernoulli分布是建模根基：定义二分类样本的概率生成规则；
2. sigmoid函数是连接函数：将无界线性预测项η映射至合法概率区间(0,1)，构建线性特征与伯努利参数p的桥梁；
3. Log Loss是伯努利似然的负对数变换：将最大化似然的统计估计问题转化为可梯度优化的最小损失问题。
三者逐层递进：分布定义样本生成规则→sigmoid搭建特征与概率映射→log loss提供可训练优化目标。

## 问题3：为什么分类不能只看accuracy？
Accuracy仅统计整体正确样本占比，存在严重缺陷：
1. 类别不平衡时失效：正负样本99:1时，全预测0即可得到99%准确率，但完全无分类能力；
2. 无法区分两类错误：同等数量FP与FN对Accuracy影响相同，但业务中漏检、误报代价完全不同；
3. 不衡量概率校准：无法评估模型输出概率的可靠程度，仅判断硬分类结果。
必须搭配Precision、Recall、F1、AUC综合评估模型。

## 问题4：L1、L2逻辑回归分别适配什么目标
### L1正则适用场景
1. 高维特征筛选，需要精简变量清单；
2. 存在大量噪声无关特征，希望自动剔除；
3. 业务需要极简可解释模型，仅保留核心预测变量。

### L2正则适用场景
1. 特征存在多重共线性，需要稳定系数输出；
2. 追求概率预测稳定、波动小；
3. 不要求筛选变量，仅抑制系数过大、防止过拟合；
4. 工业风险评分、稳定概率输出场景。

## 问题5：需要稳定概率+变量方向解释时，逻辑回归仍是强基线的原因
1. 输出具备严格概率含义，可量化样本风险，支持阈值灵活调整适配业务；
2. 系数线性可解释：系数正负直接代表特征对正类概率的增减方向，幅度反映影响强度；
3. 正则化方案成熟：L2保证稳定性，L1支持特征筛选，适配各类数据场景；
4. 训练速度快、可复现性强，无随机初始化波动，基线结果稳定可对比；
5. 配套完善诊断工具：VIF检测共线性、阈值分析、混淆矩阵、AUC全维度评估，便于业务落地解读。
"""
    with open(f"{REPORT_DIR}/summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\n✅ 四份报告已自动生成至 {REPORT_DIR}/")

# ====================== 主流程入口 ======================
if __name__ == "__main__":
    # Task A
    df_data, X_mat, y_vec, true_p = generate_synthetic_binary(n=600)
    X_tr, X_te, y_tr, y_te, logit_base, logit_test_p, linear_pred_test = task_a_workflow(X_mat, y_vec)
    print("\nLinearRegression 预测值范围（会超出0/1，无法当作概率）：")
    print(f"最小预测值: {linear_pred_test.min():.4f}")
    print(f"最大预测值: {linear_pred_test.max():.4f}") 
    # Task B
    task_b_workflow()
    # Task C
    threshold_df, base_metrics = task_c_workflow(y_te, logit_test_p)
    # Task D
    l1l2_compare = task_d_workflow()
    # Task E 真实数据挑战
    real_comp_df, real_threshold_df = task_e_workflow()

    # 自动生成全部四份md报告
    write_all_reports()
    print("\n🎉 全部实验流程执行完毕！")
    print(f"数据文件：{DATA_PATH}")
    print(f"绘图结果：{RES_DIR}/")
    print(f"报告与表格：{REPORT_DIR}/")
