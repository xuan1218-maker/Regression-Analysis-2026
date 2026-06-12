# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector

# 你自己的工具类
from utils.transformers import CustomStandardScaler
from utils.metrics import calculate_rmse, calculate_mae
from utils.models import forward_selection_cv

# ====================== 路径 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ====================== Task A1 模拟共线性数据 ======================
np.random.seed(42)
n_samples = 500

x1 = np.random.normal(0, 1, n_samples)
x2 = 0.9 * x1 + np.random.normal(0, 0.12, n_samples)
x3 = 0.85 * x1 + np.random.normal(0, 0.15, n_samples)
x4 = np.random.normal(0, 1, n_samples)
x5 = np.random.normal(0, 1, n_samples)
x6 = np.random.normal(0, 1, n_samples)
x7 = np.random.normal(0, 0.8, n_samples)
x8 = np.random.normal(0, 0.8, n_samples)

y = 2.5 * x1 + 1.8 * x4 + np.random.normal(0, 1.2, n_samples)

feat_names = [f"feat_{i+1}" for i in range(8)]
X = pd.DataFrame(np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8]), columns=feat_names)
y = pd.Series(y, name="target")

synth_data = pd.concat([X, y], axis=1)
synth_data.to_csv(os.path.join(DATA_DIR, "synthetic_correlated.csv"), index=False)
print("✅ Task A1：模拟数据已保存")

# ====================== Task A3 系数稳定性（50次切分）======================
print("🔁 Task A3：50次随机切分系数稳定性计算")
corr_feats = ["feat_1", "feat_2", "feat_3"]

c1_ols, c2_ols, c3_ols = [], [], []
c1_rdg, c2_rdg, c3_rdg = [], [], []

for seed in range(50):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    scaler = CustomStandardScaler()
    scaler.fit(X_tr)
    X_tr_s = scaler.transform(X_tr)

    ols = LinearRegression().fit(X_tr_s, y_tr)
    ridge = Ridge(alpha=10).fit(X_tr_s, y_tr)

    co = ols.coef_[:3]
    cr = ridge.coef_[:3]

    c1_ols.append(co[0])
    c2_ols.append(co[1])
    c3_ols.append(co[2])
    c1_rdg.append(cr[0])
    c2_rdg.append(cr[1])
    c3_rdg.append(cr[2])

plt.figure(figsize=(10, 5))
plt.boxplot([c1_ols, c2_ols, c3_ols], positions=[1,2,3], widths=0.3, patch_artist=True, boxprops=dict(facecolor='tab:blue'))
plt.boxplot([c1_rdg, c2_rdg, c3_rdg], positions=[1.4,2.4,3.4], widths=0.3, patch_artist=True, boxprops=dict(facecolor='tab:orange'))
plt.xticks([1.2,2.2,3.2], corr_feats)
plt.title("Coefficient Stability: OLS vs Ridge")
plt.legend([plt.Line2D([0],[0],color='tab:blue',lw=4), plt.Line2D([0],[0],color='tab:orange',lw=4)], ["OLS","Ridge"])
plt.savefig(os.path.join(RES_DIR, "coef_stability.png"), dpi=250, bbox_inches='tight')
plt.close()

# ====================== A 部分训练与测试 ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler_a = CustomStandardScaler()
scaler_a.fit(X_train)
X_train_s = scaler_a.transform(X_train)
X_test_s = scaler_a.transform(X_test)

cv = KFold(5, shuffle=True, random_state=42)

# Ridge
ridge = Ridge(random_state=42)
grid_ridge = GridSearchCV(ridge, {"alpha": np.logspace(-4,3,50)}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_ridge.fit(X_train_s, y_train)

# Lasso
lasso = Lasso(random_state=42, max_iter=50000)
grid_lasso = GridSearchCV(lasso, {"alpha": np.logspace(-4,3,50)}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_lasso.fit(X_train_s, y_train)

# ElasticNet
enet = ElasticNet(random_state=42, max_iter=50000)
grid_enet = GridSearchCV(enet, {"alpha": np.logspace(-3,2,25), "l1_ratio": np.linspace(0.1,0.9,9)}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_enet.fit(X_train_s, y_train)

# OLS
ols = LinearRegression().fit(X_train_s, y_train)

# CV 曲线
cv_df = pd.DataFrame(grid_ridge.cv_results_)
plt.figure(figsize=(10,5))
plt.semilogx(cv_df.param_alpha.astype(float), -cv_df.mean_test_score, "o-")
plt.axvline(grid_ridge.best_params_["alpha"], c="r", linestyle="--", label=f"best α = {grid_ridge.best_params_['alpha']:.2f}")
plt.xlabel("alpha (log)")
plt.ylabel("CV RMSE")
plt.title("Ridge CV Curve")
plt.legend()
plt.savefig(os.path.join(RES_DIR, "ridge_cv.png"), dpi=250, bbox_inches='tight')
plt.close()

# 评估函数
def evaluate(model, X, y):
    yp = model.predict(X)
    return {"RMSE": round(calculate_rmse(y, yp), 4), "MAE": round(calculate_mae(y, yp), 4)}

print("\n===== Task A 测试集性能 =====")
res_ols = evaluate(ols, X_test_s, y_test)
res_ridge = evaluate(grid_ridge.best_estimator_, X_test_s, y_test)
res_lasso = evaluate(grid_lasso.best_estimator_, X_test_s, y_test)
res_enet = evaluate(grid_enet.best_estimator_, X_test_s, y_test)
print(f"OLS:    {res_ols}")
print(f"Ridge:  {res_ridge}")
print(f"Lasso:  {res_lasso}")
print(f"ENet:   {res_enet}")

# 系数表
coef_df = pd.DataFrame({
    "feature": feat_names,
    "OLS": ols.coef_,
    "Ridge": grid_ridge.best_estimator_.coef_,
    "Lasso": grid_lasso.best_estimator_.coef_,
    "ENet": grid_enet.best_estimator_.coef_
})
coef_df.to_csv(os.path.join(RES_DIR, "synthetic_coef.csv"), index=False)

# A4 特征选择
fs_sel = forward_selection_cv(X_train, y_train, max_features=4)
lasso_sel = coef_df.loc[coef_df.Lasso.abs() > 1e-4, "feature"].tolist()
print("\n===== Task A4 特征筛选 =====")
print(f"前向选择: {fs_sel}")
print(f"Lasso非零: {lasso_sel}")

# ====================== Task B 真实数据：kaggle_student.csv ======================
print("\n=====================================")
print("📊 Task B 真实数据集实验")
print("=====================================")

df = pd.read_csv(os.path.join(DATA_DIR, "kaggle_student.csv"))
y_kag = df["Exam_Score"]
X_kag = df.drop("Exam_Score", axis=1)

# 自动区分数值/分类
num_cols = make_column_selector(dtype_include=np.number)(X_kag)
cat_cols = make_column_selector(dtype_exclude=np.number)(X_kag)

# 数值特征标准化
scaler_b = CustomStandardScaler()
X_num = scaler_b.fit_transform(X_kag[num_cols])

# 分类特征独热
enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_cat = enc.fit_transform(X_kag[cat_cols])
cat_feat_names = enc.get_feature_names_out(cat_cols)

# 合并
X_all = np.hstack([X_num, X_cat])
feat_all = num_cols + list(cat_feat_names)

# 划分
Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(X_all, y_kag, test_size=0.2, random_state=42)

# 建模
cv_k = KFold(5, shuffle=True, random_state=42)

# Ridge
rk = Ridge(random_state=42)
grk = GridSearchCV(rk, {"alpha": np.logspace(-2,4,30)}, cv=cv_k, n_jobs=-1)
grk.fit(Xk_tr, yk_tr)

# Lasso
lk = Lasso(random_state=42, max_iter=50000)
glk = GridSearchCV(lk, {"alpha": np.logspace(-3,2,30)}, cv=cv_k, n_jobs=-1)
glk.fit(Xk_tr, yk_tr)

# ElasticNet
ek = ElasticNet(random_state=42, max_iter=50000)
gek = GridSearchCV(ek, {"alpha": np.logspace(-3,2,20), "l1_ratio": np.linspace(0.1,0.9,9)}, cv=cv_k, n_jobs=-1)
gek.fit(Xk_tr, yk_tr)

# OLS
ok = LinearRegression().fit(Xk_tr, yk_tr)

# 真实性能输出
print("\n===== Task B 测试集真实性能 =====")
b_ols = evaluate(ok, Xk_te, yk_te)
b_ridge = evaluate(grk.best_estimator_, Xk_te, yk_te)
b_lasso = evaluate(glk.best_estimator_, Xk_te, yk_te)
b_enet = evaluate(gek.best_estimator_, Xk_te, yk_te)
print(f"OLS:    {b_ols}")
print(f"Ridge:  {b_ridge}")
print(f"Lasso:  {b_lasso}")
print(f"ENet:   {b_enet}")

# Lasso 真实重要特征
coef_kag = pd.DataFrame({
    "feature": feat_all,
    "coef": glk.best_estimator_.coef_
})
coef_kag["abs_coef"] = coef_kag.coef.abs()
top_features = coef_kag.sort_values("abs_coef", ascending=False).head(5)
top_features.to_csv(os.path.join(RES_DIR, "kaggle_top5.csv"), index=False)

print("\n===== Lasso 真实 Top5 关键特征 =====")
print(top_features[["feature", "coef"]].to_string(index=False))

print("\n🎉 Week13 全部任务真实运行完成！")