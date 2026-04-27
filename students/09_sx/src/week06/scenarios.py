import numpy as np
from sklearn.linear_model import LinearRegression
from engine import CustomOLS
from utils import *

def scenario_A(results_path):
    print("\n" + "="*50)
    print("场景A: 合成数据测试")
    print("="*50)
    
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    y = 10 + X @ [2.5, -1.5, 3.0] + np.random.randn(1000) * 2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    custom = CustomOLS().fit(X_train, y_train)
    sklearn = LinearRegression().fit(X_train, y_train)
    
    r2_custom = custom.score(X_test, y_test)
    r2_sklearn = sklearn.score(X_test, y_test)
    
    save_residual_plot(custom, X_test, y_test, "Synthetic", 
                       results_path / "figures" / "synthetic_residuals.png")
    
    report = f"""# 合成数据验证报告

## 模型对比
| 模型 | R² |
|------|-----|
| CustomOLS | {r2_custom:.4f} |
| sklearn | {r2_sklearn:.4f} |

## 系数对比
| 参数 | 真实值 | CustomOLS | sklearn |
|------|--------|-----------|---------|
| 截距 | 10.0 | {custom.coef_[0]:.4f} | {sklearn.intercept_:.4f} |
| β1 | 2.5 | {custom.coef_[1]:.4f} | {sklearn.coef_[0]:.4f} |
| β2 | -1.5 | {custom.coef_[2]:.4f} | {sklearn.coef_[1]:.4f} |
| β3 | 3.0 | {custom.coef_[3]:.4f} | {sklearn.coef_[2]:.4f} |
"""
    with open(results_path / "synthetic_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"场景A完成，CustomOLS R²: {r2_custom:.4f}")

def scenario_B(results_path):
    print("\n" + "="*50)
    print("场景B: 真实营销数据")
    print("="*50)
    
    df = load_data()
    df_na, df_eu = split_region(df)
    
    cols = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday']
    
    # 北美市场
    X_na, y_na = df_na[cols].values, df_na['Sales'].values
    X_na_train, X_na_test, y_na_train, y_na_test = train_test_split(X_na, y_na)
    model_na = CustomOLS().fit(X_na_train, y_na_train)
    r2_na = model_na.score(X_na_test, y_na_test)
    print(f"北美市场 R²: {r2_na:.4f}")
    
    # 欧洲市场
    X_eu, y_eu = df_eu[cols].values, df_eu['Sales'].values
    X_eu_train, X_eu_test, y_eu_train, y_eu_test = train_test_split(X_eu, y_eu)
    model_eu = CustomOLS().fit(X_eu_train, y_eu_train)
    r2_eu = model_eu.score(X_eu_test, y_eu_test)
    print(f"欧洲市场 R²: {r2_eu:.4f}")
    
    # F检验：广告预算整体有效性
    C = np.array([[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0]])
    na_f = model_na.f_test(C, np.zeros(3))
    eu_f = model_eu.f_test(C, np.zeros(3))
    
    print(f"\n北美市场 F检验: F={na_f['f_stat']:.4f}, p={na_f['p_value']:.6f}")
    print(f"欧洲市场 F检验: F={eu_f['f_stat']:.4f}, p={eu_f['p_value']:.6f}")
    
    # 保存图表
    save_residual_plot(model_na, X_na_test, y_na_test, "NA Market", 
                       results_path / "figures" / "na_residuals.png")
    save_residual_plot(model_eu, X_eu_test, y_eu_test, "EU Market", 
                       results_path / "figures" / "eu_residuals.png")
    
    na_result = "显著" if na_f['p_value'] < 0.05 else "不显著"
    eu_result = "显著" if eu_f['p_value'] < 0.05 else "不显著"
    
    report = f"""# 真实营销数据报告

## 数据概况
| 市场 | 样本量 |
|------|--------|
| 北美 | {len(df_na)} |
| 欧洲 | {len(df_eu)} |

## 模型性能
| 市场 | R² |
|------|-----|
| 北美 | {r2_na:.4f} |
| 欧洲 | {r2_eu:.4f} |

## 回归系数
| 变量 | 北美 | 欧洲 |
|------|------|------|
| 截距 | {model_na.coef_[0]:.4f} | {model_eu.coef_[0]:.4f} |
| TV_Budget | {model_na.coef_[1]:.4f} | {model_eu.coef_[1]:.4f} |
| Radio_Budget | {model_na.coef_[2]:.4f} | {model_eu.coef_[2]:.4f} |
| SocialMedia_Budget | {model_na.coef_[3]:.4f} | {model_eu.coef_[3]:.4f} |
| Is_Holiday | {model_na.coef_[4]:.4f} | {model_eu.coef_[4]:.4f} |

## F检验（广告预算整体有效性）
| 市场 | F统计量 | p值 | 结论 |
|------|---------|-----|------|
| 北美 | {na_f['f_stat']:.4f} | {na_f['p_value']:.6f} | {na_result} |
| 欧洲 | {eu_f['f_stat']:.4f} | {eu_f['p_value']:.6f} | {eu_result} |
"""
    with open(results_path / "real_world_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n场景B完成")