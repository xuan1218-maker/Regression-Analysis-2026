# main.py
from simulation import loop, analysis

if __name__ == "__main__":
    # 1. 运行 100 次模拟实验，生成结果矩阵
    df_results = loop(n_sim=100, n=100)
    
    # 2. 执行统计分析并打印最终模型结果（Bias、方差、ANOVA等）
    analysis(df_results)
    
    print("\n[提示] 实验结果已输出。请手动将上述结果整理至 docs/weed_02_report.md")