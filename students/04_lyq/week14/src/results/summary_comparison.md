
# Lasso vs PCR 核心总结

## 1. Sparse truth（少量真实变量）
- Lasso 更好：自动筛选真正变量
- PCR 压缩所有方向，不关心变量是否重要

## 2. Latent-factor truth（潜在因子）
- PCR 更稳、更干净
- 数据是低秩线性结构，压缩比筛选更自然

## 3. 核心区别
- Lasso：做 selection（谁留下）
- PCR：做 compression（信息压缩）

## 4. 业务选择
- 要短名单 → Lasso输出非零系数，直接给出关键变量。
- 要稳定预测 → PCR抗共线性、抗噪声，预测更稳健

## 5. 前向/后向选择
- 属于变量筛选路线
- 高维下不稳定
