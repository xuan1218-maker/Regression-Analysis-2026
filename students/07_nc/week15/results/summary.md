# Week15 总结报告

## 1. 为什么逻辑回归不是“线性回归后面接一个 sigmoid”这么简单？

逻辑回归确实使用线性得分和 sigmoid，但它的统计含义不只是“套一个函数”。它假设目标变量服从 Bernoulli 分布，模型输出的是 `P(Y=1|X)`，训练目标对应 Bernoulli likelihood 的最大化，也就是最小化 log loss。因此逻辑回归从模型输出、概率解释到优化目标都是为二分类问题设计的。

## 2. sigmoid、Bernoulli likelihood、log loss 三者之间是什么关系？

sigmoid 把线性得分映射成 0 到 1 的概率 `p`；Bernoulli likelihood 用这个 `p` 给真实标签分配概率；log loss 则是 Bernoulli likelihood 的负对数形式。三者连起来就是：线性特征组合 -> 概率输出 -> 最大似然估计 -> log loss 优化。

## 3. 为什么分类模型不能只看 accuracy？

accuracy 只看总体分类正确率，但不同错误类型的业务成本可能差别很大。比如疾病初筛中，FN 的代价通常高于 FP；信用违约中，FP 和 FN 也对应不同的资金损失和机会成本。因此还要看 precision、recall、F1、ROC-AUC、log loss，并结合阈值分析。

## 4. L1 和 L2 逻辑回归分别更适合什么目标？

L1 更适合变量筛选和得到较短变量名单，因为它能把一部分系数压到 0。L2 更适合稳定建模和缓解共线性，因为它通常保留全部变量但缩小系数。如果业务方想要稀疏解释，可以优先考虑 L1；如果更看重稳定概率输出，可以优先考虑 L2。

## 5. 为什么逻辑回归仍然是一个很强的 baseline？

逻辑回归训练快、结果稳定、输出概率、系数方向容易解释，还能通过 L1/L2 正则化适应高维分类任务。即使在工业场景中，它也常被用作强基线模型，尤其适合需要概率解释、阈值调整和业务沟通的二分类任务。

## 6. 本周结果文件

- `synthetic_report.md`：模拟数据、LinearRegression 与 LogisticRegression 对比；
- `threshold_report.md`：Bernoulli/log loss、混淆矩阵与阈值扫描；
- `regularization_report.md`：L1/L2 正则化逻辑回归；
- `real_data_report.md`：真实乳腺癌筛查数据；
- `figures/`：所有报告嵌入图片。
