# Kaggle 房价回归报告

## 数据集信息
- 名称：California Housing Prices
- 来源：https://www.kaggle.com/datasets/camnugent/california-housing-prices
- 目标：median_house_value（地区房屋中位数价格）
-- 每条样本：加州某一地理统计区块，记录该片区整体区位、人口、户型规模、居民收入与片区房屋均价
字段名	中文释义
longitude	经度
latitude	纬度
housing_median_age	房屋建成中位年限
total_rooms	片区总房间数
total_bedrooms	片区总卧室数
population	片区总人口数
households	片区住户户数
median_income	居民中位收入
median_house_value	房屋中位价格（预测目标变量）
ocean_proximity	临海区位类别（分类特征）
- 选择理由：本次实验数据样本量充足，数值特征跨度大，房屋年龄、人口、房间数量分布不均，存在明显数值差异，符合真实房产统计特征，部分字段存在空缺值与极端偏大异常样本。

## 数据问题
- 缺失值：total_bedrooms
- 异常值：total_rooms、population
- 共线性：房间数 ↔ 卧室数
- 类别变量：ocean_proximity（已独热编码）

## 流程说明
- 类别变量编码：ocean_proximity 独热编码
- 缺失值处理：total_bedrooms 均值填充
- 异常值处理：5%~95% 分位数截尾
- 预处理：CustomStandardScaler
- 模型：AnalyticalOLS
- 评估：RMSE、MAE、MAPE
- 诊断：VIF 共线性
- 验证：5折无泄露交叉验证

## VIF 结果
- longitude: 0.24
- latitude: 2.91
- housing_median_age: 1.31
- total_rooms: 12.34
- total_bedrooms: 27.18
- population: 6.34
- households: 28.12
- median_income: 1.72
- ocean_proximity_INLAND: 1.86
- ocean_proximity_ISLAND: 1.00
- ocean_proximity_NEAR BAY: 1.49
- ocean_proximity_NEAR OCEAN: 1.14

说明：
-households、total_bedrooms、total_rooms 存在极强共线性
-这三个变量高度重复，会导致模型系数不可信、解释困难
-其他变量（收入、经纬度、房龄、区位）非常稳定、无干扰

## 5 折结果
### 自研 AnalyticalOLS
RMSE: 67946.46  
MAE: 50476.47  
MAPE: 29.01%
含义：

RMSE: 67946.46模型平均误差约 6.8 万美元，受极端高价房影响较大。
MAE: 50476.47预测值与真实房价平均相差 5 万美元，是直观预测精度。
MAPE: 29.01%预测值平均偏离真实房价 29%，说明：

    房价受复杂因素影响
    线性模型预测能力有限
    只能做粗略估值，不能精准定价


### Sklearn 基线
RMSE: 67946.46  
MAE: 50476.47  
MAPE: 29.01%



## 推测结论
- 最稳定变量：median_income（收入越高房价越高，关系极稳定）
- 不稳定变量：total_rooms / total_bedrooms / households（强共线性，系数不稳定）
- 共线性/异常值问题：存在严重共线性（VIF>20），异常值会拉高预测误差
- 业务误差解释：MAPE≈28%，房价受地段、宏观影响大，线性模型难以完全精准
- 上线风险：共线性导致特征不可解释；极端房价预测偏差大

## 结果综合解释
1. **系数方向解释**
整体特征系数符合现实业务逻辑：居民收入越高、临海区位越好，房屋价格越高；内陆区域房价普遍偏低，地理区位对房价存在显著影响。

2. **误差指标含义解释**
RMSE受极端房价影响偏大，对异常样本敏感度高；MAE更贴合普通房源预测误差；MAPE接近三成，代表模型整体预测值与真实房价平均偏差在合理范围。

3. **共线性风险解释**
VIF大于10判定为中度共线性，大于20判定为严重共线性；本次卧室总数、住户数量VIF超标严重，二者信息高度重叠，会造成回归系数波动、经济学解释失效，仅适合做预测，不适合做变量归因分析。
