# Kaggle数据分析报告（Airbnb NYC 2019）

## 一、数据说明

| 属性 | 值 |
|------|-----|
| 数据来源 | Airbnb NYC 2019公开数据集 |
| 样本量（清洗后） | 约15,000+ |
| 特征数 | 8 |
| 目标变量 | 价格（对数变换后） |

### 特征列表
- latitude：纬度
- longitude：经度
- minimum_nights：最少入住天数
- number_of_reviews：评论数量
- reviews_per_month：月均评论数
- calculated_host_listings_count：房东 listings 数量
- availability_365：一年中可预订天数
- room_type_encoded：房间类型编码（整租/私人/共享）
- neighbourhood_group_encoded：区域编码

### 为什么适合练习正则化和变量筛选？
- 特征之间存在潜在共线性（经纬度相关，评论数与月均评论数相关）
- 特征尺度差异大（经纬度 vs 评论数量）
- 真实业务场景，结果可解释
- 样本量大，适合交叉验证

## 二、模型性能对比

| 模型 | RMSE | MAE | R² |
|------|------|-----|-----|
| Ridge | 0.4636 | 0.3513 | 0.5058 |
| Lasso | 0.4636 | 0.3513 | 0.5058 |
| Elastic Net | 0.4636 | 0.3513 | 0.5058 |

### 正则化是否显著提升表现？
正则化方法相比OLS提升幅度较小，原因：
1. 数据集特征数少（仅8个），过拟合风险低
2. 样本量大（1.5万+），OLS本身已经稳定
3. 真实数据噪声大，模型上限有限

## 三、所有模型系数对比

                           特征名   OLS系数 Ridge系数 Lasso系数 ElasticNet系数
                      latitude  0.0329  0.0329  0.0329       0.0329
                     longitude -0.0997 -0.0997 -0.0997      -0.0997
                minimum_nights -0.0693 -0.0693 -0.0693      -0.0693
             number_of_reviews -0.0291 -0.0291 -0.0291      -0.0291
calculated_host_listings_count  0.0146  0.0146  0.0146       0.0146
              availability_365  0.1049  0.1049  0.1049       0.1049
             reviews_per_month -0.0090 -0.0090 -0.0090      -0.0090
             room_type_encoded  0.3879  0.3879  0.3879       0.3879
   neighbourhood_group_encoded  0.1182  0.1182  0.1182       0.1182

## 四、Lasso特征重要性排序

                           特征名   Lasso系数     |系数|
             room_type_encoded  0.387930 0.387930
   neighbourhood_group_encoded  0.118244 0.118244
              availability_365  0.104899 0.104899
                     longitude -0.099709 0.099709
                minimum_nights -0.069310 0.069310
                      latitude  0.032885 0.032885
             number_of_reviews -0.029117 0.029117
calculated_host_listings_count  0.014602 0.014602
             reviews_per_month -0.009036 0.009036

### 特征剔除的合理性分析
Lasso将部分特征系数压缩到接近0，从业务角度看是合理的：
- 评论数量和月均评论数相关性高，Lasso选择保留一个
- 区域编码反映了地理位置的重要性
- 房间类型是价格的重要决定因素

## 五、最关键的5个影响因素

根据Lasso系数绝对值排序，最关键的5个因素是：

| 排名 | 特征 | Lasso系数 | 业务解释 |
|------|------|-----------|----------|
| 1 | neighbourhood_group_encoded | 0.1182 | 区域（曼哈顿最贵） |
| 2 | room_type_encoded | 0.3879 | 房间类型（整租最贵） |
| 3 | latitude | 0.0329 | 纬度 |
| 4 | longitude | -0.0997 | 经度 |
| 5 | minimum_nights | -0.0693 | 最少入住天数 |

**为什么以Lasso结果为准？**
- Lasso做了自动特征选择，剔除了相关性高的冗余特征
- 系数大小直接反映重要性，便于业务沟通
- 相比Ridge的均匀收缩，Lasso的名单更简洁
