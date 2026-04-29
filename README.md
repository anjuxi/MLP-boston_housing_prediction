# **全连接前馈神经网络**-波士顿房价预测

本项目使用多层全连接神经网络对波士顿房价进行回归预测。数据集采用经典的 Boston Housing 数据集，包含 13 个特征用于预测房屋价格中位数。

数据集来自：`https://www.kaggle.com/datasets/altavish/boston-housing-dataset`

## 项目特点

- **深度学习框架**：使用 PyTorch 构建神经网络
- **多层网络结构**：包含 BatchNorm 和 Dropout 提升泛化能力
- **数据可视化**：支持特征分布图、相关性热力图、预测散点图、残差分析图
- **早停机制**：防止过拟合，自动保存最佳模型
- **学习率衰减**：自适应调整学习率优化训练
- **继续训练**：支持从已有模型继续训练

## 数据集

**Boston Housing 数据集**包含 506 条记录，13 个特征：

| 特征     | 说明                                             |
| -------- | ------------------------------------------------ |
| CRIM     | 城镇人均犯罪率                                   |
| ZN       | 住宅用地超过 25000 sq.ft. 的比例                 |
| INDUS    | 城镇非零售业务地区的比例                         |
| CHAS     | 查尔斯河虚拟变量（临近河流为 1，否则为 0）       |
| NOX      | 一氧化氮浓度（千万分之一）                       |
| RM       | 每个住宅的平均房间数                             |
| AGE      | 1940 年之前建成的自住房屋的比例                  |
| DIS      | 到波士顿五个就业中心的加权距离                   |
| RAD      | 径向公路的可达性指数                             |
| TAX      | 每 10000 美元的全值财产税率                      |
| PTRATIO  | 城镇师生比例                                     |
| B        | 1000(Bk - 0.63)^2，其中 Bk 是城镇黑人比例        |
| LSTAT    | 人口中地位较低人群的百分数                       |
| **MEDV** | **自住房屋的中位数价值（单位：千美元）【标签】** |

## 文件说明

| 文件                           | 说明                                  |
| ------------------------------ | ------------------------------------- |
| `house_price_prediction.ipynb` | Jupyter Notebook 源码，适合学习和调试 |
| `HousingData.csv`              | Boston Housing 数据集                 |
| `house_price_model.pth`        | 训练好的模型权重文件                  |

## 使用方法

```bash
jupyter notebook house_price_prediction.ipynb
```

按顺序执行各个 Cell 即可。

## 模型架构

```
输入层 (13) → 隐藏层1 (128) → 隐藏层2 (64) → 隐藏层3 (32) → 输出层 (1)
                ↓               ↓               ↓
            BatchNorm       BatchNorm       BatchNorm
            ReLU            ReLU            ReLU
            Dropout(0.2)    Dropout(0.2)    Dropout(0.2)
```

**总参数量**：约 12,609

## 训练参数

| 参数                    | 值    | 说明         |
| ----------------------- | ----- | ------------ |
| BATCH_SIZE              | 256   | 批次大小     |
| LEARNING_RATE           | 0.001 | 初始学习率   |
| EPOCHS                  | 300   | 最大训练轮数 |
| EARLY_STOPPING_PATIENCE | 30    | 早停耐心值   |
| TEST_SIZE               | 0.2   | 测试集占比   |

运行后会生成以下可视化图表：

1. **特征分布直方图** - 展示各特征的分布情况
2. **相关性热力图** - 展示特征间的相关性
3. **预测散点图** - 真实值 vs 预测值对比
4. **残差分析图** - 残差分布和残差 vs 预测值

## 模型评估指标

- **MSE** (均方误差)
- **RMSE** (均方根误差)
- **MAE** (平均绝对误差)
- **R² 分数**

## 继续训练

如需从已有模型继续训练，修改代码中的参数：

```python
CONTINUE_TRAIN = True  # 设置为 True
```

程序会自动加载 `house_price_model.pth` 继续训练。

## 单条预测示例

```python
# 输入 13 个特征值
price = predict_single(model, 0.00632, 18.0, 2.31, 0, 0.538, 6.575, 
                       65.2, 4.09, 1, 296, 15.3, 396.9, 4.98)
print(f"预测房价: {price:.2f} 千美元")
```

## 联系方式

**谙弆悕博士（Ailan Anjuxi）**

- 邮箱：anjuxi.ME@outlook.com
- SIP 电话：sip:anjuxi@sip.linphone.org


