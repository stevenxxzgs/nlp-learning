import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 从 CSV 文件加载数据
pred_df = pd.read_csv('predictions.csv', header=None)  # 预测概率
yt_df = pd.read_csv('yt.csv', header=None)      # 真实标签

# 将 DataFrame 转换为 NumPy 数组
predictions = pred_df[0].values  # 假设预测概率在第一列
yt = yt_df[0].values             # 假设真实标签在第一列

# 定义阈值范围
thresholds = np.arange(0.2, 0.9, 0.01)  # 从 0 到 1，步长为 0.01
accuracies = []

# 计算每个阈值下的准确率
for threshold in thresholds:
    # 根据阈值将预测概率转换为类别
    y_pred = (predictions >= threshold).astype(int)
    
    # 计算准确率
    acc = accuracy_score(yt, y_pred)
    accuracies.append(acc)

# 绘制阈值与准确率的关系图
plt.figure(figsize=(8, 5))
plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='b')
plt.title('Threshold vs Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()