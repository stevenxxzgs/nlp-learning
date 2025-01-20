import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
# 读取两个 CSV 文件
file1 = 'classes.csv'
file2 = 'yt.csv'

# 加载 CSV 文件，假设文件没有表头
df1 = pd.read_csv(file1, header=None)
df2 = pd.read_csv(file2, header=None)

# 确保两个文件的行数相同
if len(df1) != len(df2):
    raise ValueError("两个 CSV 文件的行数不一致，无法计算相似度。")

# 提取列数据
col1 = df1[0].values  # 第一列数据
col2 = df2[0].values  # 第二列数据

# 计算 Jaccard 相似系数
jaccard_sim = jaccard_score(col1, col2)
accuracy = accuracy_score(col2, col1)

# 输出结果
print(f"Jaccard 相似系数: {jaccard_sim:.4f}")
print(f"acc 准确率: {accuracy:.4f}")
