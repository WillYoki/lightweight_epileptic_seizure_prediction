import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = 'G:/cnntest/pythonProject/event/chb01/test1/5Channel+SEMD+FusionCF'
channels_numbers = 3
x_file_name = f'test_slice_{channels_numbers}_5Channel+SEMD+FusionCF.csv'
x_array_path = os.path.join(path, x_file_name)

df = pd.read_csv(x_array_path)
x_array = df[f'pred_ratio_actual'].values
y_array = df[f'label'].values

# 参数设置
segment_length = 300
step = 1

# 计算每段的统计特征
segments = []
for start in range(0, len(x_array) - segment_length + 1, step):
    segment = x_array[start:start + segment_length]

    mean_val = np.mean(segment)
    greater_than_mean = np.sum(segment > mean_val)
    variance = np.var(segment)
    n1 = np.sum(segment > 0.9)
    n2 = np.sum(segment <0.5)
    # 添加统计特征
    segments.append([mean_val, greater_than_mean, variance, n1, n2])

# 转换为numpy数组
segments = np.array(segments)

# 标准化特征
# scaler = StandardScaler()
# segments_scaled = scaler.fit_transform(segments)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=4)
# labels = kmeans.fit_predict(segments_scaled)
labels = kmeans.fit_predict(segments)

# 输出每个聚类的中心点
cluster_centers = kmeans.cluster_centers_
print("聚类中心点：",cluster_centers)

# 绘制聚类结果
plt.figure(figsize=(16, 10))

# # 绘制KMeans聚类结果
# plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis', marker='x', label='KMeans Clusters')
#
# # 绘制实际标签
# plt.plot(range(len(y_array)), y_array, color='red', linestyle='-', marker='.', alpha=0.1, label='True Labels')
# 绘制KMeans聚类结果
plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis', marker='x')

# 绘制实际标签
plt.plot(range(len(y_array)), y_array, color='red', linestyle='-', marker='.', alpha=0.1)

# 设置标题和标签
plt.title('KMeans Clustering with True Labels')
# plt.xlabel('Segment Index')
# plt.ylabel('Cluster / True Label')

# 添加图例
# plt.legend()

# 显示图像
plt.show()
