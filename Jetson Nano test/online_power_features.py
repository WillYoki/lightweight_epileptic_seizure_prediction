import numpy as np
import os
from scipy.signal import welch
from tqdm import tqdm
from joblib import Parallel, delayed

# 加载数据
# 读取.npy文件中的数据
path = 'path'
print(path)
positive_data_path = os.path.join(path, 'positive_data.npy')
negative_data_path = os.path.join(path, 'negative_data.npy')
test_data_path = os.path.join(path, 'test_data.npy')
# 加载数据
positive_data = np.load(positive_data_path)
negative_data = np.load(negative_data_path)
test_data = np.load(test_data_path)

print(positive_data.shape)

def compute_band_energy_ratio(signal, fs, band):
    # 计算信号的功率谱密度（PSD）
    freqs, psd = welch(signal, fs)

    # 计算指定频段内的能量
    band_energy = np.trapz(psd[(freqs >= band[0]) & (freqs <= band[1])],
                           freqs[(freqs >= band[0]) & (freqs <= band[1])])

    # 计算总能量
    total_energy = np.trapz(psd, freqs)

    # 检查总能量是否为零
    if total_energy == 0:
        return 0.0, 0.0

    # 计算能量占比
    band_energy_ratio = band_energy / total_energy

    return band_energy, band_energy_ratio

def compute_ber_matrix(matrix, fs):
    num_channels, num_samples = matrix.shape
    num_features = 16  # 每个通道的特征数量
    features_matrix = np.zeros((num_channels, num_features))

    for channel in range(num_channels):
        channel_data = matrix[channel, :]
        features_matrix[channel, 0], features_matrix[channel, 1] = compute_band_energy_ratio(channel_data, fs, (0.5, 4))
        features_matrix[channel, 2], features_matrix[channel, 3] = compute_band_energy_ratio(channel_data, fs, (4, 8))
        features_matrix[channel, 4], features_matrix[channel, 5] = compute_band_energy_ratio(channel_data, fs, (8, 12))
        features_matrix[channel, 6], features_matrix[channel, 7] = compute_band_energy_ratio(channel_data, fs, (12, 30))
        features_matrix[channel, 8], features_matrix[channel, 9] = compute_band_energy_ratio(channel_data, fs, (30, 40))
        features_matrix[channel, 10], features_matrix[channel, 11] = compute_band_energy_ratio(channel_data, fs, (40, 58))
        features_matrix[channel, 12], features_matrix[channel, 13] = compute_band_energy_ratio(channel_data, fs, (62, 80))
        features_matrix[channel, 14], features_matrix[channel, 15] = compute_band_energy_ratio(channel_data, fs, (80, 100))

    return features_matrix


def process_data(data, index, sample_rate):
    data_f_ratio = compute_ber_matrix(data, sample_rate)
    return index, data_f_ratio

def parallel_processing(data, sample_rate, desc):
    results = Parallel(n_jobs=-1)(delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
    feature_length = results[0][1].shape[1]  # 获取特征向量的总长度
    num_features = len(results[0][1])  # 获取特征的数量
    features_matrix = np.zeros((data.shape[0], num_features, feature_length))
    for index, features in results:
        features_matrix[index, :, :] = features
    return features_matrix

# 处理正样本数据
positive_data_features = parallel_processing(positive_data, 256, "Processing positive data")
print("Shape:", positive_data_features.shape)
np.save(os.path.join(path, 'positive_features.npy'), positive_data_features)

# 处理负样本数据
negative_data_features = parallel_processing(negative_data, 256, "Processing negative data")
print("Shape:", negative_data_features.shape)
np.save(os.path.join(path, 'negative_features.npy'), negative_data_features)



