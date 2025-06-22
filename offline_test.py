from PyEMD import EMD
import numpy as np
import os
import pandas as pd
from scipy.signal import welch
import time
from tensorflow.keras.models import load_model
from joblib import load


# 函数调用
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

def compute_sodp_ellipse_area(signal):
    """
    根据 SODP 方法计算椭圆面积。

    参数:
    signal: numpy array, 输入的一维信号。

    返回:
    Ae: float, SODP 椭圆面积。
    """
    # 计算一阶和二阶差分
    y1 = np.diff(signal)
    y2 = np.diff(y1)

    # 总样本数
    N = len(signal)

    # 根据公式计算 κ1, κ2, κ12
    kappa_1 = np.sqrt(np.sum(y1[:-1] ** 2) / (N - 2))
    kappa_2 = np.sqrt(np.sum(y2 ** 2) / (N - 2))
    kappa_12 = np.sum(y1[:-1] * y2) / (N - 2)

    # 计算 δ
    delta = np.sqrt((kappa_1 ** 2 + kappa_2 ** 2) ** 2 - 4 * (kappa_1 ** 2 * kappa_2 ** 2 - kappa_12 ** 2))

    # 计算 α 和 β
    alpha = 1.7321 * np.sqrt(kappa_1 ** 2 + kappa_2 ** 2 + delta)
    beta = 1.7321 * np.sqrt(kappa_1 ** 2 + kappa_2 ** 2 - delta)

    # 椭圆面积 Ae
    Ae = np.pi * alpha * beta

    return Ae


def compute_SEMD(matrix):
    num_channels, num_samples = matrix.shape
    num_features = 3 * 3  # 每个 IMF 包含 SODP, Vc, Fi 三个特征

    # 添加头尾 32 个采样点
    padding_length = 32
    padded_channels = []
    for channel in range(num_channels):
        channel_data = matrix[channel, :]
        padded_data = np.concatenate([
            channel_data[:padding_length][::-1],  # 头部 32 个点翻转
            channel_data,
            channel_data[-padding_length:][::-1]  # 尾部 32 个点翻转
        ])
        padded_channels.append(padded_data)

    # 将所有通道拼接为一维信号
    concatenated_data = np.concatenate(padded_channels)

    # 进行 EMD 分解
    emd = EMD()
    IMFs = emd.emd(concatenated_data)

    # 初始化特征矩阵
    features_matrix = np.zeros((num_channels, num_features))

    # 还原各通道并提取特征
    channel_lengths = [num_samples + 2 * padding_length] * num_channels
    start_idx = 0

    for channel in range(num_channels):
        channel_length = channel_lengths[channel]
        end_idx = start_idx + channel_length

        channel_IMFs = [imf[start_idx + padding_length:end_idx - padding_length] for imf in IMFs]
        start_idx = end_idx

        # 提取每个通道的 IMF 特征
        for j in range(min(len(channel_IMFs), 3)):  # 最多计算 6 个 IMF
            imf_component = channel_IMFs[j]
            N = len(imf_component)
            mean_value = np.mean(imf_component)

            # SODP 特征
            features_matrix[channel, j*3] = compute_sodp_ellipse_area(imf_component)

            # 方差 (Vc) 特征
            Vc = np.sum((imf_component - mean_value) ** 2) / (N - 1)
            features_matrix[channel, 1 + j*3] = Vc

            # 波动指数 (Fi) 特征
            Fi = np.sum(np.abs(np.diff(imf_component))) / (N - 1)
            features_matrix[channel, 2 + j*3] = Fi

    return features_matrix


def feature_importance_out(input_matrix, weight_matrix):
    # 获取输入矩阵的形状
    num_samples, num_channels, num_features = input_matrix.shape

    # 初始化输出矩阵
    output_matrix = np.zeros((num_samples, num_features))
    output_matrix_2 = np.zeros((num_samples, num_channels))

    # 遍历每个样本
    for i in range(num_samples):
        # 获取当前样本的所有通道
        sample_channels = input_matrix[i]

        # 初始化当前样本的新特征
        new_feature = 0
        new_features_2 = np.zeros(num_channels)

        # 遍历每个通道
        for j in range(num_channels):
            # 获取当前通道的特征
            channel_features = sample_channels[j]

            # 点乘权重矩阵的对应行
            weighted_features = channel_features * weight_matrix[j]

            # 按行相加（累加每个特征）
            new_feature += weighted_features
            # 按列相加（累加每个通道的特征）
            new_features_2[j] = np.sum(weighted_features)

        # 将新特征存储到输出矩阵中
        output_matrix[i] = new_feature
        output_matrix_2[i] = new_features_2

    return output_matrix, output_matrix_2

# 加载数据
# 基础路径
base_path = r'path'
# test_folder_list = [1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23]

# 通道数列表
channels_numbers_list = [3, 6, 9]
# channels_numbers_list = [9]

# 遍历 chb01 到 chb12
# for chb in test_folder_list:
#     chb_folder = f'chb{chb:02d}'  # 格式化 chb 文件夹名称，如 chb01, chb02, ..., chb12
try:
    chb_folder = 'chb01'

    path = os.path.join(base_path, chb_folder)

    # 获取 test 文件夹列表
    test_folders = [folder for folder in os.listdir(path) if folder.startswith('test')]

    # 遍历每个 test 文件夹
    for test_folder in test_folders:
        test_path = os.path.join(path, test_folder)

        # 遍历每个通道数
        for channels_numbers in channels_numbers_list:

            print(f'Processing {chb_folder}/{test_folder} with channels_numbers={channels_numbers}')


            test_data_path = os.path.join(test_path, 'test_data.npy')
            y_label_path = os.path.join(test_path, 'test_slice_index.xlsx')

            df = pd.read_excel(y_label_path)
            # 提取标签列，假设列名为 'label'
            y_test = df['Label'].values
            target_dir = os.path.join(test_path, "5Channel+SEMD+FusionCF")
            sorted_channel = np.load(os.path.join(target_dir, f'sorted_channel_{channels_numbers}.npy'))
            print(sorted_channel)
            feature_importance = np.load(os.path.join(target_dir, f'feature_importance_{channels_numbers}.npy'))

            # 加载 MinMaxScaler
            scaler = load(os.path.join(target_dir, f"minmax_scaler_{channels_numbers}.joblib"))

            # 加载数据
            test_data = np.load(test_data_path)
            test_data = test_data[:, sorted_channel, :]

            # 测试、计时
            model = load_model(f'{test_path}/5Channel+SEMD+FusionCF/best_model_{channels_numbers}.h5')
            # model.load_weights(f'{test_path}/5Channel+SEMD+FusionCF/best_model_{channels_numbers}.h5')
            print('start test:')
            No_test_data = test_data.shape[0]
            print('No_test_data:', No_test_data)
            test_data_features = np.zeros((test_data.shape[0], test_data.shape[1], 25))
            start_time = time.time()
            for i in range(test_data.shape[0]):
                test_data_features[i, :, :16] = compute_ber_matrix(test_data[i, :, :], 256)
                test_data_features[i, :, 16:] = compute_SEMD(test_data[i, :, :])

            X_test_0, X_test_1 = feature_importance_out(test_data_features, feature_importance)
            X_test = np.concatenate((X_test_0, X_test_1), axis=1)

            # X_test_scaled = scaler.transform(X_test)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            test_loss, test_acc, test_pre, test_recall = model.evaluate(X_test, y_test)

            # 评估模型
            time_total = time.time() - start_time
            time_average = time_total / No_test_data

            np.save(os.path.join(f'{test_path}/5Channel+SEMD+FusionCF', f'test_features_{channels_numbers}.npy'), X_test)

            print('Test accuracy:', test_acc)
            print('Average time:', time_average)
            with open(f"{test_path}/5Channel+SEMD+FusionCF/run_time_{channels_numbers}.txt", 'w') as file:
                # 写入文件
                file.write(
                    f"time_total:\n{time_total}\n\ntime_average:\n{time_average}\n\ntest_loss:\n{test_loss}\n\ntest_acc:\n{test_acc}\n\ntest_pre:\n{test_pre}\n\ntest_recall:\n{test_recall}")

            # 使用模型对测试数据进行预测
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)  # 将概率转换为二进制分类

            # 合并为一个元组
            stacked_arrays = (y_test, y_pred_binary, y_pred)

            # 使用np.column_stack()函数按列合并数组
            test = np.column_stack(stacked_arrays)

            df = pd.DataFrame(test, columns=['label', f'predict_ratio',
                                             f'pred_ratio_actual'])
            # 保存到CSV文件
            csv_name = f'test_slice_{channels_numbers}_5Channel+SEMD+FusionCF.csv'
            csv_path = os.path.join(f'{test_path}/5Channel+SEMD+FusionCF', csv_name)
            df.to_csv(csv_path, index=False)

except IndentationError as e:
    print(f"错误: {e}")