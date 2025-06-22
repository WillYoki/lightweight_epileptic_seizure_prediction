from tqdm import tqdm
from joblib import Parallel, delayed
from PyEMD import EMD
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.signal import welch
import csv


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

def rf_channel_importance(pos_samples, neg_samples, num_top_channels=18):
    # 组合正负样本，标签正样本为1，负样本为0
    X = np.vstack([pos_samples, neg_samples])  # (num_samples, 18, N)
    y = np.hstack([np.ones(len(pos_samples)), np.zeros(len(neg_samples))])  # 样本标签

    # 将每个样本展开为一维矩阵，方便分类
    num_samples, num_channels, num_features = X.shape
    X_reshaped = X.reshape(num_samples, -1)  # (num_samples, 18 * N)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # 使用随机森林进行分类
    rf = RandomForestClassifier(n_estimators=200, random_state=2)
    rf.fit(X_train, y_train)

    # 获取每个特征的重要性
    feature_importances = rf.feature_importances_  # 长度为18*N的向量

    # 将重要性重新组织为(18, N)的矩阵形式
    channel_importances = feature_importances.reshape(num_channels, num_features)

    # 计算每个通道的重要性贡献度，即对每个通道的特征重要性进行求和
    channel_contributions = np.sum(channel_importances, axis=1)  # (18,)

    # 按贡献度从大到小对通道进行排序
    sorted_channel_indices = np.argsort(channel_contributions)[::-1]  # 从大到小排序

    channel_indices = sorted_channel_indices[:num_top_channels]

    return channel_indices  # 返回通道排序

def rf_channel_importance_heatmap(pos_samples, neg_samples):
    # 组合正负样本，标签正样本为1，负样本为0
    X = np.vstack([pos_samples, neg_samples])  # (num_samples, 18, N)
    y = np.hstack([np.ones(len(pos_samples)), np.zeros(len(neg_samples))])  # 样本标签

    # 将每个样本展开为一维矩阵，方便分类
    num_samples, num_channels, num_features = X.shape
    X_reshaped = X.reshape(num_samples, -1)  # (num_samples, 18 * N)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # 使用随机森林进行分类
    rf = RandomForestClassifier(n_estimators=200, random_state=2)
    rf.fit(X_train, y_train)

    # 获取每个特征的重要性
    feature_importances = rf.feature_importances_  # 长度为18*N的向量

    # 将重要性重新组织为(18, N)的矩阵形式
    channel_importances = feature_importances.reshape(num_channels, num_features)

    return channel_importances

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
# channels_numbers_list = [6]

# 遍历 chb01 到 chb12
# for chb in test_folder_list:
#     chb_folder = f'chb{chb:02d}'  # 格式化 chb 文件夹名称，如 chb01, chb02, ..., chb12
try:
    chb_folder = 'chb01'

    path = os.path.join(base_path, chb_folder)

    # 获取 test 文件夹列表
    test_folders = [folder for folder in os.listdir(path) if folder.startswith('test1')]

    # 遍历每个 test 文件夹
    for test_folder in test_folders:
        test_path = os.path.join(path, test_folder)

        # 遍历每个通道数
        for channels_numbers in channels_numbers_list:

            print(f'Processing {chb_folder}/{test_folder} with channels_numbers={channels_numbers}')

            positive_features_path = os.path.join(test_path, 'positive_features.npy')
            negative_features_path = os.path.join(test_path, 'negative_features.npy')
            test_data_path = os.path.join(test_path, 'test_data.npy')
            y_label_path = os.path.join(test_path, 'test_slice_index.xlsx')
            # 加载数据
            positive_features_f = np.load(positive_features_path)
            negative_features_f = np.load(negative_features_path)

            positive_features_f = positive_features_f[:, :, 0:16]
            negative_features_f = negative_features_f[:, :, 0:16]
            sorted_channel = rf_channel_importance(positive_features_f, negative_features_f, channels_numbers)
            print(sorted_channel)
            # 创建目录（如果不存在）
            target_dir = os.path.join(test_path, "5Channel+SEMD+FusionCF")
            os.makedirs(target_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
            np.save(os.path.join(target_dir,f'sorted_channel_{channels_numbers}.npy'), sorted_channel)
            txt_name = f'sorted_channel_{channels_numbers}.txt'
            txt_path = os.path.join(f'{test_path}/5Channel+SEMD+FusionCF', txt_name)
            # 确保目录存在
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            # 将 sorted_channel 的值保存到指定路径的 txt 文件中
            with open(txt_path, 'w') as file:
                for item in sorted_channel:
                    file.write(f"{item}\n")

            positive_feature_1 = positive_features_f[:, sorted_channel, 0:16]
            negative_feature_1 = negative_features_f[:, sorted_channel, 0:16]

            positive_data_path = os.path.join(test_path, 'positive_data.npy')
            negative_data_path = os.path.join(test_path, 'negative_data.npy')
            test_data_path = os.path.join(test_path, 'test_data.npy')

            # 加载数据
            positive_data_2 = np.load(positive_data_path)
            negative_data_2 = np.load(negative_data_path)
            test_data = np.load(test_data_path)
            positive_data_2 = positive_data_2[:, sorted_channel, :]
            negative_data_2 = negative_data_2[:, sorted_channel, :]
            test_data = test_data[:, sorted_channel, :]

            def process_data(data, index, sample_rate):
                combined_features = compute_SEMD(data)
                return index, combined_features

            def parallel_processing(data, sample_rate, desc):
                results = Parallel(n_jobs=8)(delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
                # results = Parallel(n_jobs=-1)(delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
                feature_length = results[0][1].shape[1]  # 获取特征向量的总长度
                num_features = len(results[0][1])  # 获取特征的数量
                features_matrix = np.zeros((data.shape[0], num_features, feature_length))
                for index, features in results:
                    features_matrix[index, :, :] = features
                return features_matrix

            # 处理正样本数据
            positive_data_features = parallel_processing(positive_data_2, 256, "Processing positive data")
            print("Shape:", positive_data_features.shape)
            positive_features = np.concatenate((positive_feature_1, positive_data_features), axis=2)

            # 处理负样本数据
            negative_data_features = parallel_processing(negative_data_2, 256, "Processing negative data")
            print("Shape:", negative_data_features.shape)
            negative_features = np.concatenate((negative_feature_1, negative_data_features), axis=2)

            feature_importance = rf_channel_importance_heatmap(positive_features, negative_features)
            np.save(os.path.join(target_dir,f'feature_importance_{channels_numbers}.npy'), feature_importance)

            positive_data_feature_0, positive_data_feature_1 = feature_importance_out(positive_features, feature_importance)
            negative_data_feature_0, negative_data_feature_1 = feature_importance_out(negative_features, feature_importance)
            positive_data_feature = np.concatenate((positive_data_feature_0, positive_data_feature_1), axis=1)
            negative_data_feature = np.concatenate((negative_data_feature_0, negative_data_feature_1), axis=1)

            csv_name = f'test_slice_{channels_numbers}_5Channel+SEMD+FusionCF.csv'

            # 标记正负样本
            positive_labels = np.ones(len(positive_data_feature))
            negative_labels = np.zeros(len(negative_data_feature))

            # 合并正负样本的训练集
            X_train_val = np.concatenate((positive_data_feature, negative_data_feature), axis=0)
            y_train_val = np.concatenate((positive_labels, negative_labels), axis=0)

            # 按 4:1 的比例将剩余数据划分为训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 5)

            # 确保数据形状正确
            print(f'Train set shape: {X_train.shape}')
            print(f'Validation set shape: {X_val.shape}')

            # Reshape the data for CNN input (if required)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

            # 创建模型
            model = models.Sequential([
                layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1), padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=1),

                layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=1),

                layers.Conv1D(128, kernel_size=2, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=1),

                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])

            precision_metric = tf.keras.metrics.Precision(name='precision')
            recall_metric = tf.keras.metrics.Recall(name='recall')

            # 编译模型
            model.compile(optimizer=Adam(1e-3),
                          loss='binary_crossentropy',
                          metrics=['accuracy', precision_metric, recall_metric])

            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=10,
                min_lr=1e-6
            )


            # 初始化自定义回调函数
            class EarlyStoppingWithBestModel(tf.keras.callbacks.Callback):
                def __init__(self, patience=30):
                    super(EarlyStoppingWithBestModel, self).__init__()
                    self.patience = patience
                    self.best_weights = None
                    self.best_val_acc = 0
                    self.best_epoch = 0
                    self.wait = 0

                def on_epoch_end(self, epoch, logs=None):
                    current_val_acc = logs.get('val_accuracy')
                    if current_val_acc is not None:
                        if current_val_acc > self.best_val_acc:
                            self.best_val_acc = current_val_acc
                            self.best_epoch = epoch  # 更新最佳epoch
                            self.wait = 0
                            self.best_weights = self.model.get_weights()
                        else:
                            self.wait += 1
                            if self.wait >= self.patience:
                                print(
                                    f'\nEarly stopping: restoring best model weights from epoch {self.best_epoch}')  # 打印最佳epoch
                                self.model.set_weights(self.best_weights)
                                self.model.stop_training = True


            checkpoint_callback = ModelCheckpoint(
                filepath=f'{test_path}/5Channel+SEMD+FusionCF/best_model_{channels_numbers}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )

            early_stopping_callback = EarlyStoppingWithBestModel(patience=30)

            # 训练模型
            history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_val, y_val),
                                callbacks=[lr_reducer, early_stopping_callback, checkpoint_callback])

            # 定义 CSV 文件路径
            csv_file_path = f'{test_path}/5Channel+SEMD+FusionCF/training_history_{channels_numbers}.csv'

            # 写入表头
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'epoch', 'loss', 'accuracy', 'precision', 'recall',
                    'val_loss', 'val_accuracy', 'val_precision', 'val_recall'
                ])

            # 逐条写入每个 epoch 的结果
            for epoch in range(len(history.history['loss'])):
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        epoch + 1,  # Epoch 编号
                        history.history['loss'][epoch],
                        history.history['accuracy'][epoch],
                        history.history['precision'][epoch],
                        history.history['recall'][epoch],
                        history.history['val_loss'][epoch],
                        history.history['val_accuracy'][epoch],
                        history.history['val_precision'][epoch],
                        history.history['val_recall'][epoch]
                    ])

            # 将最优模型的结果追加到 CSV 文件中
            best_epoch = early_stopping_callback.best_epoch
            best_results = {
                'epoch': best_epoch + 1,  # Epoch 编号从 1 开始
                'loss': history.history['loss'][best_epoch],
                'accuracy': history.history['accuracy'][best_epoch],
                'precision': history.history['precision'][best_epoch],
                'recall': history.history['recall'][best_epoch],
                'val_loss': history.history['val_loss'][best_epoch],
                'val_accuracy': history.history['val_accuracy'][best_epoch],
                'val_precision': history.history['val_precision'][best_epoch],
                'val_recall': history.history['val_recall'][best_epoch]
            }
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Best Model Results'])
                writer.writerow([
                    'epoch', 'loss', 'accuracy', 'precision', 'recall',
                    'val_loss', 'val_accuracy', 'val_precision', 'val_recall'
                ])
                writer.writerow([
                    best_results['epoch'],
                    best_results['loss'],
                    best_results['accuracy'],
                    best_results['precision'],
                    best_results['recall'],
                    best_results['val_loss'],
                    best_results['val_accuracy'],
                    best_results['val_precision'],
                    best_results['val_recall']
                ])


            # 可视化训练过程
            def plot_training_history(history):
                plt.figure(figsize=(12, 4))

                # 绘制损失曲线
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Loss Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # 绘制准确率曲线
                plt.subplot(1, 2, 2)
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Accuracy Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()

                plt.tight_layout()
                plt.savefig(f'{test_path}/5Channel+SEMD+FusionCF/training_curves_{channels_numbers}.png')
                plt.show()


except IndentationError as e:
    print(f"错误: {e}")