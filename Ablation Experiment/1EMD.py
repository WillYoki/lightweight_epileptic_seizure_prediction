from tqdm import tqdm
from joblib import Parallel, delayed
from PyEMD import EMD
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.signal import welch
import time
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

def compute_EMD(matrix):
    num_channels, num_samples = matrix.shape
    num_features = 3 * 3  # n 个 IMF，每个 IMF 包含 SODP, Vc, Fi 三个特征
    features_matrix = np.zeros((num_channels, num_features))

    for channel in range(num_channels):
        channel_data = matrix[channel, :]
        emd = EMD()
        IMF = emd.emd(channel_data)

        # 处理存在的 IMF 分量
        for j in range(min(len(IMF), 3)):
            imf_component = IMF[j]
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


# 基础路径
base_path = 'G:/cnntest/pythonProject/event'

test_folder_list = [2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23]

# 通道数列表
channels_numbers_list = [3, 6, 9]
# channels_numbers_list = [9]

# 遍历 chb01 到 chb12
# for chb in test_folder_list:
#     chb_folder = f'chb{chb:02d}'  # 格式化 chb 文件夹名称，如 chb01, chb02, ..., chb12
try:
    chb_folder = 'chb09'

    path = os.path.join(base_path, chb_folder)

    # 获取 test 文件夹列表
    test_folders = [folder for folder in os.listdir(path) if folder.startswith('test')]

    # 遍历每个 test 文件夹
    for test_folder in test_folders:
        test_path = os.path.join(path, test_folder)

        print(f'Processing {chb_folder}/{test_folder}')

        positive_data_path = os.path.join(test_path, 'positive_data.npy')
        negative_data_path = os.path.join(test_path, 'negative_data.npy')
        test_data_path = os.path.join(test_path, 'test_data.npy')
        y_label_path = os.path.join(test_path, 'test_slice_index.xlsx')
        # 加载数据
        positive_data = np.load(positive_data_path)
        negative_data = np.load(negative_data_path)
        test_data = np.load(test_data_path)

        df = pd.read_excel(y_label_path)
        # 提取标签列，假设列名为 'label'
        y_test = df['Label'].values


        def process_data(data, index, sample_rate):
            test_features_1 = compute_ber_matrix(data, 256)
            test_features_2 = compute_EMD(data)
            combined_features = np.hstack((test_features_1, test_features_2))
            return index, combined_features


        def parallel_processing(data, sample_rate, desc):
            results = Parallel(n_jobs=8)(delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
            # results = Parallel(n_jobs=8)(
            #     delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(100), desc=desc))
            # results = Parallel(n_jobs=-1)(delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
            feature_length = results[0][1].shape[1]  # 获取特征向量的总长度
            num_features = len(results[0][1])  # 获取特征的数量
            features_matrix = np.zeros((data.shape[0], num_features, feature_length))
            # features_matrix = np.zeros((100, num_features, feature_length))
            for index, features in results:
                features_matrix[index, :, :] = features
            return features_matrix


        # 处理正样本数据
        positive_data_features = parallel_processing(positive_data, 256, "Processing positive data")
        print("Shape:", positive_data_features.shape)

        # 处理负样本数据
        negative_data_features = parallel_processing(negative_data, 256, "Processing negative data")
        print("Shape:", negative_data_features.shape)

        positive_data_features = positive_data_features.reshape(positive_data_features.shape[0], -1)
        negative_data_features = negative_data_features.reshape(negative_data_features.shape[0], -1)
        # positive_data_features = positive_data_features.reshape(100, -1)
        # negative_data_features = negative_data_features.reshape(100, -1)

        # 标记正负样本
        positive_labels = np.ones(len(positive_data_features))
        negative_labels = np.zeros(len(negative_data_features))

        # 合并正负样本的训练集
        X_train_val = np.concatenate((positive_data_features, negative_data_features), axis=0)
        y_train_val = np.concatenate((positive_labels, negative_labels), axis=0)

        # 按 4:1 的比例将剩余数据划分为训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 5)

        # 确保数据形状正确
        print(f'Train set shape: {X_train.shape}')
        print(f'Validation set shape: {X_val.shape}')

        # 数据归一化
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Reshape the data for CNN input (if required)
        X_train = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val_scaled.reshape(X_val.shape[0], X_val.shape[1], 1)


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
            monitor='val_loss',  # 监控验证集上的损失
            factor=0.7,  # 学习率减小的因子
            patience=10,  # 当验证损失在指定轮数内没有改善时，减少学习率
            min_lr=1e-6  # 学习率的下限
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
            filepath=f'{test_path}/1EMD/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        early_stopping_callback = EarlyStoppingWithBestModel(patience=30)

        # 训练模型
        history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_val, y_val),
                  callbacks=[lr_reducer, early_stopping_callback, checkpoint_callback])

        # 定义 CSV 文件路径
        csv_file_path = f'{test_path}/1EMD/training_history.csv'

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
            plt.savefig(f'{test_path}/1EMD/training_curves.png')
            plt.show()


        # 调用可视化函数
        plot_training_history(history)

        # 测试、计时
        model.load_weights(f'{test_path}/1EMD/best_model.h5')
        print('start test:')
        No_test_data = test_data.shape[0]
        print('No_test_data:', No_test_data)
        # test_data_features = np.zeros((test_data.shape[0], test_data.shape[1], 25))
        start_time = time.time()
        # for i in range(test_data.shape[0]):
        #     test_data_features[i, :, :16] = compute_ber_matrix(test_data[i, :, :], 256)
        #     test_data_features[i, :, 16:] = compute_EMD(test_data[i, :, :])

        def process_test_data(data, index, sample_rate):
            test_features_1 = compute_ber_matrix(data, 256)
            test_features_2 = compute_EMD(data)
            combined_features = np.hstack((test_features_1, test_features_2))
            return index, combined_features


        def parallel_processing_test(data, sample_rate, desc):
            results = Parallel(n_jobs=8)(
                delayed(process_test_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
            # results = Parallel(n_jobs=-1)(delayed(process_data)(data[i, :], i, sample_rate) for i in tqdm(range(data.shape[0]), desc=desc))
            feature_length = results[0][1].shape[1]  # 获取特征向量的总长度
            num_features = len(results[0][1])  # 获取特征的数量
            features_matrix = np.zeros((data.shape[0], num_features, feature_length))
            for index, features in results:
                features_matrix[index, :, :] = features
            return features_matrix

        test_data_features = parallel_processing_test(test_data, 256, "Processing test data")

        X_test = test_data_features.reshape(test_data_features.shape[0], -1)

        X_test_scaled = scaler.transform(X_test)
        # X_test_scaled = scaler.transform(test_data_features)
        X_test = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1)

        test_loss, test_acc, test_pre, test_recall = model.evaluate(X_test, y_test)
        # 评估模型
        time_total = time.time() - start_time
        time_average = time_total / No_test_data

        np.save(os.path.join(f'{test_path}/1EMD', 'test_features.npy'), X_test)

        print('Test accuracy:', test_acc)
        print('Average time:', time_average)
        with open(f"{test_path}/1EMD/run_time.txt", 'w') as file:
            # 写入文件
            file.write(f"time_total:\n{time_total}\n\ntime_average:\n{time_average}\n\ntest_loss:\n{test_loss}\n\ntest_acc:\n{test_acc}\n\ntest_pre:\n{test_pre}\n\ntest_recall:\n{test_recall}")

        # 使用模型对测试数据进行预测
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)  # 将概率转换为二进制分类

        # 合并为一个元组
        stacked_arrays = (y_test, y_pred_binary, y_pred)

        # 使用np.column_stack()函数按列合并数组
        test = np.column_stack(stacked_arrays)
        # 保存到CSV文件
        # np.savetxt('test.csv', test, delimiter=',', fmt='%d')
        # 创建一个DataFrame，并添加表头
        # df = pd.DataFrame(test, columns=['label', 'predict', 'pred_actual'])
        df = pd.DataFrame(test, columns=['label', f'predict_ratio',
                                         f'pred_ratio_actual'])
        # 保存到CSV文件
        csv_name = f'test_slice_1EMD.csv'
        csv_path = os.path.join(f'{test_path}/1EMD', csv_name)
        df.to_csv(csv_path, index=False)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred_binary)

        # 绘制热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

except IndentationError as e:
    print(f"错误: {e}")