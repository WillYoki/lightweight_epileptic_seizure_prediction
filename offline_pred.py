import os
import numpy as np
import pandas as pd

# 基础路径
base_path = 'path'


# 参数设置
segment_length = 300
step = 1

# 遍历 chb01 到 chb12
test_folder_list = [1,3,4,5,8,9,10,13,14,16,18,20,21,23]

# 通道数列表
channels_numbers_list = [3, 6, 9]

# # 遍历 chb01 到 chb12
# for chb in test_folder_list:
#     chb_folder = f'chb{chb:02d}'  # 格式化 chb 文件夹名称，如 chb01, chb02, ..., chb12
try:
    chb_folder = 'chb01'
    chb_path = os.path.join(base_path, chb_folder)
    condition_df = pd.read_csv(r'csv path') #csv

    # 当前病人的阈值
    current_threshold = condition_df[condition_df['患者'] == chb_folder].iloc[0]
    # 条件判断函数
    def is_condition_met(segment):
        return (
                (segment[0] > current_threshold['平均值']) &
                (segment[1] > current_threshold['>平均值']) &
                (segment[2] < current_threshold['方差']) &
                (segment[3] > current_threshold['>0.9']) &
                (segment[4] < current_threshold['<0.5'])
        )


    # 获取 test 文件夹列表
    test_folders = [folder for folder in os.listdir(chb_path) if folder.startswith('test')]

    # 遍历每个 test 文件夹
    for test_folder in test_folders:
        folder = "5Channel+SEMD+FusionCF"
        test_path = os.path.join(chb_path, f'{test_folder}/{folder}')
        for channels_numbers in channels_numbers_list:
            x_file_name = f'test_slice_{channels_numbers}_{folder}.csv'
            x_array_path = os.path.join(test_path, x_file_name)

            # 读取 CSV 文件
            df = pd.read_csv(x_array_path)
            x_array = df[f'pred_ratio_actual'].values
            y_array = df[f'label'].values

            # 计算每段的统计特征
            segments = []
            for start in range(0, len(x_array) - segment_length + 1, step):
                segment = x_array[start:start + segment_length]

                mean_val = np.mean(segment)
                greater_than_mean = np.sum(segment > mean_val)
                variance = np.var(segment)
                n1 = np.sum(segment > 0.9)
                n2 = np.sum(segment < 0.5)
                # 添加统计特征
                segments.append([mean_val, greater_than_mean, variance, n1, n2])

                # 转换为numpy数组
                segments = np.array(segments)
                z_array = np.zeros_like(x_array)

            for i in range(segment_length, x_array.shape[0]):
                if is_condition_met(segments[i - segment_length]):
                    z_array[i] = 1

                # 保存到CSV文件
                df = pd.DataFrame(z_array, columns=[f'pred_{channels_numbers}'])
                csv_name = f'pred_{channels_numbers}.csv'
                csv_path = os.path.join(test_path, csv_name)
                df.to_csv(csv_path, index=False)

                print(f'Processed {chb_folder}/{test_folder} with channels_numbers={channels_numbers}')

            # 保存到CSV文件
            df = pd.DataFrame(z_array, columns=[f'pred'])
            csv_name = f'pred.csv'
            csv_path = os.path.join(test_path, csv_name)
            df.to_csv(csv_path, index=False)

except IndentationError as e:
    print(f"缩进错误: {e}")