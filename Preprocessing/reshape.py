import numpy as np
import h5py
import os

# 加载数据
# with h5py.File('G:/cnntest/pythonProject1/data/positive_3d_30_2_huashan_m3.mat', 'r') as f:
#     positive_data = f['all_features_3d'][:]
#     positive_data = np.transpose(positive_data, (2, 1, 0))
#
# with h5py.File('G:/cnntest/pythonProject1/data/negative_3d_30_2_huashan_m3.mat', 'r') as f:
#     negative_data = f['all_features_3d'][:]
#     negative_data = np.transpose(negative_data, (2, 1, 0))
#
# # 保存数组为.npy文件
# np.save('G:/cnntest/pythonProject1/data/positive_data_30_2_huashan_m3.npy', positive_data)
# np.save('G:/cnntest/pythonProject1/data/negative_data_30_2_huashan_m3.npy', negative_data)

path = 'G:/cnntest/pythonProject/event/chb09/test4'
print(path)
positive_data_path = os.path.join(path, 'preictal_3d_sliced.mat')
negative_data_path = os.path.join(path, 'interictal_3d_sliced.mat')
test_data_path = os.path.join(path, 'test_3d.mat')

with h5py.File(positive_data_path, 'r') as f:
    positive_data = f['all_features_3d'][:]
    positive_data = np.transpose(positive_data, (2, 1, 0))

with h5py.File(negative_data_path, 'r') as f:
    negative_data = f['all_features_3d'][:]
    negative_data = np.transpose(negative_data, (2, 1, 0))

# 保存数组为.npy文件
np.save(os.path.join(path, 'positive_data.npy'), positive_data)
np.save(os.path.join(path, 'negative_data.npy'), negative_data)

with h5py.File(test_data_path, 'r') as f:
    test_data = f['all_features_3d'][:]
    test_data = np.transpose(test_data, (2, 1, 0))

# 保存数组为.npy文件
np.save(os.path.join(path, 'test_data.npy'), test_data)