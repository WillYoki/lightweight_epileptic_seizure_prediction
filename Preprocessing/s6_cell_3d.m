clear all;
% 获取当前目录下所有以 'test' 开头的文件夹
test_folders = dir('test*');

% 遍历每个文件夹
for k = 1:length(test_folders)
    folder_name = test_folders(k).name;
    folder_path = fullfile(test_folders(k).folder, folder_name);
    
    % 处理 interictal 数据
    interictal_file = fullfile(folder_path, 'merged_interictal_data_sliced.mat');
    if exist(interictal_file, 'file')
        load(interictal_file);
        num_samples = length(slices);
        [rows, cols] = size(slices{1});
        all_features_3d = zeros(num_samples, rows, cols);
        for i = 1:num_samples
            current_data = slices{i};
            all_features_3d(i, :, :) = current_data(1:rows, :);
        end
        save(fullfile(folder_path, 'interictal_3d_sliced.mat'), 'all_features_3d');
    end
    
    % 处理 preictal 数据
    preictal_file = fullfile(folder_path, 'merged_preictal_data_sliced.mat');
    if exist(preictal_file, 'file')
        load(preictal_file);
        num_samples = length(slices);
        [rows, cols] = size(slices{1});
        all_features_3d = zeros(num_samples, rows, cols);
        for i = 1:num_samples
            current_data = slices{i};
            all_features_3d(i, :, :) = current_data(1:rows, :);
        end
        save(fullfile(folder_path, 'preictal_3d_sliced.mat'), 'all_features_3d');
    end
    
    % 处理 test 数据
    test_file = fullfile(folder_path, 'output_slices.mat');
    if exist(test_file, 'file')
        load(test_file);
        num_samples = length(slice_data);
        [rows, cols] = size(slice_data{1});
        all_features_3d = zeros(num_samples, rows, cols);
        for i = 1:num_samples
            current_data = slice_data{i};
            all_features_3d(i, :, :) = current_data(1:rows, :);
        end
        save(fullfile(folder_path, 'test_3d.mat'), 'all_features_3d');
    end
end
