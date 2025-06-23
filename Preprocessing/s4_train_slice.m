clear all;
base_directory = '.'; % 指定基础文件夹路径
test_folders = dir(fullfile(base_directory, 'test*')); % 获取以 "test" 开头的文件夹列表

window_size = 4 * 256; % 4秒，采样率为256Hz

for folder = test_folders'
    if folder.isdir
        directory = fullfile(base_directory, folder.name);
        files = dir(fullfile(directory, '*.mat')); % 获取当前文件夹中的 .mat 文件列表

        for file = files'
            mat_data = load(fullfile(directory, file.name));
            data = mat_data.mergedDataAll;
% 
%             % 检查行数，并只保留前18行
%             if size(data, 1) > 18
%                 data = data(1:18, :);
%             end

            % 根据文件名后缀确定切片方式
            if contains(file.name, '_interictal_data')
                overlap = 0; % 不重叠
            elseif contains(file.name, '_preictal_data')
                overlap = window_size / 2; % 50% 重叠
            else
                continue; % 如果文件名不符合要求，跳过
            end

            % 切片参数
            step = window_size - overlap;
            num_slices = floor((size(data, 2) - window_size) / step) + 1;

            % 创建一个 cell 数组来存储所有切片
            slices = cell(num_slices, 1);

            % 对数据进行切片并存储在 cell 数组中
            for i = 0:num_slices-1
                start_idx = i * step + 1;
                end_idx = start_idx + window_size - 1;
                slices{i + 1, 1} = data(:, start_idx:end_idx); % 存储数据切片
            end

            % 将切片的数据保存为你需要的形式（如新的 mat 文件）
            save(fullfile(directory, [file.name(1:end-4) '_sliced.mat']), 'slices');
        end
    end
end
