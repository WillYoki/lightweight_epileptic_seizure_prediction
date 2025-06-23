clear all;
% 指定父文件夹路径
parent_folder = '.'; % 父文件夹路径

% 获取所有以 "seizure" 开头的文件夹
folder_list = dir(fullfile(parent_folder, 'seizure*'));

% 遍历每个文件夹
for k = 1:length(folder_list)
    folder_path = fullfile(parent_folder, folder_list(k).name);
    
    % 提取文件夹名中的数字部分
    folder_name = regexp(folder_path, '\d+', 'match');
    if isempty(folder_name)
        error('文件夹路径中没有找到数字部分');
    end
    folder_number = folder_name{1};
    
    % 生成输出文件路径
    output_folder = ['.\test' folder_number];
    output_mat_file = fullfile(output_folder, 'output_slices.mat');
    excel_file = fullfile(output_folder, 'test_slice_index.xlsx');
    
    % 定义文件顺序
    order = {'_interictal', '_excluded', '_preictal', '_ictal', '_postictal'};
    mat_files = dir(fullfile(folder_path, '*.mat'));
    
    % 初始化变量
    slice_data = {}; % 用于存储切片数据
    slice_index = {}; % 用于存储Excel记录
    total_time = 0; % 用于追踪累计时间
    
    slice_duration = 4; % 切片持续时间（秒）
    sample_rate = 256; % 采样率（Hz）
    
    % 遍历指定的文件顺序
    for j = 1:length(order)
        % 根据顺序筛选文件
        files_in_order = mat_files(arrayfun(@(f) contains(f.name, order{j}), mat_files));
        
        % 对于每个筛选出的文件进行切片处理
        for i = 1:length(files_in_order)
            mat_file = fullfile(folder_path, files_in_order(i).name);
            load(mat_file); % 加载mat文件中的数据
            
            % 确定当前文件的重叠和标签
            if strcmp(order{j}, '_interictal')
                overlap = 0.5;
                label = 0;
            elseif strcmp(order{j}, '_preictal') || strcmp(order{j}, '_ictal') || strcmp(order{j}, '_excluded') || strcmp(order{j}, '_postictal')
                overlap = 0.5; % 50%重叠
                label = 1;
            else
                continue; 
            end
            
            % 假设数据在变量data中，通道 x 采样点
            data = mergedData; % 这里需要替换为实际的变量名
            
            num_channels = size(data, 1);
            num_samples = size(data, 2);
            step_size = slice_duration * sample_rate * (1 - overlap);
            slice_length = slice_duration * sample_rate;
            
            % 生成切片
            for start_idx = 1:step_size:num_samples - slice_length + 1
                end_idx = start_idx + slice_length - 1;
                slice = data(:, start_idx:end_idx);
                
                % 存储切片到cell数组中
                slice_data{end+1, 1} = slice;
                
                % 记录切片的时间信息并取整开始时间
                start_time = floor(total_time); % 取整开始时间
                end_time = start_time + slice_duration;
                total_time = end_time; % 更新累计时间
                
                slice_index(end+1, :) = {files_in_order(i).name, start_time, end_time, label};
            end
        end
    end
    
    % 保存切片数据到MAT文件
    save(output_mat_file, 'slice_data');
    
    % 将切片索引写入Excel文件
    header = {'FileName', 'StartTime', 'EndTime', 'Label'};
    slice_index = [header; slice_index];
    writecell(slice_index, excel_file);
end