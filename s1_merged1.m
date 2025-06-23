clear all;
% 定义父文件夹路径
parentFolderPath = '.'; % 替换为实际的父文件夹路径

% 定义不同后缀的文件类型
suffixes = {'_interictal', '_excluded', '_preictal', '_ictal', '_postictal'};

% 获取所有以 'seizure' 前缀命名的文件夹
seizureFolders = dir(fullfile(parentFolderPath, 'seizure*'));

% 遍历每个以 'seizure' 前缀命名的文件夹
for f = 1:length(seizureFolders)
    folderPath = fullfile(parentFolderPath, seizureFolders(f).name);
    
    % 遍历每个后缀类型
    for s = 1:length(suffixes)
        currentSuffix = suffixes{s};
        
        % 获取匹配该后缀的所有 .mat 文件
        matFiles = dir(fullfile(folderPath, ['*' currentSuffix '.mat']));
        
        % 初始化一个空数组，用于存储合并后的 data
        mergedData = [];
        
        % 遍历当前后缀的所有 .mat 文件
        for i = 1:length(matFiles)
            % 构建完整文件路径
            filePath = fullfile(folderPath, matFiles(i).name);
            
            % 加载 .mat 文件中的 data 变量
            loadedData = load(filePath, 'data');
            
            % 如果 data 存在，合并到 mergedData
            if isfield(loadedData, 'data')
                currentData = loadedData.data;
                
                % 检查 mergedData 是否为空，若为空则直接赋值
                if isempty(mergedData)
                    mergedData = currentData;
                else
                    % 否则在时间维度上（第2维）进行合并
                    mergedData = cat(2, mergedData, currentData);
                end
            else
                warning('File %s does not contain a "data" variable.', matFiles(i).name);
            end
        end
        
        % 如果合并后的数据不为空，保存为新的 .mat 文件
        if ~isempty(mergedData)
            saveFileName = ['merged' currentSuffix '.mat'];
            save(fullfile(folderPath, saveFileName), 'mergedData');
            fprintf('Merged data saved as: %s in folder: %s\n', saveFileName, seizureFolders(f).name);
        else
            fprintf('No data to merge for suffix: %s in folder: %s\n', currentSuffix, seizureFolders(f).name);
        end
    end
end