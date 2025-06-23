clear all;
% 自动生成文件夹路径列表
folderPrefix = 'seizure';
folderList = {};
n = 1; % 假设从seizure1开始
while true
    folderName = [folderPrefix num2str(n)];
    if exist(folderName, 'dir')
        folderList{end+1} = folderName;
        n = n + 1;
    else
        break;
    end
end

% 留一法交叉验证
for leaveOutIndex = 1:length(folderList)
    % 排除当前文件夹
    trainFolders = folderList;
    trainFolders(leaveOutIndex) = [];
    
    % 输出文件夹名相应改变
    outputFolder = fullfile(['test' num2str(leaveOutIndex)]);
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    % 处理 *_preictal.mat 文件
    fileSuffix = '*_preictal.mat'; % 指定的文件后缀
    outputFileName = 'merged_preictal_data.mat'; % 保存的文件名
    
    % 初始化空数组来存储所有合并的mergedData
    mergedDataAll = [];
    
    % 遍历每个训练文件夹
    for i = 1:length(trainFolders)
        currentFolder = trainFolders{i};
        % 获取当前文件夹中指定后缀的.mat文件
        matFiles = dir(fullfile(currentFolder, fileSuffix));
        
        % 遍历每个.mat文件
        for j = 1:length(matFiles)
            matFilePath = fullfile(currentFolder, matFiles(j).name);
            % 加载.mat文件
            matData = load(matFilePath);
            
            % 检查是否存在mergedData字段，并确保数组符合预期
            if isfield(matData, 'mergedData')
                % 连接mergedData数组到总数组
                mergedDataAll = cat(2, mergedDataAll, matData.mergedData);
            else
                warning(['File ' matFiles(j).name ' does not contain mergedData.']);
            end
        end
    end
    
    % mergedDataAll 现在包含了所有训练文件中连接后的数据
    % 保存合并后的mergedDataAll到指定文件夹
    outputFilePath = fullfile(outputFolder, outputFileName);
    save(outputFilePath, 'mergedDataAll');
    disp(['Merged data saved to ' outputFilePath]);
    
    % 处理 *_interictal.mat 文件
    fileSuffix = '*_interictal.mat'; % 指定的文件后缀
    outputFileName = 'merged_interictal_data.mat'; % 保存的文件名
    
    % 初始化空数组来存储所有合并的mergedData
    mergedDataAll = [];
    
    % 遍历每个训练文件夹
    for i = 1:length(trainFolders)
        currentFolder = trainFolders{i};
        % 获取当前文件夹中指定后缀的.mat文件
        matFiles = dir(fullfile(currentFolder, fileSuffix));
        
        % 遍历每个.mat文件
        for j = 1:length(matFiles)
            matFilePath = fullfile(currentFolder, matFiles(j).name);
            % 加载.mat文件
            matData = load(matFilePath);
            
            % 检查是否存在mergedData字段，并确保数组符合预期
            if isfield(matData, 'mergedData')
                % 连接mergedData数组到总数组
                mergedDataAll = cat(2, mergedDataAll, matData.mergedData);
            else
                warning(['File ' matFiles(j).name ' does not contain mergedData.']);
            end
        end
    end
    
    % mergedDataAll 现在包含了所有训练文件中连接后的数据
    % 保存合并后的mergedDataAll到指定文件夹
    outputFilePath = fullfile(outputFolder, outputFileName);
    save(outputFilePath, 'mergedDataAll');
    disp(['Merged data saved to ' outputFilePath]);
end