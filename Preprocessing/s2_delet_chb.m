clear all;
% 指定要处理的文件夹
folderPath = '.';

% 获取文件夹中的所有子文件夹
subFolders = dir(folderPath);
subFolders = subFolders([subFolders.isdir] & ~strcmp({subFolders.name}, '.') & ~strcmp({subFolders.name}, '..'));

% 遍历每个子文件夹
for i = 1:length(subFolders)
    % 获取当前子文件夹的路径
    subFolderPath = fullfile(folderPath, subFolders(i).name);
    
    % 获取当前子文件夹中的所有文件
    files = dir(fullfile(subFolderPath, '*.mat'));
    
    % 遍历每个文件
    for j = 1:length(files)
        % 获取文件名
        fileName = files(j).name;
        
        % 检查文件名是否以 'chb' 开头
        if startsWith(fileName, 'chb')
            % 构建完整的文件路径
            filePath = fullfile(subFolderPath, fileName);
            
            % 删除文件
            delete(filePath);
            
            % 输出删除信息
            fprintf('Deleted file: %s\n', filePath);
        end
    end
end
