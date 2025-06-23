% 1. 读取Excel文件
filePath_o = 'G:\cnntest\pythonProject\event\chb18\test5';
filePath_1 = fullfile(filePath_o, '5Channel+SEMD+FusionCF');
fileName = 'test_slice_index.xlsx';
filePath = fullfile(filePath_o, fileName);
data = readtable(filePath);
numChannel = 3;
fileName_1 = ['test_slice_', num2str(numChannel), '_5Channel+SEMD+FusionCF.csv'];
filePath_1 = fullfile(filePath_1, fileName_1);
data_1 = readtable(filePath_1);
% 获取 FileName 和 Predict 列吧 
fileNames = data.FileName;
fieldName = 'pred_ratio_actual';
predictValues = data_1.(fieldName);

% 2. 提取时期信息
numSamples = height(data); 
timePerSample = 4;       % 每条数据代表4秒
overlap = 0.5;           % 重叠比例
sampleTime = timePerSample * (1 - overlap); 

% 生成时间轴
timeAxis = (0:numSamples-1) * sampleTime;

% 根据 FileName 提取时期，并标记分段
periods = strings(numSamples, 1);
for i = 1:numSamples
    % 提取下划线后的时期，并去除多余空格
    file = char(fileNames{i});
    [~, remain] = strtok(file, '_');
    [period, ~] = strtok(remain, '.');
    period = strtrim(period);  % 去除前后空格
    period = regexprep(period, '^_', ''); % 去除前导下划线
    periods(i) = period;
end

% 3. 定义颜色映射
colorMap = containers.Map({'interictal', 'excluded', 'preictal', 'ictal', 'postictal'}, ...
                          {[0, 1, 0], [0.5, 0.5, 0.5], [1, 1, 0], [1, 0, 0], [1, 0.5, 0]});

% 4. 绘制图形
figure;
hold on;

% 绘制 Predict 的折线图
plot(timeAxis, predictValues, 'k', 'LineWidth', 1.5); % 折线图用黑色

% 5. 为不同时期上色
uniquePeriods = unique(periods);
for i = 1:length(uniquePeriods)
    period = uniquePeriods(i);
    idx = strcmp(periods, period);
    
    % 检查 period 是否存在于 colorMap 中
    if ~isKey(colorMap, period)
        warning('未找到 %s 的颜色映射。', period);
        continue; % 跳过无效的时期
    end
    
    % 获取当前时期的时间范围
    x_start = timeAxis(find(idx, 1, 'first'));
    x_end = timeAxis(find(idx, 1, 'last')) + sampleTime;
    y_min = min(0);
    y_max = max(1.2);
    
    % 获取当前时期的颜色
    color = colorMap(period);
    
    % 使用 patch 标记不同时期的背景
    patch([x_start, x_end, x_end, x_start], [y_min, y_min, y_max, y_max], color, ...
          'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% 添加图例和标签
ylim([0, 1.2]);
legend(['Predict', uniquePeriods'], 'Location', 'best');
xlabel('time (s)');
ylabel('Predict');
title('Predict');
hold off;
