clc;
clear;

% 从 Excel 文件读取数据
data = readtable('CHBMIT_seizures_new_30_4.xlsx');
patientName = 'chb01';

% 筛选指定病人的数据
patientData = data(strcmp(data.Subject, patientName), :);
%处理重复文件名与下划线转义字符
uniqueFileNames = unique(patientData.FileName);
uniqueFileNames = strrep(uniqueFileNames, '_', '\_');

% 初始化存储不同阶段时间长度的数组
stages = {'ictal', 'preictal', 'excluded', 'interictal', 'postictal'};
stageTimes = zeros(height(uniqueFileNames), length(stages));

row_count = size(patientData, 1);
temp_file_name =patientData(1, :).FileName{1};
temp_index = 1;
% 计算每个文件的各阶段时间
for i = 1:row_count
    startTime = datetime(patientData(i,:).StartTime, 'Format', 'HH:mm:ss');
    endTime = datetime(patientData(i,:).EndTime, 'Format', 'HH:mm:ss');
    durationInMinutes = minutes(endTime - startTime);
    %凌晨时间处理
    if(durationInMinutes<0)
        durationInMinutes = durationInMinutes +24*60;
    end
    if(~strcmp(patientData(i,:).FileName, temp_file_name))
    temp_index =temp_index+1;
    temp_file_name=patientData(i,:).FileName;
    end
    if(strcmp(patientData(i,:).type, 'ictal'))
    stageTimes(temp_index,1) = durationInMinutes;
    elseif(strcmp(patientData(i,:).type, 'preictal'))
    stageTimes(temp_index,2) = -durationInMinutes;
    elseif(strcmp(patientData(i,:).type, 'excluded'))
    stageTimes(temp_index,3) = -durationInMinutes;
    elseif(strcmp(patientData(i,:).type, 'interictal'))
    stageTimes(temp_index,4) = -durationInMinutes;
    elseif(strcmp(patientData(i,:).type, 'postictal'))
    stageTimes(temp_index,5) = durationInMinutes;
    end
end


% 创建堆叠的水平条形图
h=barh(stageTimes,'stacked');
barWidth = 0.1; % 调整宽度，可以根据需要调整
for i = 1:length(h)
    h(i).BarWidth = barWidth;
end

set(h(1), 'FaceColor', [1, 0, 0]);  % 设置ictal的颜色
set(h(2), 'FaceColor', [1, 1, 0]);  % 设置preictal的颜色
set(h(3), 'FaceColor', [0.5, 0.5, 0.5]);  % 设置excluded的颜色
set(h(4), 'FaceColor', [0, 1, 0]);  % 设置interictal的颜色
set(h(5), 'FaceColor', [1 0.647 0]);  % 设置postictal的颜色

% 设置轴标签
set(gca, 'YTick', 1:height(uniqueFileNames), 'YTickLabel', uniqueFileNames);
ylabel('File Name');
xlabel('Time (min)');

% 添加图例
legend(stages, 'Location', 'bestoutside');

% 添加标题
title(['Epileptic Stage Durations for Patient ', patientName]);
