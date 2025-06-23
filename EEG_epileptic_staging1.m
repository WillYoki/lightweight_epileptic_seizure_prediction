function [out_schedule_path]=EEG_epileptic_staging(in_schedule_path, in_seizure_path, preictal_duration, interictal_ex_duration)
    % 如果 preictal_duration 没有提供，默认设置为 30
    if nargin < 3 || isempty(preictal_duration)
        preictal_duration = 30;
    end

    % 如果 interictal_ex_duration 没有提供，默认设置为 2
    if nargin < 4 || isempty(interictal_ex_duration)
        interictal_ex_duration = 2;
    end

    preictal_duration_second = preictal_duration*60;
    interictal_ex_duration_second = interictal_ex_duration*60*60;
        
    schedule_data = readtable(in_schedule_path);
    seizure_data = readtable(in_seizure_path);
    
    ictal_data = table();
    new_table = table();
    row_count = size(schedule_data, 1);
    
    temp_end =0;
    temp_subject = "";
    
    % 创建进度条
    h = waitbar(0, 'Processing...'); 
    
    %循环数据集基础信息表 将发作信息存储到ictal_table中
    %进行预处理 判断两次发作是否间隔30分钟 不符合的保留前一项
    %Start
    for i=1:row_count
       current_row =schedule_data(i,:);
       %判断是否有发作
        if(current_row.SeizureNumbers)
           %根据发作次数进行循环生成数据
    
            for j=1:current_row.SeizureNumbers
                %基础信息表数据获取
                fileName = current_row.FileName;
                schedule_start_time =current_row.StartTime;
                schedule_end_time =current_row.EndTime;
                %通过FileName匹配获取发作信息
                matching_index= find(strcmp(seizure_data.FileName, fileName));
                matching_seizure = seizure_data(matching_index(j),:);
                %发作起始秒数
                start_second = matching_seizure.StartSecond;
                %发作结束秒数
                end_second = matching_seizure.EndSecond;
                %发作持续秒数
                duration_second = matching_seizure.DurationSecond;
                % 计算发作结束时间
                seizure_end_time_number = schedule_start_time + (end_second / (24 * 60 * 60));
                % 计算发作开始时间
                seizure_start_time_number = seizure_end_time_number - (duration_second / (24 * 60 * 60));
                
                %情况一：每次去判断和上一次是否相差30分钟（只保留第一次符合的）
                if(strcmp(current_row.Subject, temp_subject) && seizure_start_time_number-temp_end>=0 && seizure_start_time_number-temp_end<=minTo24Hour(30))
                      continue;
                end
    
                % 转化为字符串时分秒
                %seizure_start_time =formatToHHMMSS(seizure_start_time_number);
                %seizure_end_time=formatToHHMMSS(seizure_end_time_number);
                %更新临时结束时间
                temp_end=seizure_end_time_number;
                temp_subject=current_row.Subject;
                %发作期记录存储 不可用createTableRow替换 SamplingRate_Hz_和FrameNumbers属性名不同！！
                ictal_row = table(current_row.Subject,current_row.FileName,start_second,end_second, ...
                    current_row.SamplingRate_Hz_,(current_row.SamplingRate_Hz_*duration_second),{'ictal'}, ...
                    'VariableNames',{'Subject','FileName','StartTime','EndTime','SamplingRate_Hz_','FrameNumbers','type'});
    
                ictal_data=[ictal_data; ictal_row];
                
            end
        end
       waitbar(i / row_count, h, sprintf('ictal Data Processing... %d/%d', i, row_count));
    end
    %End
    
    %数据预分组 将Subject相同的分组一起处理
    groupedData = findgroups(schedule_data.Subject);
    groupedSeizuresData = findgroups(ictal_data.Subject);
    row_count = max(groupedData);
    timing_start_index = 2;
    %主要处理分期部分
    %Start
    for v=1:row_count
        start_index =1;
        currentGroup = schedule_data(groupedData == v, :);
        seizureGroup = ictal_data(groupedSeizuresData == v,:);
        ictal_data_index = 1; 
        initial_array=[];
        %0 不存在 1存在 
        %-1发作ictal 2发作前期preictal 3发作后期postictal 4发作间期interictal 5排除期excluded
        %Start
        for j = 1:height(currentGroup)
            current_file = currentGroup.FileName{j};
            current_start_time = currentGroup.StartTime(j);
            current_end_time = currentGroup.EndTime(j);
   

             % 针对每一行数据进行处理
             file_duration = hourNumberToSecond(current_end_time - current_start_time);
             o_file_start_index = start_index;
             file_end_index = o_file_start_index + file_duration;
           while (ictal_data_index <= height(seizureGroup) && strcmp(current_file, seizureGroup.FileName{ictal_data_index}))
                ictal_row = seizureGroup(ictal_data_index, :); 
                % 处理发作期间
                seizure_start_second = ictal_row.StartTime;
                seizure_end_second = ictal_row.EndTime;
                
                %% 第二次循环file_start_index不应该改变
                file_start_index = o_file_start_index;
                
                
                seizure_start_index = file_start_index -1 +seizure_start_second;
                seizure_end_index = file_start_index -1 +seizure_end_second;
                end_index = seizure_start_index;
                % 标记发作前、发作期间和发作后的时间段
                initial_array(start_index:end_index) = 1;
                start_index = end_index+1;
                
                end_index = seizure_end_index+1;
                initial_array(start_index:end_index) = -1;
                start_index = end_index+1;
               
                ictal_data_index = ictal_data_index + 1;

           end
           % 处理非发作期间
           end_index = file_end_index;
           % 标记非发作期间的时间段
           initial_array(start_index:end_index) = 1;
           start_index = end_index +1;
            % 检查是否是最后一次迭代
            if j < height(currentGroup)
                durationZeroSecond = hourNumberToSecond(currentGroup.StartTime(j+1) - current_end_time) -1;
                end_index = start_index + durationZeroSecond-1;
                if(end_index>start_index)
                    initial_array(start_index:end_index) = 0;
                end
                start_index = end_index +1;
            end
        end
        %End
        %待处理数组 时间线数组 仅保留0和1
        temp_array=initial_array;
        temp_array(temp_array == -1) = 1;
        preictal_array=temp_array;
        postictal_array=temp_array;
        interictal_array=temp_array;
        excluded_array=temp_array;
    
        % 找到值为 -1 的元素的索引
        indices = find(initial_array == -1);
    
        % 根据索引找到 -1 的段的起始和结束索引
        ictal_start_indices = indices([true, diff(indices) > 1]);
        ictal_end_indices = indices([diff(indices) > 1, true]);
        %边界值首项处理
        if(ictal_start_indices(:,1)-preictal_duration_second<1)
             preictal_array(1:ictal_start_indices(:,1))=preictal_array(1:ictal_start_indices(:,1))*2;
        else
            preictal_array(ictal_start_indices(:,1)-preictal_duration_second:ictal_start_indices(:,1))=preictal_array(ictal_start_indices(:,1)-preictal_duration_second:ictal_start_indices(:,1))*2;
        end
        if(ictal_start_indices(:,1)-interictal_ex_duration_second<1)
            excluded_array(1:ictal_start_indices(:,1)-preictal_duration_second)=excluded_array(1:ictal_start_indices(:,1)-preictal_duration_second)*5;
        else
            excluded_array(ictal_start_indices(:,1)-interictal_ex_duration_second:ictal_start_indices(:,1)-preictal_duration_second)=excluded_array(ictal_start_indices(:,1)-interictal_ex_duration_second:ictal_start_indices(:,1)-preictal_duration_second)*5;
        end
        %核心情况分期
        %Start
        for k = 1:length(ictal_end_indices)-1
        %临时本次的发作后期结束时间
        temp_current_postictal_end_number = ictal_end_indices(:,k) + interictal_ex_duration_second;
        %临时下一次的发作前期+排除期开始时间
        temp_next_interictal_start_number = ictal_start_indices(:,k+1) - interictal_ex_duration_second;
            %情况二：间隔大于前期定义值，小于前期定义值+后期定义值 压缩本次前期范围做为本次排除期
            if(ictal_start_indices(:,k+1)-ictal_start_indices(:,k)>30*60&&...
                ictal_start_indices(:,k+1)-ictal_start_indices(:,k)<=interictal_ex_duration_second +preictal_duration_second)
                 %判断后期结束时间是否大于下一次发作开始
%                 if(temp_current_postictal_end_number>=ictal_start_indices(:,k+1))
%                     %无发作前期
%                     excluded_array(ictal_end_indices(:,k):ictal_start_indices(:,k+1))=excluded_array(ictal_end_indices(:,k):ictal_start_indices(:,k+1))*5;
%                 else
                    %有发作前期
                    excluded_array(ictal_end_indices(:,k):(ictal_start_indices(:,k+1)-1800))=excluded_array(ictal_end_indices(:,k):(ictal_start_indices(:,k+1)-1800))*5;
                    preictal_array((ictal_start_indices(:,k+1)-1800):ictal_start_indices(:,k+1))=preictal_array((ictal_start_indices(:,k+1)-1800):ictal_start_indices(:,k+1))*2;
%                 end
            %情况三：间隔大于前期定义值+后期定义值，小于2倍的后期定义值 压缩上一次的后期范围作为本次的排除期
            elseif(ictal_start_indices(:,k+1)-ictal_start_indices(:,k)>interictal_ex_duration_second +preictal_duration_second&&...
                ictal_start_indices(:,k+1)-ictal_start_indices(:,k)<=2*interictal_ex_duration_second)
                    %本次后期
                    postictal_array(ictal_end_indices(:,k):temp_next_interictal_start_number)=postictal_array(ictal_end_indices(:,k):temp_next_interictal_start_number)*3;
                    %下一次发作间期（无）
                    %下一次排除期
                    excluded_array(temp_next_interictal_start_number:ictal_start_indices(:,k+1)-preictal_duration_second)=excluded_array(temp_next_interictal_start_number:ictal_start_indices(:,k+1)-preictal_duration_second)*5;
                    %下一次前期
                    preictal_array(ictal_start_indices(:,k+1)-preictal_duration_second:ictal_start_indices(:,k+1))=preictal_array(ictal_start_indices(:,k+1)-preictal_duration_second:ictal_start_indices(:,k+1))*2;
            %情况四：标准情况
            elseif(ictal_start_indices(:,k+1)-ictal_start_indices(:,k)>2*interictal_ex_duration_second)
                    %本次后期
                    postictal_array(ictal_end_indices(:,k):temp_current_postictal_end_number)=postictal_array(ictal_end_indices(:,k):temp_current_postictal_end_number)*3;
                    %下一次发作间期
                    interictal_array(temp_current_postictal_end_number:temp_next_interictal_start_number)=interictal_array(temp_current_postictal_end_number:temp_next_interictal_start_number)*4;
                    %下一次排除期
                    excluded_array(temp_next_interictal_start_number:ictal_start_indices(:,k+1)-preictal_duration_second)=excluded_array(temp_next_interictal_start_number:ictal_start_indices(:,k+1)-preictal_duration_second)*5;
                    %下一次前期
                    preictal_array(ictal_start_indices(:,k+1)-preictal_duration_second:ictal_start_indices(:,k+1))=preictal_array(ictal_start_indices(:,k+1)-preictal_duration_second:ictal_start_indices(:,k+1))*2;
            end
        end
        %End
        %边界值尾项处理
        if(ictal_end_indices(:,end)+interictal_ex_duration_second>=length(postictal_array))
            postictal_array(ictal_end_indices(:,end):length(postictal_array))=postictal_array(ictal_end_indices(:,end):length(postictal_array))*3;
        else
            postictal_array(ictal_end_indices(:,end):ictal_end_indices(:,end)+interictal_ex_duration_second)=postictal_array(ictal_end_indices(:,end):ictal_end_indices(:,end)+interictal_ex_duration_second)*3;
        end
        %将所有数组重叠 未覆盖到的区域标记为发作间期
        final_array = initial_array.*preictal_array.*postictal_array.*interictal_array.*excluded_array;
    
        start_index = 1;
            
    
        %最终处理输出excel
        %Start
        for z = 1:height(currentGroup)
            currentRow = currentGroup(z,:);
            durationNonSeizure = hourNumberToSecond(currentRow.EndTime - currentRow.StartTime);
            end_index = start_index + durationNonSeizure;
            start_time = formatToHHMMSS(currentRow.StartTime);
            % Check for each state and store results in different variables
            
            % 生成interictal的清单
            if any(final_array(start_index:end_index) == 1)
                indices = find(final_array(start_index:end_index) == 1);
                ictal_start_indices = indices([true, diff(indices) > 1]);
                ictal_end_indices = indices([diff(indices) > 1, true]);
                for i=1:length(ictal_start_indices)
                    if(ictal_start_indices(i)~=1)
                        ictal_start_indices(i)=ictal_start_indices(i)-1;
                    end
                    if(ictal_end_indices(i)~=end_index-start_index+1)
                        ictal_end_indices(i)=ictal_end_indices(i)+1;
                    end
                    final_array_start_time = addSeconds(start_time,ictal_start_indices(i));
                    final_array_end_time = addSeconds(start_time,ictal_end_indices(i));
                    duration=ictal_end_indices(i)-ictal_start_indices(i);
                    if(duration ~= 0)
                        final_array_row = createTableRow(currentRow.Subject,currentRow.FileName, ...
                            final_array_start_time,final_array_end_time, ...
                            currentRow.SamplingRate_Hz_,duration*currentRow.SamplingRate_Hz_, ...
                            'interictal');
                        new_table = [new_table;final_array_row];
                    end
                end
            end
            
            % 生成interictal的清单
            if any(interictal_array(start_index:end_index) == 4)
                indices = find(interictal_array(start_index:end_index) == 4);
                ictal_start_indices = indices([true, diff(indices) > 1]);
                ictal_end_indices = indices([diff(indices) > 1, true]);
                for i=1:length(ictal_start_indices)
                    interictal_start_time = addSeconds(start_time,ictal_start_indices(i));
                    interictal_end_time = addSeconds(start_time,ictal_end_indices(i));
                    duration=ictal_end_indices(i)-ictal_start_indices(i);
                    if(duration ~= 0)
                        interictal_row = createTableRow(currentRow.Subject,currentRow.FileName, ...
                            interictal_start_time,interictal_end_time, ...
                            currentRow.SamplingRate_Hz_,duration*currentRow.SamplingRate_Hz_, ...
                            'interictal');
                        new_table = [new_table;interictal_row];
                    end
                end
            end
           
           
            % 生成excluded的清单
            if any(excluded_array(start_index:end_index) == 5)
                indices = find(excluded_array(start_index:end_index) == 5);
                ictal_start_indices = indices([true, diff(indices) > 1]);
                ictal_end_indices = indices([diff(indices) > 1, true]);
                for i=1:length(ictal_start_indices)
                    excluded_start_time =addSeconds(start_time,ictal_start_indices(i));
                    excluded_end_time = addSeconds(start_time,ictal_end_indices(i));
                    duration = ictal_end_indices(i)-ictal_start_indices(i);
                    if(duration ~= 0)
                        excluded_row = createTableRow(currentRow.Subject,currentRow.FileName, ...
                            excluded_start_time,excluded_end_time, ...
                            currentRow.SamplingRate_Hz_,duration*currentRow.SamplingRate_Hz_, ...
                            'excluded');
                        new_table = [new_table;excluded_row];
                    end
                end
    
            end
            
            % 生成preictal的清单
            if any(preictal_array(start_index:end_index) == 2)
                indices = find(preictal_array(start_index:end_index) == 2);
                ictal_start_indices = indices([true, diff(indices) > 1]);
                ictal_end_indices = indices([diff(indices) > 1, true]);
                for i=1:length(ictal_start_indices)
                    preictal_start_time = addSeconds(start_time,ictal_start_indices(i));
                    preictal_end_time = addSeconds(start_time,ictal_end_indices(i));
                    duration=ictal_end_indices(i)-ictal_start_indices(i);
                    if(duration ~= 0)
                        preictal_row = createTableRow(currentRow.Subject,currentRow.FileName, ...
                            preictal_start_time,preictal_end_time, ...
                            currentRow.SamplingRate_Hz_,duration*currentRow.SamplingRate_Hz_, ...
                            'preictal');
                    new_table = [new_table;preictal_row];
                    end
                end
            end
           
         
            % 生成ictal的清单
            if any(initial_array(start_index:end_index) == -1)
                indices = find(initial_array(start_index:end_index) == -1);
                ictal_start_indices = indices([true, diff(indices) > 1]);
                ictal_end_indices = indices([diff(indices) > 1, true]);
                for i=1:length(ictal_start_indices)
                initial_start_time = addSeconds(start_time,ictal_start_indices(i));
                initial_end_time = addSeconds(start_time,ictal_end_indices(i));
                duration = ictal_end_indices(i)-ictal_start_indices(i);
                if(duration ~= 0)
                    initial_row = createTableRow(currentRow.Subject,currentRow.FileName, ...
                                              initial_start_time,initial_end_time, ...
                                              currentRow.SamplingRate_Hz_,duration*currentRow.SamplingRate_Hz_, ...
                                              'ictal');
                    new_table = [new_table;initial_row];
                end
                end
            end
          
            % 生成postictal的清单
            if any(postictal_array(start_index:end_index) == 3)
                indices = find(postictal_array(start_index:end_index) == 3);
                ictal_start_indices = indices([true, diff(indices) > 1]);
                ictal_end_indices = indices([diff(indices) > 1, true]);
                for i=1:length(ictal_start_indices)
                    postictal_start_time = addSeconds(start_time,ictal_start_indices(i));
                    postictal_end_time = addSeconds(start_time,ictal_end_indices(i));
                    duration = ictal_end_indices(i)-ictal_start_indices(i);
                    if(duration ~= 0)
                        postictal_row = createTableRow(currentRow.Subject,currentRow.FileName, ...
                            postictal_start_time,postictal_end_time, ...
                            currentRow.SamplingRate_Hz_,duration*currentRow.SamplingRate_Hz_, ...
                            'postictal');
                        new_table = [new_table;postictal_row];
                    end
                end
            end
            %时间顺序调整 冒泡排序
            for ii=1:(height(new_table)-(timing_start_index-1)+1)-1
                for i=timing_start_index:height(new_table)
                    if(datetime(new_table(i,:).StartTime, 'Format', 'HH:mm:ss')<datetime(new_table(i-1,:).StartTime, 'Format', 'HH:mm:ss') && ...
                    datetime(new_table(i,:).StartTime, 'Format', 'HH:mm:ss')~=datetime(new_table(i-1,:).EndTime, 'Format', 'HH:mm:ss'))
                    tempRow = new_table(i, :);
                    new_table(i, :) = new_table(i-1, :);
                    new_table(i-1, :) = tempRow;
                    elseif(datetime(new_table(i,:).StartTime, 'Format', 'HH:mm:ss')-datetime(new_table(i-1,:).EndTime, 'Format', 'HH:mm:ss'))>0.5
                    tempRow = new_table(i, :);
                    new_table(i, :) = new_table(i-1, :);
                    new_table(i-1, :) = tempRow;
                    end
                end
            end
            timing_start_index=height(new_table)+2;
            if(z+1<=height(currentGroup))
                nextRow=currentGroup(z+1,:);
                duration = hourNumberToSecond(nextRow.StartTime-currentRow.EndTime);
                start_index=end_index+duration;
            end
              waitbar(v / row_count, h, sprintf('Final Data Processing... %d/%d', v, row_count));
        end
        
        %End
    end
    
    %End
    % 构建文件名
    file_name = ['CHB_new_test_', num2str(preictal_duration), '_', num2str(interictal_ex_duration), '.xlsx'];

    % 构建完整的文件路径
    out_schedule_path = fullfile(pwd, file_name);

    % 使用 writetable 写入文件
    writetable(new_table, out_schedule_path);





    %% 给表格着色
    [xls_row,xls_col] = size(new_table);
    new_table_cell = table2cell(new_table);
    % Connect to Excel
    Excel = actxserver('excel.application');
    % Get Workbook object
    WB = Excel.Workbooks.Open(fullfile(pwd, file_name),0,false);
    % Set the color of cell "A1" of Sheet 1 to Yellow
    % color index:
    % 1-black,2-white,3-red,4-green,5-blue,6-yellow,45-orange,43-lightgreen,15-lightgrey
    for ii=1:xls_row
        if strcmp(new_table_cell(ii,7), {'preictal'})
            str2 = int2str(ii+1);
            poistions_range= strcat('C',str2,':G',str2);
            WB.Worksheets.Item(1).Range(poistions_range).Interior.ColorIndex = 6;       %color-yellow
        elseif strcmp(new_table_cell(ii,7), {'postictal'})
            str2 = int2str(ii+1);
            poistions_range= strcat('C',str2,':G',str2);
            WB.Worksheets.Item(1).Range(poistions_range).Interior.ColorIndex = 45;       %color-orange
        elseif strcmp(new_table_cell(ii,7), {'ictal'})
            str2 = int2str(ii+1);
            poistions_range= strcat('C',str2,':G',str2);
            WB.Worksheets.Item(1).Range(poistions_range).Interior.ColorIndex = 3;       %color-red
        elseif strcmp(new_table_cell(ii,7), {'excluded'})
            str2 = int2str(ii+1);
            poistions_range= strcat('C',str2,':G',str2);
            WB.Worksheets.Item(1).Range(poistions_range).Interior.ColorIndex = 15;       %color-lightgrey
        elseif strcmp(new_table_cell(ii,7), {'interictal'})
            str2 = int2str(ii+1);
            poistions_range= strcat('C',str2,':G',str2);
            WB.Worksheets.Item(1).Range(poistions_range).Interior.ColorIndex = 43;       %color-lightgreen
        end
        waitbar(ii / xls_row, h, sprintf('Colour Processing... %d/%d', ii, xls_row));
    end


    % Save Workbook
    WB.Save();
    % Close Workbook
    WB.Close();
    % Quit Excel
    Excel.Quit();
    close(h);







    
    return;
    









        

    function result = minTo24Hour(min)
        result = 1/24*min/60;
    end
    function result = hourNumberToSecond(input24Hour)
        if(input24Hour<0)
            input24Hour = 1+input24Hour;
        elseif(input24Hour>1)
            input24Hour = input24Hour-1;
        end
        result = round(input24Hour*24*3600);
    end
    function result = formatToHHMMSS(input24Hour)
        result = string(datestr(input24Hour, 'HH:MM:SS'));
    end
    %根据输入时分秒和秒数 确定目标时分秒
    function newTimeStr = addSeconds(startTimeStr, secondsToAdd)
        % 将HH:MM:SS格式的时间字符串和秒数相加，返回新的时间字符串
        startTimeComponents = sscanf(startTimeStr, '%d:%d:%d', [1, 3]);
        startHours = startTimeComponents(1);
        startMinutes = startTimeComponents(2);
        startSeconds = startTimeComponents(3);
        
    
        % 计算新的总秒数 （减一与数组结构有关）
        totalSeconds = startHours * 3600 + startMinutes * 60 + startSeconds + secondsToAdd -1;
    
        % 计算新的时刻
        hoursPart = floor(totalSeconds / 3600);
        minutesPart = floor((totalSeconds - hoursPart * 3600) / 60);
        secondsPart = totalSeconds - hoursPart * 3600 - minutesPart * 60;
        hoursPart=mod(hoursPart,24);
        % 格式化为字符串
        newTimeStr = sprintf('%02d:%02d:%02d', hoursPart, minutesPart, secondsPart);
    end
    %表格写入
    function newRow = createTableRow(subject, fileName, startTime, endTime, samplingRate, frameNumbers, rowType)
    newRow = table(subject, fileName, {startTime}, {endTime}, ...
                       samplingRate, frameNumbers, {rowType}, ...
                       'VariableNames', {'Subject', 'FileName', 'StartTime', 'EndTime', ...
                       'Sampling rate（Hz）', 'Frame numbers', 'type'});
    end
end