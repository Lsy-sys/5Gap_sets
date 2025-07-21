% 学校栅格模型与信号传播仿真
clear;
clc;
close all;

%% 参数设置
grid_size = [68, 95];  % 栅格大小 [行数, 列数]
num_stations = 5;      % 基站数量
num_iterations = 100; % 蒙特卡洛迭代次数
min_signal_strength = -100;  % 最小信号强度要求(dBm)，调整为更合理的负值
initial_power = 46;    % 基站初始信号强度(dBm)

%% 物体类型及其衰减特性
% 物体编号说明：
% 0: 空地
% 1: 建筑物
% 2: 树木
% 3: 道路
% 4: 水体
object_types = struct(...
    'empty', 0, ...
    'building', 1, ...
    'tree', 2, ...
    'road', 3, ...
    'water', 4);

% 不同物体的衰减系数 (dB/m)
attenuation_coeffs = [0.01, 104.17, 0.25, 3, 0.02];  % 对应每种物体类型

%% 导入栅格数据
% 从CSV文件导入栅格数据
try
    % MATLAB Engine的当前工作目录在Python中被设置为项目根目录
    % school_grid.csv 通常由 image_to_grid_python.py 保存到 static/uploads/
    grid_data = readmatrix('static/uploads/school_grid.csv'); 
    if ~isequal(size(grid_data), grid_size)
        error('栅格数据大小与设定不符');
    end
    % 验证数据有效性
    if any(grid_data(:) < 0) || any(grid_data(:) > 4)
        error('栅格数据包含无效值（应为0-4）');
    end
    fprintf('成功导入栅格数据: static/uploads/school_grid.csv\n');
catch ME
    fprintf('无法读取栅格数据文件或数据无效，使用示例数据...\n');
    fprintf('错误信息: %s\n', ME.message);
    % 生成示例栅格数据
    grid_data = zeros(grid_size);
    % 添加一些示例建筑物和设施
    grid_data(10:20, 10:20) = 1;  % 建筑物1
    grid_data(30:35, 30:45) = 1;  % 建筑物2
    grid_data(5:45, 25) = 3;      % 主路
    grid_data(25, 5:45) = 3;      % 交叉路
    grid_data(40:45, 5:10) = 2;   % 树木区域1
    grid_data(5:10, 40:45) = 2;   % 树木区域2
    grid_data(15:20, 35:40) = 4;  % 水体
end

%% 蒙特卡洛模拟
best_coverage = -inf;
best_stations = zeros(num_stations, 2);
coverage_history = zeros(num_iterations, 1);  % 记录每次迭代的覆盖率

for iter = 1:num_iterations
    % 随机生成基站位置
    stations = zeros(num_stations, 2);
    for i = 1:num_stations
        valid_position = false;
        attempts = 0;
        max_attempts = 500;  % 防止无限循环，增加尝试次数
        
        while ~valid_position && attempts < max_attempts
            % 正确生成随机位置 (行和列都应在栅格范围内)
            row = randi(grid_size(1));
            col = randi(grid_size(2));
            
            % 确保基站不在建筑物内（类型1）且与其他基站保持最小距离
            is_building = (grid_data(row, col) == object_types.building);
            
            min_dist_satisfied = true;
            if i > 1 % 对第一个基站不需要检查与之前基站的距离
                distances_to_others = sqrt(sum((stations(1:i-1,:) - [row, col]).^2, 2));
                if any(distances_to_others < 5) % 最小间距为5个栅格
                    min_dist_satisfied = false;
                end
            end

            if ~is_building && min_dist_satisfied
                stations(i,:) = [row, col];
                valid_position = true;
            end
            attempts = attempts + 1;
        end
        
        if attempts >= max_attempts
            fprintf('警告：第%d个基站位置选择达到最大尝试次数，可能未能找到最佳位置。\n', i);
        end
    end
    
    % 计算总信号覆盖
    total_signal = zeros(grid_size);
    for i = 1:num_stations
        % 确保只计算已成功定位的基站
        if all(stations(i,:) ~= 0) 
            signal = calculate_signal_strength(grid_data, stations(i,:), attenuation_coeffs, initial_power);
            total_signal = max(total_signal, signal);
        end
    end
    
    % 计算有效覆盖率
    coverage = sum(total_signal(:) >= min_signal_strength) / numel(total_signal);
    coverage_history(iter) = coverage;
    
    % 更新最佳结果
    if coverage > best_coverage
        best_coverage = coverage;
        best_stations = stations;
    end
    
    % 显示进度
    if mod(iter, 100) == 0
        fprintf('完成迭代次数: %d/%d, 当前最佳覆盖率: %.2f%%\n', ...
                iter, num_iterations, best_coverage * 100);
    end
end

%% 可视化结果
% 创建新图窗并设置大小，确保图像足够大，避免文字重叠
figure('Position', [100, 100, 1200, 900], 'PaperPositionMode', 'auto'); 

% 绘制栅格数据
subplot(2,2,1);
imagesc(grid_data);
colormap('jet');
colorbar;
hold on;
% 标记最佳基站位置
plot(best_stations(:,2), best_stations(:,1), 'w*', 'MarkerSize', 10, 'LineWidth', 2);
title('学校栅格模型与基站分布');
xlabel('X坐标');
ylabel('Y坐标');
legend('基站位置', 'Location', 'bestoutside');

% 添加栅格类型说明（放置在图外或留出足够空间）
text(grid_size(2) + 2, 1, '栅格类型说明:', 'FontSize', 8);
text(grid_size(2) + 2, 3, '0: 空地', 'FontSize', 8);
text(grid_size(2) + 2, 5, '1: 建筑物', 'FontSize', 8);
text(grid_size(2) + 2, 7, '2: 树木', 'FontSize', 8);
text(grid_size(2) + 2, 9, '3: 道路', 'FontSize', 8);
text(grid_size(2) + 2, 11, '4: 水体', 'FontSize', 8);


% 绘制信号强度分布
subplot(2,2,2);
% 重新计算final_total_signal以确保是最终结果的信号强度
final_total_signal = zeros(grid_size);
for i = 1:num_stations
    if all(best_stations(i,:) ~= 0) % 确保只计算有效基站
        signal = calculate_signal_strength(grid_data, best_stations(i,:), attenuation_coeffs, initial_power);
        final_total_signal = max(final_total_signal, signal);
    end
end
imagesc(final_total_signal);
colormap('jet');
colorbar;
title('信号强度分布 (dBm)');
xlabel('X坐标');
ylabel('Y坐标');


% 绘制覆盖率历史
subplot(2,2,3);
plot(1:num_iterations, coverage_history * 100, 'b-');
title('覆盖率优化历史');
xlabel('迭代次数');
ylabel('覆盖率 (%)');
grid on;


% 绘制信号强度直方图
subplot(2,2,4);
histogram(final_total_signal(:), 50);
title('信号强度分布直方图');
xlabel('信号强度 (dBm)');
ylabel('频次');
grid on;

drawnow; % 确保图窗更新，以便后续保存

%% **保存图表和统计信息**
% 定义保存目录，确保与Flask应用的'static/results/'路径对应
% MATLAB的当前工作目录在Python中被设置为项目根目录
output_dir_rel = 'static/results/'; 
% 检查并创建目录
if ~exist(output_dir_rel, 'dir')
    mkdir(output_dir_rel);
    fprintf('已创建输出目录: %s\n', output_dir_rel);
end

% 定义图片文件名 (用于独立保存，虽然上面也有一个组合图)
output_filenames = {
    'grid_and_stations.png',
    'signal_strength_map.png',
    'coverage_history.png',
    'signal_histogram.png'
};

% 确保保存所有子图为独立文件
% 注意：这里为每个子图创建了一个不可见的临时Figure来保存，以确保它们是独立的图片。
fig1 = figure('Visible', 'off'); imagesc(grid_data); colormap('jet'); colorbar; hold on; plot(best_stations(:,2), best_stations(:,1), 'w*', 'MarkerSize', 10, 'LineWidth', 2); title('学校栅格模型与基站分布'); xlabel('X坐标'); ylabel('Y坐标'); saveas(fig1, fullfile(output_dir_rel, output_filenames{1})); close(fig1);
fprintf('已保存: %s\n', fullfile(output_dir_rel, output_filenames{1}));

fig2 = figure('Visible', 'off'); imagesc(final_total_signal); colormap('jet'); colorbar; title('信号强度分布 (dBm)'); xlabel('X坐标'); ylabel('Y坐标'); saveas(fig2, fullfile(output_dir_rel, output_filenames{2})); close(fig2);
fprintf('已保存: %s\n', fullfile(output_dir_rel, output_filenames{2}));

fig3 = figure('Visible', 'off'); plot(1:num_iterations, coverage_history * 100, 'b-'); title('覆盖率优化历史'); xlabel('迭代次数'); ylabel('覆盖率 (%)'); grid on; saveas(fig3, fullfile(output_dir_rel, output_filenames{3})); close(fig3);
fprintf('已保存: %s\n', fullfile(output_dir_rel, output_filenames{3}));

fig4 = figure('Visible', 'off'); histogram(final_total_signal(:), 50); title('信号强度分布直方图'); xlabel('信号强度 (dBm)'); ylabel('频次'); grid on; saveas(fig4, fullfile(output_dir_rel, output_filenames{4})); close(fig4);
fprintf('已保存: %s\n', fullfile(output_dir_rel, output_filenames{4}));


% 保存统计信息到文本文件
stats_filename = fullfile(output_dir_rel, 'simulation_stats.txt');
fid = fopen(stats_filename, 'wt');
if fid == -1
    error('无法创建统计文件: %s', stats_filename);
end
fprintf(fid, '最佳覆盖率: %.2f%%\n', best_coverage * 100);
fprintf(fid, '平均信号强度: %.2f dBm\n', mean(final_total_signal(:)));
fprintf(fid, '信号强度标准差: %.2f dB\n', std(final_total_signal(:)));
fprintf(fid, '基站位置:\n');
for i = 1:num_stations
    fprintf(fid, '基站 %d: (%d, %d)\n', i, best_stations(i,1), best_stations(i,2));
end
fclose(fid);
fprintf('统计信息已保存到: %s\n', stats_filename);


% 输出统计信息到MATLAB命令行 (可选，因为已经保存到文件)
fprintf('\n=== 优化结果统计 ===\n');
fprintf('最佳覆盖率: %.2f%%\n', best_coverage * 100);
fprintf('平均信号强度: %.2f dBm\n', mean(final_total_signal(:)));
fprintf('信号强度标准差: %.2f dB\n', std(final_total_signal(:)));
fprintf('\n基站位置:\n');
for i = 1:num_stations
    fprintf('基站 %d: (%d, %d)\n', i, best_stations(i,1), best_stations(i,2));
end


%% **函数定义 (必须放在文件末尾)**

% 信号传播模型
function signal_strength = calculate_signal_strength(grid, station_pos, attenuation_coeffs, initial_power)
    [rows, cols] = size(grid);
    signal_strength = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            % 计算到基站的距离
            distance = sqrt((i-station_pos(1))^2 + (j-station_pos(2))^2);
            
            % 自由空间路径损耗 (使用对数尺度)
            % 避免 log10(0)，确保距离至少为 1
            path_loss = 20 * log10(max(distance, 1));  
            
            % 计算穿过的栅格和对应的衰减
            % Bresenham算法生成的点可能超出栅格边界，需要在访问grid前检查
            [x_line, y_line] = bresenham(station_pos(1), station_pos(2), i, j);
            total_attenuation = 0;
            
            for k = 1:length(x_line)
                curr_x = x_line(k);
                curr_y = y_line(k);
                
                % 确保点在栅格范围内
                if curr_x >= 1 && curr_x <= rows && curr_y >= 1 && curr_y <= cols
                    object_type = grid(curr_x, curr_y);
                    % 确保object_type是有效的索引 (0-4)
                    if object_type >= 0 && object_type <= 4
                        total_attenuation = total_attenuation + attenuation_coeffs(object_type + 1);
                    end
                end
            end
            
            % 计算总信号强度
            signal_strength(i,j) = initial_power - path_loss - total_attenuation;
        end
    end
end

%% Bresenham直线算法
function [x_out, y_out] = bresenham(x1, y1, x2, y2)
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    steep = dy > dx;
    
    if steep
        [x1, y1] = swap(x1, y1);
        [x2, y2] = swap(x2, y2);
    end
    
    if x1 > x2
        [x1, x2] = swap(x1, x2);
        [y1, y2] = swap(y1, y2);
    end
    
    dx = x2 - x1;
    dy = abs(y2 - y1);
    error_val = dx / 2;
    
    if y1 < y2
        ystep = 1;
    else
        ystep = -1;
    end
    
    y_curr = y1;
    x_coords = (x1:x2)';
    y_out_temp = zeros(size(x_coords));
    
    for i = 1:length(x_coords)
        if steep
            y_out_temp(i) = x_coords(i);
            x_out(i) = y_curr;
        else
            y_out_temp(i) = y_curr;
            x_out(i) = x_coords(i);
        end
        
        error_val = error_val - dy;
        if error_val < 0
            y_curr = y_curr + ystep;
            error_val = error_val + dx;
        end
    end
    
    y_out = y_out_temp;
end

function [a_out, b_out] = swap(a, b)
    temp = a;
    a_out = b;
    b_out = temp;
end