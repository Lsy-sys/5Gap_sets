% 学校栅格模型与信号传播仿真
clear;
clc;
close all;

%% 参数设置
grid_size = [68, 95];  % 栅格大小 [行数, 列数]
num_stations = 5;      % 基站数量
num_iterations = 10000; % 蒙特卡洛迭代次数
min_signal_strength = 10;  % 最小信号强度要求(dBm)
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
    grid_data = readmatrix('school_grid.csv');
    if ~isequal(size(grid_data), grid_size)
        error('栅格数据大小与设定不符');
    end
    % 验证数据有效性
    if any(grid_data(:) < 0) || any(grid_data(:) > 4)
        error('栅格数据包含无效值（应为0-4）');
    end
catch
    fprintf('无法读取栅格数据文件或数据无效，使用示例数据...\n');
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

%% 信号传播模型
function signal_strength = calculate_signal_strength(grid, station_pos, attenuation_coeffs, initial_power)
    [rows, cols] = size(grid);
    signal_strength = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            % 计算到基站的距离
            distance = sqrt((i-station_pos(1))^2 + (j-station_pos(2))^2);
            
            % 自由空间路径损耗 (使用对数尺度)
            path_loss = 20 * log10(max(distance, 1));  % 避免log(0)
            
            % 计算穿过的栅格和对应的衰减
            [x, y] = bresenham(station_pos(1), station_pos(2), i, j);
            total_attenuation = 0;
            
            for k = 1:length(x)
                if x(k) >= 1 && x(k) <= rows && y(k) >= 1 && y(k) <= cols
                    object_type = grid(x(k), y(k));
                    % 确保object_type是有效的索引
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
        max_attempts = 100;  % 防止无限循环
        
        while ~valid_position && attempts < max_attempts
            % 正确生成随机位置
            row = randi(grid_size(1));
            col = randi(grid_size(2));
            % 确保基站不在建筑物内且与其他基站保持最小距离
            if grid_data(row, col) ~= object_types.building && ...
               all(sqrt(sum((stations(1:i-1,:) - [row, col]).^2, 2)) > 5)  % 最小间距为5个栅格
                stations(i,:) = [row, col];
                valid_position = true;
            end
            attempts = attempts + 1;
        end
        
        if attempts >= max_attempts
            fprintf('警告：第%d个基站位置选择达到最大尝试次数\n', i);
        end
    end
    
    % 计算总信号覆盖
    total_signal = zeros(grid_size);
    for i = 1:num_stations
        signal = calculate_signal_strength(grid_data, stations(i,:), attenuation_coeffs, initial_power);
        total_signal = max(total_signal, signal);
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
% 创建新图窗并设置大小
figure('Position', [100, 100, 1200, 800]);

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
legend('基站位置');

% 添加栅格类型说明
text(grid_size(2)+2, 1, '栅格类型说明:', 'FontSize', 8);
text(grid_size(2)+2, 3, '0: 空地', 'FontSize', 8);
text(grid_size(2)+2, 5, '1: 建筑物', 'FontSize', 8);
text(grid_size(2)+2, 7, '2: 树木', 'FontSize', 8);
text(grid_size(2)+2, 9, '3: 道路', 'FontSize', 8);
text(grid_size(2)+2, 11, '4: 水体', 'FontSize', 8);

% 绘制信号强度分布
subplot(2,2,2);
total_signal = zeros(grid_size);
for i = 1:num_stations
    signal = calculate_signal_strength(grid_data, best_stations(i,:), attenuation_coeffs, initial_power);
    total_signal = max(total_signal, signal);
end
imagesc(total_signal);
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
histogram(total_signal(:), 50);
title('信号强度分布直方图');
xlabel('信号强度 (dBm)');
ylabel('频次');
grid on;

% 输出统计信息
fprintf('\n=== 优化结果统计 ===\n');
fprintf('最佳覆盖率: %.2f%%\n', best_coverage * 100);
fprintf('平均信号强度: %.2f dBm\n', mean(total_signal(:)));
fprintf('信号强度标准差: %.2f dB\n', std(total_signal(:)));
fprintf('\n基站位置:\n');
for i = 1:num_stations
    fprintf('基站 %d: (%d, %d)\n', i, best_stations(i,1), best_stations(i,2));
end

%% Bresenham直线算法
function [x, y] = bresenham(x1, y1, x2, y2)
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
    error = dx / 2;
    
    if y1 < y2
        ystep = 1;
    else
        ystep = -1;
    end
    
    y = y1;
    x = (x1:x2)';
    y_out = zeros(size(x));
    
    for i = 1:length(x)
        if steep
            y_out(i) = x(i);
            x(i) = y;
        else
            y_out(i) = y;
        end
        
        error = error - dy;
        if error < 0
            y = y + ystep;
            error = error + dx;
        end
    end
    
    y = y_out;
end

function [a, b] = swap(a, b)
    temp = a;
    a = b;
    b = temp;
end 
