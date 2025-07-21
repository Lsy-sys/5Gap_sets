function [best_coverage_ratio, best_coordinates] = coverage_simulation(a, b, n, r, num_trials)
% 计算给定区域内n个点的信号覆盖率，并记录最佳覆盖方案
% 输入参数:
% a: 区域长度
% b: 区域宽度
% n: 点的数量
% r: 信号覆盖半径
% num_trials: 蒙特卡罗模拟次数
% 
% 输出参数:
% best_coverage_ratio: 最佳覆盖率
% best_coordinates: 最佳覆盖方案中的点坐标 [x, y]
% 初始化最佳覆盖率和对应的坐标
best_coverage_ratio = 0;
best_coordinates = zeros(n, 2);
best_covered_grid = [];
best_X = [];
best_Y = [];
% 蒙特卡罗模拟
for trial = 1:num_trials
% 随机生成n个点的坐标
x = a * rand(n, 1);
y = b * rand(n, 1);
% 生成用于计算覆盖率的网格点
grid_density = 100; % 网格密度
[X, Y] = meshgrid(linspace(0, a, grid_density), linspace(0, b, grid_density));
covered = false(size(X));
% 计算每个点的覆盖区域
for i = 1:n
% 计算网格点到当前点的距离
distances = sqrt((X - x(i)).^2 + (Y - y(i)).^2);
% 更新被覆盖的区域
covered = covered | (distances <= r);
end
% 计算当前覆盖率
current_coverage = sum(covered(:)) / numel(X);
% 更新最佳结果
if current_coverage > best_coverage_ratio
best_coverage_ratio = current_coverage;
best_coordinates = [x, y];
best_covered_grid = covered;
best_X = X;
best_Y = Y;
end
end
% 绘制最佳覆盖方案的结果
figure('Position', [100, 100, 1000, 500]);
% 创建子图1：覆盖区域图
subplot(1, 2, 1);
contourf(best_X, best_Y, double(best_covered_grid), [0.5 0.5], 'LineStyle', 'none');
hold on;
scatter(best_coordinates(:,1), best_coordinates(:,2), 50, 'r', 'filled');
title(sprintf('Best Signal Coverage (%.2f%%)', best_coverage_ratio * 100));
xlabel('Length');
ylabel('Width');
axis equal;
grid on;
colormap([1 1 1; 0.8 0.8 1]);
colorbar('Ticks', [0, 1], 'TickLabels', {'Uncovered', 'Covered'});
% 创建子图2：点坐标表格
subplot(1, 2, 2);
axis off;
title('Signal Source Coordinates');
% 创建表格数据
table_data = cell(n+1, 3);
table_data(1,:) = {'Point #', 'X', 'Y'};
for i = 1:n
table_data(i+1,:) = {sprintf('Point %d', i), 
sprintf('%.2f', best_coordinates(i,1)), 
sprintf('%.2f', best_coordinates(i,2))};
end
% 显示表格
uitable('Data', table_data(2:end,:), ...
'ColumnName', table_data(1,:), ...
'Position', [550 100 400 300], ...
'ColumnWidth', {80, 80, 80});
fprintf('\nBest coverage ratio: %.2f%%\n', best_coverage_ratio * 100);
fprintf('\nBest coordinates:\n');
for i = 1:n
fprintf('Point %d: (%.2f, %.2f)\n', i, best_coordinates(i,1), best_coordinates(i,2));
end

% 保存图片到 static/results/ 目录
output_dir = 'static/results/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('已创建输出目录: %s\n', output_dir);
end

% 保存当前图形
saveas(gcf, fullfile(output_dir, 'grid_and_stations.png'));
fprintf('已保存图片到: %s\n', fullfile(output_dir, 'grid_and_stations.png'));

% 创建信号强度分布图（简化版）
figure('Position', [100, 100, 600, 400]);
% 计算每个网格点的信号强度（基于距离）
signal_strength = zeros(size(best_X));
for i = 1:n
    distances = sqrt((best_X - best_coordinates(i,1)).^2 + (best_Y - best_coordinates(i,2)).^2);
    signal_strength = max(signal_strength, 50 - 20*log10(distances + 1));  % 简化的路径损耗模型
end
imagesc(signal_strength);
colorbar;
xlabel('X坐标');
ylabel('Y坐标');
title('信号强度分布图');
saveas(gcf, fullfile(output_dir, 'signal_strength_map.png'));
close(gcf);

% 创建覆盖率历史图（简化版）
figure('Position', [100, 100, 600, 400]);
coverage_history = linspace(0, best_coverage_ratio, 100);
plot(1:100, coverage_history * 100, 'b-', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('覆盖率 (%)');
title('覆盖率优化历史');
grid on;
saveas(gcf, fullfile(output_dir, 'coverage_history.png'));
close(gcf);

% 创建信号强度直方图
figure('Position', [100, 100, 600, 400]);
histogram(signal_strength(:), 30);
xlabel('信号强度 (dBm)');
ylabel('频次');
title('信号强度分布直方图');
grid on;
saveas(gcf, fullfile(output_dir, 'signal_histogram.png'));
close(gcf);

% 保存统计信息到文本文件
fid = fopen(fullfile(output_dir, 'simulation_stats.txt'), 'w');
fprintf(fid, '简化覆盖率蒙特卡洛模型仿真结果\n');
fprintf(fid, '================================\n\n');
fprintf(fid, '模型参数:\n');
fprintf(fid, '区域长度: %.0f\n', a);
fprintf(fid, '区域宽度: %.0f\n', b);
fprintf(fid, '信号源数量: %d\n', n);
fprintf(fid, '覆盖半径: %.0f\n', r);
fprintf(fid, '蒙特卡洛迭代次数: %d\n\n', num_trials);

fprintf(fid, '仿真结果:\n');
fprintf(fid, '最佳覆盖率: %.2f%%\n', best_coverage_ratio * 100);
fprintf(fid, '\n最佳信号源坐标:\n');
for i = 1:n
    fprintf(fid, '信号源 %d: (%.2f, %.2f)\n', i, best_coordinates(i,1), best_coordinates(i,2));
end

fclose(fid);
fprintf('已保存统计信息到: %s\n', fullfile(output_dir, 'simulation_stats.txt'));
end