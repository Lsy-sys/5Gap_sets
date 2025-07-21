function COST231_Walfisch_Ikegami_Model()
    % COST231 Walfisch Ikegami模型参数
    freq = 2600000000; % 信号频率(Hz)
    h_b = 15.25;    % 基站天线有效高度(m)
    h_m = 1.5;   % 移动台天线高度(m)
    h_av = 10;    % 街道平均建筑物高度(m)
    w_av = 10;   % 街道平均宽度(m)
    
    % 街道方向修正因子 (0: 平行于街道方向, 1: 垂直于街道方向)
    f_c = 0;     % 街道方向修正因子
    
    % 定义信号强度阈值
    SIGNAL_GOOD = -90;    % 信号质量边界值(dBm)
    SIGNAL_POOR = -100;   % 信号接收极限值(dBm)
    
    % 参数有效性检查
    if freq < 800e6 || freq > 6e9
        error('频率必须在800MHz到6GHz之间');
    end
    if h_b <= h_m
        error('基站高度必须大于移动台高度');
    end
    
    % 计算模型参数
    lambda = 3e8 / freq; % 信号波长
    h_r = h_av * (1 - exp(-w_av/(2*h_av))); % 有效反射点高度
    h_b_eff = h_b - h_r; % 基站天线有效高度修正
    h_m_eff = h_m - h_r; % 移动台天线有效高度修正
    A_h = 5 * log10(h_av / h_r); % 建筑物高度修正因子
    A_w = 5 * log10(w_av / 20); % 街道宽度修正因子
    A_f = 10 * f_c; % 街道方向修正因子
    
    % 计算路径损耗
    d = linspace(max(0.1, lambda), 5000, 100); % 基站与移动台之间的距离(m)
    PL = zeros(size(d));
    for i = 1:length(d)
        if d(i) >= 20 % 确保距离大于最小有效距离
            PL(i) = 46.3 + 33.9 * log10(freq / 1e9) - 13.82 * log10(h_b) ...
                    - A_h - A_w - A_f + (44.9 - 6.55 * log10(h_b)) * log10(d(i) / 1000);
        else
            PL(i) = NaN; % 对于无效距离返回NaN
        end
    end
    
    % 信号强度计算
    P_t = 36; % 发射功率(dBm)
    P_r = P_t - PL; % 接收功率(dBm)
    
    % 3D可视化
    [D, Theta] = meshgrid(d, linspace(0, 2*pi, 100));
    X = D .* cos(Theta);
    Y = D .* sin(Theta);
    Z = repmat(P_r, length(Theta), 1);
    
    % 创建图形并设置属性
    figure('Position', [100, 100, 800, 600]);
    
    % 使用自定义颜色图显示不同信号强度区域
    colormap([
        1 0 0;      % 红色 - 无信号区域 (<-100dBm)
        1 1 0;      % 黄色 - 弱信号区域 (-100到-90dBm)
        0 1 0       % 绿色 - 良好信号区域 (>-90dBm)
    ]);
    
    % 绘制3D表面
    surf(X, Y, Z, 'EdgeColor', 'none');
    
    % 设置颜色刻度范围和标签
    caxis([SIGNAL_POOR-10, max(max(Z))]); % 设置颜色范围
    c = colorbar;
    ylabel(c, '信号强度 (dBm)');
    
    % 在colorbar上添加信号质量指示
    c.Ticks = [SIGNAL_POOR, SIGNAL_GOOD, max(max(Z))];
    c.TickLabels = {sprintf('%.0f\n无信号', SIGNAL_POOR), ...
                    sprintf('%.0f\n弱信号', SIGNAL_GOOD), ...
                    sprintf('%.0f\n强信号', max(max(Z)))};
    
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Signal Strength (dBm)');
    title('COST231 Walfisch Ikegami Model - 信号覆盖分析');
    grid on;
    axis tight;
    view(45, 30);
    
    % 添加信息文本
    text_str = sprintf(['频率: %.1f GHz\n' ...
                       '基站高度: %.1f m\n' ...
                       '移动台高度: %.1f m\n' ...
                       '信号覆盖说明:\n' ...
                       '绿色: 良好 (>%.0f dBm)\n' ...
                       '黄色: 较差 (%.0f到%.0f dBm)\n' ...
                       '红色: 无信号 (<%.0f dBm)'], ...
                      freq/1e9, h_b, h_m, ...
                      SIGNAL_GOOD, SIGNAL_POOR, SIGNAL_GOOD, SIGNAL_POOR);
    annotation('textbox', [0.02, 0.75, 0.25, 0.2], 'String', text_str, ...
               'EdgeColor', 'none', 'BackgroundColor', 'white', ...
               'FitBoxToText', 'on');
               
    % 计算并显示有效覆盖半径
    coverage_radius = interp1(P_r, d, SIGNAL_POOR);
    if ~isnan(coverage_radius)
        text_coverage = sprintf('有效覆盖半径: %.1f m', coverage_radius);
        annotation('textbox', [0.02, 0.7, 0.25, 0.05], 'String', text_coverage, ...
                   'EdgeColor', 'none', 'BackgroundColor', 'white', ...
                   'FitBoxToText', 'on');
    end
end