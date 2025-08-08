import numpy as np
from PIL import Image
import os
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def extract_color_features(patch):
    """
    提取颜色特征
    """
    if patch.size == 0:
        return np.zeros(6)
    
    if len(patch.shape) == 3:
        # RGB特征
        r_mean = np.mean(patch[:,:,0])
        g_mean = np.mean(patch[:,:,1])
        b_mean = np.mean(patch[:,:,2])
        
        # HSV特征
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        h_mean = np.mean(hsv[:,:,0])
        s_mean = np.mean(hsv[:,:,1])
        v_mean = np.mean(hsv[:,:,2])
        
        return np.array([r_mean, g_mean, b_mean, h_mean, s_mean, v_mean])
    else:
        # 灰度图
        return np.array([np.mean(patch)] * 6)


def calculate_texture_features(patch):
    """
    计算纹理特征 - 改进版
    """
    if patch.size == 0:
        return np.zeros(4)
    
    # 转换为灰度图
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch
    
    # 1. 梯度特征
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)
    
    # 2. 局部二值模式 (LBP) 简化版
    lbp = calculate_lbp(gray)
    lbp_entropy = calculate_entropy(lbp)
    
    # 3. 灰度共生矩阵特征 (简化)
    glcm_contrast = calculate_glcm_contrast(gray)
    
    return np.array([gradient_mean, gradient_std, lbp_entropy, glcm_contrast])


def calculate_lbp(gray):
    """
    计算局部二值模式
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros_like(gray)
    
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            code = 0
            # 8邻域
            neighbors = [
                gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                gray[i+1, j-1], gray[i, j-1]
            ]
            for k, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    code |= (1 << k)
            lbp[i, j] = code
    
    return lbp


def calculate_entropy(lbp):
    """
    计算LBP的熵
    """
    if lbp.size == 0:
        return 0
    
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # 移除零值
    if len(hist) == 0:
        return 0
    
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-10))
    return entropy


def calculate_glcm_contrast(gray):
    """
    计算灰度共生矩阵的对比度特征
    """
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0
    
    # 简化的GLCM计算
    gray = gray.astype(np.uint8)
    contrast = 0
    count = 0
    
    for i in range(gray.shape[0]-1):
        for j in range(gray.shape[1]-1):
            # 水平方向
            diff = abs(int(gray[i, j]) - int(gray[i, j+1]))
            contrast += diff * diff
            count += 1
            # 垂直方向
            diff = abs(int(gray[i, j]) - int(gray[i+1, j]))
            contrast += diff * diff
            count += 1
    
    return contrast / max(count, 1)


def detect_edges_improved(patch):
    """
    改进的边缘检测
    """
    if patch.size == 0:
        return np.zeros(3)
    
    # 转换为灰度图
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch
    
    # 1. Canny边缘
    edges_canny = cv2.Canny(gray, 50, 150)
    edge_density_canny = np.sum(edges_canny > 0) / edges_canny.size
    
    # 2. Sobel边缘
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_density_sobel = np.mean(sobel_magnitude) / 255
    
    # 3. Laplacian边缘
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_density_laplacian = np.mean(np.abs(laplacian)) / 255
    
    return np.array([edge_density_canny, edge_density_sobel, edge_density_laplacian])


def analyze_shape_features_improved(patch):
    """
    改进的形状特征分析
    """
    if patch.size == 0:
        return np.zeros(4)
    
    # 转换为灰度图
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch
    
    # 自适应阈值
    binary = cv2.adaptiveThreshold(
        gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.array([0, 0, 0, 0])
    
    # 计算形状特征
    total_area = patch.shape[0] * patch.shape[1]
    contour_areas = [cv2.contourArea(c) for c in contours]
    total_contour_area = sum(contour_areas)
    
    # 1. 形状复杂度
    shape_complexity = len(contours) / max(1, total_contour_area / total_area)
    
    # 2. 矩形度
    if len(contour_areas) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        rect_area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_rect_area = w * h
        rectangularity = rect_area / max(1, bounding_rect_area)
    else:
        rectangularity = 0
    
    # 3. 圆形度
    if len(contour_areas) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / max(perimeter**2, 1)
    else:
        circularity = 0
    
    # 4. 面积占比
    area_ratio = total_contour_area / max(total_area, 1)
    
    return np.array([shape_complexity, rectangularity, circularity, area_ratio])


def improved_classification(patch, neighbors=None):
    """
    改进的分类算法
    """
    if patch.size == 0:
        return 0
    
    # 提取特征
    color_features = extract_color_features(patch)
    texture_features = calculate_texture_features(patch)
    edge_features = detect_edges_improved(patch)
    shape_features = analyze_shape_features_improved(patch)
    
    # 合并所有特征
    features = np.concatenate([color_features, texture_features, edge_features, shape_features])
    
    # 分类逻辑
    scores = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # 颜色特征分析
    r, g, b, h, s, v = color_features
    
    # 空地评分 - 高亮度，低饱和度，均匀颜色
    if v > 180 and s < 50:
        scores[0] += 15
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:  # 灰色
        scores[0] += 8
    if texture_features[0] < 15:  # 低纹理
        scores[0] += 8
    if edge_features[0] < 0.05:  # 少边缘
        scores[0] += 5
    
    # 建筑物评分 - 中等亮度，高矩形度，中等纹理
    if 80 < v < 160:
        scores[1] += 8
    if shape_features[1] > 0.6:  # 高矩形度
        scores[1] += 12
    if 10 < texture_features[0] < 30:  # 中等纹理
        scores[1] += 6
    if edge_features[0] < 0.1:  # 较少边缘
        scores[1] += 5
    if abs(r - g) < 25 and abs(g - b) < 25:  # 偏灰色
        scores[1] += 6
    
    # 树木评分 - 绿色，高纹理，高复杂度
    if g > r + 15 and g > b + 15:  # 偏绿色
        scores[2] += 15
    if 80 < h < 120:  # HSV绿色范围
        scores[2] += 10
    if texture_features[0] > 25:  # 高纹理
        scores[2] += 12
    if shape_features[0] > 2:  # 高复杂度
        scores[2] += 8
    if edge_features[1] > 0.05:  # 较多边缘
        scores[2] += 6
    if s > 80:  # 较高饱和度
        scores[2] += 5
    
    # 道路评分 - 灰色，低纹理，高矩形度，直线特征（更严格的条件）
    if abs(r - g) < 10 and abs(g - b) < 10 and abs(r - b) < 10:  # 严格灰色
        scores[3] += 15
    if texture_features[0] < 15:  # 低纹理
        scores[3] += 8
    if shape_features[1] > 0.8:  # 高矩形度
        scores[3] += 12
    if edge_features[0] > 0.08 and edge_features[0] < 0.2:  # 中等边缘
        scores[3] += 8
    if v > 100 and v < 160:  # 中等亮度
        scores[3] += 5
    if s < 60:  # 低饱和度
        scores[3] += 5
    
    # 水体评分 - 蓝色，低纹理，低边缘，高饱和度
    if b > r + 15 and b > g + 15:  # 偏蓝色
        scores[4] += 15
    if 200 < h < 250:  # HSV蓝色范围
        scores[4] += 10
    if texture_features[0] < 10:  # 低纹理
        scores[4] += 8
    if edge_features[0] < 0.05:  # 少边缘
        scores[4] += 8
    if s > 100:  # 高饱和度
        scores[4] += 5
    if v < 120:  # 低亮度
        scores[4] += 5
    
    # 上下文分析
    if neighbors is not None and len(neighbors) > 0:
        neighbor_counts = {}
        for neighbor in neighbors:
            if neighbor in neighbor_counts:
                neighbor_counts[neighbor] += 1
            else:
                neighbor_counts[neighbor] = 1
        
        # 如果邻居主要是某种类型，增加该类型的评分
        if neighbor_counts:
            most_common = max(neighbor_counts, key=neighbor_counts.get)
            if neighbor_counts[most_common] >= 4:  # 至少4个邻居相同
                scores[most_common] += 5
    
    # 返回得分最高的类型
    return max(scores, key=scores.get)


def spatial_consistency_filter(grid_data, window_size=3):
    """
    空间一致性滤波
    """
    rows, cols = grid_data.shape
    filtered = grid_data.copy()
    
    for r in range(rows):
        for c in range(cols):
            # 获取周围窗口
            r_start = max(0, r - window_size // 2)
            r_end = min(rows, r + window_size // 2 + 1)
            c_start = max(0, c - window_size // 2)
            c_end = min(cols, c + window_size // 2 + 1)
            
            window = grid_data[r_start:r_end, c_start:c_end]
            
            # 统计窗口中的类型
            unique, counts = np.unique(window, return_counts=True)
            most_common = unique[np.argmax(counts)]
            
            # 如果当前像素与周围大多数像素不同，则改为最常见的类型
            if grid_data[r, c] != most_common and counts[np.argmax(counts)] >= (window_size * window_size) // 2:
                filtered[r, c] = most_common
    
    return filtered


def morphological_post_processing(grid_data):
    """
    形态学后处理
    """
    # 转换为uint8
    grid_uint8 = grid_data.astype(np.uint8)
    
    # 对每种类型进行形态学操作
    processed = grid_uint8.copy()
    
    for terrain_type in range(5):
        # 创建二值图像
        binary = (grid_uint8 == terrain_type).astype(np.uint8)
        
        # 形态学闭运算（填充小洞）
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 形态学开运算（去除小噪点）
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 更新结果
        processed[binary == 1] = terrain_type
    
    return processed


def enhanced_image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir='static/uploads/'):
    """
    改进的图像转栅格算法
    """
    try:
        # 1. 读取图像
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"错误: 无法读取图像文件 {image_path}: {e}")
        return None

    # 2. 初始化栅格数据
    grid_data = np.zeros((target_grid_rows, target_grid_cols), dtype=int)
    
    # 3. 图像缩放参数
    img_rows, img_cols, _ = img_np.shape
    row_scale = img_rows / target_grid_rows
    col_scale = img_cols / target_grid_cols
    
    print("正在使用改进算法转换图像...")
    
    # 统计识别结果
    terrain_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # 第一遍：初始分类
    for r in range(target_grid_rows):
        for c in range(target_grid_cols):
            # 计算当前栅格单元对应的图像区域
            start_row = int(round(r * row_scale))
            end_row = int(round((r + 1) * row_scale))
            start_col = int(round(c * col_scale))
            end_col = int(round((c + 1) * col_scale))
            
            # 边界检查
            start_row = max(0, start_row)
            end_row = min(img_rows, end_row)
            start_col = max(0, start_col)
            end_col = min(img_cols, end_col)
            
            if start_row >= end_row or start_col >= end_col:
                grid_data[r, c] = 0
                terrain_counts[0] += 1
                continue
            
            # 提取图像块
            patch = img_np[start_row:end_row, start_col:end_col, :]
            
            # 获取邻居信息
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < target_grid_rows and 0 <= nc < target_grid_cols:
                        neighbors.append(grid_data[nr, nc])
            
            # 使用改进分类
            terrain_type = improved_classification(patch, neighbors)
            
            grid_data[r, c] = terrain_type
            terrain_counts[terrain_type] += 1
    
    # 4. 后处理
    print("正在进行后处理...")
    
    # 空间一致性滤波
    grid_data = spatial_consistency_filter(grid_data, window_size=3)
    
    # 形态学后处理
    grid_data = morphological_post_processing(grid_data)
    
    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'school_grid.csv')
    np.savetxt(csv_filename, grid_data, delimiter=',', fmt='%d')
    
    # 重新统计
    terrain_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for r in range(target_grid_rows):
        for c in range(target_grid_cols):
            terrain_counts[grid_data[r, c]] += 1
    
    # 打印统计信息
    total_cells = target_grid_rows * target_grid_cols
    print(f"\n改进算法识别统计:")
    terrain_names = {0: '空地', 1: '建筑物', 2: '树木', 3: '道路', 4: '水体'}
    for terrain_type, count in terrain_counts.items():
        percentage = (count / total_cells) * 100
        print(f"{terrain_names[terrain_type]}: {count} 个栅格 ({percentage:.1f}%)")
    
    print(f"改进算法转换完成，结果已保存到: {csv_filename}")
    return grid_data


def simple_effective_classification(patch):
    """
    简单有效的分类算法 - 保留作为备用
    """
    if patch.size == 0:
        return 0
    
    # 转换为灰度图
    if len(patch.shape) == 3:
        gray = np.mean(patch, axis=2)
        # 计算RGB平均值
        avg_r = np.mean(patch[:,:,0])
        avg_g = np.mean(patch[:,:,1])
        avg_b = np.mean(patch[:,:,2])
    else:
        gray = patch
        avg_r = avg_g = avg_b = np.mean(gray)
    
    # 计算基本特征
    avg_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # 简单的分类逻辑 - 基于亮度和颜色
    if avg_intensity > 200:  # 很亮的区域
        return 0  # 空地
    elif avg_intensity < 60:  # 很暗的区域
        return 4  # 水体
    elif avg_intensity > 120 and avg_intensity < 180:  # 中等偏亮
        # 判断是否为建筑物
        if std_intensity < 20:  # 相对均匀
            return 1  # 建筑物
        else:
            return 0  # 空地
    elif avg_intensity > 80 and avg_intensity < 140:  # 中等亮度
        # 判断颜色偏向
        if avg_g > avg_r + 10 and avg_g > avg_b + 10:  # 偏绿色
            return 2  # 树木
        elif abs(avg_r - avg_g) < 15 and abs(avg_g - avg_b) < 15:  # 灰色
            return 3  # 道路
        else:
            return 1  # 建筑物
    else:
        # 其他情况
        if avg_intensity > 100:
            return 0  # 空地
        else:
            return 2  # 树木


def simple_post_processing(grid_data):
    """
    简单的后处理 - 保留作为备用
    """
    rows, cols = grid_data.shape
    processed = grid_data.copy()
    
    # 简单的邻居投票
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            current_type = processed[r, c]
            
            # 检查周围8个邻居
            neighbors = [
                processed[r-1, c-1], processed[r-1, c], processed[r-1, c+1],
                processed[r, c-1], processed[r, c+1],
                processed[r+1, c-1], processed[r+1, c], processed[r+1, c+1]
            ]
            
            # 统计邻居类型
            neighbor_counts = {}
            for neighbor in neighbors:
                neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1
            
            # 确保当前类型在字典中存在
            if current_type not in neighbor_counts:
                neighbor_counts[current_type] = 0
            
            # 如果当前像素与周围大多数像素不同，则改为周围最常见的类型
            most_common_neighbor = max(neighbor_counts, key=neighbor_counts.get)
            if neighbor_counts[current_type] <= 2 and neighbor_counts[most_common_neighbor] >= 4:
                processed[r, c] = most_common_neighbor
    
    return processed


def image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir='static/uploads/'):
    """
    主函数，调用改进算法
    """
    return enhanced_image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir)


if __name__ == '__main__':
    """
    测试主函数 - 用于测试改进算法
    """
    print("=== 改进算法测试 ===")
    
    # 测试参数
    target_rows = 68
    target_cols = 95
    
    # 测试图像路径（使用已上传的图像）
    test_image_path = 'static/uploads/1.jpg'
    
    # 检查测试图像是否存在
    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像不存在: {test_image_path}")
        print("请先上传一张图像到 static/uploads/ 目录")
        exit(1)
    
    print(f"使用测试图像: {test_image_path}")
    print(f"目标栅格尺寸: {target_rows} x {target_cols}")
    print("-" * 50)
    
    # 执行转换
    try:
        converted_grid = enhanced_image_to_grid_python(test_image_path, target_rows, target_cols)
        
        if converted_grid is not None:
            print("\n=== 转换结果 ===")
            
            # 显示栅格数据统计
            unique, counts = np.unique(converted_grid, return_counts=True)
            terrain_names = {0: '空地', 1: '建筑物', 2: '树木', 3: '道路', 4: '水体'}
            
            print("地形类型统计:")
            total_cells = target_rows * target_cols
            for terrain_type, count in zip(unique, counts):
                percentage = (count / total_cells) * 100
                name = terrain_names.get(terrain_type, f'未知类型{terrain_type}')
                print(f"  {name}: {count} 个栅格 ({percentage:.1f}%)")
            
            # 显示栅格数据的前10x10部分
            print(f"\n栅格数据预览 (前10x10):")
            print(converted_grid[:10, :10])
            
            # 验证输出文件
            output_csv_path = os.path.join('static/uploads/', 'school_grid.csv')
            if os.path.exists(output_csv_path):
                print(f"\n✓ CSV文件已成功创建: {output_csv_path}")
                
                # 显示文件大小
                file_size = os.path.getsize(output_csv_path)
                print(f"文件大小: {file_size} 字节")
            else:
                print(f"\n✗ CSV文件创建失败")
            
            print("\n=== 测试完成 ===")
            
        else:
            print("✗ 图像转换失败")
            
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


