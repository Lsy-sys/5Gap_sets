import numpy as np
from PIL import Image
import os
import cv2
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.spatial.distance import cdist


def calculate_texture_features(patch):
    """
    计算纹理特征（灰度共生矩阵的简化版本）
    """
    if patch.size == 0:
        return 0
    
    # 转换为灰度图
    if len(patch.shape) == 3:
        gray = np.mean(patch, axis=2)
    else:
        gray = patch
    
    # 计算梯度
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 返回梯度强度的平均值作为纹理特征
    return np.mean(gradient_magnitude)


def detect_edges(patch):
    """
    检测边缘特征
    """
    if patch.size == 0:
        return 0
    
    # 转换为灰度图
    if len(patch.shape) == 3:
        gray = np.mean(patch, axis=2)
    else:
        gray = patch
    
    # 使用Canny边缘检测
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    return np.sum(edges > 0) / edges.size


def rgb_to_hsv(rgb_color):
    """
    将RGB颜色转换为HSV
    """
    rgb_normalized = rgb_color / 255.0
    hsv = cv2.cvtColor(np.uint8([[rgb_normalized * 255]]), cv2.COLOR_RGB2HSV)
    return hsv[0][0] / np.array([180, 255, 255])


def enhanced_image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir='static/uploads/'):
    """
    增强版图像转栅格算法，使用多特征融合识别地形类型
    
    新增特性：
    1. HSV色彩空间分析
    2. 纹理特征分析
    3. 边缘检测
    4. 自适应颜色聚类
    5. 形态学后处理
    """
    try:
        # 1. 读取图像
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"错误: 无法读取图像文件 {image_path}: {e}")
        return None

    # 2. 图像预处理
    # 转换为HSV色彩空间
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # 3. 自适应颜色聚类（发现图像中的主要颜色）
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    
    # 4. 定义增强的颜色映射（包含HSV信息）
    terrain_definitions = {
        0: {  # 空地
            'name': '空地',
            'rgb': np.array([255, 255, 255]),
            'hsv': np.array([0, 0, 1.0]),
            'texture_threshold': 0.1,
            'edge_threshold': 0.05
        },
        1: {  # 建筑物
            'name': '建筑物',
            'rgb': np.array([139, 0, 0]),
            'hsv': np.array([0, 1.0, 0.55]),
            'texture_threshold': 0.3,
            'edge_threshold': 0.15
        },
        2: {  # 树木
            'name': '树木',
            'rgb': np.array([34, 139, 34]),
            'hsv': np.array([120/180, 0.76, 0.55]),
            'texture_threshold': 0.4,
            'edge_threshold': 0.08
        },
        3: {  # 道路
            'name': '道路',
            'rgb': np.array([128, 128, 128]),
            'hsv': np.array([0, 0, 0.5]),
            'texture_threshold': 0.2,
            'edge_threshold': 0.12
        },
        4: {  # 水体
            'name': '水体',
            'rgb': np.array([0, 0, 128]),
            'hsv': np.array([240/180, 1.0, 0.5]),
            'texture_threshold': 0.05,
            'edge_threshold': 0.03
        }
    }
    
    # 5. 初始化栅格数据
    grid_data = np.zeros((target_grid_rows, target_grid_cols), dtype=int)
    
    # 6. 图像缩放参数
    img_rows, img_cols, _ = img_np.shape
    row_scale = img_rows / target_grid_rows
    col_scale = img_cols / target_grid_cols
    
    print("正在使用增强算法转换图像...")
    
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
                continue
            
            # 提取图像块
            patch_rgb = img_np[start_row:end_row, start_col:end_col, :]
            patch_hsv = img_hsv[start_row:end_row, start_col:end_col, :]
            
            # 计算多特征
            avg_rgb = np.mean(patch_rgb, axis=(0, 1))
            avg_hsv = np.mean(patch_hsv, axis=(0, 1))
            texture_score = calculate_texture_features(patch_rgb)
            edge_score = detect_edges(patch_rgb)
            
            # 多特征融合评分
            best_score = -np.inf
            best_type = 0
            
            for type_val, terrain in terrain_definitions.items():
                # 1. RGB颜色相似度
                rgb_dist = np.linalg.norm(avg_rgb - terrain['rgb'])
                rgb_score = 1.0 / (1.0 + rgb_dist / 100.0)
                
                # 2. HSV颜色相似度
                hsv_dist = np.linalg.norm(avg_hsv - terrain['hsv'])
                hsv_score = 1.0 / (1.0 + hsv_dist)
                
                # 3. 纹理特征匹配
                texture_match = 1.0 / (1.0 + abs(texture_score - terrain['texture_threshold']))
                
                # 4. 边缘特征匹配
                edge_match = 1.0 / (1.0 + abs(edge_score - terrain['edge_threshold']))
                
                # 5. 与聚类中心的相似度
                cluster_dist = np.min(cdist([avg_rgb], dominant_colors, 'euclidean'))
                cluster_score = 1.0 / (1.0 + cluster_dist / 50.0)
                
                # 综合评分（加权平均）
                total_score = (0.3 * rgb_score + 
                              0.3 * hsv_score + 
                              0.2 * texture_match + 
                              0.1 * edge_match + 
                              0.1 * cluster_score)
                
                if total_score > best_score:
                    best_score = total_score
                    best_type = type_val
            
            grid_data[r, c] = best_type
    
    # 7. 形态学后处理（改善识别结果）
    grid_data = morphological_post_processing(grid_data)
    
    # 8. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'school_grid.csv')
    np.savetxt(csv_filename, grid_data, delimiter=',', fmt='%d')
    
    print(f"增强算法转换完成，结果已保存到: {csv_filename}")
    return grid_data


def morphological_post_processing(grid_data):
    """
    形态学后处理，改善识别结果
    """
    # 创建结构元素
    kernel = np.ones((3, 3), np.uint8)
    
    # 对每种地形类型进行形态学操作
    for terrain_type in range(5):
        # 创建二值掩码
        mask = (grid_data == terrain_type).astype(np.uint8)
        
        # 小面积噪声去除
        if terrain_type in [1, 2]:  # 建筑物和树木
            # 去除孤立的像素
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # 填充小洞
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif terrain_type == 3:  # 道路
            # 连接断开的道路
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 更新栅格数据
        grid_data[mask == 1] = terrain_type
    
    return grid_data


def image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir='static/uploads/'):
    """
    主函数，调用增强版算法
    """
    return enhanced_image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir)


if __name__ == '__main__':
    # 示例用法:
    # 假设你有一张名为 'test_aerial_image.png' 的航拍图在当前目录下
    # 你需要准备一张符合颜色编码的图片进行测试

    # 创建一个简单的虚拟图片用于测试 (例如 255x255 像素)
    # 1. 纯白背景
    test_img = Image.new('RGB', (255, 255), color='white')
    # 2. 画一个红色的正方形代表建筑物
    for i in range(50, 100):
        for j in range(50, 100):
            test_img.putpixel((i, j), (139, 0, 0))  # 暗红色
    # 3. 画一个绿色的圆代表树木
    for i in range(150, 200):
        for j in range(150, 200):
            if (i - 175) ** 2 + (j - 175) ** 2 < 20 ** 2:
                test_img.putpixel((i, j), (34, 139, 34))  # 森林绿
    # 4. 画一条灰色的线代表道路
    for i in range(0, 255):
        for j in range(120, 130):
            test_img.putpixel((i, j), (128, 128, 128))  # 灰色

    test_image_path = 'test_aerial_image.png'
    test_img.save(test_image_path)

    # 定义目标栅格尺寸 (与zuizhongban.m中的grid_size匹配)
    target_rows = 68
    target_cols = 95

    converted_grid = image_to_grid_python(test_image_path, target_rows, target_cols)

    if converted_grid is not None:
        print("\n转换后的栅格数据的前5行5列:")
        print(converted_grid[:5, :5])

        # 验证输出文件是否存在
        output_csv_path = os.path.join('static/uploads/', 'school_grid.csv')
        if os.path.exists(output_csv_path):
            print(f"\nCSV文件已成功创建: {output_csv_path}")
