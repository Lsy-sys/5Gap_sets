import numpy as np
from PIL import Image
import os


def image_to_grid_python(image_path, target_grid_rows, target_grid_cols, output_dir='static/uploads/'):
    """
    将航拍图转换为栅格数据。
    根据预定义的颜色（例如空地、建筑物、树木、道路、水体），将图片中的像素映射到对应的数字代码。

    参数:
        image_path (str): 航拍图的文件路径 (例如 'path/to/your/aerial_image.png')
        target_grid_rows (int): 目标栅格的行数 (例如 68)
        target_grid_cols (int): 目标栅格的列数 (例如 95)
        output_dir (str): 输出 CSV 文件的目录。默认为 'static/uploads/'。

    返回:
        numpy.ndarray: 转换后的栅格数据矩阵，包含 0-4 的整数值。
                       同时会将此数据保存为 'school_grid.csv'。
    """
    try:
        # 1. 读取图像
        img = Image.open(image_path).convert('RGB')  # 确保转换为RGB格式
    except FileNotFoundError:
        # print(f"错误: 图像文件未找到: {image_path}")
        return None
    except Exception as e:
        # print(f"错误: 无法读取图像文件 {image_path}: {e}")
        return None

    img_np = np.array(img) / 255.0  # 将像素值归一化到 0-1 范围

    # 2. 定义颜色映射 (RGB 值, 归一化后)
    # 这些颜色值应与前端和原MATLAB脚本中描述的颜色保持一致
    # 0: 空地 - 白色 (RGB: 255, 255, 255)
    # 1: 建筑物 - 暗红色 (RGB: 139, 0, 0)
    # 2: 树木 - 森林绿 (RGB: 34, 139, 34)
    # 3: 道路 - 灰色 (RGB: 128, 128, 128)
    # 4: 水体 - 海军蓝 (RGB: 0, 0, 128)

    color_map = {
        0: np.array([1.0, 1.0, 1.0]),  # 空地 (白色)
        1: np.array([139 / 255, 0 / 255, 0 / 255]),  # 建筑物 (暗红色)
        2: np.array([34 / 255, 139 / 255, 34 / 255]),  # 树木 (森林绿)
        3: np.array([128 / 255, 128 / 255, 128 / 255]),  # 道路 (灰色)
        4: np.array([0 / 255, 0 / 255, 128 / 255])  # 水体 (海军蓝)
    }

    # 颜色容差 (允许颜色匹配时的轻微偏差)
    color_tolerance = 0.05  # 值越小，匹配越严格

    # 3. 初始化目标栅格数据
    grid_data = np.zeros((target_grid_rows, target_grid_cols), dtype=int)

    # 4. 图像缩放和颜色识别
    img_rows, img_cols, _ = img_np.shape

    row_scale = img_rows / target_grid_rows
    col_scale = img_cols / target_grid_cols

    # print("正在将图像转换为栅格数据...")

    for r in range(target_grid_rows):
        for c in range(target_grid_cols):
            # 计算当前栅格单元在原始图像中的对应区域
            start_row = int(round(r * row_scale))
            end_row = int(round((r + 1) * row_scale))
            start_col = int(round(c * col_scale))
            end_col = int(round((c + 1) * col_scale))

            # 确保索引在图像范围内
            start_row = max(0, start_row)
            end_row = min(img_rows, end_row)
            start_col = max(0, start_col)
            end_col = min(img_cols, end_col)

            # 提取该区域的像素块
            if start_row >= end_row or start_col >= end_col:
                # 区域无效，可能在图像边缘，或缩放计算导致，赋默认值
                grid_data[r, c] = 0  # 默认设置为 '空地'
                continue

            patch = img_np[start_row:end_row, start_col:end_col, :]

            # 计算该区域的平均颜色
            avg_color = np.mean(patch, axis=(0, 1))

            # 查找最接近的预定义颜色
            min_dist = np.inf
            matched_type = 0  # 默认空地

            for type_val, defined_color in color_map.items():
                # 计算颜色距离 (欧几里得距离)
                dist = np.linalg.norm(avg_color - defined_color)

                if dist < min_dist:
                    min_dist = dist
                    matched_type = type_val

            # 如果最佳匹配的颜色距离在容差范围内，则赋值
            if min_dist <= color_tolerance:
                grid_data[r, c] = matched_type
            else:
                # 如果颜色偏差过大，无法匹配任何预设类型，可以设置为默认值
                grid_data[r, c] = 0  # 默认为空地

    # print("图像转换完成。正在保存栅格数据到 school_grid.csv...")

    # 5. 保存栅格数据到 CSV 文件
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'school_grid.csv')
    np.savetxt(csv_filename, grid_data, delimiter=',', fmt='%d')

    # print(f"栅格数据已保存到: {csv_filename}")
    return grid_data


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
