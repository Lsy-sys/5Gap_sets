import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def analyze_image(image_path):
    """
    分析图像的颜色分布和特征
    """
    print("=== 图像分析 ===")
    
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    print(f"图像尺寸: {img_np.shape}")
    
    # 转换为HSV
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # 分析RGB分布
    print("\n=== RGB颜色分析 ===")
    avg_r = np.mean(img_np[:,:,0])
    avg_g = np.mean(img_np[:,:,1])
    avg_b = np.mean(img_np[:,:,2])
    print(f"平均RGB: R={avg_r:.1f}, G={avg_g:.1f}, B={avg_b:.1f}")
    
    # 分析HSV分布
    print("\n=== HSV颜色分析 ===")
    avg_h = np.mean(img_hsv[:,:,0])
    avg_s = np.mean(img_hsv[:,:,1])
    avg_v = np.mean(img_hsv[:,:,2])
    print(f"平均HSV: H={avg_h:.1f}, S={avg_s:.1f}, V={avg_v:.1f}")
    
    # 分析颜色分布
    print("\n=== 颜色分布分析 ===")
    
    # 计算颜色强度
    intensity = (avg_r + avg_g + avg_b) / 3
    print(f"平均亮度: {intensity:.1f}")
    
    # 分析主要颜色
    if avg_r > avg_g + 20 and avg_r > avg_b + 20:
        print("主要颜色: 红色系")
    elif avg_g > avg_r + 20 and avg_g > avg_b + 20:
        print("主要颜色: 绿色系")
    elif avg_b > avg_r + 20 and avg_b > avg_g + 20:
        print("主要颜色: 蓝色系")
    else:
        print("主要颜色: 灰色/中性色")
    
    # 分析饱和度
    if avg_s < 50:
        print("饱和度: 低 (灰色调)")
    elif avg_s < 100:
        print("饱和度: 中等")
    else:
        print("饱和度: 高")
    
    # 分析亮度
    if avg_v < 100:
        print("亮度: 暗")
    elif avg_v < 150:
        print("亮度: 中等")
    else:
        print("亮度: 亮")
    
    # 详细分析不同亮度区域
    print("\n=== 亮度分布分析 ===")
    gray = np.mean(img_np, axis=2)
    
    # 统计不同亮度范围的像素数量
    bright_pixels = np.sum(gray > 150)  # 很亮的区域
    medium_pixels = np.sum((gray >= 80) & (gray <= 150))  # 中等亮度
    dark_pixels = np.sum(gray < 80)  # 暗的区域
    
    total_pixels = gray.size
    print(f"很亮区域 (>150): {bright_pixels} 像素 ({bright_pixels/total_pixels*100:.1f}%)")
    print(f"中等亮度 (80-150): {medium_pixels} 像素 ({medium_pixels/total_pixels*100:.1f}%)")
    print(f"暗区域 (<80): {dark_pixels} 像素 ({dark_pixels/total_pixels*100:.1f}%)")
    
    # 分析纹理和边缘特征
    print("\n=== 纹理和边缘分析 ===")
    
    # 计算整体纹理
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_texture = np.mean(gradient_magnitude)
    print(f"平均纹理强度: {avg_texture:.2f}")
    
    # 计算边缘密度
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    print(f"边缘密度: {edge_density:.4f}")
    
    # 分析标准差分布
    std_dev = np.std(gray)
    print(f"整体标准差: {std_dev:.2f}")
    
    # 采样一些区域进行分析
    print("\n=== 区域采样分析 ===")
    
    # 采样几个区域
    regions = [
        (0, 0, 100, 100, "左上角"),
        (img_np.shape[1]//2-50, img_np.shape[0]//2-50, 100, 100, "中心"),
        (img_np.shape[1]-100, img_np.shape[0]-100, 100, 100, "右下角")
    ]
    
    for x, y, w, h, name in regions:
        region = img_np[y:y+h, x:x+w]
        region_hsv = img_hsv[y:y+h, x:x+w]
        region_gray = gray[y:y+h, x:x+w]
        
        r_avg = np.mean(region[:,:,0])
        g_avg = np.mean(region[:,:,1])
        b_avg = np.mean(region[:,:,2])
        h_avg = np.mean(region_hsv[:,:,0])
        s_avg = np.mean(region_hsv[:,:,1])
        v_avg = np.mean(region_hsv[:,:,2])
        
        intensity_avg = np.mean(region_gray)
        std_avg = np.std(region_gray)
        
        print(f"{name}区域:")
        print(f"  RGB:({r_avg:.1f},{g_avg:.1f},{b_avg:.1f}) HSV:({h_avg:.1f},{s_avg:.1f},{v_avg:.1f})")
        print(f"  亮度:{intensity_avg:.1f} 标准差:{std_avg:.1f}")

if __name__ == "__main__":
    analyze_image("static/uploads/1.jpg") 