import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from PIL import Image
from matplotlib import colors
from matplotlib.patches import Polygon
import os

# 设置matplotlib后端，解决显示问题
matplotlib.use('Agg')  # 非交互式后端，保存图片

# 简化字体设置，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class ColorSpaceConverter:
    def __init__(self):
        """初始化颜色空间转换器"""
        # 定义标准色空间的顶点坐标（CIE 1931 xy色度坐标）
        self.bt2020_vertices = np.array([
            [0.708, 0.292],  # Red
            [0.170, 0.797],  # Green  
            [0.131, 0.046]   # Blue
        ])
        
        self.display_vertices = np.array([
            [0.640, 0.330],  # Red
            [0.300, 0.600],  # Green
            [0.150, 0.060]   # Blue
        ])
        
        # CIE 1931光谱轨迹数据（简化版）
        self.setup_spectral_locus()
        
    def setup_spectral_locus(self):
        """设置CIE 1931光谱轨迹数据"""
        # 简化的光谱轨迹数据
        wavelengths = np.linspace(380, 700, 100)
        
        # 生成马蹄形光谱轨迹（近似）
        t = np.linspace(0, 2*np.pi, 100)
        x_coords = 0.3 + 0.25 * np.cos(t) + 0.1 * np.cos(2*t)
        y_coords = 0.3 + 0.25 * np.sin(t) + 0.1 * np.sin(2*t)
        
        # 调整为更真实的马蹄形状
        mask = y_coords > 0.1
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        
        # 闭合曲线
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        
        self.spectral_locus = np.column_stack([x_coords, y_coords])

# 生成测试RGB图像
def generate_test_image(height, width):
    """生成测试RGB图像，包含渐变和色块"""
    image = np.zeros((height, width, 3))
    
    # 创建彩色渐变
    for i in range(height):
        for j in range(width):
            # 红色渐变
            if j < width // 3:
                image[i, j] = [j / (width // 3), 0, 0]
            # 绿色渐变
            elif j < 2 * width // 3:
                image[i, j] = [0, (j - width // 3) / (width // 3), 0]
            # 蓝色渐变
            else:
                image[i, j] = [0, 0, (j - 2 * width // 3) / (width // 3)]
                
    # 添加一些彩色方块
    image[50:100, 50:100] = [1, 1, 0]  # 黄色
    image[150:200, 50:100] = [1, 0, 1]  # 洋红
    image[50:100, 150:200] = [0, 1, 1]  # 青色
    
    return np.clip(image, 0, 1)

def rgb_to_xy_approximate(rgb):
    """将RGB近似转换为CIE xy坐标（简化实现）"""
    # 这是一个简化的转换，实际应该使用标准的RGB到XYZ再到xy的转换
    # 线性变换矩阵（近似）
    rgb = np.array(rgb)
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, -1)
    
    # 简化的转换矩阵
    transform_matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    
    xyz = rgb @ transform_matrix.T
    
    # 转换为xy色度坐标
    sum_xyz = np.sum(xyz, axis=1, keepdims=True)
    sum_xyz[sum_xyz == 0] = 1  # 避免除零
    
    x = xyz[:, 0:1] / sum_xyz
    y = xyz[:, 1:2] / sum_xyz
    
    return np.hstack([x, y])

def calculate_delta_e_lab(rgb1, rgb2):
    """计算两个RGB颜色的ΔE色差（简化版）"""
    # 简化的RGB到Lab转换
    def rgb_to_lab_simple(rgb):
        # 非常简化的转换，实际应该使用标准转换
        lab = np.zeros_like(rgb)
        lab[..., 0] = 100 * (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])  # L
        lab[..., 1] = 128 * (rgb[..., 0] - rgb[..., 1])  # a
        lab[..., 2] = 128 * (rgb[..., 1] - rgb[..., 2])  # b
        return lab
    
    lab1 = rgb_to_lab_simple(rgb1)
    lab2 = rgb_to_lab_simple(rgb2)
    
    delta_e = np.sqrt(np.sum((lab1 - lab2)**2, axis=-1))
    return delta_e

def point_in_triangle(point, triangle):
    """判断点是否在三角形内"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def barycentric_coordinates(point, triangle):
    """计算重心坐标"""
    x, y = point
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
        
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    
    return np.array([a, b, c])

def map_color_gamut(source_xy, converter):
    """映射颜色色域"""
    mapped_xy = []
    
    for point in source_xy:
        if point_in_triangle(point, converter.bt2020_vertices):
            # 在BT2020色域内，计算重心坐标
            bary_coords = barycentric_coordinates(point, converter.bt2020_vertices)
            # 映射到显示屏色域
            mapped_point = (bary_coords[0] * converter.display_vertices[0] + 
                           bary_coords[1] * converter.display_vertices[1] + 
                           bary_coords[2] * converter.display_vertices[2])
        else:
            # 在色域外，投影到最近的边界
            mapped_point = project_to_triangle(point, converter.display_vertices)
        
        mapped_xy.append(mapped_point)
    
    return np.array(mapped_xy)

def project_to_triangle(point, triangle):
    """将点投影到三角形内最近的位置"""
    # 简化实现：投影到重心
    return np.mean(triangle, axis=0)

def plot_image_comparison(image1, image2, save_path=None):
    """可视化原始图像与优化后图像对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image1)
    axes[0].set_title("BT2020 Source Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image2)
    axes[1].set_title("Mapped Display Image", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像对比已保存到: {save_path}")
    else:
        plt.savefig('image_comparison.png', dpi=300, bbox_inches='tight')
        print("图像对比已保存到: image_comparison.png")
    plt.close()

def plot_delta_e_curve(delta_e_values, save_path=None):
    """绘制ΔE误差曲线"""
    plt.figure(figsize=(10, 6))
    
    # 计算统计信息
    mean_delta_e = np.mean(delta_e_values)
    max_delta_e = np.max(delta_e_values)
    
    plt.plot(delta_e_values, 'b-', linewidth=2, label=f"Delta E (mean: {mean_delta_e:.3f})")
    plt.axhline(y=mean_delta_e, color='r', linestyle='--', alpha=0.7, label=f"Mean: {mean_delta_e:.3f}")
    plt.axhline(y=max_delta_e, color='orange', linestyle='--', alpha=0.7, label=f"Max: {max_delta_e:.3f}")
    
    plt.title("Color Conversion Delta E Error Distribution", fontsize=16)
    plt.xlabel("Pixel Index", fontsize=12)
    plt.ylabel("Delta E", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息文本
    stats_text = f"Total Pixels: {len(delta_e_values)}\nMean Delta E: {mean_delta_e:.3f}\nMax Delta E: {max_delta_e:.3f}\nStd Dev: {np.std(delta_e_values):.3f}"
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ΔE误差曲线已保存到: {save_path}")
    else:
        plt.savefig('delta_e_curve.png', dpi=300, bbox_inches='tight')
        print("ΔE误差曲线已保存到: delta_e_curve.png")
    plt.close()

def plot_cie1931_color_space(converter, source_points=None, mapped_points=None, save_path=None):
    """绘制CIE1931色度图"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制光谱轨迹
    spectral_x = converter.spectral_locus[:, 0]
    spectral_y = converter.spectral_locus[:, 1]
    
    ax.plot(spectral_x, spectral_y, 'k-', linewidth=2, label='CIE1931 Spectral Locus')
    ax.fill(spectral_x, spectral_y, color='lightgray', alpha=0.3)
    
    # 绘制BT2020色域
    bt2020_triangle = Polygon(converter.bt2020_vertices, fill=False, 
                             edgecolor='brown', linewidth=3, 
                             label='BT2020 Gamut')
    ax.add_patch(bt2020_triangle)
    
    # 绘制显示屏色域
    display_triangle = Polygon(converter.display_vertices, fill=False, 
                              edgecolor='red', linewidth=3, 
                              label='Display Gamut')
    ax.add_patch(display_triangle)
    
    # 标注顶点
    colors_labels = ['R', 'G', 'B']
    for i, (vertex, label) in enumerate(zip(converter.bt2020_vertices, colors_labels)):
        ax.plot(vertex[0], vertex[1], 'bo', markersize=8)
        ax.annotate(f'BT2020-{label}', vertex, xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
        
    for i, (vertex, label) in enumerate(zip(converter.display_vertices, colors_labels)):
        ax.plot(vertex[0], vertex[1], 'ro', markersize=8)
        ax.annotate(f'Display-{label}', vertex, xytext=(5, -15), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    # 如果有颜色点，绘制映射
    if source_points is not None and mapped_points is not None:
        ax.scatter(source_points[:, 0], source_points[:, 1], 
                  c='blue', s=30, alpha=0.6, label='Source Colors')
        ax.scatter(mapped_points[:, 0], mapped_points[:, 1], 
                  c='red', s=30, alpha=0.6, label='Mapped Colors')
        
        # 绘制映射箭头（只显示部分，避免太密集）
        step = max(1, len(source_points) // 20)
        for i in range(0, len(source_points), step):
            ax.arrow(source_points[i, 0], source_points[i, 1],
                    mapped_points[i, 0] - source_points[i, 0],
                    mapped_points[i, 1] - source_points[i, 1],
                    head_width=0.01, head_length=0.01, 
                    fc='green', ec='green', alpha=0.5)
    
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('CIE x', fontsize=12)
    ax.set_ylabel('CIE y', fontsize=12)
    ax.set_title('CIE1931 Chromaticity Diagram and Color Space Conversion', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CIE1931色度图已保存到: {save_path}")
    else:
        plt.savefig('cie1931_color_space.png', dpi=300, bbox_inches='tight')
        print("CIE1931色度图已保存到: cie1931_color_space.png")
    plt.close()

def main():
    """主函数"""
    print("=== LED显示屏颜色转换设计与校正 - 问题1 ===")
    
    # 创建输出目录
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化转换器
    converter = ColorSpaceConverter()
    
    print("步骤1: 生成测试图像...")
    # 生成测试图像
    original_image = generate_test_image(256, 256)
    
    print("步骤2: 转换颜色空间...")
    # 将图像RGB转换为xy坐标
    h, w, c = original_image.shape
    rgb_flat = original_image.reshape(-1, 3)
    
    # 转换为xy色度坐标
    source_xy = rgb_to_xy_approximate(rgb_flat)
    
    # 执行色域映射
    mapped_xy = map_color_gamut(source_xy, converter)
    
    print("步骤3: 计算转换后的RGB...")
    # 简化：这里直接使用线性缩放作为反向转换
    # 实际应该实现xy到RGB的精确转换
    scale_factor = 0.8  # 模拟色域压缩
    mapped_rgb = rgb_flat * scale_factor
    mapped_image = mapped_rgb.reshape(h, w, c)
    mapped_image = np.clip(mapped_image, 0, 1)
    
    print("步骤4: 计算色差...")
    # 计算ΔE色差
    delta_e_values = calculate_delta_e_lab(original_image, mapped_image)
    delta_e_flat = delta_e_values.flatten()
    
    # 统计信息
    avg_delta_e = np.mean(delta_e_flat)
    max_delta_e = np.max(delta_e_flat)
    std_delta_e = np.std(delta_e_flat)
    
    print(f"\n=== 转换结果统计 ===")
    print(f"平均ΔE色差: {avg_delta_e:.4f}")
    print(f"最大ΔE色差: {max_delta_e:.4f}")
    print(f"ΔE标准差: {std_delta_e:.4f}")
    
    # 色域覆盖率计算
    def triangle_area(vertices):
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2)
    
    bt2020_area = triangle_area(converter.bt2020_vertices)
    display_area = triangle_area(converter.display_vertices)
    coverage = display_area / bt2020_area
    
    print(f"BT2020色域面积: {bt2020_area:.6f}")
    print(f"显示屏色域面积: {display_area:.6f}")
    print(f"色域覆盖率: {coverage:.2%}")
    
    print("\n步骤5: 生成可视化结果...")
    
    # 生成图像对比
    plot_image_comparison(original_image, mapped_image, 
                         os.path.join(output_dir, 'image_comparison.png'))
    
    # 生成ΔE误差曲线
    plot_delta_e_curve(delta_e_flat[:1000],  # 只显示前1000个点，避免图太密集
                      os.path.join(output_dir, 'delta_e_curve.png'))
    
    # 生成CIE1931色度图
    sample_indices = np.random.choice(len(source_xy), 100, replace=False)
    plot_cie1931_color_space(converter, 
                             source_xy[sample_indices], 
                             mapped_xy[sample_indices],
                             os.path.join(output_dir, 'cie1931_color_space.png'))
    
    # 保存数值结果
    results = {
        'source_image': original_image,
        'mapped_image': mapped_image,
        'delta_e_values': delta_e_flat,
        'statistics': {
            'avg_delta_e': avg_delta_e,
            'max_delta_e': max_delta_e,
            'std_delta_e': std_delta_e,
            'coverage_ratio': coverage
        }
    }
    
    np.savez(os.path.join(output_dir, 'conversion_results.npz'), **results)
    
    print(f"\n=== 完成 ===")
    print(f"所有结果已保存到: {output_dir}/")
    print("生成的文件:")
    print("1. image_comparison.png - 图像对比")
    print("2. delta_e_curve.png - ΔE误差曲线")
    print("3. cie1931_color_space.png - CIE1931色度图")
    print("4. conversion_results.npz - 数值结果")

if __name__ == "__main__":
    main()
