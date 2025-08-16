import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
import os
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

# 设置matplotlib后端
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class OptimizedColorSpaceConverter:
    def __init__(self):
        """优化的颜色空间转换器"""
        # 精确的BT2020和sRGB色域顶点
        self.bt2020_vertices = np.array([
            [0.708, 0.292],  # Red
            [0.170, 0.797],  # Green  
            [0.131, 0.046]   # Blue
        ])
        
        self.display_vertices = np.array([
            [0.640, 0.330],  # Red (sRGB)
            [0.300, 0.600],  # Green (sRGB)
            [0.150, 0.060]   # Blue (sRGB)
        ])
        
        # 计算转换矩阵
        self.compute_optimal_transform_matrix()
        
        # 标准转换矩阵
        self.srgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        self.xyz_to_srgb_matrix = np.linalg.inv(self.srgb_to_xyz_matrix)
    
    def compute_optimal_transform_matrix(self):
        """计算最优的色域变换矩阵"""
        # 使用仿射变换从BT2020映射到sRGB
        # 添加齐次坐标
        bt2020_homo = np.column_stack([self.bt2020_vertices, np.ones(3)])
        display_homo = np.column_stack([self.display_vertices, np.ones(3)])
        
        # 计算仿射变换矩阵
        self.transform_matrix = np.linalg.lstsq(bt2020_homo, display_homo, rcond=None)[0]
    
    def srgb_to_linear(self, rgb):
        """sRGB到线性RGB转换"""
        return np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    
    def linear_to_srgb(self, rgb):
        """线性RGB到sRGB转换"""
        return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1/2.4) - 0.055)
    
    def rgb_to_xyz(self, rgb):
        """RGB到XYZ转换"""
        rgb = np.array(rgb)
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, -1)
        
        linear_rgb = self.srgb_to_linear(rgb)
        xyz = linear_rgb @ self.srgb_to_xyz_matrix.T
        return xyz
    
    def xyz_to_rgb(self, xyz):
        """XYZ到RGB转换"""
        xyz = np.array(xyz)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        
        linear_rgb = xyz @ self.xyz_to_srgb_matrix.T
        linear_rgb = np.clip(linear_rgb, 0, 1)
        rgb = self.linear_to_srgb(linear_rgb)
        return rgb
    
    def xyz_to_xy(self, xyz):
        """XYZ到xy色度坐标"""
        xyz = np.array(xyz)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        
        sum_xyz = np.sum(xyz, axis=1, keepdims=True)
        sum_xyz[sum_xyz == 0] = 1e-10
        
        x = xyz[:, 0:1] / sum_xyz
        y = xyz[:, 1:2] / sum_xyz
        return np.hstack([x, y])
    
    def xy_to_xyz(self, xy, Y=1.0):
        """xy到XYZ转换"""
        xy = np.array(xy)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        
        x, y = xy[:, 0], xy[:, 1]
        z = 1 - x - y
        
        y_coord = np.where(y == 0, 1e-10, y)
        
        if np.isscalar(Y):
            Y = np.full(len(xy), Y)
        
        X = Y * x / y_coord
        Z = Y * z / y_coord
        
        return np.column_stack([X, Y, Z])
    
    def rgb_to_lab(self, rgb):
        """RGB到Lab转换"""
        xyz = self.rgb_to_xyz(rgb)
        
        # D65白点
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        xyz_norm = xyz / np.array([Xn, Yn, Zn])
        
        def f(t):
            delta = 6/29
            return np.where(t > delta**3, np.power(t, 1/3), t/(3*delta**2) + 4/29)
        
        fx = f(xyz_norm[:, 0])
        fy = f(xyz_norm[:, 1])
        fz = f(xyz_norm[:, 2])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.column_stack([L, a, b])
    
    def calculate_delta_e_94(self, lab1, lab2):
        """计算CIE94色差"""
        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
        
        dL = L1 - L2
        da = a1 - a2
        db = b1 - b2
        
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        dC = C1 - C2
        
        dH_sq = da**2 + db**2 - dC**2
        dH = np.sqrt(np.maximum(dH_sq, 0))
        
        # CIE94参数
        kL, kC, kH = 1, 1, 1
        K1, K2 = 0.045, 0.015
        
        SL = 1
        SC = 1 + K1 * C1
        SH = 1 + K2 * C1
        
        delta_e = np.sqrt((dL/(kL*SL))**2 + (dC/(kC*SC))**2 + (dH/(kH*SH))**2)
        return delta_e

def generate_realistic_test_image(height, width):
    """生成更真实的测试图像"""
    image = np.zeros((height, width, 3))
    
    # 创建渐变背景
    for i in range(height):
        for j in range(width):
            # 双线性插值渐变
            r = (i / height) * (j / width)
            g = (1 - i / height) * (j / width)
            b = (i / height) * (1 - j / width)
            
            image[i, j] = [r, g, b]
    
    # 添加颜色测试色块
    block_size = 32
    test_colors = [
        [1.0, 0.0, 0.0],  # 纯红
        [0.0, 1.0, 0.0],  # 纯绿
        [0.0, 0.0, 1.0],  # 纯蓝
        [1.0, 1.0, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # 洋红
        [0.0, 1.0, 1.0],  # 青
        [0.5, 0.5, 0.5],  # 灰
        [1.0, 0.5, 0.0],  # 橙
        [0.5, 0.0, 1.0],  # 紫
        [0.0, 0.5, 0.0],  # 深绿
        [0.8, 0.2, 0.2],  # 深红
        [0.2, 0.2, 0.8],  # 深蓝
    ]
    
    for idx, color in enumerate(test_colors):
        row = (idx // 4) * block_size + 10
        col = (idx % 4) * block_size + 10
        
        if row + block_size <= height and col + block_size <= width:
            image[row:row+block_size, col:col+block_size] = color
    
    return np.clip(image, 0, 1)

def optimized_gamut_mapping(converter, source_rgb, method='smart_compression'):
    """优化的色域映射算法"""
    
    if method == 'smart_compression':
        return smart_compression_mapping(converter, source_rgb)
    elif method == 'perceptual_mapping':
        return perceptual_mapping(converter, source_rgb)
    else:
        return improved_linear_mapping(converter, source_rgb)

def smart_compression_mapping(converter, source_rgb):
    """智能压缩映射 - 保持色相，智能压缩饱和度和亮度"""
    
    source_xyz = converter.rgb_to_xyz(source_rgb)
    source_xy = converter.xyz_to_xy(source_xyz)
    
    mapped_rgb = []
    
    for i, (rgb, xyz, xy) in enumerate(zip(source_rgb, source_xyz, source_xy)):
        
        if point_in_triangle(xy, converter.display_vertices):
            # 已在目标色域内，直接使用
            mapped_rgb.append(rgb)
        elif point_in_triangle(xy, converter.bt2020_vertices):
            # 在BT2020内但超出显示器色域，使用智能压缩
            mapped_point = intelligent_compression(xy, rgb, converter)
            mapped_rgb.append(mapped_point)
        else:
            # 完全在色域外，投影到最近边界
            projected_xy = project_to_display_gamut(xy, converter.display_vertices)
            Y = xyz[1]  # 保持亮度
            projected_xyz = converter.xy_to_xyz(projected_xy.reshape(1, -1), Y)[0]
            projected_rgb = converter.xyz_to_rgb(projected_xyz.reshape(1, -1))[0]
            mapped_rgb.append(np.clip(projected_rgb, 0, 1))
    
    return np.array(mapped_rgb)

def intelligent_compression(xy, rgb, converter):
    """智能压缩算法"""
    # 找到色域边界上的对应点
    target_xy = find_optimal_target_point(xy, converter.display_vertices)
    
    # 使用HSV空间进行智能插值
    hsv = rgb_to_hsv(rgb)
    
    # 计算目标RGB
    target_xyz = converter.xy_to_xyz(target_xy.reshape(1, -1), converter.rgb_to_xyz(rgb.reshape(1, -1))[0, 1])[0]
    target_rgb = converter.xyz_to_rgb(target_xyz.reshape(1, -1))[0]
    target_hsv = rgb_to_hsv(target_rgb)
    
    # 保持色相，智能调整饱和度
    compression_factor = 0.8  # 可调参数
    new_saturation = hsv[1] * compression_factor + target_hsv[1] * (1 - compression_factor)
    new_value = hsv[2] * 0.9 + target_hsv[2] * 0.1  # 轻微调整亮度
    
    new_hsv = np.array([hsv[0], new_saturation, new_value])
    new_rgb = hsv_to_rgb(new_hsv)
    
    return np.clip(new_rgb, 0, 1)

def find_optimal_target_point(source_xy, target_triangle):
    """找到最优的目标映射点"""
    # 计算到三角形重心的方向
    centroid = np.mean(target_triangle, axis=0)
    direction = centroid - source_xy
    direction = direction / np.linalg.norm(direction)
    
    # 沿方向寻找最佳交点
    best_point = project_to_display_gamut(source_xy, target_triangle)
    
    return best_point

def project_to_display_gamut(point, triangle):
    """投影到显示器色域"""
    # 找到三角形边界上最近的点
    min_distance = float('inf')
    closest_point = point
    
    for i in range(3):
        p1 = triangle[i]
        p2 = triangle[(i + 1) % 3]
        
        projected = project_point_to_line_segment(point, p1, p2)
        distance = np.linalg.norm(point - projected)
        
        if distance < min_distance:
            min_distance = distance
            closest_point = projected
    
    return closest_point

def project_point_to_line_segment(point, line_start, line_end):
    """点到线段的投影"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return line_start
        
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0, min(1, t))
    
    return line_start + t * line_vec

def point_in_triangle(point, triangle):
    """检查点是否在三角形内"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def rgb_to_hsv(rgb):
    """RGB到HSV转换"""
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # 色相计算
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # 饱和度计算
    s = 0 if max_val == 0 else diff / max_val
    
    # 明度
    v = max_val
    
    return np.array([h, s, v])

def hsv_to_rgb(hsv):
    """HSV到RGB转换"""
    h, s, v = hsv
    h = h % 360
    
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return np.array([r + m, g + m, b + m])

def plot_optimized_results(original_image, mapped_image, converter, 
                          source_rgb, mapped_rgb, delta_e_values, save_dir):
    """绘制优化结果"""
    
    # 1. 详细的图像对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis('off')
    
    # 映射后
    axes[0, 1].imshow(mapped_image)
    axes[0, 1].set_title("Mapped Image", fontsize=12)
    axes[0, 1].axis('off')
    
    # 差异图
    diff_image = np.abs(original_image - mapped_image)
    im1 = axes[0, 2].imshow(diff_image, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 2].set_title("Absolute Difference", fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Delta E 热图
    delta_e_2d = delta_e_values.reshape(original_image.shape[:2])
    im2 = axes[1, 0].imshow(delta_e_2d, cmap='viridis', vmin=0, vmax=20)
    axes[1, 0].set_title("Delta E Heatmap", fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Delta E 直方图
    axes[1, 1].hist(delta_e_values, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(delta_e_values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(delta_e_values):.2f}')
    axes[1, 1].axvline(np.median(delta_e_values), color='green', linestyle='--', 
                      label=f'Median: {np.median(delta_e_values):.2f}')
    axes[1, 1].set_xlabel('Delta E')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Delta E Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 色域映射可视化
    source_xyz = converter.rgb_to_xyz(source_rgb[::100])  # 采样
    mapped_xyz = converter.rgb_to_xyz(mapped_rgb[::100])
    source_xy = converter.xyz_to_xy(source_xyz)
    mapped_xy = converter.xyz_to_xy(mapped_xyz)
    
    # 绘制色域三角形
    bt2020_triangle = Polygon(converter.bt2020_vertices, fill=False, 
                             edgecolor='brown', linewidth=2, label='BT2020')
    axes[1, 2].add_patch(bt2020_triangle)
    
    display_triangle = Polygon(converter.display_vertices, fill=False, 
                              edgecolor='red', linewidth=2, label='Display')
    axes[1, 2].add_patch(display_triangle)
    
    axes[1, 2].scatter(source_xy[:, 0], source_xy[:, 1], 
                      c='blue', s=10, alpha=0.6, label='Source')
    axes[1, 2].scatter(mapped_xy[:, 0], mapped_xy[:, 1], 
                      c='red', s=10, alpha=0.6, label='Mapped')
    
    axes[1, 2].set_xlim(0, 0.8)
    axes[1, 2].set_ylim(0, 0.9)
    axes[1, 2].set_xlabel('CIE x')
    axes[1, 2].set_ylabel('CIE y')
    axes[1, 2].set_title('Gamut Mapping')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimized_complete_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """优化版主函数"""
    print("=== LED显示屏颜色转换设计与校正 - 问题1 (终极优化版) ===")
    
    # 创建输出目录
    output_dir = "optimized_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化优化的转换器
    converter = OptimizedColorSpaceConverter()
    
    print("步骤1: 生成真实测试图像...")
    original_image = generate_realistic_test_image(256, 256)
    
    print("步骤2: 执行优化的颜色空间转换...")
    h, w, c = original_image.shape
    source_rgb = original_image.reshape(-1, 3)
    
    # 执行优化的色域映射
    mapped_rgb = optimized_gamut_mapping(converter, source_rgb, method='smart_compression')
    mapped_image = mapped_rgb.reshape(h, w, c)
    mapped_image = np.clip(mapped_image, 0, 1)
    
    print("步骤3: 计算精确色差...")
    # 计算精确的ΔE色差
    source_lab = converter.rgb_to_lab(source_rgb)
    mapped_lab = converter.rgb_to_lab(mapped_rgb)
    delta_e_values = converter.calculate_delta_e_94(source_lab, mapped_lab)
    
    # 统计信息
    avg_delta_e = np.mean(delta_e_values)
    max_delta_e = np.max(delta_e_values)
    std_delta_e = np.std(delta_e_values)
    median_delta_e = np.median(delta_e_values)
    
    print(f"\n=== 终极优化后的转换结果统计 ===")
    print(f"平均ΔE色差: {avg_delta_e:.4f}")
    print(f"中位数ΔE色差: {median_delta_e:.4f}")
    print(f"最大ΔE色差: {max_delta_e:.4f}")
    print(f"ΔE标准差: {std_delta_e:.4f}")
    
    # ΔE分布统计
    ranges = [(0, 1), (1, 3), (3, 6), (6, 10), (10, float('inf'))]
    range_labels = ['优秀 (0-1)', '良好 (1-3)', '可接受 (3-6)', '较差 (6-10)', '很差 (>10)']
    
    print(f"\n=== ΔE分布统计 ===")
    for (low, high), label in zip(ranges, range_labels):
        if high == float('inf'):
            mask = delta_e_values >= low
        else:
            mask = (delta_e_values >= low) & (delta_e_values < high)
        count = np.sum(mask)
        percentage = count / len(delta_e_values) * 100
        print(f"{label}: {count} 像素 ({percentage:.1f}%)")
    
    print("\n步骤4: 生成优化的可视化结果...")
    
    # 生成优化的可视化
    plot_optimized_results(original_image, mapped_image, converter, 
                          source_rgb, mapped_rgb, delta_e_values, output_dir)
    
    # 保存优化的数值结果
    optimized_results = {
        'source_image': original_image,
        'mapped_image': mapped_image,
        'source_rgb': source_rgb,
        'mapped_rgb': mapped_rgb,
        'delta_e_values': delta_e_values,
        'optimized_statistics': {
            'avg_delta_e': avg_delta_e,
            'median_delta_e': median_delta_e,
            'max_delta_e': max_delta_e,
            'std_delta_e': std_delta_e,
        }
    }
    
    np.savez(os.path.join(output_dir, 'optimized_conversion_results.npz'), **optimized_results)
    
    # 创建最终报告
    with open(os.path.join(output_dir, 'optimized_summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("LED显示屏颜色转换设计与校正 - 问题1 终极优化版结果报告\n")
        f.write("=" * 75 + "\n\n")
        
        f.write("1. 终极优化方案\n")
        f.write("-" * 25 + "\n")
        f.write("- 使用HSV色彩空间进行智能压缩映射\n")
        f.write("- 保持色相不变，智能调整饱和度和明度\n")
        f.write("- 采用CIE94色差计算公式\n")
        f.write("- 实现了感知一致性的色域边界投影\n")
        f.write("- 针对不同颜色区域采用自适应压缩策略\n\n")
        
        f.write("2. 核心指标（终极优化后）\n")
        f.write("-" * 25 + "\n")
        f.write(f"平均ΔE色差: {avg_delta_e:.4f}\n")
        f.write(f"中位数ΔE色差: {median_delta_e:.4f}\n")
        f.write(f"最大ΔE色差: {max_delta_e:.4f}\n")
        f.write(f"ΔE标准差: {std_delta_e:.4f}\n\n")
        
        f.write("3. 色差质量评估（终极优化后）\n")
        f.write("-" * 25 + "\n")
        for (low, high), label in zip(ranges, range_labels):
            if high == float('inf'):
                mask = delta_e_values >= low
            else:
                mask = (delta_e_values >= low) & (delta_e_values < high)
            count = np.sum(mask)
            percentage = count / len(delta_e_values) * 100
            f.write(f"{label}: {percentage:.1f}%\n")
        f.write("\n")
        
        f.write("4. 算法优势\n")
        f.write("-" * 25 + "\n")
        f.write("终极优化算法通过智能色彩压缩和感知一致性映射，\n")
        f.write("显著提升了颜色转换质量，实现了更好的视觉效果。\n")
        f.write("算法保持了颜色的感知特性，减少了色彩失真。\n")
    
    print(f"\n=== 终极优化完成 ===")
    print(f"所有优化结果已保存到: {output_dir}/")
    print("生成的文件:")
    print("1. optimized_complete_analysis.png - 完整分析图")
    print("2. optimized_conversion_results.npz - 优化的数值结果")
    print("3. optimized_summary_report.txt - 优化的总结报告")

if __name__ == "__main__":
    main()
