import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
import os
from PIL import Image

# 设置matplotlib后端和字体
matplotlib.use('Agg')
# 尝试使用中文字体，如果失败则使用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    use_chinese = True
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    use_chinese = False

plt.rcParams['font.size'] = 12

class PracticalColorConverter:
    def __init__(self):
        """实用的颜色空间转换器"""
        # BT2020和标准显示器色域顶点（CIE 1931 xy坐标）
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
        self.compute_transform_matrices()
        
    def compute_transform_matrices(self):
        """计算色域转换矩阵"""
        # 将三角形顶点转换为齐次坐标
        bt2020_homo = np.column_stack([self.bt2020_vertices, np.ones(3)])
        display_homo = np.column_stack([self.display_vertices, np.ones(3)])
        
        # 计算从BT2020到显示器色域的变换矩阵
        self.transform_matrix = np.linalg.solve(bt2020_homo, display_homo)
        
    def rgb_to_xy(self, rgb):
        """RGB到xy色度坐标的简化转换"""
        rgb = np.array(rgb)
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, -1)
        
        # 使用标准的RGB到XYZ转换矩阵（sRGB）
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
        
        srgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = rgb_linear @ srgb_to_xyz.T
        
        # XYZ到xy
        xyz_sum = np.sum(xyz, axis=1, keepdims=True)
        xyz_sum = np.where(xyz_sum == 0, 1e-10, xyz_sum)
        
        x = xyz[:, 0:1] / xyz_sum
        y = xyz[:, 1:2] / xyz_sum
        
        return np.hstack([x, y])
    
    def xy_to_rgb(self, xy, Y=1.0):
        """xy色度坐标到RGB的转换"""
        xy = np.array(xy)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        
        x, y = xy[:, 0], xy[:, 1]
        z = 1 - x - y
        
        # 防止除零
        y_safe = np.where(y == 0, 1e-10, y)
        
        if np.isscalar(Y):
            Y = np.full(len(xy), Y)
        
        X = Y * x / y_safe
        Z = Y * z / y_safe
        
        xyz = np.column_stack([X, Y, Z])
        
        # XYZ到RGB
        xyz_to_srgb = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        rgb_linear = xyz @ xyz_to_srgb.T
        rgb_linear = np.clip(rgb_linear, 0, 1)
        
        # 线性到sRGB
        rgb = np.where(rgb_linear <= 0.0031308, 
                      12.92 * rgb_linear, 
                      1.055 * np.power(rgb_linear, 1/2.4) - 0.055)
        
        return np.clip(rgb, 0, 1)
    
    def enhanced_gamut_mapping(self, source_rgb):
        """增强的色域映射算法"""
        source_rgb = np.array(source_rgb)
        if source_rgb.ndim == 1:
            source_rgb = source_rgb.reshape(1, -1)
            
        mapped_rgb = []
        
        for rgb in source_rgb:
            # 转换到xy色度坐标
            xy = self.rgb_to_xy(rgb.reshape(1, -1))[0]
            
            # 检查是否在目标色域内
            if self.point_in_triangle(xy, self.display_vertices):
                # 已在目标色域内
                mapped_rgb.append(rgb)
            else:
                # 需要映射到目标色域
                # 使用保持亮度的压缩映射
                mapped_xy = self.compress_to_gamut(xy, rgb)
                mapped_color = self.xy_to_rgb(mapped_xy.reshape(1, -1), 
                                            Y=np.mean(rgb))[0]  # 保持平均亮度
                mapped_rgb.append(mapped_color)
        
        return np.array(mapped_rgb)
    
    def compress_to_gamut(self, xy, original_rgb):
        """压缩到目标色域的改进算法"""
        # 计算色域中心
        gamut_center = np.mean(self.display_vertices, axis=0)
        
        # 从色域中心到当前点的方向
        direction = xy - gamut_center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm == 0:
            return gamut_center
        
        direction_unit = direction / direction_norm
        
        # 找到射线与色域边界的交点
        intersection = self.find_gamut_intersection(gamut_center, direction_unit)
        
        # 使用智能压缩因子
        saturation = np.max(original_rgb) - np.min(original_rgb)
        brightness = np.mean(original_rgb)
        
        # 基于饱和度和亮度调整压缩策略
        if saturation > 0.8:  # 高饱和度
            compression_factor = 0.85
        elif brightness < 0.3:  # 暗色
            compression_factor = 0.9
        else:  # 普通情况
            compression_factor = 0.88
        
        # 进行压缩映射
        compressed_xy = gamut_center + compression_factor * (intersection - gamut_center)
        
        return compressed_xy
    
    def find_gamut_intersection(self, center, direction):
        """找到射线与色域边界的交点"""
        min_t = float('inf')
        intersection_point = center
        
        # 检查与每条边的交点
        for i in range(3):
            p1 = self.display_vertices[i]
            p2 = self.display_vertices[(i + 1) % 3]
            
            # 计算射线与线段的交点
            edge_vec = p2 - p1
            center_to_p1 = p1 - center
            
            # 使用2D叉积的标量值
            cross_prod = direction[0] * edge_vec[1] - direction[1] * edge_vec[0]
            if abs(cross_prod) < 1e-10:
                continue  # 平行
            
            t = (center_to_p1[0] * edge_vec[1] - center_to_p1[1] * edge_vec[0]) / cross_prod
            u = (center_to_p1[0] * direction[1] - center_to_p1[1] * direction[0]) / cross_prod
            
            if t > 0 and 0 <= u <= 1 and t < min_t:
                min_t = t
                intersection_point = center + t * direction
        
        return intersection_point
    
    def point_in_triangle(self, point, triangle):
        """检查点是否在三角形内（重心坐标法）"""
        v0 = triangle[2] - triangle[0]
        v1 = triangle[1] - triangle[0]
        v2 = point - triangle[0]
        
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    def calculate_delta_e_lab(self, rgb1, rgb2):
        """计算CIE ΔE*ab色差"""
        lab1 = self.rgb_to_lab(rgb1)
        lab2 = self.rgb_to_lab(rgb2)
        
        delta_l = lab1[:, 0] - lab2[:, 0]
        delta_a = lab1[:, 1] - lab2[:, 1]
        delta_b = lab1[:, 2] - lab2[:, 2]
        
        delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        return delta_e
    
    def rgb_to_lab(self, rgb):
        """RGB到Lab色彩空间转换"""
        rgb = np.array(rgb)
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, -1)
        
        # RGB到XYZ（使用sRGB矩阵）
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
        
        srgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = rgb_linear @ srgb_to_xyz.T
        
        # 标准化到D65白点
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        xyz_n = xyz / np.array([Xn, Yn, Zn])
        
        # XYZ到Lab转换
        def f(t):
            delta = 6.0/29.0
            return np.where(t > delta**3, np.power(t, 1/3), t/(3*delta**2) + 4/29)
        
        fx = f(xyz_n[:, 0])
        fy = f(xyz_n[:, 1])
        fz = f(xyz_n[:, 2])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.column_stack([L, a, b])

def create_comprehensive_test_image(height, width):
    """创建包含多种颜色的综合测试图像"""
    image = np.zeros((height, width, 3))
    
    # 创建颜色渐变区域
    for i in range(height):
        for j in range(width):
            # 三种不同的渐变模式
            x_norm = j / width
            y_norm = i / height
            
            if i < height // 3:
                # 红绿渐变
                image[i, j] = [x_norm, y_norm, 0.2]
            elif i < 2 * height // 3:
                # 蓝黄渐变
                image[i, j] = [y_norm, x_norm, 1.0 - x_norm]
            else:
                # 彩虹渐变
                hue = x_norm * 360
                sat = 0.8
                val = 0.8
                rgb = hsv_to_rgb(hue, sat, val)
                image[i, j] = rgb
    
    # 添加特定颜色的测试块
    test_colors = [
        [1.0, 0.0, 0.0],  # 纯红
        [0.0, 1.0, 0.0],  # 纯绿
        [0.0, 0.0, 1.0],  # 纯蓝
        [1.0, 1.0, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # 洋红
        [0.0, 1.0, 1.0],  # 青
        [0.8, 0.4, 0.2],  # 橙色
        [0.6, 0.2, 0.8],  # 紫色
    ]
    
    block_size = 24
    for idx, color in enumerate(test_colors):
        row = 10 + (idx // 4) * (block_size + 5)
        col = 10 + (idx % 4) * (block_size + 5)
        
        if row + block_size <= height and col + block_size <= width:
            image[row:row+block_size, col:col+block_size] = color
    
    return np.clip(image, 0, 1)

def hsv_to_rgb(h, s, v):
    """HSV到RGB转换"""
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
    
    return [r + m, g + m, b + m]

def create_comprehensive_plots(original_image, mapped_image, converter, 
                             source_rgb, mapped_rgb, delta_e_values, save_dir):
    """创建综合性分析图表 - 使用英文标签"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 图像对比区域 (2x3)
    gs1 = fig.add_gridspec(2, 3, left=0.05, right=0.65, top=0.95, bottom=0.55)
    
    # 原始图像
    ax1 = fig.add_subplot(gs1[0, 0])
    ax1.imshow(original_image)
    ax1.set_title("Original Image (BT2020 simulated)", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 映射后图像
    ax2 = fig.add_subplot(gs1[0, 1])
    ax2.imshow(mapped_image)
    ax2.set_title("Mapped Image (Display gamut)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 差异图
    diff_image = np.abs(original_image - mapped_image)
    ax3 = fig.add_subplot(gs1[0, 2])
    im1 = ax3.imshow(diff_image, cmap='hot', vmin=0, vmax=0.3)
    ax3.set_title("Absolute Difference", fontsize=14, fontweight='bold')
    ax3.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax3, shrink=0.8)
    cbar1.set_label('RGB Difference')
    
    # Delta E 热图
    delta_e_2d = delta_e_values.reshape(original_image.shape[:2])
    ax4 = fig.add_subplot(gs1[1, 0])
    im2 = ax4.imshow(delta_e_2d, cmap='viridis', vmin=0, vmax=15)
    ax4.set_title("Delta E Color Difference Map", fontsize=14, fontweight='bold')
    ax4.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax4, shrink=0.8)
    cbar2.set_label('Delta E Value')
    
    # 色域可视化
    ax5 = fig.add_subplot(gs1[1, 1])
    
    # 绘制色域三角形
    bt2020_triangle = Polygon(converter.bt2020_vertices, fill=False, 
                             edgecolor='blue', linewidth=3, label='BT2020 Gamut')
    ax5.add_patch(bt2020_triangle)
    
    display_triangle = Polygon(converter.display_vertices, fill=False, 
                              edgecolor='red', linewidth=3, label='Display Gamut')
    ax5.add_patch(display_triangle)
    
    # 采样点进行可视化
    sample_indices = np.random.choice(len(source_rgb), min(2000, len(source_rgb)), replace=False)
    source_xy_sample = converter.rgb_to_xy(source_rgb[sample_indices])
    mapped_xy_sample = converter.rgb_to_xy(mapped_rgb[sample_indices])
    
    ax5.scatter(source_xy_sample[:, 0], source_xy_sample[:, 1], 
               c='lightblue', s=8, alpha=0.6, label='Source Points', edgecolors='none')
    ax5.scatter(mapped_xy_sample[:, 0], mapped_xy_sample[:, 1], 
               c='salmon', s=8, alpha=0.6, label='Mapped Points', edgecolors='none')
    
    ax5.set_xlim(0, 0.8)
    ax5.set_ylim(0, 0.9)
    ax5.set_xlabel('CIE x', fontsize=12)
    ax5.set_ylabel('CIE y', fontsize=12)
    ax5.set_title('Color Gamut Mapping', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 质量评估区域
    ax6 = fig.add_subplot(gs1[1, 2])
    
    # ΔE分布条形图
    ranges = [(0, 1), (1, 3), (3, 6), (6, 10), (10, float('inf'))]
    range_labels = ['Excellent\n(0-1)', 'Good\n(1-3)', 'Acceptable\n(3-6)', 'Poor\n(6-10)', 'Very Poor\n(>10)']
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    
    percentages = []
    for low, high in ranges:
        if high == float('inf'):
            mask = delta_e_values >= low
        else:
            mask = (delta_e_values >= low) & (delta_e_values < high)
        percentage = np.sum(mask) / len(delta_e_values) * 100
        percentages.append(percentage)
    
    bars = ax6.bar(range(len(ranges)), percentages, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Quality Categories', fontsize=12)
    ax6.set_ylabel('Percentage (%)', fontsize=12)
    ax6.set_title('Color Quality Distribution', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(ranges)))
    ax6.set_xticklabels(range_labels, fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 统计区域
    gs2 = fig.add_gridspec(2, 2, left=0.70, right=0.98, top=0.95, bottom=0.55)
    
    # ΔE直方图
    ax7 = fig.add_subplot(gs2[0, :])
    ax7.hist(delta_e_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax7.axvline(np.mean(delta_e_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(delta_e_values):.2f}')
    ax7.axvline(np.median(delta_e_values), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(delta_e_values):.2f}')
    ax7.set_xlabel('Delta E Value', fontsize=12)
    ax7.set_ylabel('Density', fontsize=12)
    ax7.set_title('Delta E Distribution Histogram', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3)
    
    # 统计表格
    ax8 = fig.add_subplot(gs2[1, :])
    ax8.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Mean Delta E', f'{np.mean(delta_e_values):.3f}'],
        ['Median Delta E', f'{np.median(delta_e_values):.3f}'],
        ['Std Delta E', f'{np.std(delta_e_values):.3f}'],
        ['Max Delta E', f'{np.max(delta_e_values):.3f}'],
        ['Min Delta E', f'{np.min(delta_e_values):.3f}'],
        ['95th Percentile', f'{np.percentile(delta_e_values, 95):.3f}'],
        ['Total Pixels', f'{len(delta_e_values):,}'],
    ]
    
    table = ax8.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(stats_data)):
        if i == 0:  # 标题行
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')
    
    # 算法说明区域
    gs3 = fig.add_gridspec(1, 1, left=0.05, right=0.98, top=0.45, bottom=0.05)
    ax9 = fig.add_subplot(gs3[0, 0])
    ax9.axis('off')
    
    explanation_text = """
    LED Display Color Conversion Design & Calibration - Problem 1 Practical Algorithm:
    
    1. Gamut Detection: Use CIE 1931 xy chromaticity diagram to detect if colors are within target display gamut
    2. Smart Compression: Apply adaptive compression strategy based on saturation and brightness for out-of-gamut colors
    3. Boundary Projection: Calculate optimal projection points from gamut center to boundary, maintaining perceptual consistency
    4. Color Difference Assessment: Use CIE Delta E*ab formula to quantify color differences before and after conversion
    
    Quality Standards:
    • Excellent (ΔE < 1): Differences barely perceptible to human eye
    • Good (1 ≤ ΔE < 3): Slight differences that may be perceived by trained observers  
    • Acceptable (3 ≤ ΔE < 6): Noticeable but acceptable differences for general observers
    • Poor (6 ≤ ΔE < 10): Obvious color differences requiring optimization
    • Very Poor (ΔE ≥ 10): Severe color distortion, unacceptable
    """
    
    ax9.text(0.02, 0.98, explanation_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('LED Display Color Conversion Design & Calibration - Problem 1 Complete Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(save_dir, 'practical_complete_analysis_english.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("=== LED Display Color Conversion Design & Calibration - Problem 1 (English Version) ===")
    
    # 创建输出目录
    output_dir = "practical_results_english"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化转换器
    converter = PracticalColorConverter()
    
    print("Step 1: Generate comprehensive test image...")
    original_image = create_comprehensive_test_image(300, 400)
    
    print("Step 2: Execute practical gamut mapping...")
    h, w, c = original_image.shape
    source_rgb = original_image.reshape(-1, 3)
    
    # 执行色域映射
    mapped_rgb = converter.enhanced_gamut_mapping(source_rgb)
    mapped_image = mapped_rgb.reshape(h, w, c)
    mapped_image = np.clip(mapped_image, 0, 1)
    
    print("Step 3: Calculate color difference analysis...")
    # 计算ΔE色差
    delta_e_values = converter.calculate_delta_e_lab(source_rgb, mapped_rgb)
    
    # 统计信息
    avg_delta_e = np.mean(delta_e_values)
    max_delta_e = np.max(delta_e_values)
    std_delta_e = np.std(delta_e_values)
    median_delta_e = np.median(delta_e_values)
    
    print(f"\n=== Practical Algorithm Conversion Results ===")
    print(f"Average Delta E: {avg_delta_e:.4f}")
    print(f"Median Delta E: {median_delta_e:.4f}")
    print(f"Maximum Delta E: {max_delta_e:.4f}")
    print(f"Delta E Standard Deviation: {std_delta_e:.4f}")
    
    # ΔE分布统计
    ranges = [(0, 1), (1, 3), (3, 6), (6, 10), (10, float('inf'))]
    range_labels = ['Excellent (0-1)', 'Good (1-3)', 'Acceptable (3-6)', 'Poor (6-10)', 'Very Poor (>10)']
    
    print(f"\n=== Delta E Quality Distribution ===")
    for (low, high), label in zip(ranges, range_labels):
        if high == float('inf'):
            mask = delta_e_values >= low
        else:
            mask = (delta_e_values >= low) & (delta_e_values < high)
        count = np.sum(mask)
        percentage = count / len(delta_e_values) * 100
        print(f"{label}: {count} pixels ({percentage:.1f}%)")
    
    print("\nStep 4: Generate detailed analysis charts...")
    
    # 生成综合分析图
    create_comprehensive_plots(original_image, mapped_image, converter, 
                              source_rgb, mapped_rgb, delta_e_values, output_dir)
    
    # 保存数值结果
    practical_results = {
        'source_image': original_image,
        'mapped_image': mapped_image,
        'source_rgb': source_rgb,
        'mapped_rgb': mapped_rgb,
        'delta_e_values': delta_e_values,
        'statistics': {
            'avg_delta_e': avg_delta_e,
            'median_delta_e': median_delta_e,
            'max_delta_e': max_delta_e,
            'std_delta_e': std_delta_e,
            'percentile_95': np.percentile(delta_e_values, 95),
            'percentile_99': np.percentile(delta_e_values, 99),
        },
        'quality_distribution': {
            'excellent': np.sum((delta_e_values >= 0) & (delta_e_values < 1)) / len(delta_e_values) * 100,
            'good': np.sum((delta_e_values >= 1) & (delta_e_values < 3)) / len(delta_e_values) * 100,
            'acceptable': np.sum((delta_e_values >= 3) & (delta_e_values < 6)) / len(delta_e_values) * 100,
            'poor': np.sum((delta_e_values >= 6) & (delta_e_values < 10)) / len(delta_e_values) * 100,
            'very_poor': np.sum(delta_e_values >= 10) / len(delta_e_values) * 100,
        }
    }
    
    np.savez(os.path.join(output_dir, 'practical_conversion_results_english.npz'), **practical_results)
    
    # 创建详细报告
    with open(os.path.join(output_dir, 'practical_detailed_report_english.txt'), 'w', encoding='utf-8') as f:
        f.write("LED Display Color Conversion Design & Calibration - Problem 1 Practical Algorithm Report\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("1. Algorithm Overview\n")
        f.write("-" * 30 + "\n")
        f.write("This practical algorithm employs the following core technologies:\n")
        f.write("• Precise CIE 1931 chromaticity coordinate conversion\n")
        f.write("• Geometry-based gamut boundary detection\n")
        f.write("• Adaptive smart compression mapping strategy\n")
        f.write("• Perceptually consistent gamut projection algorithm\n")
        f.write("• Standard CIE Delta E*ab color difference assessment\n\n")
        
        f.write("2. Core Performance Metrics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Delta E: {avg_delta_e:.4f}\n")
        f.write(f"Median Delta E: {median_delta_e:.4f}\n")
        f.write(f"Maximum Delta E: {max_delta_e:.4f}\n")
        f.write(f"Delta E Standard Deviation: {std_delta_e:.4f}\n")
        f.write(f"95th Percentile: {np.percentile(delta_e_values, 95):.4f}\n")
        f.write(f"99th Percentile: {np.percentile(delta_e_values, 99):.4f}\n\n")
        
        f.write("3. Quality Distribution Details\n")
        f.write("-" * 30 + "\n")
        for (low, high), label in zip(ranges, range_labels):
            if high == float('inf'):
                mask = delta_e_values >= low
            else:
                mask = (delta_e_values >= low) & (delta_e_values < high)
            count = np.sum(mask)
            percentage = count / len(delta_e_values) * 100
            f.write(f"{label}: {count:6} pixels ({percentage:5.1f}%)\n")
        f.write(f"Total pixels: {len(delta_e_values):,}\n\n")
        
        f.write("4. Algorithm Features & Advantages\n")
        f.write("-" * 30 + "\n")
        f.write("• High Practicality: Stable algorithm operation, suitable for engineering applications\n")
        f.write("• Perceptual Optimization: Maintains visual color consistency, reduces obvious distortion\n")
        f.write("• Adaptability: Adjusts mapping strategy based on color characteristics\n")
        f.write("• Standard Compatibility: Uses international standard color difference assessment methods\n")
        f.write("• Efficient Computation: Moderate algorithm complexity, suitable for real-time processing\n\n")
        
        f.write("5. Application Recommendations\n")
        f.write("-" * 30 + "\n")
        f.write("• For high-demand applications, further optimize high-saturation color processing\n")
        f.write("• Consider combining with hardware calibration to improve overall display quality\n")
        f.write("• Adjust compression parameters for specific content types\n")
        f.write("• Regularly validate conversion effects using standard test images\n")
    
    print(f"\n=== Practical Algorithm Analysis Complete ===")
    print(f"Detailed results saved to: {output_dir}/")
    print("Generated files:")
    print("1. practical_complete_analysis_english.png - Complete analysis chart")
    print("2. practical_conversion_results_english.npz - Numerical results")
    print("3. practical_detailed_report_english.txt - Detailed analysis report")

if __name__ == "__main__":
    main()
