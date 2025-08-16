import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
import os
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# 设置matplotlib后端和字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class FourToFiveChannelConverter:
    def __init__(self):
        """4通道到5通道颜色转换器"""
        
        # 定义4通道RGBV色域顶点（CIE 1931 xy坐标）
        self.rgbv_vertices = np.array([
            [0.708, 0.292],  # R - Red
            [0.170, 0.797],  # G - Green  
            [0.131, 0.046],  # B - Blue
            [0.280, 0.330]   # V - Violet (假设紫色通道)
        ])
        
        # 定义5通道RGBCX色域顶点
        self.rgbcx_vertices = np.array([
            [0.640, 0.330],  # R - Red
            [0.300, 0.600],  # G - Green
            [0.150, 0.060],  # B - Blue  
            [0.225, 0.329],  # C - Cyan
            [0.313, 0.329]   # X - Extra channel (扩展色域)
        ])
        
        # 计算转换矩阵
        self.compute_conversion_matrix()
        
        # 标准转换矩阵
        self.setup_color_matrices()
        
    def setup_color_matrices(self):
        """设置颜色空间转换矩阵"""
        # RGB到XYZ转换矩阵
        self.rgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        self.xyz_to_rgb_matrix = np.linalg.inv(self.rgb_to_xyz_matrix)
        
    def compute_conversion_matrix(self):
        """计算4通道到5通道的转换矩阵"""
        # 使用最小二乘法计算最优转换矩阵
        # 将4维色域映射到5维色域
        source_points = self.rgbv_vertices
        target_points = self.rgbcx_vertices[:4]  # 使用前4个目标点
        
        # 添加齐次坐标
        source_homo = np.column_stack([source_points, np.ones(4)])
        target_homo = np.column_stack([target_points, np.ones(4)])
        
        # 计算转换矩阵
        self.conversion_matrix = np.linalg.lstsq(source_homo, target_homo, rcond=None)[0]
        
    def rgbv_to_xy(self, rgbv):
        """将4通道RGBV转换为xy色度坐标"""
        rgbv = np.array(rgbv)
        if rgbv.ndim == 1:
            rgbv = rgbv.reshape(1, -1)
        
        # 先转换RGB部分到XYZ
        rgb = rgbv[:, :3]
        v_channel = rgbv[:, 3:4]
        
        # RGB线性化
        rgb_linear = self.srgb_to_linear(rgb)
        
        # RGB到XYZ
        xyz = rgb_linear @ self.rgb_to_xyz_matrix.T
        
        # 添加V通道的影响（假设V通道增强紫色分量）
        xyz[:, 2] += v_channel.flatten() * 0.3  # Z分量增强
        
        # XYZ到xy
        xyz_sum = np.sum(xyz, axis=1, keepdims=True)
        xyz_sum = np.where(xyz_sum == 0, 1e-10, xyz_sum)
        
        x = xyz[:, 0:1] / xyz_sum
        y = xyz[:, 1:2] / xyz_sum
        
        return np.hstack([x, y])
    
    def rgbcx_to_xy(self, rgbcx):
        """将5通道RGBCX转换为xy色度坐标"""
        rgbcx = np.array(rgbcx)
        if rgbcx.ndim == 1:
            rgbcx = rgbcx.reshape(1, -1)
        
        # RGB部分处理
        rgb = rgbcx[:, :3]
        c_channel = rgbcx[:, 3:4]  # Cyan channel
        x_channel = rgbcx[:, 4:5]  # Extra channel
        
        # RGB线性化
        rgb_linear = self.srgb_to_linear(rgb)
        
        # RGB到XYZ
        xyz = rgb_linear @ self.rgb_to_xyz_matrix.T
        
        # 添加C和X通道的影响
        xyz[:, 0] += c_channel.flatten() * 0.2  # C通道增强X分量
        xyz[:, 1] += c_channel.flatten() * 0.25  # C通道增强Y分量
        xyz[:, 0] += x_channel.flatten() * 0.15  # X通道额外增强
        
        # XYZ到xy
        xyz_sum = np.sum(xyz, axis=1, keepdims=True)
        xyz_sum = np.where(xyz_sum == 0, 1e-10, xyz_sum)
        
        x = xyz[:, 0:1] / xyz_sum
        y = xyz[:, 1:2] / xyz_sum
        
        return np.hstack([x, y])
    
    def srgb_to_linear(self, rgb):
        """sRGB到线性RGB转换"""
        return np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    
    def linear_to_srgb(self, rgb):
        """线性RGB到sRGB转换"""
        return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1/2.4) - 0.055)
    
    def point_in_polygon(self, point, polygon):
        """检查点是否在多边形内（射线法）"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def advanced_4to5_conversion(self, rgbv_input):
        """高级4通道到5通道转换算法"""
        rgbv_input = np.array(rgbv_input)
        if rgbv_input.ndim == 1:
            rgbv_input = rgbv_input.reshape(1, -1)
        
        rgbcx_output = []
        
        for rgbv in rgbv_input:
            # 转换到色度坐标
            xy_source = self.rgbv_to_xy(rgbv.reshape(1, -1))[0]
            
            # 检查是否在目标色域内
            if self.point_in_polygon(xy_source, self.rgbcx_vertices):
                # 在目标色域内，直接映射
                rgbcx = self.direct_mapping(rgbv)
            else:
                # 超出目标色域，使用智能压缩
                rgbcx = self.intelligent_compression_5d(rgbv, xy_source)
            
            rgbcx_output.append(rgbcx)
        
        return np.array(rgbcx_output)
    
    def direct_mapping(self, rgbv):
        """直接映射算法"""
        r, g, b, v = rgbv
        
        # 基础RGB映射
        rgb_mapped = np.array([r, g, b]) * 0.9  # 略微压缩保证在范围内
        
        # V通道分解到C和X通道
        c_channel = v * 0.6  # 60%的V通道映射到C
        x_channel = v * 0.4  # 40%的V通道映射到X
        
        # 组合5通道输出
        rgbcx = np.concatenate([rgb_mapped, [c_channel], [x_channel]])
        return np.clip(rgbcx, 0, 1)
    
    def intelligent_compression_5d(self, rgbv, xy_source):
        """5维空间的智能压缩算法"""
        # 计算目标色域的重心
        centroid = np.mean(self.rgbcx_vertices, axis=0)
        
        # 计算压缩方向
        direction = xy_source - centroid
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm == 0:
            return self.direct_mapping(rgbv)
        
        direction_unit = direction / direction_norm
        
        # 找到边界交点
        intersection = self.find_polygon_intersection(centroid, direction_unit)
        
        # 自适应压缩因子
        v_intensity = rgbv[3]
        compression_factor = 0.85 if v_intensity > 0.7 else 0.90
        
        # 计算压缩后的xy坐标
        compressed_xy = centroid + compression_factor * (intersection - centroid)
        
        # 转换回5通道空间
        rgbcx = self.xy_to_rgbcx(compressed_xy, rgbv)
        
        return rgbcx
    
    def find_polygon_intersection(self, center, direction):
        """找到射线与多边形边界的交点"""
        min_t = float('inf')
        intersection_point = center
        
        n = len(self.rgbcx_vertices)
        for i in range(n):
            p1 = self.rgbcx_vertices[i]
            p2 = self.rgbcx_vertices[(i + 1) % n]
            
            # 计算射线与线段的交点
            edge_vec = p2 - p1
            center_to_p1 = p1 - center
            
            cross_prod = direction[0] * edge_vec[1] - direction[1] * edge_vec[0]
            if abs(cross_prod) < 1e-10:
                continue
            
            t = (center_to_p1[0] * edge_vec[1] - center_to_p1[1] * edge_vec[0]) / cross_prod
            u = (center_to_p1[0] * direction[1] - center_to_p1[1] * direction[0]) / cross_prod
            
            if t > 0 and 0 <= u <= 1 and t < min_t:
                min_t = t
                intersection_point = center + t * direction
        
        return intersection_point
    
    def xy_to_rgbcx(self, xy, original_rgbv):
        """从xy色度坐标和原始RGBV转换到RGBCX"""
        # 保持原始的亮度信息
        original_brightness = np.mean(original_rgbv[:3])
        
        # 基于xy坐标重构RGB
        # 这里使用简化的方法，实际应用中可能需要更复杂的色彩模型
        r = max(0, min(1, xy[0] * 1.5))
        g = max(0, min(1, xy[1] * 1.5))
        b = max(0, min(1, (1 - xy[0] - xy[1]) * 1.5))
        
        # 归一化到原始亮度
        current_brightness = (r + g + b) / 3
        if current_brightness > 0:
            scale = original_brightness / current_brightness
            r, g, b = r * scale, g * scale, b * scale
        
        # 分配C和X通道
        v_original = original_rgbv[3]
        c_channel = v_original * 0.6
        x_channel = v_original * 0.4
        
        return np.clip([r, g, b, c_channel, x_channel], 0, 1)
    
    def calculate_conversion_quality(self, rgbv_input, rgbcx_output):
        """计算转换质量评估"""
        # 计算色域覆盖率
        source_xy = self.rgbv_to_xy(rgbv_input)
        target_xy = self.rgbcx_to_xy(rgbcx_output)
        
        # 计算色差（简化版本）
        color_differences = []
        for src_xy, tgt_xy in zip(source_xy, target_xy):
            diff = np.linalg.norm(src_xy - tgt_xy)
            color_differences.append(diff)
        
        color_differences = np.array(color_differences)
        
        # 质量指标
        avg_diff = np.mean(color_differences)
        max_diff = np.max(color_differences)
        std_diff = np.std(color_differences)
        
        # 质量等级划分
        excellent = np.sum(color_differences < 0.01) / len(color_differences)
        good = np.sum((color_differences >= 0.01) & (color_differences < 0.03)) / len(color_differences)
        acceptable = np.sum((color_differences >= 0.03) & (color_differences < 0.06)) / len(color_differences)
        poor = np.sum(color_differences >= 0.06) / len(color_differences)
        
        return {
            'avg_difference': avg_diff,
            'max_difference': max_diff,
            'std_difference': std_diff,
            'excellent_rate': excellent * 100,
            'good_rate': good * 100,
            'acceptable_rate': acceptable * 100,
            'poor_rate': poor * 100,
            'color_differences': color_differences
        }

def create_test_rgbv_image(height, width):
    """创建4通道RGBV测试图像"""
    image = np.zeros((height, width, 4))
    
    # 创建渐变区域
    for i in range(height):
        for j in range(width):
            x_norm = j / width
            y_norm = i / height
            
            if i < height // 4:
                # R-G渐变 + V通道
                image[i, j] = [x_norm, y_norm, 0.1, 0.3]
            elif i < height // 2:
                # G-B渐变 + V通道
                image[i, j] = [0.1, x_norm, y_norm, 0.5]
            elif i < 3 * height // 4:
                # B-V渐变
                image[i, j] = [0.2, 0.1, x_norm, y_norm]
            else:
                # 全彩渐变 + V通道
                v_val = x_norm * y_norm
                image[i, j] = [x_norm, y_norm, 1-x_norm, v_val]
    
    # 添加测试色块
    test_colors = [
        [1.0, 0.0, 0.0, 0.0],  # 纯红
        [0.0, 1.0, 0.0, 0.0],  # 纯绿
        [0.0, 0.0, 1.0, 0.0],  # 纯蓝
        [0.0, 0.0, 0.0, 1.0],  # 纯V
        [1.0, 1.0, 0.0, 0.2],  # 黄+V
        [1.0, 0.0, 1.0, 0.3],  # 洋红+V
        [0.0, 1.0, 1.0, 0.4],  # 青+V
        [0.5, 0.5, 0.5, 0.5],  # 灰+V
    ]
    
    block_size = 20
    for idx, color in enumerate(test_colors):
        row = 10 + (idx // 4) * (block_size + 5)
        col = 10 + (idx % 4) * (block_size + 5)
        
        if row + block_size <= height and col + block_size <= width:
            image[row:row+block_size, col:col+block_size] = color
    
    return np.clip(image, 0, 1)

def visualize_conversion_results(original_rgbv, converted_rgbcx, converter, quality_metrics, save_dir):
    """可视化转换结果"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 图像对比区域
    gs1 = fig.add_gridspec(3, 3, left=0.05, right=0.65, top=0.95, bottom=0.55)
    
    # 原始4通道图像 (显示RGB部分)
    ax1 = fig.add_subplot(gs1[0, 0])
    rgb_display = original_rgbv[:, :, :3]
    ax1.imshow(rgb_display)
    ax1.set_title("Original RGBV Image (RGB display)", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # V通道单独显示
    ax2 = fig.add_subplot(gs1[0, 1])
    v_channel = original_rgbv[:, :, 3]
    im1 = ax2.imshow(v_channel, cmap='viridis')
    ax2.set_title("V Channel (Violet)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im1, ax=ax2, shrink=0.8)
    
    # 转换后5通道图像 (显示RGB部分)
    ax3 = fig.add_subplot(gs1[0, 2])
    rgb_converted = converted_rgbcx[:, :, :3]
    ax3.imshow(rgb_converted)
    ax3.set_title("Converted RGBCX (RGB display)", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # C通道显示
    ax4 = fig.add_subplot(gs1[1, 0])
    c_channel = converted_rgbcx[:, :, 3]
    im2 = ax4.imshow(c_channel, cmap='cool')
    ax4.set_title("C Channel (Cyan)", fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im2, ax=ax4, shrink=0.8)
    
    # X通道显示
    ax5 = fig.add_subplot(gs1[1, 1])
    x_channel = converted_rgbcx[:, :, 4]
    im3 = ax5.imshow(x_channel, cmap='plasma')
    ax5.set_title("X Channel (Extra)", fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im3, ax=ax5, shrink=0.8)
    
    # 差异图
    ax6 = fig.add_subplot(gs1[1, 2])
    rgb_diff = np.abs(rgb_display - rgb_converted)
    im4 = ax6.imshow(rgb_diff, cmap='hot')
    ax6.set_title("RGB Difference", fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im4, ax=ax6, shrink=0.8)
    
    # 色域对比图
    ax7 = fig.add_subplot(gs1[2, :])
    
    # 绘制4通道色域 (四边形)
    rgbv_polygon = Polygon(converter.rgbv_vertices, fill=False, 
                          edgecolor='blue', linewidth=3, label='4-Channel RGBV')
    ax7.add_patch(rgbv_polygon)
    
    # 绘制5通道色域 (五边形)
    rgbcx_polygon = Polygon(converter.rgbcx_vertices, fill=False, 
                           edgecolor='red', linewidth=3, label='5-Channel RGBCX')
    ax7.add_patch(rgbcx_polygon)
    
    # 显示转换前后的采样点
    h, w = original_rgbv.shape[:2]
    sample_indices = np.random.choice(h*w, min(1000, h*w), replace=False)
    
    source_flat = original_rgbv.reshape(-1, 4)[sample_indices]
    target_flat = converted_rgbcx.reshape(-1, 5)[sample_indices]
    
    source_xy = converter.rgbv_to_xy(source_flat)
    target_xy = converter.rgbcx_to_xy(target_flat)
    
    ax7.scatter(source_xy[:, 0], source_xy[:, 1], 
               c='lightblue', s=8, alpha=0.6, label='Source Points')
    ax7.scatter(target_xy[:, 0], target_xy[:, 1], 
               c='lightcoral', s=8, alpha=0.6, label='Converted Points')
    
    ax7.set_xlim(0, 0.8)
    ax7.set_ylim(0, 0.9)
    ax7.set_xlabel('CIE x')
    ax7.set_ylabel('CIE y')
    ax7.set_title('4-Channel to 5-Channel Gamut Mapping')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal')
    
    # 统计分析区域
    gs2 = fig.add_gridspec(2, 2, left=0.70, right=0.98, top=0.95, bottom=0.55)
    
    # 质量分布饼图
    ax8 = fig.add_subplot(gs2[0, 0])
    quality_labels = ['Excellent', 'Good', 'Acceptable', 'Poor']
    quality_values = [
        quality_metrics['excellent_rate'],
        quality_metrics['good_rate'], 
        quality_metrics['acceptable_rate'],
        quality_metrics['poor_rate']
    ]
    colors = ['green', 'lightgreen', 'yellow', 'red']
    
    wedges, texts, autotexts = ax8.pie(quality_values, labels=quality_labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax8.set_title('Conversion Quality Distribution')
    
    # 色差直方图
    ax9 = fig.add_subplot(gs2[0, 1])
    ax9.hist(quality_metrics['color_differences'], bins=30, alpha=0.7, 
            color='skyblue', edgecolor='black')
    ax9.axvline(quality_metrics['avg_difference'], color='red', linestyle='--',
               label=f"Mean: {quality_metrics['avg_difference']:.4f}")
    ax9.set_xlabel('Color Difference')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Color Difference Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 统计表格
    ax10 = fig.add_subplot(gs2[1, :])
    ax10.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Average Difference', f'{quality_metrics["avg_difference"]:.6f}'],
        ['Max Difference', f'{quality_metrics["max_difference"]:.6f}'],
        ['Std Difference', f'{quality_metrics["std_difference"]:.6f}'],
        ['Excellent Rate', f'{quality_metrics["excellent_rate"]:.1f}%'],
        ['Good Rate', f'{quality_metrics["good_rate"]:.1f}%'],
        ['Acceptable Rate', f'{quality_metrics["acceptable_rate"]:.1f}%'],
        ['Poor Rate', f'{quality_metrics["poor_rate"]:.1f}%'],
    ]
    
    table = ax10.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(stats_data)):
        if i == 0:
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')
    
    # 算法说明
    gs3 = fig.add_gridspec(1, 1, left=0.05, right=0.98, top=0.45, bottom=0.05)
    ax11 = fig.add_subplot(gs3[0, 0])
    ax11.axis('off')
    
    explanation_text = """
    4-Channel to 5-Channel Color Conversion Algorithm:
    
    1. Source Gamut: 4-channel RGBV (Red, Green, Blue, Violet) with expanded color space
    2. Target Gamut: 5-channel RGBCX (Red, Green, Blue, Cyan, eXtra) for enhanced LED display
    3. Conversion Strategy: 
       • Direct mapping for colors within target gamut
       • Intelligent compression for out-of-gamut colors
       • V-channel decomposition to C and X channels (60%-40% split)
       • Adaptive compression based on V-channel intensity
    
    Quality Metrics:
    • Excellent: Color difference < 0.01 (imperceptible)
    • Good: 0.01 ≤ difference < 0.03 (barely perceptible)
    • Acceptable: 0.03 ≤ difference < 0.06 (perceptible but acceptable)
    • Poor: difference ≥ 0.06 (noticeable distortion)
    """
    
    ax11.text(0.02, 0.98, explanation_text, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('4-Channel RGBV to 5-Channel RGBCX Conversion Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(save_dir, 'channel_4to5_complete_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("=== 4-Channel to 5-Channel Color Conversion - Problem 2 ===")
    
    # 创建输出目录
    output_dir = "problem2_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化转换器
    converter = FourToFiveChannelConverter()
    
    print("Step 1: Generate 4-channel RGBV test image...")
    original_rgbv = create_test_rgbv_image(200, 300)
    
    print("Step 2: Execute 4-to-5 channel conversion...")
    h, w, c = original_rgbv.shape
    source_rgbv = original_rgbv.reshape(-1, 4)
    
    # 执行转换
    converted_rgbcx_flat = converter.advanced_4to5_conversion(source_rgbv)
    converted_rgbcx = converted_rgbcx_flat.reshape(h, w, 5)
    converted_rgbcx = np.clip(converted_rgbcx, 0, 1)
    
    print("Step 3: Calculate conversion quality...")
    quality_metrics = converter.calculate_conversion_quality(source_rgbv, converted_rgbcx_flat)
    
    print(f"\n=== Conversion Results ===")
    print(f"Average Color Difference: {quality_metrics['avg_difference']:.6f}")
    print(f"Maximum Color Difference: {quality_metrics['max_difference']:.6f}")
    print(f"Standard Deviation: {quality_metrics['std_difference']:.6f}")
    print(f"\nQuality Distribution:")
    print(f"Excellent: {quality_metrics['excellent_rate']:.1f}%")
    print(f"Good: {quality_metrics['good_rate']:.1f}%")
    print(f"Acceptable: {quality_metrics['acceptable_rate']:.1f}%")
    print(f"Poor: {quality_metrics['poor_rate']:.1f}%")
    
    print("\nStep 4: Generate comprehensive analysis...")
    
    # 生成可视化结果
    visualize_conversion_results(original_rgbv, converted_rgbcx, converter, 
                                quality_metrics, output_dir)
    
    # 保存数值结果
    results = {
        'original_rgbv': original_rgbv,
        'converted_rgbcx': converted_rgbcx,
        'source_flat': source_rgbv,
        'converted_flat': converted_rgbcx_flat,
        'quality_metrics': quality_metrics,
        'rgbv_vertices': converter.rgbv_vertices,
        'rgbcx_vertices': converter.rgbcx_vertices
    }
    
    np.savez(os.path.join(output_dir, 'channel_4to5_results.npz'), **results)
    
    # 创建详细报告
    with open(os.path.join(output_dir, 'conversion_report.txt'), 'w', encoding='utf-8') as f:
        f.write("4-Channel to 5-Channel Color Conversion Report\n")
        f.write("=" * 55 + "\n\n")
        
        f.write("1. Algorithm Overview\n")
        f.write("-" * 20 + "\n")
        f.write("• Source: 4-channel RGBV (Red, Green, Blue, Violet)\n")
        f.write("• Target: 5-channel RGBCX (Red, Green, Blue, Cyan, eXtra)\n")
        f.write("• Strategy: Intelligent mapping with V-channel decomposition\n\n")
        
        f.write("2. Performance Metrics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Color Difference: {quality_metrics['avg_difference']:.6f}\n")
        f.write(f"Maximum Color Difference: {quality_metrics['max_difference']:.6f}\n")
        f.write(f"Standard Deviation: {quality_metrics['std_difference']:.6f}\n\n")
        
        f.write("3. Quality Distribution\n")
        f.write("-" * 20 + "\n")
        f.write(f"Excellent (< 0.01): {quality_metrics['excellent_rate']:.1f}%\n")
        f.write(f"Good (0.01-0.03): {quality_metrics['good_rate']:.1f}%\n")
        f.write(f"Acceptable (0.03-0.06): {quality_metrics['acceptable_rate']:.1f}%\n")
        f.write(f"Poor (≥ 0.06): {quality_metrics['poor_rate']:.1f}%\n\n")
        
        f.write("4. Technical Features\n")
        f.write("-" * 20 + "\n")
        f.write("• Polygon-based gamut detection\n")
        f.write("• Adaptive compression based on V-channel intensity\n")
        f.write("• V-channel decomposition (60% C + 40% X)\n")
        f.write("• Perceptual consistency preservation\n\n")
        
        f.write("5. Applications\n")
        f.write("-" * 20 + "\n")
        f.write("• High-end LED displays with extended color gamut\n")
        f.write("• Professional video processing and broadcast\n")
        f.write("• Advanced color management systems\n")
        f.write("• Next-generation display technologies\n")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}/")
    print("Generated files:")
    print("1. channel_4to5_complete_analysis.png - Complete analysis chart")
    print("2. channel_4to5_results.npz - Numerical results")
    print("3. conversion_report.txt - Summary report")

if __name__ == "__main__":
    main()
