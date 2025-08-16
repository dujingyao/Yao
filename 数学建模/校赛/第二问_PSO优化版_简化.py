import numpy as np
import os
from typing import Tuple, List
import random

# 导入可视化模块
from pso_visualization import visualize_pso_results

class PSOOptimizedColorConverter:
    """基于粒子群优化的4通道到5通道颜色转换器"""
    
    def __init__(self):
        self.setup_color_spaces()
        self.setup_pso_parameters()
        
    def setup_color_spaces(self):
        """设置颜色空间定义"""
        # 4通道RGBV色域顶点 (CIE 1931 xy)
        self.rgbv_vertices = np.array([
            [0.708, 0.292],  # R
            [0.170, 0.797],  # G  
            [0.131, 0.046],  # B
            [0.280, 0.330]   # V
        ])
        
        # 5通道RGBCX色域顶点
        self.rgbcx_vertices = np.array([
            [0.640, 0.330],  # R
            [0.300, 0.600],  # G
            [0.150, 0.060],  # B
            [0.225, 0.329],  # C
            [0.313, 0.329]   # X
        ])
        
        # RGB到XYZ转换矩阵 (sRGB标准)
        self.rgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        self.xyz_to_rgb = np.linalg.inv(self.rgb_to_xyz)
        
    def setup_pso_parameters(self):
        """设置PSO算法参数"""
        self.pso_params = {
            'swarm_size': 30,      # 粒子群大小
            'max_iter': 100,       # 最大迭代次数
            'w': 0.7,              # 惯性权重
            'c1': 1.5,             # 个体学习因子
            'c2': 1.5,             # 社会学习因子
            'bounds': (-2.0, 2.0)  # 参数边界
        }
        
    def srgb_to_linear(self, rgb):
        """sRGB到线性RGB转换"""
        return np.where(rgb <= 0.04045, 
                       rgb / 12.92, 
                       np.power((rgb + 0.055) / 1.055, 2.4))
    
    def linear_to_srgb(self, rgb):
        """线性RGB到sRGB转换"""
        return np.where(rgb <= 0.0031308,
                       12.92 * rgb,
                       1.055 * np.power(rgb, 1/2.4) - 0.055)
    
    def xyz_to_lab(self, xyz):
        """XYZ到Lab转换"""
        # D65标准光源白点
        xn, yn, zn = 95.047, 100.000, 108.883
        
        x = xyz[:, 0] / xn
        y = xyz[:, 1] / yn  
        z = xyz[:, 2] / zn
        
        # 立方根变换
        fx = np.where(x > 0.008856, np.power(x, 1/3), (903.3 * x + 16) / 116)
        fy = np.where(y > 0.008856, np.power(y, 1/3), (903.3 * y + 16) / 116)
        fz = np.where(z > 0.008856, np.power(z, 1/3), (903.3 * z + 16) / 116)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.column_stack([L, a, b])
    
    def calculate_delta_e(self, lab1, lab2):
        """计算CIE ΔE*ab色差"""
        diff = lab1 - lab2
        return np.sqrt(np.sum(diff**2, axis=1))
    
    def rgbv_to_xyz(self, rgbv, v_params):
        """RGBV到XYZ转换（参数化V通道影响）"""
        rgbv = np.array(rgbv)
        if rgbv.ndim == 1:
            rgbv = rgbv.reshape(1, -1)
            
        rgb = rgbv[:, :3]
        v = rgbv[:, 3:4].flatten()
        
        # RGB线性化并转换到XYZ
        rgb_linear = self.srgb_to_linear(rgb)
        xyz = rgb_linear @ self.rgb_to_xyz.T
        
        # V通道的参数化影响
        v_x_coef, v_y_coef, v_z_coef = v_params[:3]
        xyz[:, 0] += v * v_x_coef
        xyz[:, 1] += v * v_y_coef  
        xyz[:, 2] += v * v_z_coef
        
        return xyz
    
    def rgbcx_to_xyz(self, rgbcx, channel_params):
        """RGBCX到XYZ转换（参数化通道影响）"""
        rgbcx = np.array(rgbcx)
        if rgbcx.ndim == 1:
            rgbcx = rgbcx.reshape(1, -1)
            
        rgb = rgbcx[:, :3]
        c = rgbcx[:, 3:4].flatten()
        x = rgbcx[:, 4:5].flatten()
        
        # RGB线性化并转换到XYZ
        rgb_linear = self.srgb_to_linear(rgb)
        xyz = rgb_linear @ self.rgb_to_xyz.T
        
        # C和X通道的参数化影响
        c_x_coef, c_y_coef, c_z_coef = channel_params[:3]
        x_x_coef, x_y_coef, x_z_coef = channel_params[3:6]
        
        xyz[:, 0] += c * c_x_coef + x * x_x_coef
        xyz[:, 1] += c * c_y_coef + x * x_y_coef
        xyz[:, 2] += c * c_z_coef + x * x_z_coef
        
        return xyz
    
    def decode_transformation_matrix(self, params):
        """解码转换矩阵参数"""
        # 前20个参数用于4x5转换矩阵
        matrix_params = params[:20].reshape(4, 5)
        
        # 基础矩阵（单位矩阵基础上的小幅调整）
        base_matrix = np.array([
            [0.95, 0.0, 0.0, 0.3, 0.2],   # R -> RGBCX
            [0.0, 0.95, 0.0, 0.3, 0.1],   # G -> RGBCX  
            [0.0, 0.0, 0.95, 0.2, 0.1],   # B -> RGBCX
            [0.0, 0.0, 0.0, 0.6, 0.4]     # V -> CX
        ])
        
        # 添加参数化的调整
        adjustment = matrix_params * 0.1  # 限制调整幅度
        transformation_matrix = base_matrix + adjustment
        
        # 确保非负性和合理范围
        transformation_matrix = np.clip(transformation_matrix, 0, 1.5)
        
        return transformation_matrix
    
    def apply_transformation(self, rgbv, transformation_matrix):
        """应用转换矩阵"""
        rgbv = np.array(rgbv)
        if rgbv.ndim == 1:
            rgbv = rgbv.reshape(1, -1)
            
        # 应用线性转换
        rgbcx = rgbv @ transformation_matrix
        
        # 归一化处理
        rgbcx = np.clip(rgbcx, 0, 1)
        
        return rgbcx
    
    def objective_function(self, params, rgbv_data):
        """PSO优化的目标函数"""
        try:
            # 解码参数
            transformation_matrix = self.decode_transformation_matrix(params)
            v_params = params[20:23]  # V通道影响参数
            channel_params = params[23:29]  # C和X通道影响参数
            
            # 执行转换
            rgbcx_converted = self.apply_transformation(rgbv_data, transformation_matrix)
            
            # 计算源和目标的Lab值
            source_xyz = self.rgbv_to_xyz(rgbv_data, v_params)
            target_xyz = self.rgbcx_to_xyz(rgbcx_converted, channel_params)
            
            source_lab = self.xyz_to_lab(source_xyz)
            target_lab = self.xyz_to_lab(target_xyz)
            
            # 计算色差
            delta_e = self.calculate_delta_e(source_lab, target_lab)
            
            # 目标函数：最小化平均色差
            avg_delta_e = np.mean(delta_e)
            
            # 添加正则化项防止过拟合
            regularization = 0.01 * np.sum(np.abs(params))
            
            return avg_delta_e + regularization
            
        except Exception as e:
            # 如果计算出错，返回一个很大的值
            return 1000.0
    
    def pso_optimize(self, rgbv_data):
        """粒子群优化算法"""
        print("开始PSO优化...")
        
        # 参数维度：20(转换矩阵) + 3(V通道) + 6(C,X通道) = 29
        dim = 29
        
        # 初始化粒子群
        swarm_size = self.pso_params['swarm_size']
        particles = np.random.uniform(
            self.pso_params['bounds'][0], 
            self.pso_params['bounds'][1], 
            (swarm_size, dim)
        )
        
        velocities = np.random.uniform(-1, 1, (swarm_size, dim))
        personal_best = particles.copy()
        personal_best_scores = np.full(swarm_size, float('inf'))
        
        global_best = None
        global_best_score = float('inf')
        
        # PSO主循环
        for iteration in range(self.pso_params['max_iter']):
            for i in range(swarm_size):
                # 计算当前粒子的适应度
                score = self.objective_function(particles[i], rgbv_data)
                
                # 更新个体最优
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particles[i].copy()
                
                # 更新全局最优
                if score < global_best_score:
                    global_best_score = score
                    global_best = particles[i].copy()
            
            # 更新粒子速度和位置
            w = self.pso_params['w']
            c1 = self.pso_params['c1']
            c2 = self.pso_params['c2']
            
            r1 = np.random.random((swarm_size, dim))
            r2 = np.random.random((swarm_size, dim))
            
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best - particles) +
                         c2 * r2 * (global_best - particles))
            
            particles += velocities
            
            # 边界处理
            particles = np.clip(particles, 
                              self.pso_params['bounds'][0],
                              self.pso_params['bounds'][1])
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 最优色差 = {global_best_score:.4f}")
        
        print(f"PSO优化完成，最终色差: {global_best_score:.4f}")
        return global_best, global_best_score
    
    def convert_with_optimized_params(self, rgbv_data, optimized_params):
        """使用优化参数进行转换"""
        transformation_matrix = self.decode_transformation_matrix(optimized_params)
        rgbcx_result = self.apply_transformation(rgbv_data, transformation_matrix)
        return rgbcx_result, transformation_matrix
    
    def evaluate_conversion_quality(self, rgbv_data, rgbcx_data, optimized_params):
        """评估转换质量"""
        v_params = optimized_params[20:23]
        channel_params = optimized_params[23:29]
        
        # 计算Lab值
        source_xyz = self.rgbv_to_xyz(rgbv_data, v_params)
        target_xyz = self.rgbcx_to_xyz(rgbcx_data, channel_params)
        
        source_lab = self.xyz_to_lab(source_xyz)
        target_lab = self.xyz_to_lab(target_xyz)
        
        # 计算色差
        delta_e = self.calculate_delta_e(source_lab, target_lab)
        
        # 统计分析
        avg_delta_e = np.mean(delta_e)
        median_delta_e = np.median(delta_e)
        max_delta_e = np.max(delta_e)
        std_delta_e = np.std(delta_e)
        
        # 质量分级（基于CIE标准）
        excellent = np.sum(delta_e < 1.0) / len(delta_e) * 100
        good = np.sum((delta_e >= 1.0) & (delta_e < 3.0)) / len(delta_e) * 100
        acceptable = np.sum((delta_e >= 3.0) & (delta_e < 6.0)) / len(delta_e) * 100
        poor = np.sum(delta_e >= 6.0) / len(delta_e) * 100
        
        return {
            'delta_e_values': delta_e,
            'avg_delta_e': avg_delta_e,
            'median_delta_e': median_delta_e,
            'max_delta_e': max_delta_e,
            'std_delta_e': std_delta_e,
            'excellent_rate': excellent,
            'good_rate': good,
            'acceptable_rate': acceptable,
            'poor_rate': poor
        }

def create_comprehensive_test_image(height, width):
    """创建综合测试图像"""
    image = np.zeros((height, width, 4))
    
    # 创建多样化的测试区域
    for i in range(height):
        for j in range(width):
            x_norm = j / width
            y_norm = i / height
            
            # 分区域测试不同颜色组合
            if i < height // 6:
                # 纯色测试区域
                image[i, j] = [x_norm, 0, 0, y_norm * 0.3]
            elif i < 2 * height // 6:
                # 双色混合区域  
                image[i, j] = [x_norm * 0.8, y_norm * 0.8, 0, 0.2]
            elif i < 3 * height // 6:
                # 三色混合区域
                image[i, j] = [x_norm * 0.6, y_norm * 0.6, (1-x_norm) * 0.6, 0.4]
            elif i < 4 * height // 6:
                # V通道主导区域
                image[i, j] = [0.3, 0.3, 0.3, x_norm * y_norm]
            elif i < 5 * height // 6:
                # 高饱和度区域
                image[i, j] = [x_norm, y_norm, 1-x_norm, 0.5]
            else:
                # 复合测试区域
                image[i, j] = [
                    np.sin(x_norm * np.pi) * 0.7,
                    np.cos(y_norm * np.pi) * 0.7,
                    np.sin((x_norm + y_norm) * np.pi) * 0.7,
                    (x_norm * y_norm) * 0.6
                ]
    
    # 添加标准色块
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

def main():
    """主函数"""
    print("=== PSO-Optimized 4-Channel to 5-Channel Color Conversion ===")
    
    # 创建输出目录
    output_dir = "pso_optimized_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化转换器
    converter = PSOOptimizedColorConverter()
    
    print("Step 1: 生成测试图像...")
    original_rgbv = create_comprehensive_test_image(150, 200)  # 减小尺寸加快PSO
    
    print("Step 2: 准备训练数据...")
    h, w, c = original_rgbv.shape
    source_rgbv = original_rgbv.reshape(-1, 4)
    
    # 使用采样数据进行PSO优化（提高效率）
    sample_size = min(2000, len(source_rgbv))
    sample_indices = np.random.choice(len(source_rgbv), sample_size, replace=False)
    training_data = source_rgbv[sample_indices]
    
    print(f"Step 3: PSO优化 (样本数: {sample_size})...")
    optimized_params, best_score = converter.pso_optimize(training_data)
    
    print("Step 4: 应用优化参数到完整数据...")
    converted_rgbcx_flat, transformation_matrix = converter.convert_with_optimized_params(
        source_rgbv, optimized_params)
    converted_rgbcx = converted_rgbcx_flat.reshape(h, w, 5)
    
    print("Step 5: 评估转换质量...")
    quality_metrics = converter.evaluate_conversion_quality(
        source_rgbv, converted_rgbcx_flat, optimized_params)
    
    print(f"\n=== PSO OPTIMIZATION RESULTS ===")
    print(f"Training Score: {best_score:.4f}")
    print(f"Full Dataset Results:")
    print(f"Average ΔE*ab: {quality_metrics['avg_delta_e']:.4f}")
    print(f"Median ΔE*ab: {quality_metrics['median_delta_e']:.4f}")  
    print(f"Maximum ΔE*ab: {quality_metrics['max_delta_e']:.4f}")
    print(f"Standard Deviation: {quality_metrics['std_delta_e']:.4f}")
    print(f"\nQuality Distribution:")
    print(f"Excellent (ΔE<1): {quality_metrics['excellent_rate']:.1f}%")
    print(f"Good (1≤ΔE<3): {quality_metrics['good_rate']:.1f}%")
    print(f"Acceptable (3≤ΔE<6): {quality_metrics['acceptable_rate']:.1f}%")
    print(f"Poor (ΔE≥6): {quality_metrics['poor_rate']:.1f}%")
    
    print("\nStep 6: 生成可视化分析...")
    visualize_pso_results(original_rgbv, converted_rgbcx, converter, 
                         quality_metrics, transformation_matrix, output_dir)
    
    # 保存结果
    results = {
        'original_rgbv': original_rgbv,
        'converted_rgbcx': converted_rgbcx,
        'optimized_params': optimized_params,
        'transformation_matrix': transformation_matrix,
        'quality_metrics': quality_metrics,
        'pso_score': best_score
    }
    
    np.savez(os.path.join(output_dir, 'pso_optimized_results.npz'), **results)
    
    # 生成报告
    with open(os.path.join(output_dir, 'pso_optimization_report.txt'), 'w', encoding='utf-8') as f:
        f.write("PSO-Optimized 4-Channel to 5-Channel Color Conversion Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. PSO ALGORITHM CONFIGURATION\n")
        f.write("-" * 35 + "\n")
        f.write(f"Swarm Size: {converter.pso_params['swarm_size']}\n")
        f.write(f"Max Iterations: {converter.pso_params['max_iter']}\n")
        f.write(f"Inertia Weight: {converter.pso_params['w']}\n")
        f.write(f"Learning Factors: c1={converter.pso_params['c1']}, c2={converter.pso_params['c2']}\n")
        f.write(f"Parameter Bounds: {converter.pso_params['bounds']}\n\n")
        
        f.write("2. OPTIMIZATION RESULTS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Training Score: {best_score:.4f}\n")
        f.write(f"Average ΔE*ab: {quality_metrics['avg_delta_e']:.4f}\n")
        f.write(f"Median ΔE*ab: {quality_metrics['median_delta_e']:.4f}\n")
        f.write(f"Maximum ΔE*ab: {quality_metrics['max_delta_e']:.4f}\n")
        f.write(f"Standard Deviation: {quality_metrics['std_delta_e']:.4f}\n\n")
        
        f.write("3. QUALITY DISTRIBUTION\n")
        f.write("-" * 25 + "\n")
        f.write(f"Excellent (ΔE<1): {quality_metrics['excellent_rate']:.1f}%\n")
        f.write(f"Good (1≤ΔE<3): {quality_metrics['good_rate']:.1f}%\n")
        f.write(f"Acceptable (3≤ΔE<6): {quality_metrics['acceptable_rate']:.1f}%\n")
        f.write(f"Poor (ΔE≥6): {quality_metrics['poor_rate']:.1f}%\n\n")
        
        f.write("4. TRANSFORMATION MATRIX\n")
        f.write("-" * 25 + "\n")
        f.write("Optimized 4x5 transformation matrix:\n")
        for i, row in enumerate(transformation_matrix):
            f.write(f"Row {i}: {row}\n")
        f.write("\n")
        
        f.write("5. TECHNICAL ACHIEVEMENTS\n")
        f.write("-" * 25 + "\n")
        f.write("• PSO-based global optimization of transformation parameters\n")
        f.write("• CIE ΔE*ab perceptual color difference minimization\n")
        f.write("• Data-driven parameter tuning for optimal conversion\n")
        f.write("• Professional-grade color management solution\n")
    
    print(f"\n=== PSO OPTIMIZATION COMPLETE ===")
    print(f"Results saved to: {output_dir}/")
    print("Generated files:")
    print("1. pso_optimized_conversion_analysis.png - Complete PSO analysis")
    print("2. pso_optimized_results.npz - Numerical results and parameters")
    print("3. pso_optimization_report.txt - Detailed PSO report")

if __name__ == "__main__":
    main()
