import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, XYZColor, LabColor
from colormath.color_conversions import convert_color

class OptimizedFourToFiveChannelConverter:
    def __init__(self, excel_file_path='LED数据.xlsx'):
        self.excel_file = excel_file_path
        self.led_data = {}
        self.rgbv_data = None
        self.rgbcx_data = None
        self.conversion_matrix = None
        self.optimization_history = []
    
    def load_led_data(self):
        """加载LED光谱响应数据"""
        sheet_names = ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']
        print("📂 正在加载LED数据...")
        for sheet in sheet_names:
            try:
                df = pd.read_excel(self.excel_file, sheet_name=sheet, header=None)
                self.led_data[sheet] = df.values
                print(f"✅ 成功加载 {sheet}: {df.shape}")
            except Exception as e:
                print(f"❌ 加载 {sheet} 失败: {e}")
        return len(self.led_data) == 9

    def build_rgbv_channels(self):
        """构建RGBV四通道数据"""
        print("🔧 构建RGBV四通道数据...")
        R_channel = self.led_data['R_R']
        G_channel = self.led_data['G_G']
        B_channel = self.led_data['B_B']
        V_channel = 0.299 * R_channel + 0.587 * G_channel + 0.114 * B_channel
        
        height, width = R_channel.shape
        rgbv_array = np.zeros((height, width, 4))
        rgbv_array[:, :, 0] = R_channel
        rgbv_array[:, :, 1] = G_channel
        rgbv_array[:, :, 2] = B_channel
        rgbv_array[:, :, 3] = V_channel
        
        self.rgbv_data = rgbv_array
        print(f"✅ RGBV数据构建完成: {rgbv_array.shape}")
        return rgbv_array
    
    def build_rgbcx_channels(self):
        """构建RGBCX五通道目标数据"""
        print("🎯 构建RGBCX五通道目标数据...")
        R_target = self.led_data['R_R']
        G_target = self.led_data['G_G']
        B_target = self.led_data['B_B']
        C_channel = 0.6 * self.led_data['G_B'] + 0.4 * self.led_data['B_G']
        X_channel = (0.2 * self.led_data['R_G'] + 
                    0.3 * self.led_data['G_R'] + 
                    0.3 * self.led_data['B_R'] +
                    0.2 * self.led_data['R_B'])
        
        height, width = R_target.shape
        rgbcx_array = np.zeros((height, width, 5))
        rgbcx_array[:, :, 0] = R_target
        rgbcx_array[:, :, 1] = G_target
        rgbcx_array[:, :, 2] = B_target
        rgbcx_array[:, :, 3] = C_channel
        rgbcx_array[:, :, 4] = X_channel
        
        self.rgbcx_data = rgbcx_array
        print(f"✅ RGBCX数据构建完成: {rgbcx_array.shape}")
        return rgbcx_array
    
    def initialize_conversion_matrix(self):
        """初始化转换矩阵"""
        initial_matrix = np.array([
            [0.95, 0.0,  0.0,  0.3,  0.2],
            [0.0,  0.95, 0.0,  0.3,  0.1],
            [0.0,  0.0,  0.95, 0.2,  0.1],
            [0.0,  0.0,  0.0,  0.6,  0.4]
        ])
        self.conversion_matrix = initial_matrix
        return initial_matrix
    
    def rgb_to_xyz(self, rgb):
        """RGB转XYZ颜色空间"""
        if np.max(rgb) > 1:
            rgb = rgb / 255.0
        
        # 伽马校正
        def gamma_correct(channel):
            return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

        rgb_linear = np.array([gamma_correct(c) for c in rgb])
        
        # sRGB到XYZ转换
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        return np.dot(M, rgb_linear) * 100
    
    def xyz_to_lab(self, xyz):
        """XYZ转Lab颜色空间"""
        xyz_n = np.array([95.047, 100.000, 108.883])  # D65标准光源
        xyz_norm = xyz / xyz_n
        
        def f_func(t):
            delta = 6/29
            return np.power(t, 1/3) if t > delta**3 else t/(3*delta**2) + 4/29
        
        fx, fy, fz = f_func(xyz_norm[0]), f_func(xyz_norm[1]), f_func(xyz_norm[2])
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.array([L, a, b])
    
    def calculate_delta_e(self, lab1, lab2):
        """计算CIEDE2000色差"""
        return np.sqrt(np.sum((lab1 - lab2) ** 2))

    def fitness_function(self, params):
        """PSO适应度函数"""
        try:
            matrix_params = params[:20].reshape(4, 5)
            v_xyz_factors = params[20:23]
            cx_xyz_factors = params[23:29].reshape(2, 3)
            
            base_matrix = self.initialize_conversion_matrix()
            conversion_matrix = np.clip(base_matrix + matrix_params * 0.1, 0, 1.5)
            
            rgbv_flat = self.rgbv_data.reshape(-1, 4)
            predicted_rgbcx = np.dot(rgbv_flat, conversion_matrix)
            
            # 应用通道特定优化因子
            v_influence = rgbv_flat[:, 3:4] * v_xyz_factors.reshape(1, 3)
            predicted_rgbcx[:, 3] *= cx_xyz_factors[0, 0]
            predicted_rgbcx[:, 4] *= cx_xyz_factors[1, 0]
            
            predicted_rgbcx = predicted_rgbcx.reshape(self.rgbcx_data.shape)
            actual_rgbcx = self.rgbcx_data
            
            # 计算色差指标
            total_delta_e, valid_pixels = 0, 0
            height, width, _ = predicted_rgbcx.shape
            
            for i in range(0, height, 4):
                for j in range(0, width, 4):
                    pred_rgb = predicted_rgbcx[i, j, :3]
                    pred_lab = self.xyz_to_lab(self.rgb_to_xyz(pred_rgb))
                    
                    actual_rgb = actual_rgbcx[i, j, :3]
                    actual_lab = self.xyz_to_lab(self.rgb_to_xyz(actual_rgb))
                    
                    total_delta_e += self.calculate_delta_e(pred_lab, actual_lab)
                    valid_pixels += 1
            
            if valid_pixels == 0:
                return 1000.0
                
            avg_delta_e = total_delta_e / valid_pixels
            matrix_penalty = np.sum(np.abs(matrix_params)) * 0.01
            fitness = avg_delta_e + matrix_penalty
            self.optimization_history.append(fitness)
            
            return fitness
            
        except Exception as e:
            print(f"适应度计算错误: {e}")
            return 1000.0
    
    def pso_optimize(self, swarm_size=30, max_iterations=100, w=0.7, c1=1.5, c2=1.5):
        """PSO优化算法核心实现"""
        print(f"🚀 开始PSO优化 (粒子数: {swarm_size}, 最大迭代: {max_iterations})")
        
        # 参数边界设置
        n_params = 29
        bounds = [(-0.5, 0.5)] * 20 + [(0.5, 1.5)] * 9
        
        # 粒子群初始化
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(swarm_size):
            particle = np.array([np.random.uniform(low, high) for low, high in bounds])
            velocity = np.random.uniform(-0.1, 0.1, n_params)
            
            particles.append(particle)
            velocities.append(velocity)
            
            fitness = self.fitness_function(particle)
            personal_best.append(particle.copy())
            personal_best_fitness.append(fitness)
        
        # 全局最优初始化
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        print(f"初始最优适应度: {global_best_fitness:.4f}")
        
        # PSO迭代优化
        stagnation_count = 0
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(swarm_size):
                # 速度更新
                r1, r2 = np.random.random(n_params), np.random.random(n_params)
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -0.2, 0.2)
                
                # 位置更新
                particles[i] += velocities[i]
                for j in range(n_params):
                    low, high = bounds[j]
                    particles[i][j] = np.clip(particles[i][j], low, high)
                
                # 适应度评估
                fitness = self.fitness_function(particles[i])
                
                # 更新个体最优
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # 更新全局最优
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                        improved = True
                        stagnation_count = 0
            
            # 早停检测
            if not improved:
                stagnation_count += 1
                if stagnation_count >= 30:
                    print(f"早停：连续{stagnation_count}代无改善")
                    break
            
            # 进度输出
            if iteration % 10 == 0:
                avg_fitness = np.mean(personal_best_fitness)
                print(f"第{iteration:3d}代: 最佳={global_best_fitness:.2f}, "
                      f"平均={avg_fitness:.2f}, 停滞={stagnation_count}")
        
        print(f"✅ PSO优化完成！最终适应度: {global_best_fitness:.4f}")
        return global_best, global_best_fitness
    
    def apply_optimized_conversion(self, optimal_params):
        """应用优化后的转换参数"""
        print("🔧 应用优化转换参数...")
        
        matrix_params = optimal_params[:20].reshape(4, 5)
        v_xyz_factors = optimal_params[20:23]
        cx_xyz_factors = optimal_params[23:29].reshape(2, 3)
        
        base_matrix = self.initialize_conversion_matrix()
        final_matrix = np.clip(base_matrix + matrix_params * 0.1, 0, 1.5)
        
        rgbv_flat = self.rgbv_data.reshape(-1, 4)
        converted_rgbcx = np.dot(rgbv_flat, final_matrix)
        
        # 应用通道优化因子
        converted_rgbcx[:, 3] *= cx_xyz_factors[0, 0]  # C通道
        converted_rgbcx[:, 4] *= cx_xyz_factors[1, 0]  # X通道
        
        converted_rgbcx = converted_rgbcx.reshape(self.rgbcx_data.shape)
        self.conversion_matrix = final_matrix
        print("✅ 转换完成！")
        return converted_rgbcx
    
    def comprehensive_optimization(self):
        """综合优化流程控制"""
        print("🎯 开始执行完整优化流程...")
        
        if not self.load_led_data():
            return None
        
        self.build_rgbv_channels()
        self.build_rgbcx_channels()
        self.initialize_conversion_matrix()
        
        # 执行PSO优化
        optimal_params, best_fitness = self.pso_optimize(
            swarm_size=30, 
            max_iterations=100
        )
        
        # 应用优化结果
        converted_data = self.apply_optimized_conversion(optimal_params)
        evaluation_results = self.evaluate_conversion_quality(converted_data)
        
        # 结果整合
        results = {
            'optimal_parameters': optimal_params,
            'best_fitness': best_fitness,
            'conversion_matrix': self.conversion_matrix,
            'converted_data': converted_data,
            'evaluation': evaluation_results,
            'optimization_history': self.optimization_history
        }
        
        print("🎉 完整优化流程执行完成！")
        return results
    
    def evaluate_conversion_quality(self, converted_data):
        """评估转换质量"""
        print("📊 评估转换质量...")
        
        delta_e_values = []
        actual_data = self.rgbcx_data
        height, width, _ = converted_data.shape
        
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                conv_rgb = converted_data[i, j, :3]
                conv_lab = self.xyz_to_lab(self.rgb_to_xyz(conv_rgb))
                
                actual_rgb = actual_data[i, j, :3]
                actual_lab = self.xyz_to_lab(self.rgb_to_xyz(actual_rgb))
                
                delta_e = self.calculate_delta_e(conv_lab, actual_lab)
                delta_e_values.append(delta_e)
        
        delta_e_values = np.array(delta_e_values)
        
        # 质量指标计算
        quality_metrics = {
            'mean_delta_e': np.mean(delta_e_values),
            'median_delta_e': np.median(delta_e_values),
            'std_delta_e': np.std(delta_e_values),
            'max_delta_e': np.max(delta_e_values),
            'min_delta_e': np.min(delta_e_values),
            'percentile_95': np.percentile(delta_e_values, 95),
            'excellent_ratio': np.sum(delta_e_values < 1.0) / len(delta_e_values),
            'good_ratio': np.sum(delta_e_values < 3.0) / len(delta_e_values),
            'acceptable_ratio': np.sum(delta_e_values < 6.0) / len(delta_e_values)
        }
        
        print(f"✅ 质量评估完成 - 平均ΔE: {quality_metrics['mean_delta_e']:.4f}")
        return quality_metrics

if __name__ == "__main__":
    # 实例化并执行优化
    converter = OptimizedFourToFiveChannelConverter('LED数据.xlsx')
    results = converter.comprehensive_optimization()
    
    # 输出优化结果
    if results:
        print("\n🎊 优化结果总结:")
        print(f"最优适应度: {results['best_fitness']:.4f}")
        print(f"平均色差: {results['evaluation']['mean_delta_e']:.4f}")
        print(f"优秀级别比例: {results['evaluation']['excellent_ratio']:.2%}")
        print(f"良好级别比例: {results['evaluation']['good_ratio']:.2%}")