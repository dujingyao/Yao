```python
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
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
        print("🔧 构建RGBV四通道数据...")
       
        R_channel = self.led_data['R_R']  # 红色LED的红色响应
        G_channel = self.led_data['G_G']  # 绿色LED的绿色响应  
        B_channel = self.led_data['B_B']  # 蓝色LED的蓝色响应
        
        V_channel = (0.299 * self.led_data['R_R'] + 
                    0.587 * self.led_data['G_G'] + 
                    0.114 * self.led_data['B_B'])
        # 组合成RGBV数组
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
       
        print("🎯 构建RGBCX五通道目标数据...")
        
        # 基础RGB通道
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
       
        initial_matrix = np.array([
            [0.95, 0.0,  0.0,  0.3,  0.2],   # R → RGBCX
            [0.0,  0.95, 0.0,  0.3,  0.1],   # G → RGBCX  
            [0.0,  0.0,  0.95, 0.2,  0.1],   # B → RGBCX
            [0.0,  0.0,  0.0,  0.6,  0.4]    # V → CX
        ])
        
        self.conversion_matrix = initial_matrix
        return initial_matrix
    
    def rgb_to_xyz(self, rgb):
        if np.max(rgb) > 1:
            rgb = rgb / 255.0
        
        # 伽马校正
        def gamma_correct(channel):
            if channel <= 0.04045:
                return channel / 12.92
            else:
                return np.power((channel + 0.055) / 1.055, 2.4)
        
        rgb_linear = np.array([gamma_correct(c) for c in rgb])
        
        # sRGB到XYZ转换矩阵
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = np.dot(M, rgb_linear) * 100
        return xyz
    
    def xyz_to_lab(self, xyz):
        # D65标准光源
        xyz_n = np.array([95.047, 100.000, 108.883])
        xyz_norm = xyz / xyz_n
        
        def f_func(t):
            delta = 6/29
            if t > delta**3:
                return np.power(t, 1/3)
            else:
                return t / (3 * delta**2) + 4/29
        
        fx = f_func(xyz_norm[0])
        fy = f_func(xyz_norm[1])
        fz = f_func(xyz_norm[2])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.array([L, a, b])
    
    def calculate_delta_e(self, lab1, lab2):
        return np.sqrt(np.sum((lab1 - lab2)**2))
    
    def fitness_function(self, params):
        try:
            matrix_params = params[:20].reshape(4, 5)
            v_xyz_factors = params[20:23]
            cx_xyz_factors = params[23:29].reshape(2, 3)
            
            base_matrix = self.initialize_conversion_matrix()
            conversion_matrix = base_matrix + matrix_params * 0.1  # 10%的调整幅度
           
            conversion_matrix = np.clip(conversion_matrix, 0, 1.5)
          
            rgbv_flat = self.rgbv_data.reshape(-1, 4)
            predicted_rgbcx = np.dot(rgbv_flat, conversion_matrix)
            
            v_influence = rgbv_flat[:, 3:4] * v_xyz_factors.reshape(1, 3)
          
            predicted_rgbcx[:, 3] *= cx_xyz_factors[0, 0]  # C通道
            predicted_rgbcx[:, 4] *= cx_xyz_factors[1, 0]  # X通道
            
            predicted_rgbcx = predicted_rgbcx.reshape(self.rgbcx_data.shape)
            actual_rgbcx = self.rgbcx_data
         
            total_delta_e = 0
            valid_pixels = 0
            
            height, width, _ = predicted_rgbcx.shape
            for i in range(0, height, 4):  # 采样以提高效率
                for j in range(0, width, 4):
                    try:
                        pred_rgb = predicted_rgbcx[i, j, :3]
                        pred_xyz = self.rgb_to_xyz(pred_rgb)
                        pred_lab = self.xyz_to_lab(pred_xyz)
                    
                        actual_rgb = actual_rgbcx[i, j, :3]
                        actual_xyz = self.rgb_to_xyz(actual_rgb)
                        actual_lab = self.xyz_to_lab(actual_xyz)
                        
                        delta_e = self.calculate_delta_e(pred_lab, actual_lab)
                        total_delta_e += delta_e
                        valid_pixels += 1
                        
                    except:
                        continue
            
            if valid_pixels == 0:
                return 1000.0
            
            avg_delta_e = total_delta_e / valid_pixels
        
            matrix_penalty = np.sum(np.abs(matrix_params)) * 0.01
            
            fitness = avg_delta_e + matrix_penalty
           
            self.optimization_history.append(fitness)
            
            return fitness
            
        except Exception as e:
            return 1000.0  # 返回大值表示失败
    
    def pso_optimize(self, swarm_size=30, max_iterations=100, w=0.7, c1=1.5, c2=1.5):
        print(f"🚀 开始PSO优化 (粒子数: {swarm_size}, 最大迭代: {max_iterations})")

        n_params = 29
        bounds = []
        
        # 转换矩阵参数边界
        for i in range(20):
            bounds.append((-0.5, 0.5))
        
        # V通道XYZ因子边界
        for i in range(3):
            bounds.append((0.5, 1.5))
            
        # C和X通道因子边界
        for i in range(6):
            bounds.append((0.5, 1.5))
        
        # 初始化粒子群
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(swarm_size):
            # 随机初始化粒子位置
            particle = []
            velocity = []
            
            for low, high in bounds:
                particle.append(np.random.uniform(low, high))
                velocity.append(np.random.uniform(-0.1, 0.1))
            
            particles.append(np.array(particle))
            velocities.append(np.array(velocity))
            
            # 计算初始适应度
            fitness = self.fitness_function(particles[-1])
            personal_best.append(particles[-1].copy())
            personal_best_fitness.append(fitness)
        
        # 找到全局最优
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        print(f"初始最优适应度: {global_best_fitness:.4f}")
        
        # 迭代优化
        stagnation_count = 0
        prev_best_fitness = global_best_fitness
        
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(swarm_size):
                # 更新速度
                r1, r2 = np.random.random(n_params), np.random.random(n_params)
                
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                
                velocities[i] = np.clip(velocities[i], -0.2, 0.2)
               
                particles[i] = particles[i] + velocities[i]
                
                for j, (low, high) in enumerate(bounds):
                    particles[i][j] = np.clip(particles[i][j], low, high)
             
                fitness = self.fitness_function(particles[i])
               
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                        improved = True
            
            # 早停检查
            if not improved:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # 每10代输出进度
            if iteration % 10 == 0:
                avg_fitness = np.mean(personal_best_fitness)
                print(f"第{iteration:3d}代: 最佳={global_best_fitness:.2f}, "
                      f"平均={avg_fitness:.2f}, 停滞={stagnation_count}")
            
            # 早停条件
            if stagnation_count >= 30:
                print(f"早停：连续{stagnation_count}代无改善")
                break
        
        print(f"✅ PSO优化完成！最终适应度: {global_best_fitness:.4f}")
        
        return global_best, global_best_fitness
    
    def apply_optimized_conversion(self, optimal_params):
    
        print("🔧 应用优化转换参数...")
        
        matrix_params = optimal_params[:20].reshape(4, 5)
        v_xyz_factors = optimal_params[20:23]
        cx_xyz_factors = optimal_params[23:29].reshape(2, 3)
        
        base_matrix = self.initialize_conversion_matrix()
        final_matrix = base_matrix + matrix_params * 0.1
        final_matrix = np.clip(final_matrix, 0, 1.5)
        
        rgbv_flat = self.rgbv_data.reshape(-1, 4)
        converted_rgbcx = np.dot(rgbv_flat, final_matrix)
        
        converted_rgbcx[:, 3] *= cx_xyz_factors[0, 0]  # C通道
        converted_rgbcx[:, 4] *= cx_xyz_factors[1, 0]  # X通道
        
        converted_rgbcx = converted_rgbcx.reshape(self.rgbcx_data.shape)
        
        self.conversion_matrix = final_matrix
        print("✅ 转换完成！")
        
        return converted_rgbcx
    
    def comprehensive_optimization(self):
        
        print("🎯 开始执行完整优化流程...")
        
        if not self.load_led_data():
            return None
        
        self.build_rgbv_channels()
        self.build_rgbcx_channels()
        
        self.initialize_conversion_matrix()
        
        optimal_params, best_fitness = self.pso_optimize(
            swarm_size=30, 
            max_iterations=100
        )
        converted_data = self.apply_optimized_conversion(optimal_params)
        
        evaluation_results = self.evaluate_conversion_quality(converted_data)
        
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
        
        print("📊 评估转换质量...")
        
        delta_e_values = []
        actual_data = self.rgbcx_data
        
        height, width, _ = converted_data.shape
        
        for i in range(0, height, 2):  # 采样评估
            for j in range(0, width, 2):
                try:
                    conv_rgb = converted_data[i, j, :3]
                    conv_xyz = self.rgb_to_xyz(conv_rgb)
                    conv_lab = self.xyz_to_lab(conv_xyz)
                    
                    actual_rgb = actual_data[i, j, :3]
                    actual_xyz = self.rgb_to_xyz(actual_rgb)
                    actual_lab = self.xyz_to_lab(actual_xyz)
                    
                    delta_e = self.calculate_delta_e(conv_lab, actual_lab)
                    delta_e_values.append(delta_e)
                    
                except:
                    continue
        
        delta_e_values = np.array(delta_e_values)
        
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
   
    converter = OptimizedFourToFiveChannelConverter('LED数据.xlsx')
    
    results = converter.comprehensive_optimization()
    
    if results:
        print("\n🎊 优化结果总结:")
        print(f"最优适应度: {results['best_fitness']:.4f}")
        print(f"平均色差: {results['evaluation']['mean_delta_e']:.4f}")
        print(f"优秀级别比例: {results['evaluation']['excellent_ratio']:.2%}")
        print(f"良好级别比例: {results['evaluation']['good_ratio']:.2%}")
```

## B.2 V通道分解算法

```python
def
    if method == 'adaptive':
  
        def calculate_adaptive_weights(rgb_pixel):
            """根据RGB比例动态调整权重"""
            total = np.sum(rgb_pixel)
            if total == 0:
                return np.array([0.299, 0.587, 0.114])  # 默认权重
            
            rgb_norm = rgb_pixel / total
            
            dominant_channel = np.argmax(rgb_norm)
            weights = np.array([0.299, 0.587, 0.114])
            
            weights[dominant_channel] *= 1.2
            weights = weights / np.sum(weights)  # 重新归一化
            
            return weights
        
        height, width, _ = rgb_data.shape
        v_channel = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                rgb_pixel = rgb_data[i, j]
                weights = calculate_adaptive_weights(rgb_pixel)
                v_channel[i, j] = np.dot(weights, rgb_pixel)
        
        return v_channel
    
    elif method == 'luminance':
        # 标准亮度分解
        luminance_weights = np.array([0.299, 0.587, 0.114])
        return np.dot(rgb_data, luminance_weights)
    
    elif method == 'contrast':
        contrast_weights = np.array([0.25, 0.50, 0.25])
        base_v = np.dot(rgb_data, contrast_weights)
        
        mean_v = np.mean(base_v)
        enhanced_v = mean_v + 1.2 * (base_v - mean_v)
        
        return np.clip(enhanced_v, 0, 255)

def v_channel_quality_assessment(v_channel, rgb_original):
    
    rgb_luminance = 0.299 * rgb_original[:,:,0] + 0.587 * rgb_original[:,:,1] + 0.114 * rgb_original[:,:,2]
    correlation = np.corrcoef(v_channel.flatten(), rgb_luminance.flatten())[0,1]
    
    def calculate_entropy(data):
        hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 255))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # 移除零值
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    v_entropy = calculate_entropy(v_channel)
    rgb_entropy = np.mean([calculate_entropy(rgb_original[:,:,i]) for i in range(3)])
    
    v_range = np.max(v_channel) - np.min(v_channel)
    rgb_range = np.mean([np.max(rgb_original[:,:,i]) - np.min(rgb_original[:,:,i]) for i in range(3)])
    
    quality_metrics = {
        'luminance_correlation': correlation,
        'entropy_ratio': v_entropy / rgb_entropy,
        'dynamic_range_ratio': v_range / rgb_range,
        'uniformity_index': 1 / (1 + np.std(v_channel) / np.mean(v_channel))
    }
    
    return quality_metrics
```

## B.3 差分进化优化器

```python
class DifferentialEvolutionOptimizer:
   
    def __init__(self, converter):
        self.converter = converter
        
    def optimize(self, bounds, population_size=50, max_generations=200, F=0.5, CR=0.7):
       
        print(f"🧬 开始差分进化优化 (种群: {population_size}, 代数: {max_generations})")
        
        n_params = len(bounds)
        
        population = []
        fitness_values = []
        
        for _ in range(population_size):
            individual = []
            for low, high in bounds:
                individual.append(np.random.uniform(low, high))
            
            individual = np.array(individual)
            population.append(individual)
            
            fitness = self.converter.fitness_function(individual)
            fitness_values.append(fitness)
        
        population = np.array(population)
        fitness_values = np.array(fitness_values)
        
        best_idx = np.argmin(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        print(f"初始最优适应度: {best_fitness:.4f}")
        
        for generation in range(max_generations):
            new_population = []
            new_fitness = []
            
            for i in range(population_size):
                # 选择三个不同的个体进行变异
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                mutant = population[a] + F * (population[b] - population[c])
                
                for j, (low, high) in enumerate(bounds):
                    mutant[j] = np.clip(mutant[j], low, high)
                
                trial = population[i].copy()
                crossover_points = np.random.random(n_params) < CR
                trial[crossover_points] = mutant[crossover_points]
                
                if not np.any(crossover_points):
                    j = np.random.randint(n_params)
                    trial[j] = mutant[j]
                
                trial_fitness = self.converter.fitness_function(trial)
                
                if trial_fitness < fitness_values[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                    
                    if trial_fitness < best_fitness:
                        best_individual = trial.copy()
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness_values[i])
            
            population = np.array(new_population)
            fitness_values = np.array(new_fitness)
            
            if generation % 20 == 0:
                avg_fitness = np.mean(fitness_values)
                print(f"第{generation:3d}代: 最佳={best_fitness:.4f}, 平均={avg_fitness:.4f}")
        
        print(f"✅ 差分进化完成！最终适应度: {best_fitness:.4f}")
        return best_individual, best_fitness

def advanced_gamut_mapping(source_channels, target_channels):
   
    def calculate_gamut_boundary(channels):
        xy_points = []
        for i in range(0, channels.shape[0], 10):  # 采样
            for j in range(0, channels.shape[1], 10):
                rgb = channels[i, j, :3]
                if np.sum(rgb) > 10:  # 忽略暗像素
                    # 转换到XYZ
                    xyz = rgb_to_xyz_simple(rgb)
                    if np.sum(xyz) > 0:
                        x = xyz[0] / np.sum(xyz)
                        y = xyz[1] / np.sum(xyz)
                        xy_points.append([x, y])
        
        return np.array(xy_points)
    
    def rgb_to_xyz_simple(rgb):
        rgb_norm = rgb / 255.0
        M = np.array([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9503]
        ])
        return np.dot(M, rgb_norm)
    
    source_gamut = calculate_gamut_boundary(source_channels)
    target_gamut = calculate_gamut_boundary(target_channels)
    
    if len(source_gamut) == 0 or len(target_gamut) == 0:
        return target_channels  # 如果无法计算色域，返回原始目标
    
    source_center = np.mean(source_gamut, axis=0)
    target_center = np.mean(target_gamut, axis=0)
    
    source_extent = np.max(source_gamut, axis=0) - np.min(source_gamut, axis=0)
    target_extent = np.max(target_gamut, axis=0) - np.min(target_gamut, axis=0)
    
    scale_factors = target_extent / (source_extent + 1e-8)
    translation = target_center - source_center * scale_factors
    
    mapped_channels = target_channels.copy()
    
    for i in range(3):
        channel_mean = np.mean(source_channels[:, :, i])
        target_mean = np.mean(target_channels[:, :, i])
        if channel_mean > 0:
            adjustment_factor = target_mean / channel_mean
            mapped_channels[:, :, i] = source_channels[:, :, i] * adjustment_factor
    
    return mapped_channels
```

## B.4 算法性能评估

```python
def evaluate_pso_performance(optimization_history, evaluation_results):
    """
    评估PSO算法性能
    
    参数:
        optimization_history: PSO优化历史记录
        evaluation_results: 转换质量评估结果
    
    返回:
        PSO性能评估报告
    """
    performance_metrics = {
        'convergence_analysis': {},
        'optimization_efficiency': {},
        'quality_assessment': {}
    }
    
    # 收敛分析
    if optimization_history:
        initial_fitness = optimization_history[0]
        final_fitness = optimization_history[-1]
        improvement_ratio = (initial_fitness - final_fitness) / initial_fitness
        
        # 计算收敛速度
        half_improvement = initial_fitness - (initial_fitness - final_fitness) / 2
        convergence_generation = 0
        for i, fitness in enumerate(optimization_history):
            if fitness <= half_improvement:
                convergence_generation = i
                break
        
        performance_metrics['convergence_analysis'] = {
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'improvement_ratio': improvement_ratio,
            'convergence_speed': convergence_generation,
            'total_generations': len(optimization_history),
            'convergence_stability': np.std(optimization_history[-20:]) if len(optimization_history) >= 20 else 0
        }
    
    # 优化效率
    performance_metrics['optimization_efficiency'] = {
        'parameter_space_dimension': 29,
        'exploration_effectiveness': improvement_ratio > 0.5,
        'exploitation_balance': np.std(optimization_history) / np.mean(optimization_history) if optimization_history else 0
    }
    
    # 质量评估整合
    if evaluation_results:
        performance_metrics['quality_assessment'] = {
            'color_accuracy': evaluation_results.get('mean_delta_e', 0),
            'consistency': 1 / (1 + evaluation_results.get('std_delta_e', 1)),
            'excellent_coverage': evaluation_results.get('excellent_ratio', 0),
            'overall_grade': 'A' if evaluation_results.get('excellent_ratio', 0) > 0.8 else 
                           'B' if evaluation_results.get('good_ratio', 0) > 0.9 else 'C'
        }
    
    return performance_metrics

def calculate_conversion_matrix_properties(conversion_matrix):
    """
    计算转换矩阵的数学性质
    
    参数:
        conversion_matrix: 4×5转换矩阵
    
    返回:
        矩阵性质分析结果
    """
    properties = {
        'matrix_analysis': {},
        'numerical_stability': {},
        'physical_constraints': {}
    }
    
    # 矩阵分析
    matrix_rank = np.linalg.matrix_rank(conversion_matrix)
    condition_number = np.linalg.cond(conversion_matrix[:, :4])  # 取4×4子矩阵
    
    properties['matrix_analysis'] = {
        'matrix_rank': matrix_rank,
        'condition_number': condition_number,
        'determinant': np.linalg.det(conversion_matrix[:, :4]),
        'frobenius_norm': np.linalg.norm(conversion_matrix, 'fro')
    }
    
    # 数值稳定性
    eigenvalues = np.linalg.eigvals(conversion_matrix[:, :4])
    properties['numerical_stability'] = {
        'eigenvalues': eigenvalues.tolist(),
        'spectral_radius': np.max(np.abs(eigenvalues)),
        'stability_indicator': np.all(np.abs(eigenvalues) < 2.0)
    }
    
    # 物理约束检查
    non_negative_ratio = np.sum(conversion_matrix >= 0) / conversion_matrix.size
    reasonable_range_ratio = np.sum((conversion_matrix >= 0) & (conversion_matrix <= 1.5)) / conversion_matrix.size
    
    properties['physical_constraints'] = {
        'non_negative_ratio': non_negative_ratio,
        'reasonable_range_ratio': reasonable_range_ratio,
        'energy_conservation': np.sum(conversion_matrix, axis=1).tolist(),
        'physical_validity': non_negative_ratio > 0.8 and reasonable_range_ratio > 0.9
    }
    
    return properties

def generate_optimization_report(converter_results):
    """
    生成完整的优化报告
    
    参数:
        converter_results: 转换器完整结果
    
    返回:
        格式化的优化报告
    """
    if not converter_results:
        return "优化失败，无法生成报告"
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("PSO优化4通道到5通道颜色转换 - 性能报告")
    report_lines.append("=" * 60)
    
    # 基本信息
    report_lines.append(f"\n📊 基本信息:")
    report_lines.append(f"   最优适应度: {converter_results['best_fitness']:.4f}")
    report_lines.append(f"   优化参数维度: 29维")
    report_lines.append(f"   转换矩阵规模: 4×5")
    
    # 质量评估
    evaluation = converter_results['evaluation']
    report_lines.append(f"\n🎯 转换质量:")
    report_lines.append(f"   平均色差ΔE: {evaluation['mean_delta_e']:.4f}")
    report_lines.append(f"   优秀级别比例: {evaluation['excellent_ratio']:.2%}")
    report_lines.append(f"   良好级别比例: {evaluation['good_ratio']:.2%}")
    report_lines.append(f"   可接受比例: {evaluation['acceptable_ratio']:.2%}")
    
    # 性能等级
    excellent_ratio = evaluation['excellent_ratio']
    if excellent_ratio >= 0.8:
        grade = "A+ (卓越)"
    elif excellent_ratio >= 0.6:
        grade = "A (优秀)"
    elif excellent_ratio >= 0.4:
        grade = "B (良好)"
    else:
        grade = "C (需改进)"
    
    report_lines.append(f"\n⭐ 整体评级: {grade}")
    
    # 收敛分析
    history = converter_results['optimization_history']
    if history:
        improvement = (history[0] - history[-1]) / history[0] * 100
        report_lines.append(f"\n📈 收敛性能:")
        report_lines.append(f"   适应度改善: {improvement:.1f}%")
        report_lines.append(f"   迭代次数: {len(history)}")
        report_lines.append(f"   收敛稳定性: {'稳定' if np.std(history[-10:]) < 0.1 else '需优化'}")
    
    # 矩阵性质
    matrix_props = calculate_conversion_matrix_properties(converter_results['conversion_matrix'])
    report_lines.append(f"\n🔧 转换矩阵:")
    report_lines.append(f"   数值稳定性: {'良好' if matrix_props['numerical_stability']['stability_indicator'] else '需注意'}")
    report_lines.append(f"   物理有效性: {'有效' if matrix_props['physical_constraints']['physical_validity'] else '需检查'}")
    
    report_lines.append(f"\n" + "=" * 60)
    
    return "\n".join(report_lines)
```
