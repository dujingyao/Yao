#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED校准系统 - 根据用户指导优化版本
问题三：基于用户分析指导的精确校准解决方案

核心改进：
1. 目标值调整：[220, 220, 200] - 平衡绿/蓝差异
2. 权重优化：红色误差×2，强化红色抑制
3. 验证标准：红≤220，绿≈220，蓝180-200
4. 均匀性保证：散点分布不偏移

作者：GitHub Copilot (基于用户指导)
版本：优化版本 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import List, Tuple
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedLEDCalibrator:
    """优化的LED校准器 - 基于用户指导"""
    
    def __init__(self, target_rgb=(220, 220, 200)):
        """
        初始化校准器
        target_rgb: 目标RGB值 [220, 220, 200] - 平衡绿/蓝差异
        """
        self.target_rgb = np.array(target_rgb)
        self.rgb_data = None
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []
        
        # 优化的遗传算法参数
        self.population_size = 80  # 增大种群
        self.generations = 150     # 增加代数
        self.cx_prob = 0.8        # 提高交叉概率
        self.mut_prob = 0.15      # 降低变异概率，提高稳定性
        
        # 权重设置
        self.channel_weights = np.array([2.0, 1.0, 1.2])  # R×2, G×1, B×1.2
        
    def load_real_led_data(self, file_path: str):
        """加载真实LED数据"""
        try:
            print(f"正在加载真实LED数据: {file_path}")
            
            # 读取RGB目标值
            target_df = pd.read_excel(file_path, sheet_name='RGB目标值')
            
            # 读取各通道响应数据
            led_responses = {}
            sheet_names = ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']
            
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    numeric_data = df.select_dtypes(include=[np.number]).values.flatten()
                    led_responses[sheet_name] = numeric_data
                    print(f"  {sheet_name}: {len(numeric_data)} 数据点")
                except Exception as e:
                    print(f"  警告: 无法加载 {sheet_name}: {e}")
            
            if len(led_responses) >= 9:
                n_pixels = len(led_responses['R_R'])
                rgb_matrix = np.zeros((n_pixels, 3))
                
                # 使用主要通道数据
                rgb_matrix[:, 0] = led_responses['R_R']  # 红色
                rgb_matrix[:, 1] = led_responses['G_G']  # 绿色
                rgb_matrix[:, 2] = led_responses['B_B']  # 蓝色
                
                self.rgb_data = rgb_matrix
                
                print(f"\\n真实LED数据加载成功:")
                print(f"  数据形状: {rgb_matrix.shape}")
                print(f"  红色范围: [{rgb_matrix[:, 0].min():.1f}, {rgb_matrix[:, 0].max():.1f}]")
                print(f"  绿色范围: [{rgb_matrix[:, 1].min():.1f}, {rgb_matrix[:, 1].max():.1f}]")
                print(f"  蓝色范围: [{rgb_matrix[:, 2].min():.1f}, {rgb_matrix[:, 2].max():.1f}]")
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"加载真实数据失败: {e}")
            return False
    
    def generate_realistic_data(self, n_pixels=500):
        """生成更真实的LED数据 - 模拟红色过饱和问题"""
        print(f"生成模拟LED数据 ({n_pixels} 像素)...")
        
        np.random.seed(42)
        rgb_data = np.zeros((n_pixels, 3))
        
        for i in range(n_pixels):
            # 位置效应
            position_factor = 1.0 + 0.2 * np.sin(4 * np.pi * i / n_pixels)
            
            # 模拟实际问题：红色过饱和，蓝色不足
            base_values = [235, 215, 175]  # 红色偏高，绿色适中，蓝色偏低
            
            # 通道特定的噪声和偏差
            noise = np.random.normal(0, 15, 3)  # 增加噪声模拟真实情况
            
            # 额外的通道偏差
            channel_bias = [1.1, 1.0, 0.85]  # 红色偏高，蓝色偏低
            
            for c in range(3):
                rgb_data[i, c] = base_values[c] * position_factor * channel_bias[c] + noise[c]
        
        # 确保数据在合理范围
        rgb_data = np.clip(rgb_data, 150, 300)
        self.rgb_data = rgb_data
        
        print(f"模拟数据特征:")
        print(f"  红色: 均值={rgb_data[:, 0].mean():.1f}, 范围=[{rgb_data[:, 0].min():.1f}, {rgb_data[:, 0].max():.1f}]")
        print(f"  绿色: 均值={rgb_data[:, 1].mean():.1f}, 范围=[{rgb_data[:, 1].min():.1f}, {rgb_data[:, 1].max():.1f}]")
        print(f"  蓝色: 均值={rgb_data[:, 2].mean():.1f}, 范围=[{rgb_data[:, 2].min():.1f}, {rgb_data[:, 2].max():.1f}]")
        
        return rgb_data
    
    def advanced_fitness_function(self, individual: List[float]) -> float:
        """高级适应度函数 - 基于用户指导优化"""
        C_R, C_G, C_B = individual
        
        # 应用校正
        corrected_rgb = self.rgb_data * np.array([C_R, C_G, C_B])
        
        # 分通道计算目标偏差
        target_matrix = np.tile(self.target_rgb, (len(self.rgb_data), 1))
        diff = corrected_rgb - target_matrix
        
        # 基础误差（加权）
        channel_errors = np.abs(diff).mean(axis=0)  # 每通道平均误差
        weighted_error = np.sum(channel_errors * self.channel_weights)
        
        # 特殊目标惩罚
        penalty = 0
        
        # 红色抑制：超过220的严重惩罚
        red_excess = np.maximum(0, corrected_rgb[:, 0] - 220)
        penalty += np.sum(red_excess) * 5  # 红色超标重罚
        
        # 绿色精确性：偏离220的惩罚
        green_deviation = np.abs(corrected_rgb[:, 1] - 220)
        penalty += np.sum(green_deviation) * 1.5
        
        # 蓝色范围：不在180-200区间的惩罚
        blue_low_penalty = np.maximum(0, 180 - corrected_rgb[:, 2])
        blue_high_penalty = np.maximum(0, corrected_rgb[:, 2] - 200)
        penalty += np.sum(blue_low_penalty + blue_high_penalty) * 2
        
        # 均匀性惩罚
        uniformity_penalty = 0
        for channel in range(3):
            values = corrected_rgb[:, channel]
            if values.mean() > 0:
                cv = values.std() / values.mean()
                uniformity_penalty += cv * 500 * (channel + 1)  # 红色均匀性更重要
        
        # 空间连续性（如果数据足够大）
        spatial_penalty = 0
        if len(corrected_rgb) >= 64*64:  # 假设是64x64布局
            try:
                for channel in range(3):
                    matrix = corrected_rgb[:4096, channel].reshape(64, 64)
                    grad_x = np.abs(np.diff(matrix, axis=1)).mean()
                    grad_y = np.abs(np.diff(matrix, axis=0)).mean()
                    spatial_penalty += (grad_x + grad_y) * 0.1
            except:
                pass
        
        total_fitness = weighted_error + penalty + uniformity_penalty + spatial_penalty
        return total_fitness
    
    def create_individual(self) -> List[float]:
        """创建个体 - 智能初始化"""
        # 基于当前数据特征智能设置初始范围
        if self.rgb_data is not None:
            current_mean = self.rgb_data.mean(axis=0)
            target_ratios = self.target_rgb / current_mean
            
            # 在合理范围内随机化
            factors = []
            for i, ratio in enumerate(target_ratios):
                base = np.clip(ratio, 0.3, 2.0)
                noise = random.uniform(-0.1, 0.1)
                factors.append(np.clip(base + noise, 0.3, 2.0))
            
            return factors
        else:
            return [random.uniform(0.5, 1.5) for _ in range(3)]
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """改进的交叉操作"""
        # 模拟二进制交叉
        eta = 2.0  # 分布指数
        children = []
        
        for _ in range(2):
            child = []
            for i in range(3):
                if random.random() < 0.5:
                    u = random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child.append(np.clip(c1, 0.3, 2.0))
                else:
                    child.append(parent1[i] if random.random() < 0.5 else parent2[i])
            children.append(child)
        
        return children[0], children[1]
    
    def mutate(self, individual: List[float]) -> List[float]:
        """改进的变异操作"""
        mutated = individual.copy()
        
        for i in range(3):
            if random.random() < 0.3:  # 30%变异概率
                # 高斯变异
                sigma = 0.05  # 小的变异步长
                delta = random.gauss(0, sigma)
                mutated[i] = np.clip(mutated[i] + delta, 0.3, 2.0)
        
        return mutated
    
    def tournament_selection(self, population: List[List[float]], 
                           fitnesses: List[float], k=5) -> List[float]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        return population[winner_index].copy()
    
    def optimize(self):
        """执行优化算法"""
        if self.rgb_data is None:
            raise ValueError("请先加载或生成LED数据")
        
        print("\\n=== 开始优化校准算法 ===")
        print(f"优化目标: {self.target_rgb}")
        print(f"通道权重: 红×{self.channel_weights[0]}, 绿×{self.channel_weights[1]}, 蓝×{self.channel_weights[2]}")
        print(f"种群大小: {self.population_size}, 进化代数: {self.generations}")
        
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]
        self.fitness_history = []
        
        best_ever_fitness = float('inf')
        best_ever_individual = None
        stagnation_count = 0
        
        for gen in range(self.generations):
            # 计算适应度
            fitnesses = [self.advanced_fitness_function(ind) for ind in population]
            
            # 跟踪最优解
            current_best_fitness = min(fitnesses)
            current_best_index = fitnesses.index(current_best_fitness)
            
            if current_best_fitness < best_ever_fitness:
                best_ever_fitness = current_best_fitness
                best_ever_individual = population[current_best_index].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            self.fitness_history.append(current_best_fitness)
            
            # 显示进度
            if gen % 15 == 0 or gen == self.generations - 1:
                avg_fitness = np.mean(fitnesses)
                print(f"第 {gen:3d} 代: 最佳={current_best_fitness:8.2f}, 平均={avg_fitness:8.2f}, 停滞={stagnation_count}")
            
            # 早停机制
            if stagnation_count > 30:
                print(f"算法在第 {gen} 代收敛，提前停止")
                break
            
            # 生成新一代
            new_population = []
            
            # 精英保留
            elite_count = max(1, self.population_size // 10)
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成其余个体
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                if random.random() < self.cx_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < self.mut_prob:
                    child1 = self.mutate(child1)
                if random.random() < self.mut_prob:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # 保存最优结果
        self.best_individual = best_ever_individual
        self.best_fitness = best_ever_fitness
        
        print(f"\\n优化完成!")
        print(f"最佳校正因子: R={self.best_individual[0]:.4f}, G={self.best_individual[1]:.4f}, B={self.best_individual[2]:.4f}")
        print(f"最优适应度: {self.best_fitness:.2f}")
        
        return self.best_individual, self.best_fitness
    
    def apply_calibration(self):
        """应用校准"""
        if self.best_individual is None:
            raise ValueError("请先运行优化")
        
        return self.rgb_data * np.array(self.best_individual)
    
    def comprehensive_analysis(self):
        """全面分析校准效果"""
        if self.best_individual is None:
            raise ValueError("请先运行优化")
        
        original_rgb = self.rgb_data
        corrected_rgb = self.apply_calibration()
        
        print("\\n" + "="*60)
        print("           全面校准效果分析报告")
        print("="*60)
        
        # 基础统计
        original_mean = original_rgb.mean(axis=0)
        original_std = original_rgb.std(axis=0)
        corrected_mean = corrected_rgb.mean(axis=0)
        corrected_std = corrected_rgb.std(axis=0)
        
        print("\\n📊 基础统计对比:")
        channels = ['红色', '绿色', '蓝色']
        for i, ch in enumerate(channels):
            print(f"  {ch}:")
            print(f"    校准前: 均值={original_mean[i]:6.1f}, 标准差={original_std[i]:5.1f}")
            print(f"    校准后: 均值={corrected_mean[i]:6.1f}, 标准差={corrected_std[i]:5.1f}")
            print(f"    校正因子: {self.best_individual[i]:6.4f}")
        
        # 目标达成验证
        print("\\n🎯 目标达成验证:")
        red_check = "✓ 达成" if corrected_mean[0] <= 220 else "✗ 未达成"
        green_check = "✓ 达成" if abs(corrected_mean[1] - 220) <= 15 else "✗ 未达成"
        blue_check = "✓ 达成" if 180 <= corrected_mean[2] <= 200 else "✗ 未达成"
        
        print(f"  红色 ≤ 220:     {corrected_mean[0]:6.1f}  {red_check}")
        print(f"  绿色 ≈ 220:     {corrected_mean[1]:6.1f}  {green_check}")
        print(f"  蓝色 180-200:   {corrected_mean[2]:6.1f}  {blue_check}")
        
        # 均匀性分析
        print("\\n📐 均匀性分析:")
        for i, ch in enumerate(channels):
            original_cv = original_std[i] / original_mean[i]
            corrected_cv = corrected_std[i] / corrected_mean[i]
            improvement = (original_cv - corrected_cv) / original_cv * 100
            
            print(f"  {ch}: CV {original_cv:.4f} → {corrected_cv:.4f} (改善 {improvement:+.1f}%)")
        
        # 整体评估
        overall_targets_met = sum([
            corrected_mean[0] <= 220,
            abs(corrected_mean[1] - 220) <= 15,
            180 <= corrected_mean[2] <= 200
        ])
        
        overall_cv = corrected_std.mean() / corrected_mean.mean()
        
        print("\\n🏆 整体评估:")
        print(f"  目标达成度: {overall_targets_met}/3")
        print(f"  整体变异系数: {overall_cv:.4f}")
        print(f"  校准质量: {'优秀' if overall_targets_met >= 3 and overall_cv < 0.1 else '良好' if overall_targets_met >= 2 else '需改进'}")
        
        return {
            'original_stats': {'mean': original_mean, 'std': original_std},
            'corrected_stats': {'mean': corrected_mean, 'std': corrected_std},
            'correction_factors': self.best_individual,
            'targets_met': overall_targets_met,
            'overall_cv': overall_cv
        }

def create_comprehensive_visualization(calibrator, output_dir="led_optimized_results"):
    """创建全面的可视化报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    original_rgb = calibrator.rgb_data
    corrected_rgb = calibrator.apply_calibration()
    target_rgb = calibrator.target_rgb
    
    # 1. 优化历史
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(calibrator.fitness_history, 'b-', linewidth=2)
    plt.title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
    plt.xlabel('进化代数')
    plt.ylabel('最佳适应度')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(calibrator.fitness_history, 'r-', linewidth=2)
    plt.title('适应度对数图', fontsize=14, fontweight='bold')
    plt.xlabel('进化代数')
    plt.ylabel('适应度 (对数尺度)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 详细对比分析
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    channels = ['红色', '绿色', '蓝色']
    colors = ['red', 'green', 'blue']
    
    # 均值对比
    original_mean = original_rgb.mean(axis=0)
    corrected_mean = corrected_rgb.mean(axis=0)
    
    x = np.arange(len(channels))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, original_mean, width, label='校准前', color=colors, alpha=0.7)
    bars2 = axes[0, 0].bar(x + width/2, corrected_mean, width, label='校准后', color=colors, alpha=0.9)
    
    # 添加目标线
    for i, target in enumerate(target_rgb):
        axes[0, 0].axhline(y=target, color='black', linestyle='--', alpha=0.8)
        axes[0, 0].text(i, target + 5, f'目标:{target}', ha='center', fontsize=10)
    
    axes[0, 0].set_title('RGB均值对比', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('RGB值')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(channels)
    axes[0, 0].legend()
    
    # 标准差对比
    original_std = original_rgb.std(axis=0)
    corrected_std = corrected_rgb.std(axis=0)
    
    axes[0, 1].bar(x - width/2, original_std, width, label='校准前', alpha=0.7)
    axes[0, 1].bar(x + width/2, corrected_std, width, label='校准后', alpha=0.7)
    axes[0, 1].set_title('RGB标准差对比', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('标准差')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(channels)
    axes[0, 1].legend()
    
    # 校正因子
    factors = calibrator.best_individual
    bars = axes[0, 2].bar(channels, factors, color=colors, alpha=0.8)
    axes[0, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='无校正线')
    axes[0, 2].set_title('RGB校正因子', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('校正因子')
    axes[0, 2].legend()
    
    # 添加数值标签
    for bar, factor in zip(bars, factors):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{factor:.3f}', ha='center', fontweight='bold')
    
    # 分布散点图
    n_samples = min(100, len(original_rgb))
    sample_indices = np.linspace(0, len(original_rgb)-1, n_samples, dtype=int)
    
    for i, (ch, color) in enumerate(zip(channels, colors)):
        axes[1, i].scatter(sample_indices, original_rgb[sample_indices, i], 
                          alpha=0.6, color=color, s=20, label='校准前', marker='o')
        axes[1, i].scatter(sample_indices, corrected_rgb[sample_indices, i], 
                          alpha=0.8, color=color, s=20, label='校准后', marker='x')
        
        axes[1, i].axhline(y=target_rgb[i], color='black', linestyle='--', alpha=0.8)
        axes[1, i].set_title(f'{ch}通道分布', fontsize=14, fontweight='bold')
        axes[1, i].set_xlabel('像素索引（采样）')
        axes[1, i].set_ylabel('RGB值')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'均值: {corrected_mean[i]:.1f}\\n标准差: {corrected_std[i]:.1f}'
        axes[1, i].text(0.02, 0.98, stats_text, transform=axes[1, i].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\\n📈 可视化报告已生成:")
    print(f"  - {output_dir}/optimization_history.png")
    print(f"  - {output_dir}/comprehensive_analysis.png")

def main():
    """主函数 - 执行优化的LED校准流程"""
    print("="*80)
    print("          LED显示屏智能校准系统")
    print("       基于用户指导的优化解决方案")
    print("="*80)
    print("\\n🎯 校准目标:")
    print("  • 红色通道: 抑制至 ≤220")
    print("  • 绿色通道: 逼近至 ≈220") 
    print("  • 蓝色通道: 提升至 180-200")
    print("  • 整体要求: 分布均匀，无偏移")
    
    # 创建优化校准器
    calibrator = OptimizedLEDCalibrator(target_rgb=(220, 220, 200))
    
    # 尝试加载真实数据
    print("\\n🔄 步骤1: 数据加载...")
    data_file = "B题附件：RGB数值.xlsx"
    
    if os.path.exists(data_file):
        print("发现真实LED数据文件，尝试加载...")
        if calibrator.load_real_led_data(data_file):
            print("✓ 真实LED数据加载成功")
        else:
            print("✗ 真实数据加载失败，使用模拟数据")
            calibrator.generate_realistic_data(n_pixels=800)
    else:
        print("未找到真实数据文件，使用高保真模拟数据...")
        calibrator.generate_realistic_data(n_pixels=800)
    
    # 执行优化
    print("\\n🚀 步骤2: 智能优化校准...")
    best_individual, best_fitness = calibrator.optimize()
    
    # 全面分析
    print("\\n📊 步骤3: 校准效果分析...")
    analysis_results = calibrator.comprehensive_analysis()
    
    # 生成可视化
    print("\\n📈 步骤4: 生成可视化报告...")
    create_comprehensive_visualization(calibrator)
    
    # 保存结果
    output_dir = "led_optimized_results"
    np.save(f"{output_dir}/correction_factors.npy", best_individual)
    np.save(f"{output_dir}/corrected_rgb_data.npy", calibrator.apply_calibration())
    
    print("\\n" + "="*80)
    print("🎉 校准完成！")
    print(f"📁 所有结果已保存至: {output_dir}/")
    print("="*80)
    
    return analysis_results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ 运行时出错: {e}")
        import traceback
        traceback.print_exc()
