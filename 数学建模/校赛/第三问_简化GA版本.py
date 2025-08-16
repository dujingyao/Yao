#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED校准测试版本 - 简化的遗传算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleGeneticCalibrator:
    """简化的遗传算法LED校准器"""
    
    def __init__(self, target_rgb=(220, 220, 200)):  # 修改目标值：平衡绿/蓝差异
        self.target_rgb = np.array(target_rgb)
        self.rgb_data = None
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []
        
        # 遗传算法参数
        self.population_size = 50
        self.generations = 100
        self.cx_prob = 0.7
        self.mut_prob = 0.2
        
    def generate_sample_data(self, n_pixels=100):
        """生成示例LED数据"""
        print(f"生成 {n_pixels} 个像素的示例LED数据...")
        
        np.random.seed(42)
        target = self.target_rgb[0]
        
        rgb_data = np.zeros((n_pixels, 3))
        
        for i in range(n_pixels):
            # 位置相关偏差
            position_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / n_pixels)
            
            # 随机噪声
            noise = np.random.normal(1.0, 0.1, 3)
            
            # 通道偏差
            channel_bias = [0.95, 1.0, 1.05]  # R偏低，G正常，B偏高
            
            for c in range(3):
                rgb_data[i, c] = target * position_factor * noise[c] * channel_bias[c]
        
        rgb_data = np.clip(rgb_data, 150, 280)
        self.rgb_data = rgb_data
        
        print(f"数据范围: R[{rgb_data[:, 0].min():.1f}, {rgb_data[:, 0].max():.1f}]")
        print(f"         G[{rgb_data[:, 1].min():.1f}, {rgb_data[:, 1].max():.1f}]")
        print(f"         B[{rgb_data[:, 2].min():.1f}, {rgb_data[:, 2].max():.1f}]")
        
        return rgb_data
    
    def fitness_function(self, individual: List[float]) -> float:
        """优化的适应度函数 - 对红色误差施加更高权重"""
        C_R, C_G, C_B = individual
        
        # 应用校正
        corrected_rgb = self.rgb_data * np.array([C_R, C_G, C_B])
        
        # 计算与目标的差异（分通道计算以应用不同权重）
        target_matrix = np.tile(self.target_rgb, (len(self.rgb_data), 1))
        diff = corrected_rgb - target_matrix
        
        # 分通道计算误差
        red_errors = np.abs(diff[:, 0])    # 红色通道误差
        green_errors = np.abs(diff[:, 1])  # 绿色通道误差  
        blue_errors = np.abs(diff[:, 2])   # 蓝色通道误差
        
        # 应用权重：红色误差×2，绿色×1，蓝色×1.2（稍微增强）
        weighted_error = (2.0 * np.sum(red_errors) + 
                         1.0 * np.sum(green_errors) + 
                         1.2 * np.sum(blue_errors))
        
        # 添加均匀性惩罚：鼓励各通道内部的均匀分布
        uniformity_penalty = 0
        for channel in range(3):
            channel_values = corrected_rgb[:, channel]
            cv = np.std(channel_values) / np.mean(channel_values) if np.mean(channel_values) > 0 else 0
            uniformity_penalty += cv * 1000  # 变异系数惩罚
        
        return weighted_error + uniformity_penalty
    
    def create_individual(self) -> List[float]:
        """创建个体（校正因子）"""
        return [random.uniform(0.5, 1.5) for _ in range(3)]
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """交叉操作"""
        alpha = 0.5
        child1 = []
        child2 = []
        
        for i in range(3):
            beta = random.uniform(-alpha, 1 + alpha)
            child1.append(beta * parent1[i] + (1 - beta) * parent2[i])
            child2.append(beta * parent2[i] + (1 - beta) * parent1[i])
        
        return child1, child2
    
    def mutate(self, individual: List[float]) -> List[float]:
        """变异操作"""
        mutated = individual.copy()
        for i in range(3):
            if random.random() < 0.2:  # 变异概率
                mutated[i] += random.gauss(0, 0.1)
                mutated[i] = np.clip(mutated[i], 0.3, 2.0)
        return mutated
    
    def tournament_selection(self, population: List[List[float]], fitnesses: List[float], k=3) -> List[float]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        return population[winner_index].copy()
    
    def optimize(self):
        """遗传算法优化"""
        if self.rgb_data is None:
            raise ValueError("请先生成或加载LED数据")
        
        print("\\n=== 开始遗传算法优化 ===")
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {self.generations}")
        
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]
        
        self.fitness_history = []
        
        for gen in range(self.generations):
            # 计算适应度
            fitnesses = [self.fitness_function(ind) for ind in population]
            
            # 记录最佳适应度
            best_fitness = min(fitnesses)
            best_index = fitnesses.index(best_fitness)
            self.fitness_history.append(best_fitness)
            
            # 显示进度
            if gen % 10 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"第 {gen:3d} 代: 最佳适应度 = {best_fitness:8.2f}, 平均适应度 = {avg_fitness:8.2f}")
            
            # 创建新一代
            new_population = []
            
            # 保留最佳个体（精英策略）
            new_population.append(population[best_index].copy())
            
            # 生成其余个体
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # 交叉
                if random.random() < self.cx_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                if random.random() < self.mut_prob:
                    child1 = self.mutate(child1)
                if random.random() < self.mut_prob:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 确保种群大小不超过限制
            population = new_population[:self.population_size]
        
        # 最终结果
        final_fitnesses = [self.fitness_function(ind) for ind in population]
        best_fitness = min(final_fitnesses)
        best_index = final_fitnesses.index(best_fitness)
        
        self.best_individual = population[best_index]
        self.best_fitness = best_fitness
        
        print(f"\\n优化完成!")
        print(f"最佳校正因子: R={self.best_individual[0]:.4f}, G={self.best_individual[1]:.4f}, B={self.best_individual[2]:.4f}")
        print(f"最小颜色误差: {self.best_fitness:.2f}")
        
        return self.best_individual, self.best_fitness
    
    def apply_calibration(self):
        """应用校准"""
        if self.best_individual is None:
            raise ValueError("请先运行优化算法")
        
        return self.rgb_data * np.array(self.best_individual)
    
    def analyze_results(self):
        """分析结果"""
        if self.best_individual is None:
            raise ValueError("请先运行优化算法")
        
        original_rgb = self.rgb_data
        corrected_rgb = self.apply_calibration()
        
        # 计算统计量
        original_mean = np.mean(original_rgb, axis=0)
        original_std = np.std(original_rgb, axis=0)
        original_cv = original_std / original_mean
        
        corrected_mean = np.mean(corrected_rgb, axis=0)
        corrected_std = np.std(corrected_rgb, axis=0)
        corrected_cv = corrected_std / corrected_mean
        
        cv_improvement = (original_cv - corrected_cv) / original_cv * 100
        
        print("\\n=== 校准改善分析 ===")
        channels = ['Red', 'Green', 'Blue']
        for i, ch in enumerate(channels):
            print(f"{ch}通道:")
            print(f"  原始: 均值={original_mean[i]:.1f}, 标准差={original_std[i]:.1f}, CV={original_cv[i]:.4f}")
            print(f"  校正: 均值={corrected_mean[i]:.1f}, 标准差={corrected_std[i]:.1f}, CV={corrected_cv[i]:.4f}")
            print(f"  改善: CV改善={cv_improvement[i]:.1f}%, 校正因子={self.best_individual[i]:.4f}")
        
        # 新增：验证校准效果是否符合指导标准
        print("\\n=== 校准效果验证 ===")
        print("目标设定验证:")
        red_target_check = "✓" if corrected_mean[0] <= 220 else "✗"
        green_target_check = "✓" if abs(corrected_mean[1] - 220) <= 10 else "✗"
        blue_target_check = "✓" if 180 <= corrected_mean[2] <= 200 else "✗"
        
        print(f"  红色≤220: {corrected_mean[0]:.1f} {red_target_check}")
        print(f"  绿色≈220: {corrected_mean[1]:.1f} {green_target_check}")
        print(f"  蓝色180-200: {corrected_mean[2]:.1f} {blue_target_check}")
        
        # 分布均匀性验证
        overall_uniformity = np.mean(1 - corrected_cv)
        uniformity_check = "✓" if overall_uniformity > 0.8 else "✗"
        print(f"  整体均匀性: {overall_uniformity:.3f} {uniformity_check}")
        
        return {
            'original_mean': original_mean,
            'corrected_mean': corrected_mean,
            'cv_improvement': cv_improvement,
            'correction_factors': self.best_individual,
            'validation_results': {
                'red_target_met': corrected_mean[0] <= 220,
                'green_target_met': abs(corrected_mean[1] - 220) <= 10,
                'blue_target_met': 180 <= corrected_mean[2] <= 200,
                'uniformity_met': overall_uniformity > 0.8
            }
        }
    
    def plot_results(self, output_dir="led_simple_ga"):
        """绘制结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 优化历史
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', linewidth=2)
        plt.title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
        plt.xlabel('进化代数')
        plt.ylabel('最佳适应度（总误差）')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_dir, 'optimization_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"优化历史图保存至: {save_path}")
        plt.close()
        
        # 2. RGB对比
        original_rgb = self.rgb_data
        corrected_rgb = self.apply_calibration()
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 均值对比
        original_mean = np.mean(original_rgb, axis=0)
        corrected_mean = np.mean(corrected_rgb, axis=0)
        
        x = np.arange(len(channels))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, original_mean, width, label='校准前', color=colors, alpha=0.7)
        axes[0, 0].bar(x + width/2, corrected_mean, width, label='校准后', color=colors, alpha=0.9)
        axes[0, 0].axhline(y=self.target_rgb[0], color='black', linestyle='--', label=f'目标值({self.target_rgb[0]})')
        axes[0, 0].set_title('RGB平均值对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(channels)
        axes[0, 0].legend()
        
        # 标准差对比
        original_std = np.std(original_rgb, axis=0)
        corrected_std = np.std(corrected_rgb, axis=0)
        
        axes[0, 1].bar(x - width/2, original_std, width, label='校准前', alpha=0.7)
        axes[0, 1].bar(x + width/2, corrected_std, width, label='校准后', alpha=0.7)
        axes[0, 1].set_title('RGB标准差对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(channels)
        axes[0, 1].legend()
        
        # 校正因子
        axes[1, 0].bar(channels, self.best_individual, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('RGB校正因子')
        axes[1, 0].set_ylabel('校正因子')
        
        # 数据分布散点图（取样显示）
        n_samples = min(50, len(original_rgb))
        sample_indices = np.linspace(0, len(original_rgb)-1, n_samples, dtype=int)
        
        for i, (ch, color) in enumerate(zip(channels, colors)):
            axes[1, 1].scatter(sample_indices, original_rgb[sample_indices, i], 
                             alpha=0.6, color=color, s=30, marker='o', label=f'原始{ch}')
            axes[1, 1].scatter(sample_indices, corrected_rgb[sample_indices, i], 
                             alpha=0.8, color=color, s=30, marker='x', label=f'校正{ch}')
        
        axes[1, 1].axhline(y=self.target_rgb[0], color='black', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('RGB数据分布对比')
        axes[1, 1].set_xlabel('像素索引（采样）')
        axes[1, 1].set_ylabel('RGB值')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'rgb_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RGB对比图保存至: {save_path}")
        plt.close()

def main():
    """主函数"""
    print("=== LED显示屏遗传算法校准系统（简化版本） ===")
    print("问题三：基于遗传算法的LED显示屏校准解决方案")
    print("=" * 60)
    
    # 创建校准器 - 使用优化的目标值
    calibrator = SimpleGeneticCalibrator(target_rgb=(220, 220, 200))  # 平衡绿/蓝差异
    
    # 生成测试数据
    print("\\n步骤1: 生成测试LED数据...")
    calibrator.generate_sample_data(n_pixels=200)
    
    # 运行优化
    print("\\n步骤2: 遗传算法优化...")
    best_individual, best_fitness = calibrator.optimize()
    
    # 分析结果
    print("\\n步骤3: 分析校准效果...")
    results = calibrator.analyze_results()
    
    # 生成图表
    print("\\n步骤4: 生成可视化结果...")
    calibrator.plot_results()
    
    print("\\n=== 处理完成 ===")
    print(f"所有结果已保存至: led_simple_ga/")
    
    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()
