#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED显示屏色彩均匀性校准系统 - 基于遗传算法的解决方案
问题三：使用遗传算法优化LED显示屏校正因子

参考思路：基于用户提供的遗传算法校准方案
核心算法：遗传算法 + 欧几里得距离误差最小化

作者：GitHub Copilot
日期：2025年7月28日
版本：GA-1.0 (遗传算法版本)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from deap import base, creator, tools, algorithms
import random
import os
from typing import Tuple, List, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端和中文字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class LEDGeneticCalibrator:
    """基于遗传算法的LED校准器"""
    
    def __init__(self, target_rgb=(220, 220, 220)):
        self.target_rgb = np.array(target_rgb)
        self.rgb_data = None
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []
        self.population_size = 50
        self.generations = 100
        self.cx_prob = 0.7  # 交叉概率
        self.mut_prob = 0.2  # 变异概率
        
        # 设置DEAP框架
        self._setup_deap()
    
    def _setup_deap(self):
        """设置DEAP遗传算法框架"""
        # 清除之前的定义（如果存在）
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0.5, 1.5)  # 校正因子范围
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.attr_float, self.toolbox.attr_float, self.toolbox.attr_float), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._fitness)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def load_led_data_from_excel(self, file_path: str):
        """从Excel文件加载LED数据"""
        print(f"正在加载LED数据: {file_path}")
        
        try:
            # 读取RGB目标值工作表
            target_df = pd.read_excel(file_path, sheet_name='RGB目标值')
            print(f"目标数据形状: {target_df.shape}")
            
            # 读取各个LED响应工作表
            led_responses = {}
            sheet_names = ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']
            
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    # 转换为数值数据并展平
                    numeric_data = df.select_dtypes(include=[np.number]).values.flatten()
                    led_responses[sheet_name] = numeric_data
                    print(f"加载 {sheet_name}: {len(numeric_data)} 个数据点")
                except Exception as e:
                    print(f"警告: 无法加载工作表 {sheet_name}: {e}")
            
            # 重构RGB数据
            if len(led_responses) >= 9:
                # 假设所有工作表的数据长度相同
                n_pixels = len(led_responses['R_R'])
                
                # 构建RGB矩阵：每个像素的RGB值
                rgb_matrix = np.zeros((n_pixels, 3))
                
                # 使用主要通道的数据（R_R, G_G, B_B）
                rgb_matrix[:, 0] = led_responses['R_R']  # 红色通道
                rgb_matrix[:, 1] = led_responses['G_G']  # 绿色通道  
                rgb_matrix[:, 2] = led_responses['B_B']  # 蓝色通道
                
                self.rgb_data = rgb_matrix
                print(f"LED RGB数据形状: {rgb_matrix.shape}")
                print(f"RGB数据范围: R[{rgb_matrix[:, 0].min():.1f}, {rgb_matrix[:, 0].max():.1f}], "
                      f"G[{rgb_matrix[:, 1].min():.1f}, {rgb_matrix[:, 1].max():.1f}], "
                      f"B[{rgb_matrix[:, 2].min():.1f}, {rgb_matrix[:, 2].max():.1f}]")
                
                return True
            else:
                print("错误: 无法找到足够的LED响应数据")
                return False
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
    
    def generate_sample_data(self, n_pixels=100):
        """生成示例数据用于测试"""
        print(f"生成 {n_pixels} 个像素的示例LED数据...")
        
        np.random.seed(42)
        
        # 生成具有不均匀性的RGB数据
        target = self.target_rgb[0]  # 假设目标值为220
        
        # 模拟LED不均匀性：不同位置有不同的亮度偏差
        rgb_data = np.zeros((n_pixels, 3))
        
        for i in range(n_pixels):
            # 位置相关的偏差
            position_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / n_pixels)
            
            # 随机噪声
            noise = np.random.normal(1.0, 0.1, 3)
            
            # 通道间的差异
            channel_bias = [0.95, 1.0, 1.05]  # R偏低，G正常，B偏高
            
            for c in range(3):
                rgb_data[i, c] = target * position_factor * noise[c] * channel_bias[c]
        
        # 确保数据在合理范围内
        rgb_data = np.clip(rgb_data, 150, 280)
        
        self.rgb_data = rgb_data
        print(f"示例数据生成完成，形状: {rgb_data.shape}")
        print(f"数据范围: R[{rgb_data[:, 0].min():.1f}, {rgb_data[:, 0].max():.1f}], "
              f"G[{rgb_data[:, 1].min():.1f}, {rgb_data[:, 1].max():.1f}], "
              f"B[{rgb_data[:, 2].min():.1f}, {rgb_data[:, 2].max():.1f}]")
        
        return rgb_data
    
    def _fitness(self, individual):
        """适应度函数：计算校正后的RGB误差"""
        if self.rgb_data is None:
            return (float('inf'),)
        
        # 获取校正因子
        C_R, C_G, C_B = individual
        
        # 应用校正
        corrected_rgb = self.rgb_data * np.array([C_R, C_G, C_B])
        
        # 计算每个像素与目标RGB的欧几里得距离
        target_matrix = np.tile(self.target_rgb, (len(self.rgb_data), 1))
        errors = np.sqrt(np.sum((corrected_rgb - target_matrix) ** 2, axis=1))
        
        # 返回总误差
        total_error = np.sum(errors)
        
        return (total_error,)
    
    def optimize(self):
        """运行遗传算法优化"""
        if self.rgb_data is None:
            raise ValueError("请先加载LED数据")
        
        print("\\n=== 开始遗传算法优化 ===")
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {self.generations}")
        print(f"交叉概率: {self.cx_prob}")
        print(f"变异概率: {self.mut_prob}")
        
        # 初始化种群
        population = self.toolbox.population(n=self.population_size)
        
        # 记录适应度历史
        self.fitness_history = []
        
        # 进化过程
        for gen in range(self.generations):
            # 评估种群适应度
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # 选择下一代
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 变异操作
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    # 确保校正因子在合理范围内
                    for i in range(len(mutant)):
                        mutant[i] = np.clip(mutant[i], 0.3, 2.0)
                    del mutant.fitness.values
            
            # 评估变异后的个体
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_individuals))
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
            
            # 更新种群
            population[:] = offspring
            
            # 记录最佳适应度
            best_individual = tools.selBest(population, 1)[0]
            best_fitness = best_individual.fitness.values[0]
            
            self.fitness_history.append(best_fitness)
            
            # 显示进度
            if gen % 10 == 0:
                avg_fitness = np.mean([ind.fitness.values[0] for ind in population])
                print(f"第 {gen} 代: 最佳适应度 = {best_fitness:.2f}, 平均适应度 = {avg_fitness:.2f}")
        
        # 保存最优结果
        self.best_individual = tools.selBest(population, 1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        print(f"\\n优化完成!")
        print(f"最佳校正因子: R={self.best_individual[0]:.4f}, G={self.best_individual[1]:.4f}, B={self.best_individual[2]:.4f}")
        print(f"最小颜色误差: {self.best_fitness:.2f}")
        
        return self.best_individual, self.best_fitness
    
    def apply_calibration(self):
        """应用校准并返回校正后的RGB数据"""
        if self.best_individual is None:
            raise ValueError("请先运行优化算法")
        
        corrected_rgb = self.rgb_data * np.array(self.best_individual)
        return corrected_rgb
    
    def analyze_improvement(self):
        """分析校准改善效果"""
        if self.best_individual is None:
            raise ValueError("请先运行优化算法")
        
        # 原始数据统计
        original_mean = np.mean(self.rgb_data, axis=0)
        original_std = np.std(self.rgb_data, axis=0)
        original_cv = original_std / original_mean
        
        # 校正后数据统计
        corrected_rgb = self.apply_calibration()
        corrected_mean = np.mean(corrected_rgb, axis=0)
        corrected_std = np.std(corrected_rgb, axis=0)
        corrected_cv = corrected_std / corrected_mean
        
        # 计算改善程度
        cv_improvement = (original_cv - corrected_cv) / original_cv * 100
        
        # 计算与目标的偏差
        target_deviation_original = np.abs(original_mean - self.target_rgb)
        target_deviation_corrected = np.abs(corrected_mean - self.target_rgb)
        deviation_improvement = (target_deviation_original - target_deviation_corrected) / target_deviation_original * 100
        
        analysis_results = {
            'original_stats': {
                'mean': original_mean.tolist(),
                'std': original_std.tolist(),
                'cv': original_cv.tolist()
            },
            'corrected_stats': {
                'mean': corrected_mean.tolist(),
                'std': corrected_std.tolist(),
                'cv': corrected_cv.tolist()
            },
            'improvements': {
                'cv_improvement': cv_improvement.tolist(),
                'deviation_improvement': deviation_improvement.tolist()
            },
            'correction_factors': self.best_individual.copy()
        }
        
        print("\\n=== 校准改善分析 ===")
        channels = ['Red', 'Green', 'Blue']
        for i, ch in enumerate(channels):
            print(f"{ch}通道:")
            print(f"  变异系数改善: {cv_improvement[i]:.1f}%")
            print(f"  目标偏差改善: {deviation_improvement[i]:.1f}%")
            print(f"  校正因子: {self.best_individual[i]:.4f}")
        
        return analysis_results

class LEDVisualizationGA:
    """LED校准结果可视化（遗传算法版本）"""
    
    def __init__(self, output_dir="led_ga_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_optimization_history(self, fitness_history, save_path=None):
        """绘制优化历史"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
        plt.xlabel('进化代数')
        plt.ylabel('最佳适应度（总误差）')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(fitness_history, 'r-', linewidth=2)
        plt.title('适应度对数图', fontsize=14, fontweight='bold')
        plt.xlabel('进化代数')
        plt.ylabel('最佳适应度（对数尺度）')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'ga_optimization_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"优化历史图保存至: {save_path}")
        plt.close()
    
    def plot_rgb_comparison(self, original_rgb, corrected_rgb, target_rgb, save_path=None):
        """绘制校准前后RGB对比"""
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 平均值对比
        original_mean = np.mean(original_rgb, axis=0)
        corrected_mean = np.mean(corrected_rgb, axis=0)
        
        x = np.arange(len(channels))
        width = 0.25
        
        axes[0, 0].bar(x - width, original_mean, width, label='校准前', color=colors, alpha=0.7)
        axes[0, 0].bar(x, corrected_mean, width, label='校准后', color=colors, alpha=0.9)
        axes[0, 0].axhline(y=target_rgb[0], color='black', linestyle='--', label=f'目标值({target_rgb[0]})')
        axes[0, 0].set_title('RGB平均值对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('RGB值')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(channels)
        axes[0, 0].legend()
        
        # 2. 标准差对比
        original_std = np.std(original_rgb, axis=0)
        corrected_std = np.std(corrected_rgb, axis=0)
        
        axes[0, 1].bar(x - width/2, original_std, width, label='校准前', color='lightcoral', alpha=0.7)
        axes[0, 1].bar(x + width/2, corrected_std, width, label='校准后', color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('RGB标准差对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('标准差')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(channels)
        axes[0, 1].legend()
        
        # 3. 数据分布直方图（仅显示前100个像素避免过密）
        n_samples = min(100, len(original_rgb))
        sample_indices = np.linspace(0, len(original_rgb)-1, n_samples, dtype=int)
        
        for i, (ch, color) in enumerate(zip(channels, colors)):
            axes[1, 0].scatter(sample_indices, original_rgb[sample_indices, i], 
                             alpha=0.6, color=color, s=20, label=f'原始{ch}')
        
        axes[1, 0].axhline(y=target_rgb[0], color='black', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('校准前RGB分布', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('像素索引')
        axes[1, 0].set_ylabel('RGB值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, (ch, color) in enumerate(zip(channels, colors)):
            axes[1, 1].scatter(sample_indices, corrected_rgb[sample_indices, i], 
                             alpha=0.6, color=color, s=20, label=f'校正{ch}')
        
        axes[1, 1].axhline(y=target_rgb[0], color='black', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('校准后RGB分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('像素索引')
        axes[1, 1].set_ylabel('RGB值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'rgb_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RGB对比图保存至: {save_path}")
        plt.close()
    
    def plot_correction_factors(self, correction_factors, save_path=None):
        """绘制校正因子图"""
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(channels, correction_factors, color=colors, alpha=0.7)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='无校正线(1.0)')
        
        # 添加数值标签
        for bar, factor in zip(bars, correction_factors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{factor:.3f}', ha='center', fontweight='bold')
        
        plt.title('RGB通道校正因子', fontsize=14, fontweight='bold')
        plt.ylabel('校正因子')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correction_factors.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"校正因子图保存至: {save_path}")
        plt.close()

def main():
    """主函数 - LED遗传算法校准完整流程"""
    print("=== LED显示屏色彩均匀性校准系统（遗传算法版本） ===")
    print("问题三：基于遗传算法的LED显示屏校准解决方案")
    print("核心算法：遗传算法 + 欧几里得距离误差最小化")
    print("=" * 75)
    
    # 创建校准器
    target_rgb = (220, 220, 220)
    calibrator = LEDGeneticCalibrator(target_rgb=target_rgb)
    
    # 步骤1: 加载数据
    print("\\n步骤1: 加载LED数据...")
    
    data_file = "B题附件：RGB数值.xlsx"
    if os.path.exists(data_file):
        success = calibrator.load_led_data_from_excel(data_file)
        if not success:
            print("Excel数据加载失败，使用示例数据")
            calibrator.generate_sample_data(n_pixels=1000)
    else:
        print("Excel文件不存在，使用示例数据")
        calibrator.generate_sample_data(n_pixels=1000)
    
    # 步骤2: 遗传算法优化
    print("\\n步骤2: 遗传算法优化校正因子...")
    best_individual, best_fitness = calibrator.optimize()
    
    # 步骤3: 分析改善效果
    print("\\n步骤3: 分析校准改善效果...")
    analysis_results = calibrator.analyze_improvement()
    
    # 步骤4: 生成可视化结果
    print("\\n步骤4: 生成可视化结果...")
    visualizer = LEDVisualizationGA()
    
    # 绘制优化历史
    visualizer.plot_optimization_history(calibrator.fitness_history)
    
    # 绘制RGB对比
    original_rgb = calibrator.rgb_data
    corrected_rgb = calibrator.apply_calibration()
    visualizer.plot_rgb_comparison(original_rgb, corrected_rgb, target_rgb)
    
    # 绘制校正因子
    visualizer.plot_correction_factors(best_individual)
    
    # 步骤5: 保存结果
    print("\\n步骤5: 保存结果...")
    
    output_dir = visualizer.output_dir
    
    # 保存校正因子
    np.save(os.path.join(output_dir, 'correction_factors.npy'), best_individual)
    
    # 保存校正后数据
    np.save(os.path.join(output_dir, 'corrected_rgb_data.npy'), corrected_rgb)
    
    # 保存分析报告
    with open(os.path.join(output_dir, 'ga_analysis_report.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print("\\n=== 处理完成 ===")
    print(f"最佳校正因子: R={best_individual[0]:.4f}, G={best_individual[1]:.4f}, B={best_individual[2]:.4f}")
    print(f"最小颜色误差: {best_fitness:.2f}")
    print(f"所有结果已保存至: {output_dir}/")
    
    print("\\n生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")
    
    return analysis_results

if __name__ == "__main__":
    main()
