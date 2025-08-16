#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED显示屏色彩均匀性校准系统 - 遗传算法简化版
问题三：基于用户推荐思路的LED显示屏校准解决方案

核心思路：
1. 使用遗传算法优化RGB校正因子
2. 最小化校正后RGB与目标RGB的欧几里得距离
3. 简化的校正模型：每个通道使用统一的校正因子

作者：GitHub Copilot
日期：2025年7月28日
基于用户推荐的遗传算法思路
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
import os
import json
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端和中文字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class SimpleGeneticCalibrator:
    """简化的遗传算法LED校准器"""
    
    def __init__(self, target_rgb=(220, 220, 220)):
        self.target_rgb = np.array(target_rgb)
        self.rgb_data = None
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []
        
        # 遗传算法参数
        self.population_size = 50
        self.generations = 100
        self.cx_prob = 0.7  # 交叉概率
        self.mut_prob = 0.2  # 变异概率
        self.mutation_strength = 0.1  # 变异强度
        
    def load_led_data_from_excel(self, file_path: str):
        """从Excel文件加载LED数据"""
        print(f"正在加载LED数据: {file_path}")
        
        try:
            # 尝试读取Excel文件的所有工作表
            xl_file = pd.ExcelFile(file_path)
            print(f"发现工作表: {xl_file.sheet_names}")
            
            # 查找RGB相关的工作表
            rgb_sheets = {}
            for sheet_name in xl_file.sheet_names:
                if any(x in sheet_name for x in ['R_R', 'G_G', 'B_B', 'RGB']):
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    rgb_sheets[sheet_name] = df
                    print(f"加载工作表 {sheet_name}: {df.shape}")
            
            # 如果找到RGB相关数据，构建RGB矩阵
            if 'R_R' in rgb_sheets and 'G_G' in rgb_sheets and 'B_B' in rgb_sheets:
                # 使用主对角线数据（R_R, G_G, B_B）
                r_data = rgb_sheets['R_R'].select_dtypes(include=[np.number]).values.flatten()
                g_data = rgb_sheets['G_G'].select_dtypes(include=[np.number]).values.flatten()
                b_data = rgb_sheets['B_B'].select_dtypes(include=[np.number]).values.flatten()
                
                # 确保数据长度一致
                min_len = min(len(r_data), len(g_data), len(b_data))
                rgb_matrix = np.column_stack([
                    r_data[:min_len],
                    g_data[:min_len],
                    b_data[:min_len]
                ])
                
                self.rgb_data = rgb_matrix
                print(f"成功构建RGB矩阵: {rgb_matrix.shape}")
                print(f"RGB数据范围: R[{rgb_matrix[:, 0].min():.1f}, {rgb_matrix[:, 0].max():.1f}], "
                      f"G[{rgb_matrix[:, 1].min():.1f}, {rgb_matrix[:, 1].max():.1f}], "
                      f"B[{rgb_matrix[:, 2].min():.1f}, {rgb_matrix[:, 2].max():.1f}]")
                return True
            else:
                # 尝试其他加载方式
                df = pd.read_excel(file_path, sheet_name=0)  # 读取第一个工作表
                if df.shape[1] >= 3:
                    self.rgb_data = df.iloc[:, :3].values
                    print(f"使用前三列作为RGB数据: {self.rgb_data.shape}")
                    return True
                else:
                    print("无法识别RGB数据格式")
                    return False
                    
        except Exception as e:
            print(f"加载Excel数据时出错: {e}")
            return False
    
    def generate_sample_data(self, n_pixels=500):
        """生成示例数据（按照用户提供的思路）"""
        print(f"生成 {n_pixels} 个像素的示例LED数据...")
        
        # 设置随机种子以保证可重复性
        np.random.seed(42)
        
        # 按照用户代码的思路：模拟显示器色度差异
        rgb_data = np.random.randint(180, 250, (n_pixels, 3))
        
        self.rgb_data = rgb_data.astype(float)
        
        print(f"示例数据生成完成，形状: {rgb_data.shape}")
        print(f"数据范围: R[{rgb_data[:, 0].min():.1f}, {rgb_data[:, 0].max():.1f}], "
              f"G[{rgb_data[:, 1].min():.1f}, {rgb_data[:, 1].max():.1f}], "
              f"B[{rgb_data[:, 2].min():.1f}, {rgb_data[:, 2].max():.1f}]")
        
        return rgb_data
    
    def fitness(self, individual):
        """
        适应度函数：计算每个个体（即校正因子）的适应度
        individual：校正因子 [C_R, C_G, C_B]
        返回：校正后RGB与目标RGB的误差总和
        """
        if self.rgb_data is None:
            return float('inf')
        
        # 获取校正因子
        C_R, C_G, C_B = individual
        
        # 校正RGB信号
        corrected_rgb = self.rgb_data * np.array([C_R, C_G, C_B])
        
        # 计算每个像素的颜色误差（欧几里得距离）
        target_matrix = np.tile(self.target_rgb, (len(self.rgb_data), 1))
        error = np.sqrt(np.sum((corrected_rgb - target_matrix) ** 2, axis=1))
        
        # 返回所有像素的误差总和
        return np.sum(error)
    
    def create_individual(self):
        """创建一个个体（校正因子在0.8到1.2之间）"""
        return [random.uniform(0.8, 1.2) for _ in range(3)]
    
    def create_population(self):
        """创建初始种群"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def crossover(self, parent1, parent2):
        """交叉操作（混合交叉）"""
        alpha = 0.5
        child1 = []
        child2 = []
        
        for i in range(3):
            # 混合交叉
            child1.append(alpha * parent1[i] + (1 - alpha) * parent2[i])
            child2.append((1 - alpha) * parent1[i] + alpha * parent2[i])
        
        return child1, child2
    
    def mutate(self, individual):
        """变异操作（高斯变异）"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mut_prob:
                mutated[i] += random.gauss(0, self.mutation_strength)
                # 确保校正因子在合理范围内
                mutated[i] = max(0.3, min(2.0, mutated[i]))
        return mutated
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            # 随机选择tournament_size个个体
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # 选择适应度最好的（误差最小的）
            winner_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
            selected.append(population[winner_index].copy())
        
        return selected
    
    def optimize(self):
        """运行遗传算法优化"""
        if self.rgb_data is None:
            raise ValueError("请先加载LED数据")
        
        print("\\n=== 开始遗传算法优化 ===")
        print(f"目标RGB: {self.target_rgb}")
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {self.generations}")
        print(f"交叉概率: {self.cx_prob}")
        print(f"变异概率: {self.mut_prob}")
        
        # 初始化种群
        population = self.create_population()
        
        # 记录最佳适应度历史
        self.fitness_history = []
        
        # 进化过程
        for gen in range(self.generations):
            # 评估种群适应度
            fitnesses = [self.fitness(individual) for individual in population]
            
            # 记录最佳适应度
            best_fitness = min(fitnesses)
            best_index = fitnesses.index(best_fitness)
            best_individual = population[best_index].copy()
            
            self.fitness_history.append(best_fitness)
            
            # 显示进度
            if gen % 20 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"第 {gen} 代: 最佳适应度 = {best_fitness:.2f}, 平均适应度 = {avg_fitness:.2f}")
                print(f"  当前最佳校正因子: R={best_individual[0]:.4f}, G={best_individual[1]:.4f}, B={best_individual[2]:.4f}")
            
            # 选择下一代
            offspring = self.tournament_selection(population, fitnesses)
            
            # 交叉和变异
            new_population = []
            for i in range(0, len(offspring), 2):
                parent1 = offspring[i]
                parent2 = offspring[i + 1] if i + 1 < len(offspring) else offspring[0]
                
                # 交叉操作
                if random.random() < self.cx_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异操作
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群（保留最佳个体）
            population = new_population[:self.population_size]
            
            # 精英保留策略：确保最佳个体不丢失
            population[0] = best_individual
        
        # 最终评估并保存最优结果
        final_fitnesses = [self.fitness(individual) for individual in population]
        best_final_fitness = min(final_fitnesses)
        best_final_index = final_fitnesses.index(best_final_fitness)
        self.best_individual = population[best_final_index]
        self.best_fitness = best_final_fitness
        
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
        
        # 原始数据与目标的误差
        target_matrix = np.tile(self.target_rgb, (len(self.rgb_data), 1))
        original_errors = np.sqrt(np.sum((self.rgb_data - target_matrix) ** 2, axis=1))
        original_total_error = np.sum(original_errors)
        
        # 校正后数据与目标的误差
        corrected_rgb = self.apply_calibration()
        corrected_errors = np.sqrt(np.sum((corrected_rgb - target_matrix) ** 2, axis=1))
        corrected_total_error = np.sum(corrected_errors)
        
        # 计算改善程度
        error_improvement = (original_total_error - corrected_total_error) / original_total_error * 100
        
        # 统计分析
        original_mean = np.mean(self.rgb_data, axis=0)
        corrected_mean = np.mean(corrected_rgb, axis=0)
        
        original_std = np.std(self.rgb_data, axis=0)
        corrected_std = np.std(corrected_rgb, axis=0)
        
        cv_original = original_std / original_mean
        cv_corrected = corrected_std / corrected_mean
        cv_improvement = (cv_original - cv_corrected) / cv_original * 100
        
        results = {
            'correction_factors': self.best_individual.copy(),
            'error_improvement': float(error_improvement),
            'original_total_error': float(original_total_error),
            'corrected_total_error': float(corrected_total_error),
            'cv_improvement': cv_improvement.tolist(),
            'original_mean': original_mean.tolist(),
            'corrected_mean': corrected_mean.tolist(),
            'target_deviation_original': (np.abs(original_mean - self.target_rgb)).tolist(),
            'target_deviation_corrected': (np.abs(corrected_mean - self.target_rgb)).tolist()
        }
        
        print("\\n=== 校准改善分析 ===")
        print(f"总误差改善: {error_improvement:.1f}%")
        channels = ['Red', 'Green', 'Blue']
        for i, ch in enumerate(channels):
            print(f"{ch}通道:")
            print(f"  变异系数改善: {cv_improvement[i]:.1f}%")
            print(f"  校正因子: {self.best_individual[i]:.4f}")
            print(f"  目标偏差: {np.abs(original_mean[i] - self.target_rgb[i]):.1f} → {np.abs(corrected_mean[i] - self.target_rgb[i]):.1f}")
        
        return results

def plot_results(calibrator, output_dir):
    """绘制结果图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 优化历史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(calibrator.fitness_history, 'b-', linewidth=2)
    plt.title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
    plt.xlabel('进化代数')
    plt.ylabel('最佳适应度（总误差）')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(calibrator.fitness_history, 'r-', linewidth=2)
    plt.title('适应度对数图', fontsize=14, fontweight='bold')
    plt.xlabel('进化代数')
    plt.ylabel('最佳适应度（对数尺度）')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 原始与校正后RGB值对比（按照用户代码的思路）
    corrected_rgb = calibrator.apply_calibration()
    
    labels = ['Red', 'Green', 'Blue']
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始RGB信号
    ax.bar(x - 0.2, np.mean(calibrator.rgb_data, axis=0), 0.4, label='原始RGB值', 
           color=['lightcoral', 'lightgreen', 'lightblue'], alpha=0.7)
    
    # 绘制校正后的RGB信号
    ax.bar(x + 0.2, np.mean(corrected_rgb, axis=0), 0.4, label='校正后RGB值',
           color=['red', 'green', 'blue'], alpha=0.9)
    
    # 添加目标线
    ax.axhline(y=calibrator.target_rgb[0], color='black', linestyle='--', 
               label=f'目标值({calibrator.target_rgb[0]})', alpha=0.8)
    
    # 设置图形参数
    ax.set_ylabel('RGB值')
    ax.set_title('原始与校正后RGB值对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rgb_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 校正因子图
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    bars = plt.bar(labels, calibrator.best_individual, color=colors, alpha=0.7)
    
    # 添加基准线
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='无校正线(1.0)')
    
    # 添加数值标签
    for bar, factor in zip(bars, calibrator.best_individual):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{factor:.3f}', ha='center', fontweight='bold')
    
    plt.title('RGB通道校正因子', fontsize=14, fontweight='bold')
    plt.ylabel('校正因子')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correction_factors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"所有图表已保存至: {output_dir}/")

def main():
    """主函数 - 按照用户推荐思路的LED校准解决方案"""
    print("=== LED显示屏色彩均匀性校准系统（遗传算法版本） ===")
    print("基于用户推荐的遗传算法思路")
    print("核心算法：遗传算法 + 欧几里得距离误差最小化")
    print("=" * 65)
    
    # 创建校准器
    target_rgb = (220, 220, 220)
    calibrator = SimpleGeneticCalibrator(target_rgb=target_rgb)
    
    # 步骤1: 加载数据
    print("\\n步骤1: 加载LED数据...")
    
    data_file = "B题附件：RGB数值.xlsx"
    if os.path.exists(data_file):
        success = calibrator.load_led_data_from_excel(data_file)
        if not success:
            print("Excel数据加载失败，使用示例数据")
            calibrator.generate_sample_data(n_pixels=100)  # 按照用户示例：10个像素，这里用100个
    else:
        print("Excel文件不存在，使用示例数据")
        calibrator.generate_sample_data(n_pixels=100)
    
    # 步骤2: 遗传算法优化
    print("\\n步骤2: 遗传算法优化校正因子...")
    best_individual, best_fitness = calibrator.optimize()
    
    # 步骤3: 分析改善效果
    print("\\n步骤3: 分析校准改善效果...")
    analysis_results = calibrator.analyze_improvement()
    
    # 步骤4: 可视化结果
    print("\\n步骤4: 生成可视化结果...")
    output_dir = "led_ga_simple_results"
    plot_results(calibrator, output_dir)
    
    # 步骤5: 保存结果
    print("\\n步骤5: 保存结果...")
    
    # 保存校正因子
    np.save(os.path.join(output_dir, 'correction_factors.npy'), best_individual)
    
    # 保存校正后数据
    corrected_rgb = calibrator.apply_calibration()
    np.save(os.path.join(output_dir, 'corrected_rgb_data.npy'), corrected_rgb)
    
    # 保存分析报告
    with open(os.path.join(output_dir, 'analysis_report.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print("\\n=== 处理完成 ===")
    print(f"最佳校正因子: R校正: {best_individual[0]:.4f}, G校正: {best_individual[1]:.4f}, B校正: {best_individual[2]:.4f}")
    print(f"最小颜色误差: {best_fitness:.2f}")
    print(f"总误差改善: {analysis_results['error_improvement']:.1f}%")
    
    print("\\n生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")
    
    return analysis_results

if __name__ == "__main__":
    main()
