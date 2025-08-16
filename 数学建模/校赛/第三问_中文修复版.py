#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED显示屏智能校准系统 - 中文显示修复版
基于用户指导的优化解决方案
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 强制设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 验证字体设置
def verify_chinese_font():
    """验证中文字体设置"""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '中文测试：红绿蓝校准', ha='center', va='center', fontsize=16)
        ax.set_title('字体测试')
        plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 中文字体设置成功")
        return True
    except Exception as e:
        print(f"✗ 字体设置失败: {e}")
        return False

class OptimizedLEDCalibrator:
    """优化的LED校准器"""
    
    def __init__(self, target_rgb: np.ndarray = np.array([220, 220, 200]), 
                 channel_weights: np.ndarray = np.array([2.0, 1.0, 1.2])):
        """
        初始化校准器
        target_rgb: 目标RGB值 [红色≤220, 绿色≈220, 蓝色180-200]
        channel_weights: 通道权重 [红×2, 绿×1, 蓝×1.2]
        """
        self.target_rgb = target_rgb
        self.channel_weights = channel_weights
        self.original_data = None
        self.correction_factors = None
        self.corrected_data = None
        
    def load_real_data(self, filepath: str) -> bool:
        """加载真实LED数据"""
        try:
            print(f"正在加载真实LED数据: {filepath}")
            
            # 读取Excel文件的所有工作表
            excel_data = pd.read_excel(filepath, sheet_name=None)
            
            # 提取RGB数据矩阵
            rgb_matrices = {}
            for sheet_name, data in excel_data.items():
                if any(color in sheet_name.upper() for color in ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']):
                    # 只取数值数据并展平
                    numeric_data = data.select_dtypes(include=[np.number]).values.flatten()
                    rgb_matrices[sheet_name] = numeric_data
                    print(f"  {sheet_name}: {len(numeric_data)} 数据点")
            
            # 构建RGB数据 - 使用最小长度对齐
            if len(rgb_matrices) >= 9:
                # 找到最小数据长度
                min_length = min(len(data) for data in rgb_matrices.values())
                print(f"  对齐数据长度: {min_length}")
                
                # 构建RGB矩阵，每个像素点对应一个RGB值
                rgb_data = []
                for i in range(min_length):
                    # 从R_R, G_G, B_B中取对应位置的值作为该像素的RGB
                    r_val = rgb_matrices['R_R'][i] if i < len(rgb_matrices['R_R']) else 200
                    g_val = rgb_matrices['G_G'][i] if i < len(rgb_matrices['G_G']) else 200  
                    b_val = rgb_matrices['B_B'][i] if i < len(rgb_matrices['B_B']) else 200
                    rgb_data.append([r_val, g_val, b_val])
                
                self.original_data = np.array(rgb_data)
                
                print(f"\n真实LED数据加载成功:")
                print(f"  数据形状: {self.original_data.shape}")
                print(f"  红色范围: [{self.original_data[:, 0].min():.1f}, {self.original_data[:, 0].max():.1f}]")
                print(f"  绿色范围: [{self.original_data[:, 1].min():.1f}, {self.original_data[:, 1].max():.1f}]")
                print(f"  蓝色范围: [{self.original_data[:, 2].min():.1f}, {self.original_data[:, 2].max():.1f}]")
                return True
            else:
                raise ValueError("Excel文件中RGB矩阵数量不足")
                
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return False
    
    def advanced_fitness_function(self, factors: np.ndarray, data: np.ndarray) -> float:
        """高级适应度函数"""
        corrected = data * factors
        
        # 目标偏差（加权）
        mean_corrected = np.mean(corrected, axis=0)
        target_error = np.sum(self.channel_weights * (mean_corrected - self.target_rgb) ** 2)
        
        # 均匀性惩罚
        uniformity_penalty = np.sum([np.var(corrected[:, i]) for i in range(3)]) * 0.1
        
        # 范围约束惩罚
        range_penalty = 0
        if mean_corrected[0] > 220:  # 红色不能超过220
            range_penalty += (mean_corrected[0] - 220) ** 2 * 10
        if mean_corrected[2] < 180 or mean_corrected[2] > 200:  # 蓝色范围180-200
            if mean_corrected[2] < 180:
                range_penalty += (180 - mean_corrected[2]) ** 2 * 5
            else:
                range_penalty += (mean_corrected[2] - 200) ** 2 * 5
        
        return target_error + uniformity_penalty + range_penalty
    
    def optimize_calibration(self, population_size: int = 80, generations: int = 150) -> Dict:
        """执行校准优化"""
        if self.original_data is None:
            raise ValueError("请先加载数据")
        
        print(f"\n=== 开始优化校准算法 ===")
        print(f"优化目标: {self.target_rgb}")
        print(f"通道权重: 红×{self.channel_weights[0]}, 绿×{self.channel_weights[1]}, 蓝×{self.channel_weights[2]}")
        print(f"种群大小: {population_size}, 进化代数: {generations}")
        
        # 智能初始化种群
        population = []
        for _ in range(population_size):
            factors = np.random.uniform(0.7, 1.3, 3)  # 校正因子范围
            population.append(factors)
        
        history = {'fitness': [], 'best_factors': [], 'mean_fitness': []}
        best_fitness = float('inf')
        best_factors = None
        stagnation_count = 0
        
        for generation in range(generations):
            # 计算适应度
            fitness_scores = []
            for factors in population:
                fitness = self.advanced_fitness_function(factors, self.original_data)
                fitness_scores.append(fitness)
            
            # 记录历史
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_factors = population[current_best_idx].copy()
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_factors = current_best_factors.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            history['fitness'].append(best_fitness)
            history['best_factors'].append(best_factors.copy())
            history['mean_fitness'].append(np.mean(fitness_scores))
            
            # 显示进度
            if generation % 15 == 0 or generation == generations - 1:
                print(f"第 {generation:3d} 代: 最佳={best_fitness:.2f}, 平均={np.mean(fitness_scores):.2f}, 停滞={stagnation_count}")
            
            # 早停条件
            if stagnation_count > 30:
                print(f"第 {generation} 代早停：连续{stagnation_count}代无改善")
                break
            
            # 选择和变异
            sorted_indices = np.argsort(fitness_scores)
            elite_size = population_size // 4
            new_population = [population[i].copy() for i in sorted_indices[:elite_size]]
            
            # 生成新个体
            while len(new_population) < population_size:
                if np.random.random() < 0.7:  # 交叉
                    parent1 = population[sorted_indices[np.random.randint(0, elite_size)]]
                    parent2 = population[sorted_indices[np.random.randint(0, elite_size)]]
                    child = (parent1 + parent2) / 2
                else:  # 变异
                    parent = population[sorted_indices[np.random.randint(0, elite_size)]]
                    child = parent + np.random.normal(0, 0.05, 3)
                
                # 约束校正因子范围
                child = np.clip(child, 0.5, 1.5)
                new_population.append(child)
            
            population = new_population
        
        self.correction_factors = best_factors
        self.corrected_data = self.original_data * best_factors
        
        print(f"\n优化完成!")
        print(f"最佳校正因子: R={best_factors[0]:.4f}, G={best_factors[1]:.4f}, B={best_factors[2]:.4f}")
        print(f"最优适应度: {best_fitness:.2f}")
        
        return history
    
    def comprehensive_analysis(self) -> Dict:
        """全面校准效果分析"""
        if self.corrected_data is None:
            raise ValueError("请先执行校准优化")
        
        original_mean = np.mean(self.original_data, axis=0)
        original_std = np.std(self.original_data, axis=0)
        corrected_mean = np.mean(self.corrected_data, axis=0)
        corrected_std = np.std(self.corrected_data, axis=0)
        
        # 变异系数 (CV)
        original_cv = original_std / original_mean
        corrected_cv = corrected_std / corrected_mean
        
        # 目标达成验证
        target_achievement = {
            'red_target': corrected_mean[0] <= 220,
            'green_target': abs(corrected_mean[1] - 220) <= 5,
            'blue_target': 180 <= corrected_mean[2] <= 200
        }
        
        analysis = {
            'original': {'mean': original_mean, 'std': original_std, 'cv': original_cv},
            'corrected': {'mean': corrected_mean, 'std': corrected_std, 'cv': corrected_cv},
            'correction_factors': self.correction_factors,
            'target_achievement': target_achievement,
            'overall_cv': np.mean(corrected_cv)
        }
        
        # 打印详细报告
        print(f"\n" + "="*60)
        print(f"           全面校准效果分析报告")
        print(f"=" * 60)
        
        print(f"\n基础统计对比:")
        for i, color in enumerate(['红色', '绿色', '蓝色']):
            print(f"  {color}:")
            print(f"    校准前: 均值={original_mean[i]:6.1f}, 标准差={original_std[i]:5.1f}")
            print(f"    校准后: 均值={corrected_mean[i]:6.1f}, 标准差={corrected_std[i]:5.1f}")
            print(f"    校正因子: {self.correction_factors[i]:.4f}")
        
        print(f"\n目标达成验证:")
        print(f"  红色 ≤ 220:      {corrected_mean[0]:5.1f}  {'✓ 达成' if target_achievement['red_target'] else '✗ 未达成'}")
        print(f"  绿色 ≈ 220:      {corrected_mean[1]:5.1f}  {'✓ 达成' if target_achievement['green_target'] else '✗ 未达成'}")
        print(f"  蓝色 180-200:    {corrected_mean[2]:5.1f}  {'✓ 达成' if target_achievement['blue_target'] else '✗ 未达成'}")
        
        print(f"\n均匀性分析:")
        for i, color in enumerate(['红色', '绿色', '蓝色']):
            cv_change = ((corrected_cv[i] - original_cv[i]) / original_cv[i]) * 100
            print(f"  {color}: CV {original_cv[i]:.4f} → {corrected_cv[i]:.4f} (改善 {cv_change:+.1f}%)")
        
        achievement_count = sum(target_achievement.values())
        print(f"\n整体评估:")
        print(f"  目标达成度: {achievement_count}/3")
        print(f"  整体变异系数: {analysis['overall_cv']:.4f}")
        print(f"  校准质量: {'优秀' if achievement_count == 3 else '良好' if achievement_count >= 2 else '需改进'}")
        
        return analysis
    
    def create_visualizations(self, history: Dict, output_dir: str = "led_optimized_results"):
        """创建可视化报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 优化历史图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LED显示屏校准优化报告', fontsize=16, fontweight='bold')
        
        # 适应度收敛
        axes[0, 0].plot(history['fitness'], 'b-', linewidth=2, label='最佳适应度')
        axes[0, 0].plot(history['mean_fitness'], 'r--', alpha=0.7, label='平均适应度')
        axes[0, 0].set_title('适应度收敛曲线')
        axes[0, 0].set_xlabel('进化代数')
        axes[0, 0].set_ylabel('适应度值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 校正因子演化
        factors_history = np.array(history['best_factors'])
        for i, (color, style) in enumerate([('红色', 'r-'), ('绿色', 'g-'), ('蓝色', 'b-')]):
            axes[0, 1].plot(factors_history[:, i], style, linewidth=2, label=f'{color}校正因子')
        axes[0, 1].set_title('校正因子演化历程')
        axes[0, 1].set_xlabel('进化代数')
        axes[0, 1].set_ylabel('校正因子')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RGB值对比（箱线图）
        data_comparison = [
            self.original_data[:, 0], self.corrected_data[:, 0],
            self.original_data[:, 1], self.corrected_data[:, 1],
            self.original_data[:, 2], self.corrected_data[:, 2]
        ]
        box_labels = ['红色(原)', '红色(校准)', '绿色(原)', '绿色(校准)', '蓝色(原)', '蓝色(校准)']
        
        box_plot = axes[1, 0].boxplot(data_comparison, labels=box_labels, patch_artist=True)
        colors = ['lightcoral', 'red', 'lightgreen', 'green', 'lightblue', 'blue']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_title('RGB值分布对比')
        axes[1, 0].set_ylabel('像素值')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 目标达成情况
        mean_corrected = np.mean(self.corrected_data, axis=0)
        target_bars = axes[1, 1].bar(['红色通道', '绿色通道', '蓝色通道'], 
                                   mean_corrected, 
                                   color=['red', 'green', 'blue'], 
                                   alpha=0.7)
        
        # 添加目标线
        axes[1, 1].axhline(y=220, color='red', linestyle='--', alpha=0.8, label='红色目标(≤220)')
        axes[1, 1].axhline(y=220, color='green', linestyle='--', alpha=0.8, label='绿色目标(≈220)')
        axes[1, 1].axhspan(180, 200, alpha=0.2, color='blue', label='蓝色目标范围(180-200)')
        
        axes[1, 1].set_title('目标达成情况')
        axes[1, 1].set_ylabel('像素均值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(target_bars, mean_corrected)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 综合分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RGB通道详细分析报告', fontsize=16, fontweight='bold')
        
        # 原始vs校准后分布对比
        colors = ['red', 'green', 'blue']
        channel_names = ['红色', '绿色', '蓝色']
        
        for i in range(3):
            row, col = i // 2, i % 2 if i < 2 else 0
            if i == 2:  # 第三个图放在第二行中间
                axes[1, 1].hist(self.original_data[:, i], bins=50, alpha=0.6, 
                               color=colors[i], label=f'{channel_names[i]}原始分布', density=True)
                axes[1, 1].hist(self.corrected_data[:, i], bins=50, alpha=0.6, 
                               color=colors[i], label=f'{channel_names[i]}校准后分布', 
                               density=True, linestyle='--')
                axes[1, 1].set_title(f'{channel_names[i]}通道分布对比')
                axes[1, 1].set_xlabel('像素值')
                axes[1, 1].set_ylabel('密度')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[row, col].hist(self.original_data[:, i], bins=50, alpha=0.6, 
                                  color=colors[i], label=f'{channel_names[i]}原始分布', density=True)
                axes[row, col].hist(self.corrected_data[:, i], bins=50, alpha=0.6, 
                                  color=colors[i], label=f'{channel_names[i]}校准后分布', 
                                  density=True, linestyle='--')
                axes[row, col].set_title(f'{channel_names[i]}通道分布对比')
                axes[row, col].set_xlabel('像素值')
                axes[row, col].set_ylabel('密度')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        # 删除空白子图
        fig.delaxes(axes[1, 0])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        np.save(f"{output_dir}/correction_factors.npy", self.correction_factors)
        np.save(f"{output_dir}/corrected_rgb_data.npy", self.corrected_data)
        
        print(f"\n可视化报告已生成:")
        print(f"  - {output_dir}/optimization_history.png")
        print(f"  - {output_dir}/comprehensive_analysis.png")

def main():
    """主函数"""
    print("=" * 80)
    print("          LED显示屏智能校准系统")
    print("       基于用户指导的优化解决方案")
    print("=" * 80)
    
    # 验证中文字体
    if not verify_chinese_font():
        print("警告: 中文字体可能无法正常显示")
    
    # 初始化校准器
    print(f"\n目标设定:")
    print(f"  • 红色通道: 抑制至 ≤220")
    print(f"  • 绿色通道: 逼近至 ≈220") 
    print(f"  • 蓝色通道: 提升至 180-200")
    print(f"  • 整体要求: 分布均匀，无偏移")
    
    calibrator = OptimizedLEDCalibrator(
        target_rgb=np.array([220, 220, 200]),
        channel_weights=np.array([2.0, 1.0, 1.2])
    )
    
    # 数据加载
    print(f"\n步骤1: 数据加载...")
    data_file = "B题附件：RGB数值.xlsx"
    if os.path.exists(data_file):
        print("发现真实LED数据文件，尝试加载...")
        if calibrator.load_real_data(data_file):
            print("✓ 真实LED数据加载成功")
        else:
            print("✗ 真实数据加载失败，程序退出")
            return
    else:
        print(f"✗ 未找到数据文件: {data_file}")
        return
    
    # 校准优化
    print(f"\n步骤2: 智能优化校准...")
    try:
        history = calibrator.optimize_calibration(population_size=80, generations=150)
    except Exception as e:
        print(f"✗ 优化失败: {e}")
        return
    
    # 效果分析
    print(f"\n步骤3: 校准效果分析...")
    try:
        analysis = calibrator.comprehensive_analysis()
    except Exception as e:
        print(f"✗ 分析失败: {e}")
        return
    
    # 可视化
    print(f"\n步骤4: 生成可视化报告...")
    try:
        calibrator.create_visualizations(history)
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        return
    
    print(f"\n" + "=" * 80)
    print(f"🎉 校准完成！")
    print(f"📁 所有结果已保存至: led_optimized_results/")
    print(f"=" * 80)

if __name__ == "__main__":
    main()
