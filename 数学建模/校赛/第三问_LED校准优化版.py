#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED显示屏色彩均匀性校准与优化系统
问题三：基于PSO优化算法的LED显示屏校准解决方案

作者：GitHub Copilot
日期：2025年7月28日
版本：1.0

功能说明：
1. 读取LED显示屏RGB数据
2. 分析色彩不均匀性
3. 应用PSO优化算法进行校准
4. 生成校准系数和可视化结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import os
from typing import Tuple, List, Dict
import random
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端和中文字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

@dataclass
class LEDCalibrationConfig:
    """LED校准配置参数"""
    target_brightness: float = 220.0  # 目标亮度值
    display_size: Tuple[int, int] = (64, 64)  # 显示屏尺寸
    max_correction_factor: float = 1.5  # 最大校正系数
    min_correction_factor: float = 0.5  # 最小校正系数
    uniformity_threshold: float = 0.05  # 均匀性阈值
    pso_swarm_size: int = 30  # PSO粒子群大小
    pso_max_iter: int = 100  # PSO最大迭代次数

class LEDDisplayAnalyzer:
    """LED显示屏数据分析器"""
    
    def __init__(self, config: LEDCalibrationConfig = None):
        self.config = config or LEDCalibrationConfig()
        self.rgb_data = None
        self.uniformity_metrics = {}
        
    def load_led_data(self, file_path: str) -> pd.DataFrame:
        """加载LED显示屏RGB数据"""
        try:
            print(f"正在加载LED数据文件: {file_path}")
            
            # 读取Excel文件
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            # 验证数据格式
            expected_pixels = self.config.display_size[0] * self.config.display_size[1]
            if df.shape[0] != expected_pixels:
                print(f"警告: 数据像素数({df.shape[0]})与预期({expected_pixels})不匹配")
            
            self.rgb_data = df
            return df
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def analyze_uniformity(self) -> Dict:
        """分析LED显示屏均匀性"""
        if self.rgb_data is None:
            raise ValueError("请先加载LED数据")
        
        print("\n=== LED显示屏均匀性分析 ===")
        
        # 获取RGB列
        rgb_columns = []
        for col in self.rgb_data.columns:
            if any(c in col.upper() for c in ['R', 'G', 'B']):
                rgb_columns.append(col)
        
        if len(rgb_columns) < 3:
            # 假设前三列是RGB
            rgb_columns = self.rgb_data.columns[:3].tolist()
        
        print(f"RGB列: {rgb_columns}")
        
        metrics = {}
        
        for i, col in enumerate(rgb_columns[:3]):
            channel_name = ['Red', 'Green', 'Blue'][i]
            values = self.rgb_data[col].values
            
            # 基本统计
            stats = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'cv': float(values.std() / values.mean()),  # 变异系数
                'uniformity': 1.0 - (values.std() / values.mean())  # 均匀性指数
            }
            
            # 与目标值的偏差
            target_deviation = np.abs(values - self.config.target_brightness)
            stats['target_deviation_mean'] = float(target_deviation.mean())
            stats['target_deviation_max'] = float(target_deviation.max())
            
            # 空间均匀性分析（如果是64x64排列）
            if len(values) == 4096:
                matrix = values.reshape(64, 64)
                # 计算空间梯度
                grad_x = np.gradient(matrix, axis=1)
                grad_y = np.gradient(matrix, axis=0)
                spatial_gradient = np.sqrt(grad_x**2 + grad_y**2)
                stats['spatial_uniformity'] = float(1.0 / (1.0 + spatial_gradient.mean()))
            
            metrics[channel_name] = stats
            
            print(f"\n{channel_name}通道统计:")
            print(f"  均值: {stats['mean']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  变异系数: {stats['cv']:.4f}")
            print(f"  均匀性指数: {stats['uniformity']:.4f}")
            print(f"  与目标值偏差: {stats['target_deviation_mean']:.2f} ± {stats['target_deviation_max']:.2f}")
        
        # 整体均匀性评估
        overall_cv = np.mean([metrics[ch]['cv'] for ch in metrics.keys()])
        overall_uniformity = np.mean([metrics[ch]['uniformity'] for ch in metrics.keys()])
        
        metrics['Overall'] = {
            'average_cv': float(overall_cv),
            'average_uniformity': float(overall_uniformity),
            'quality_grade': self._assess_quality_grade(overall_uniformity)
        }
        
        print(f"\n整体评估:")
        print(f"  平均变异系数: {overall_cv:.4f}")
        print(f"  平均均匀性: {overall_uniformity:.4f}")
        print(f"  质量等级: {metrics['Overall']['quality_grade']}")
        
        self.uniformity_metrics = metrics
        return metrics
    
    def _assess_quality_grade(self, uniformity: float) -> str:
        """评估质量等级"""
        if uniformity >= 0.95:
            return "优秀"
        elif uniformity >= 0.90:
            return "良好" 
        elif uniformity >= 0.80:
            return "一般"
        else:
            return "较差"

class PSOLEDCalibrator:
    """基于PSO算法的LED校准优化器"""
    
    def __init__(self, config: LEDCalibrationConfig = None):
        self.config = config or LEDCalibrationConfig()
        self.calibration_matrix = None
        self.optimization_history = []
        
    def optimize_calibration(self, rgb_data: np.ndarray) -> np.ndarray:
        """使用PSO算法优化LED校准参数"""
        print("\n=== PSO LED校准优化开始 ===")
        
        # 初始化粒子群
        swarm_size = self.config.pso_swarm_size
        max_iter = self.config.pso_max_iter
        
        # 参数维度：每个像素的RGB校正系数
        n_pixels = rgb_data.shape[0]
        n_dims = n_pixels * 3  # RGB三通道
        
        print(f"优化参数维度: {n_dims}")
        print(f"粒子群大小: {swarm_size}")
        print(f"最大迭代次数: {max_iter}")
        
        # 初始化粒子位置和速度
        particles = np.random.uniform(
            self.config.min_correction_factor,
            self.config.max_correction_factor,
            (swarm_size, n_dims)
        )
        
        velocities = np.random.uniform(-0.1, 0.1, (swarm_size, n_dims))
        
        # 个体最优和全局最优
        personal_best = particles.copy()
        personal_best_fitness = np.full(swarm_size, float('inf'))
        global_best = None
        global_best_fitness = float('inf')
        
        # PSO参数
        w = 0.7  # 惯性权重
        c1 = 1.5  # 个体学习因子
        c2 = 1.5  # 社会学习因子
        
        # 优化迭代
        for iteration in range(max_iter):
            for i in range(swarm_size):
                # 计算适应度
                fitness = self._calculate_fitness(particles[i], rgb_data)
                
                # 更新个体最优
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()
                
                # 更新全局最优
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()
            
            # 更新粒子速度和位置
            for i in range(swarm_size):
                r1, r2 = np.random.random(2)
                
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                particles[i] += velocities[i]
                
                # 边界约束
                particles[i] = np.clip(particles[i], 
                                     self.config.min_correction_factor,
                                     self.config.max_correction_factor)
            
            # 记录优化历史
            self.optimization_history.append({
                'iteration': iteration,
                'best_fitness': global_best_fitness,
                'mean_fitness': np.mean(personal_best_fitness)
            })
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 最优适应度 = {global_best_fitness:.6f}")
        
        print(f"PSO优化完成，最终适应度: {global_best_fitness:.6f}")
        
        # 将最优解重塑为校准矩阵
        self.calibration_matrix = global_best.reshape(n_pixels, 3)
        return self.calibration_matrix
    
    def _calculate_fitness(self, correction_factors: np.ndarray, rgb_data: np.ndarray) -> float:
        """计算适应度函数"""
        # 重塑校正系数
        factors = correction_factors.reshape(rgb_data.shape[0], 3)
        
        # 应用校正
        corrected_rgb = rgb_data * factors
        
        # 目标：使所有像素接近目标亮度
        target = self.config.target_brightness
        
        # 计算均匀性目标函数
        uniformity_penalty = 0.0
        
        for channel in range(3):
            values = corrected_rgb[:, channel]
            
            # 与目标值的偏差
            target_deviation = np.mean(np.abs(values - target))
            
            # 变异系数惩罚
            cv = values.std() / values.mean() if values.mean() > 0 else 1.0
            
            # 空间连续性惩罚（相邻像素差异）
            if len(values) == 4096:  # 64x64
                matrix = values.reshape(64, 64)
                spatial_penalty = self._calculate_spatial_penalty(matrix)
            else:
                spatial_penalty = 0
            
            uniformity_penalty += target_deviation + 100 * cv + spatial_penalty
        
        return uniformity_penalty
    
    def _calculate_spatial_penalty(self, matrix: np.ndarray) -> float:
        """计算空间连续性惩罚"""
        # 相邻像素差异
        diff_h = np.abs(np.diff(matrix, axis=1))  # 水平方向
        diff_v = np.abs(np.diff(matrix, axis=0))  # 垂直方向
        
        return np.mean(diff_h) + np.mean(diff_v)
    
    def apply_calibration(self, rgb_data: np.ndarray) -> np.ndarray:
        """应用校准矩阵"""
        if self.calibration_matrix is None:
            raise ValueError("请先运行优化算法生成校准矩阵")
        
        return rgb_data * self.calibration_matrix

class LEDCalibrationVisualizer:
    """LED校准可视化器"""
    
    def __init__(self, output_dir: str = "led_calibration_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_uniformity_analysis(self, metrics: Dict, save_path: str = None):
        """绘制均匀性分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取RGB通道数据
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        # 1. 均匀性指数对比
        uniformity_values = [metrics[ch]['uniformity'] for ch in channels]
        axes[0, 0].bar(channels, uniformity_values, color=colors, alpha=0.7)
        axes[0, 0].set_title('各通道均匀性指数', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('均匀性指数')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(uniformity_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. 变异系数对比
        cv_values = [metrics[ch]['cv'] for ch in channels]
        axes[0, 1].bar(channels, cv_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('各通道变异系数', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('变异系数 (CV)')
        for i, v in enumerate(cv_values):
            axes[0, 1].text(i, v + max(cv_values) * 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # 3. 目标偏差分析
        deviation_mean = [metrics[ch]['target_deviation_mean'] for ch in channels]
        deviation_max = [metrics[ch]['target_deviation_max'] for ch in channels]
        
        x = np.arange(len(channels))
        width = 0.35
        axes[1, 0].bar(x - width/2, deviation_mean, width, label='平均偏差', color='orange', alpha=0.7)
        axes[1, 0].bar(x + width/2, deviation_max, width, label='最大偏差', color='red', alpha=0.7)
        axes[1, 0].set_title('与目标值(220)的偏差', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('偏差值')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(channels)
        axes[1, 0].legend()
        
        # 4. 质量等级总结
        overall_data = metrics['Overall']
        labels = ['平均变异系数', '平均均匀性']
        values = [overall_data['average_cv'], overall_data['average_uniformity']]
        
        axes[1, 1].pie([overall_data['average_uniformity'], 1 - overall_data['average_uniformity']], 
                      labels=['均匀性', '不均匀性'], autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title(f'整体质量评估\\n等级: {overall_data["quality_grade"]}', 
                            fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"均匀性分析图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'uniformity_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"均匀性分析图保存至: {save_path}")
        
        plt.close()
    
    def plot_led_matrix(self, rgb_data: np.ndarray, title: str, save_path: str = None):
        """绘制LED矩阵热力图"""
        if rgb_data.shape[0] != 4096:
            print(f"警告: 数据不是64x64格式 (当前: {rgb_data.shape})")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        channels = ['Red', 'Green', 'Blue']
        
        for i in range(3):
            matrix = rgb_data[:, i].reshape(64, 64)
            
            im = axes[i].imshow(matrix, cmap='hot', aspect='equal')
            axes[i].set_title(f'{channels[i]} Channel', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('X Position')
            axes[i].set_ylabel('Y Position')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Brightness Value')
            
            # 添加统计信息
            stats_text = f'Mean: {matrix.mean():.1f}\\nStd: {matrix.std():.1f}\\nMin: {matrix.min():.1f}\\nMax: {matrix.max():.1f}'
            axes[i].text(0.02, 0.98, stats_text,
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LED矩阵图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, f'{title.replace(" ", "_").lower()}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LED矩阵图保存至: {save_path}")
        
        plt.close()
    
    def plot_calibration_comparison(self, original_data: np.ndarray, 
                                  calibrated_data: np.ndarray, save_path: str = None):
        """绘制校准前后对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        channels = ['Red', 'Green', 'Blue']
        
        for i in range(3):
            # 原始数据
            orig_matrix = original_data[:, i].reshape(64, 64)
            im1 = axes[0, i].imshow(orig_matrix, cmap='hot', aspect='equal')
            axes[0, i].set_title(f'Original {channels[i]}', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 校准后数据
            calib_matrix = calibrated_data[:, i].reshape(64, 64)
            im2 = axes[1, i].imshow(calib_matrix, cmap='hot', aspect='equal')
            axes[1, i].set_title(f'Calibrated {channels[i]}', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[1, i])
            
            # 添加改善统计
            orig_cv = orig_matrix.std() / orig_matrix.mean()
            calib_cv = calib_matrix.std() / calib_matrix.mean()
            improvement = (orig_cv - calib_cv) / orig_cv * 100
            
            axes[1, i].text(0.02, 0.02, f'CV改善: {improvement:.1f}%',
                           transform=axes[1, i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                           fontsize=10, fontweight='bold')
        
        plt.suptitle('LED校准前后对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准对比图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'calibration_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准对比图保存至: {save_path}")
        
        plt.close()
    
    def plot_optimization_history(self, history: List[Dict], save_path: str = None):
        """绘制优化历史曲线"""
        iterations = [h['iteration'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        mean_fitness = [h['mean_fitness'] for h in history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(iterations, best_fitness, 'b-', linewidth=2, label='最优适应度')
        plt.plot(iterations, mean_fitness, 'r--', linewidth=2, label='平均适应度')
        plt.title('PSO优化收敛曲线', fontsize=14, fontweight='bold')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.semilogy(iterations, best_fitness, 'b-', linewidth=2)
        plt.title('适应度值对数图', fontsize=14, fontweight='bold')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值 (对数尺度)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"优化历史图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'optimization_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"优化历史图保存至: {save_path}")
        
        plt.close()

def main():
    """主函数 - LED显示屏校准完整流程"""
    print("=== LED显示屏色彩均匀性校准与优化系统 ===")
    print("问题三：基于PSO优化算法的LED显示屏校准解决方案")
    print("=" * 60)
    
    # 配置参数
    config = LEDCalibrationConfig()
    
    # 创建输出目录
    output_dir = "led_calibration_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 数据加载与分析
    print("\n步骤1: 加载LED显示屏数据...")
    analyzer = LEDDisplayAnalyzer(config)
    
    # 尝试加载数据文件
    data_file = "B题附件：RGB数值.xlsx"
    if os.path.exists(data_file):
        df = analyzer.load_led_data(data_file)
        if df is not None:
            # 提取RGB数据
            rgb_columns = df.columns[:3].tolist()  # 假设前三列是RGB
            rgb_data = df[rgb_columns].values
        else:
            print("数据加载失败，使用模拟数据")
            rgb_data = generate_simulated_led_data(config)
    else:
        print("数据文件不存在，使用模拟数据")
        rgb_data = generate_simulated_led_data(config)
    
    print(f"RGB数据形状: {rgb_data.shape}")
    
    # 步骤2: 均匀性分析
    print("\n步骤2: LED显示屏均匀性分析...")
    if hasattr(analyzer, 'rgb_data') and analyzer.rgb_data is not None:
        metrics = analyzer.analyze_uniformity()
    else:
        # 直接分析RGB数据
        analyzer.rgb_data = pd.DataFrame(rgb_data, columns=['R', 'G', 'B'])
        metrics = analyzer.analyze_uniformity()
    
    # 步骤3: PSO校准优化
    print("\n步骤3: PSO校准优化...")
    calibrator = PSOLEDCalibrator(config)
    calibration_matrix = calibrator.optimize_calibration(rgb_data)
    
    # 应用校准
    calibrated_rgb = calibrator.apply_calibration(rgb_data)
    
    # 步骤4: 校准效果评估
    print("\n步骤4: 校准效果评估...")
    calibrated_analyzer = LEDDisplayAnalyzer(config)
    calibrated_analyzer.rgb_data = pd.DataFrame(calibrated_rgb, columns=['R', 'G', 'B'])
    calibrated_metrics = calibrated_analyzer.analyze_uniformity()
    
    # 计算改善效果
    print("\n=== 校准改善效果 ===")
    for channel in ['Red', 'Green', 'Blue']:
        orig_cv = metrics[channel]['cv']
        calib_cv = calibrated_metrics[channel]['cv']
        improvement = (orig_cv - calib_cv) / orig_cv * 100
        print(f"{channel}通道变异系数改善: {improvement:.1f}%")
    
    overall_orig = metrics['Overall']['average_uniformity']
    overall_calib = calibrated_metrics['Overall']['average_uniformity']
    overall_improvement = (overall_calib - overall_orig) / overall_orig * 100
    print(f"整体均匀性改善: {overall_improvement:.1f}%")
    
    # 步骤5: 可视化结果
    print("\n步骤5: 生成可视化结果...")
    visualizer = LEDCalibrationVisualizer(output_dir)
    
    # 原始数据分析图
    visualizer.plot_uniformity_analysis(metrics, 
                                       os.path.join(output_dir, 'original_uniformity_analysis.png'))
    
    # 校准后分析图
    visualizer.plot_uniformity_analysis(calibrated_metrics,
                                       os.path.join(output_dir, 'calibrated_uniformity_analysis.png'))
    
    # LED矩阵热力图
    visualizer.plot_led_matrix(rgb_data, "Original LED Matrix",
                              os.path.join(output_dir, 'original_led_matrix.png'))
    
    visualizer.plot_led_matrix(calibrated_rgb, "Calibrated LED Matrix",
                              os.path.join(output_dir, 'calibrated_led_matrix.png'))
    
    # 校准前后对比
    visualizer.plot_calibration_comparison(rgb_data, calibrated_rgb,
                                          os.path.join(output_dir, 'calibration_comparison.png'))
    
    # 优化历史
    visualizer.plot_optimization_history(calibrator.optimization_history,
                                        os.path.join(output_dir, 'optimization_history.png'))
    
    # 步骤6: 保存结果数据
    print("\n步骤6: 保存结果数据...")
    
    # 保存校准矩阵
    np.save(os.path.join(output_dir, 'calibration_matrix.npy'), calibration_matrix)
    
    # 保存校准后数据
    np.save(os.path.join(output_dir, 'calibrated_rgb_data.npy'), calibrated_rgb)
    
    # 保存评估报告
    report = {
        'original_metrics': metrics,
        'calibrated_metrics': calibrated_metrics,
        'improvement_summary': {
            'red_cv_improvement': (metrics['Red']['cv'] - calibrated_metrics['Red']['cv']) / metrics['Red']['cv'] * 100,
            'green_cv_improvement': (metrics['Green']['cv'] - calibrated_metrics['Green']['cv']) / metrics['Green']['cv'] * 100,
            'blue_cv_improvement': (metrics['Blue']['cv'] - calibrated_metrics['Blue']['cv']) / metrics['Blue']['cv'] * 100,
            'overall_uniformity_improvement': overall_improvement
        }
    }
    
    with open(os.path.join(output_dir, 'calibration_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n=== 处理完成 ===")
    print(f"所有结果已保存至: {output_dir}/")
    print("\n生成的文件:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    return report

def generate_simulated_led_data(config: LEDCalibrationConfig) -> np.ndarray:
    """生成模拟LED数据用于测试"""
    print("生成模拟LED显示屏数据...")
    
    width, height = config.display_size
    n_pixels = width * height
    target = config.target_brightness
    
    # 创建有不均匀性的模拟数据
    np.random.seed(42)
    
    # 基础亮度模式
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 模拟不同的不均匀性模式
    rgb_data = np.zeros((n_pixels, 3))
    
    for channel in range(3):
        # 不同通道有不同的不均匀性模式
        if channel == 0:  # Red
            base_pattern = target * (0.8 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))
        elif channel == 1:  # Green
            base_pattern = target * (0.9 + 0.2 * (X + Y) / 2)
        else:  # Blue
            base_pattern = target * (0.85 + 0.25 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.2))
        
        # 添加随机噪声
        noise = np.random.normal(0, target * 0.05, (height, width))
        pattern = base_pattern + noise
        
        # 确保在合理范围内
        pattern = np.clip(pattern, target * 0.5, target * 1.5)
        
        rgb_data[:, channel] = pattern.flatten()
    
    print(f"模拟数据生成完成，形状: {rgb_data.shape}")
    return rgb_data

if __name__ == "__main__":
    main()
