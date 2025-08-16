#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED显示屏色彩均匀性校准系统 - 基于真实数据的映射反演方法
问题三：针对64×64 LED显示屏的精确校准解决方案

数据结构说明：
- RGB目标值：64×68 的理想输出值（220）
- R_R, R_G, R_B：红色LED在RGB三通道下的实际输出
- G_R, G_G, G_B：绿色LED在RGB三通道下的实际输出  
- B_R, B_G, B_B：蓝色LED在RGB三通道下的实际输出

核心算法：映射反演 + 逐点校正 + ΔE最小化

作者：GitHub Copilot
日期：2025年7月28日
版本：3.0 - 真实数据版本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.optimize import minimize, least_squares
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import os
from typing import Tuple, List, Dict, Optional
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
    target_rgb: Tuple[int, int, int] = (220, 220, 220)  # 目标RGB值
    display_size: Tuple[int, int] = (64, 64)  # 显示屏尺寸
    max_correction_value: int = 255  # 最大校正值
    min_correction_value: int = 0    # 最小校正值
    delta_e_threshold: float = 1.0   # ΔE阈值
    optimization_method: str = 'SLSQP'  # 优化方法

class ColorSpaceConverter:
    """色彩空间转换器"""
    
    def __init__(self):
        # RGB到XYZ转换矩阵 (sRGB标准)
        self.rgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
    def rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """RGB到XYZ转换"""
        # 归一化到[0,1]
        rgb_norm = np.clip(rgb / 255.0, 0, 1)
        
        # 线性化RGB
        rgb_linear = np.where(rgb_norm <= 0.04045, 
                             rgb_norm / 12.92, 
                             np.power((rgb_norm + 0.055) / 1.055, 2.4))
        
        # 转换到XYZ
        if rgb_linear.ndim == 1:
            return self.rgb_to_xyz_matrix @ rgb_linear
        else:
            return (self.rgb_to_xyz_matrix @ rgb_linear.T).T
    
    def xyz_to_lab(self, xyz: np.ndarray) -> np.ndarray:
        """XYZ到Lab转换"""
        # D65白点
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        
        x_norm = xyz[..., 0] / xn
        y_norm = xyz[..., 1] / yn  
        z_norm = xyz[..., 2] / zn
        
        def f(t):
            return np.where(t > (216/24389), np.power(t, 1/3), (841/108) * t + 16/116)
        
        fx = f(x_norm)
        fy = f(y_norm)
        fz = f(z_norm)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)
    
    def calculate_delta_e_lab(self, rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
        """计算ΔE*ab色差"""
        xyz1 = self.rgb_to_xyz(rgb1)
        xyz2 = self.rgb_to_xyz(rgb2)
        
        lab1 = self.xyz_to_lab(xyz1)
        lab2 = self.xyz_to_lab(xyz2)
        
        delta_lab = lab1 - lab2
        delta_e = np.sqrt(np.sum(delta_lab**2, axis=-1))
        
        return delta_e

class LEDDataProcessor:
    """LED数据处理器"""
    
    def __init__(self, config: LEDCalibrationConfig = None):
        self.config = config or LEDCalibrationConfig()
        self.color_converter = ColorSpaceConverter()
        self.led_data = {}
        self.target_data = None
        
    def load_led_data(self, file_path: str) -> Dict:
        """加载LED显示屏数据"""
        try:
            print(f"正在加载LED数据文件: {file_path}")
            
            # 读取所有工作表
            xl_file = pd.ExcelFile(file_path)
            print(f"发现工作表: {xl_file.sheet_names}")
            
            led_data = {}
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                led_data[sheet_name] = df
                print(f"加载工作表 {sheet_name}: {df.shape}")
            
            self.led_data = led_data
            
            # 处理目标数据
            if 'RGB目标值' in led_data:
                target_df = led_data['RGB目标值']
                print(f"目标数据形状: {target_df.shape}")
                # 取第一列的值作为目标（应该都是220）
                target_values = target_df.iloc[:, 0].values
                unique_targets = np.unique(target_values)
                print(f"目标值范围: {unique_targets}")
                self.target_data = target_values[0]  # 应该是220
            
            return led_data
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def extract_rgb_matrices(self) -> Dict[str, np.ndarray]:
        """提取RGB矩阵数据"""
        if not self.led_data:
            raise ValueError("请先加载LED数据")
        
        print("\\n=== 提取RGB矩阵数据 ===")
        
        rgb_matrices = {}
        
        # 处理每个LED颜色的RGB输出
        for led_color in ['R', 'G', 'B']:
            print(f"\\n处理 {led_color} LED:")
            
            led_rgb_data = {}
            
            for output_channel in ['R', 'G', 'B']:
                sheet_name = f"{led_color}_{output_channel}"
                
                if sheet_name in self.led_data:
                    df = self.led_data[sheet_name]
                    
                    # 转换为数值矩阵
                    # 假设数据是63x64，需要转换为4032个像素的数据
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    
                    # 展平为一维数组
                    pixel_values = numeric_data.flatten()
                    
                    print(f"  {sheet_name}: {df.shape} -> {len(pixel_values)} 像素")
                    print(f"    数值范围: [{pixel_values.min():.1f}, {pixel_values.max():.1f}]")
                    print(f"    平均值: {pixel_values.mean():.2f}")
                    
                    led_rgb_data[output_channel] = pixel_values
                else:
                    print(f"  警告: 未找到工作表 {sheet_name}")
            
            if len(led_rgb_data) == 3:
                # 组合R、G、B输出数据
                n_pixels = len(led_rgb_data['R'])
                rgb_matrix = np.zeros((n_pixels, 3))
                rgb_matrix[:, 0] = led_rgb_data['R']
                rgb_matrix[:, 1] = led_rgb_data['G']  
                rgb_matrix[:, 2] = led_rgb_data['B']
                
                rgb_matrices[f"{led_color}_LED"] = rgb_matrix
                
                print(f"  {led_color} LED RGB矩阵: {rgb_matrix.shape}")
                print(f"  平均RGB: [{rgb_matrix[:, 0].mean():.1f}, {rgb_matrix[:, 1].mean():.1f}, {rgb_matrix[:, 2].mean():.1f}]")
        
        return rgb_matrices
    
    def analyze_led_uniformity(self, rgb_matrices: Dict[str, np.ndarray]) -> Dict:
        """分析LED均匀性"""
        print("\\n=== LED均匀性分析 ===")
        
        target_rgb = np.array(self.config.target_rgb)
        analysis_results = {}
        
        for led_name, rgb_matrix in rgb_matrices.items():
            print(f"\\n分析 {led_name}:")
            
            led_analysis = {}
            
            # 基本统计
            for i, channel in enumerate(['R', 'G', 'B']):
                values = rgb_matrix[:, i]
                target_val = target_rgb[i]
                
                stats = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'cv': float(values.std() / values.mean()),
                    'uniformity': 1.0 - (values.std() / values.mean()),
                    'target_deviation': float(np.abs(values - target_val).mean()),
                    'max_deviation': float(np.abs(values - target_val).max())
                }
                
                led_analysis[channel] = stats
                
                print(f"  {channel}通道: 均值={stats['mean']:.1f}, CV={stats['cv']:.4f}, 目标偏差={stats['target_deviation']:.1f}")
            
            # 整体色差分析
            target_matrix = np.tile(target_rgb, (len(rgb_matrix), 1))
            delta_e_values = self.color_converter.calculate_delta_e_lab(rgb_matrix, target_matrix)
            
            color_analysis = {
                'average_delta_e': float(delta_e_values.mean()),
                'max_delta_e': float(delta_e_values.max()),
                'excellent_pixels': int(np.sum(delta_e_values < 1.0)),
                'good_pixels': int(np.sum((delta_e_values >= 1.0) & (delta_e_values < 3.0))),
                'poor_pixels': int(np.sum(delta_e_values >= 3.0)),
                'total_pixels': len(delta_e_values)
            }
            
            color_analysis['excellent_percentage'] = color_analysis['excellent_pixels'] / color_analysis['total_pixels'] * 100
            color_analysis['good_percentage'] = color_analysis['good_pixels'] / color_analysis['total_pixels'] * 100
            color_analysis['poor_percentage'] = color_analysis['poor_pixels'] / color_analysis['total_pixels'] * 100
            
            led_analysis['color_quality'] = color_analysis
            
            print(f"  平均ΔE: {color_analysis['average_delta_e']:.3f}")
            print(f"  优秀像素: {color_analysis['excellent_percentage']:.1f}%")
            print(f"  良好像素: {color_analysis['good_percentage']:.1f}%")
            print(f"  较差像素: {color_analysis['poor_percentage']:.1f}%")
            
            analysis_results[led_name] = led_analysis
        
        return analysis_results

class MappingInversionCalibrator:
    """映射反演校准器"""
    
    def __init__(self, config: LEDCalibrationConfig = None):
        self.config = config or LEDCalibrationConfig()
        self.color_converter = ColorSpaceConverter()
        self.calibration_results = {}
        
    def calibrate_led_system(self, rgb_matrices: Dict[str, np.ndarray]) -> Dict:
        """校准LED系统"""
        print("\\n=== 开始LED系统校准 ===")
        
        target_rgb = np.array(self.config.target_rgb)
        calibration_results = {}
        
        for led_name, rgb_matrix in rgb_matrices.items():
            print(f"\\n校准 {led_name}...")
            
            # 应用映射反演算法
            calibration_matrix = self._optimize_led_calibration(rgb_matrix, target_rgb)
            
            # 计算校准后的输出
            calibrated_outputs = self._predict_calibrated_output(rgb_matrix, calibration_matrix)
            
            # 验证校准效果
            validation_results = self._validate_calibration(
                rgb_matrix, calibrated_outputs, target_rgb
            )
            
            calibration_results[led_name] = {
                'original_rgb': rgb_matrix,
                'calibration_matrix': calibration_matrix,
                'calibrated_outputs': calibrated_outputs,
                'validation': validation_results
            }
            
            print(f"  {led_name} 校准完成")
            print(f"  ΔE改善: {validation_results['delta_e_improvement']:.1f}%")
            print(f"  优秀像素提升: {validation_results['excellent_improvement']:.1f}%")
        
        self.calibration_results = calibration_results
        return calibration_results
    
    def _optimize_led_calibration(self, rgb_matrix: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        """优化LED校准参数"""
        
        n_pixels = len(rgb_matrix)
        calibration_matrix = np.zeros((n_pixels, 3))
        
        print(f"    优化 {n_pixels} 个像素的校准参数...")
        
        # 分批处理以显示进度
        batch_size = max(1, n_pixels // 10)
        
        for i in range(n_pixels):
            # 为每个像素求解最优校正值
            pixel_rgb = rgb_matrix[i]
            
            # 目标函数：最小化校准后输出与目标的色差
            def objective(correction_input):
                # 简化的像素响应模型：线性响应
                predicted_output = pixel_rgb * (correction_input / 220.0)  # 假设输入220对应当前输出
                predicted_output = np.clip(predicted_output, 0, 255)
                
                # 计算与目标的色差
                delta_e = self.color_converter.calculate_delta_e_lab(
                    predicted_output.reshape(1, -1),
                    target_rgb.reshape(1, -1)
                )[0]
                
                return delta_e
            
            # 初始猜测：基于比例反推
            initial_guess = target_rgb * (220.0 / (pixel_rgb + 1e-6))  # 避免除零
            initial_guess = np.clip(initial_guess, self.config.min_correction_value, self.config.max_correction_value)
            
            # 边界约束
            bounds = [(self.config.min_correction_value, self.config.max_correction_value) for _ in range(3)]
            
            # 优化求解
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method='L-BFGS-B',  # 更快的优化方法
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                
                if result.success:
                    calibration_matrix[i] = result.x
                else:
                    calibration_matrix[i] = initial_guess
                    
            except:
                calibration_matrix[i] = initial_guess
            
            # 显示进度
            if i % batch_size == 0 or i == n_pixels - 1:
                progress = (i + 1) / n_pixels * 100
                print(f"      进度: {progress:.1f}%")
        
        return calibration_matrix
    
    def _predict_calibrated_output(self, original_rgb: np.ndarray, calibration_matrix: np.ndarray) -> np.ndarray:
        """预测校准后的输出"""
        # 简化的响应模型
        calibrated_outputs = original_rgb * (calibration_matrix / 220.0)
        calibrated_outputs = np.clip(calibrated_outputs, 0, 255)
        return calibrated_outputs
    
    def _validate_calibration(self, original_rgb: np.ndarray, 
                            calibrated_outputs: np.ndarray, 
                            target_rgb: np.ndarray) -> Dict:
        """验证校准效果"""
        
        target_matrix = np.tile(target_rgb, (len(original_rgb), 1))
        
        # 计算校准前后的色差
        delta_e_before = self.color_converter.calculate_delta_e_lab(original_rgb, target_matrix)
        delta_e_after = self.color_converter.calculate_delta_e_lab(calibrated_outputs, target_matrix)
        
        # 质量分析
        excellent_before = np.sum(delta_e_before < 1.0)
        excellent_after = np.sum(delta_e_after < 1.0)
        
        good_before = np.sum((delta_e_before >= 1.0) & (delta_e_before < 3.0))
        good_after = np.sum((delta_e_after >= 1.0) & (delta_e_after < 3.0))
        
        poor_before = np.sum(delta_e_before >= 3.0)
        poor_after = np.sum(delta_e_after >= 3.0)
        
        total_pixels = len(delta_e_before)
        
        return {
            'delta_e_before_mean': float(delta_e_before.mean()),
            'delta_e_after_mean': float(delta_e_after.mean()),
            'delta_e_improvement': float((delta_e_before.mean() - delta_e_after.mean()) / delta_e_before.mean() * 100),
            'excellent_before': int(excellent_before),
            'excellent_after': int(excellent_after),
            'excellent_improvement': float((excellent_after - excellent_before) / total_pixels * 100),
            'good_before': int(good_before),
            'good_after': int(good_after),
            'poor_before': int(poor_before),
            'poor_after': int(poor_after),
            'delta_e_values_before': delta_e_before,
            'delta_e_values_after': delta_e_after
        }

class CalibrationVisualizer:
    """校准结果可视化器"""
    
    def __init__(self, output_dir: str = "led_calibration_results_real"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_led_analysis_summary(self, analysis_results: Dict, save_path: str = None):
        """绘制LED分析总结图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        led_names = list(analysis_results.keys())
        colors = ['red', 'green', 'blue']
        
        # 1. 各LED的ΔE对比
        delta_e_values = [analysis_results[led]['color_quality']['average_delta_e'] for led in led_names]
        bars1 = axes[0, 0].bar(led_names, delta_e_values, color=colors, alpha=0.7)
        axes[0, 0].set_title('各LED平均ΔE值', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('平均ΔE')
        
        for bar, val in zip(bars1, delta_e_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', fontweight='bold')
        
        # 2. 优秀像素百分比对比
        excellent_percentages = [analysis_results[led]['color_quality']['excellent_percentage'] for led in led_names]
        bars2 = axes[0, 1].bar(led_names, excellent_percentages, color=colors, alpha=0.7)
        axes[0, 1].set_title('各LED优秀像素百分比 (ΔE<1)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('优秀像素百分比 (%)')
        
        for bar, val in zip(bars2, excellent_percentages):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.1f}%', ha='center', fontweight='bold')
        
        # 3. RGB通道变异系数
        channel_names = ['R', 'G', 'B']
        x = np.arange(len(channel_names))
        width = 0.25
        
        for i, led_name in enumerate(led_names):
            cv_values = [analysis_results[led_name][ch]['cv'] for ch in channel_names]
            axes[1, 0].bar(x + i*width, cv_values, width, label=led_name, color=colors[i], alpha=0.7)
        
        axes[1, 0].set_title('各LED RGB通道变异系数', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('变异系数 (CV)')
        axes[1, 0].set_xlabel('RGB通道')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(channel_names)
        axes[1, 0].legend()
        
        # 4. 质量分布堆叠图
        excellent_data = [analysis_results[led]['color_quality']['excellent_percentage'] for led in led_names]
        good_data = [analysis_results[led]['color_quality']['good_percentage'] for led in led_names]
        poor_data = [analysis_results[led]['color_quality']['poor_percentage'] for led in led_names]
        
        axes[1, 1].bar(led_names, excellent_data, label='优秀 (ΔE<1)', color='green', alpha=0.7)
        axes[1, 1].bar(led_names, good_data, bottom=excellent_data, label='良好 (1≤ΔE<3)', color='orange', alpha=0.7)
        
        bottoms = [e + g for e, g in zip(excellent_data, good_data)]
        axes[1, 1].bar(led_names, poor_data, bottom=bottoms, label='较差 (ΔE≥3)', color='red', alpha=0.7)
        
        axes[1, 1].set_title('各LED质量分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('像素百分比 (%)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LED分析总结图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'led_analysis_summary.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LED分析总结图保存至: {save_path}")
        
        plt.close()
    
    def plot_calibration_results(self, calibration_results: Dict, save_path: str = None):
        """绘制校准结果对比图"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        led_names = list(calibration_results.keys())
        
        for i, led_name in enumerate(led_names):
            validation = calibration_results[led_name]['validation']
            
            # ΔE改善对比
            delta_e_before = validation['delta_e_before_mean']
            delta_e_after = validation['delta_e_after_mean']
            
            axes[i, 0].bar(['校准前', '校准后'], [delta_e_before, delta_e_after], 
                          color=['red', 'green'], alpha=0.7)
            axes[i, 0].set_title(f'{led_name} ΔE改善', fontsize=14, fontweight='bold')
            axes[i, 0].set_ylabel('平均ΔE')
            
            improvement = validation['delta_e_improvement']
            axes[i, 0].text(0.5, max(delta_e_before, delta_e_after) * 0.8,
                           f'改善: {improvement:.1f}%', ha='center', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # 质量分布对比
            categories = ['优秀', '良好', '较差']
            before_counts = [validation['excellent_before'], validation['good_before'], validation['poor_before']]
            after_counts = [validation['excellent_after'], validation['good_after'], validation['poor_after']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[i, 1].bar(x - width/2, before_counts, width, label='校准前', color='lightcoral', alpha=0.7)
            axes[i, 1].bar(x + width/2, after_counts, width, label='校准后', color='lightgreen', alpha=0.7)
            
            axes[i, 1].set_title(f'{led_name} 质量分布对比', fontsize=14, fontweight='bold')
            axes[i, 1].set_ylabel('像素数量')
            axes[i, 1].set_xticks(x)
            axes[i, 1].set_xticklabels(categories)
            axes[i, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准结果对比图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'calibration_results_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准结果对比图保存至: {save_path}")
        
        plt.close()

def main():
    """主函数 - LED显示屏校准完整流程"""
    print("=== LED显示屏色彩均匀性校准系统 - 基于真实数据 ===")
    print("问题三：64×64 LED显示屏精确校准解决方案")
    print("核心算法：映射反演 + 逐点校正 + ΔE最小化")
    print("=" * 75)
    
    # 配置参数
    config = LEDCalibrationConfig(
        target_rgb=(220, 220, 220),
        display_size=(64, 64),
        delta_e_threshold=1.0
    )
    
    output_dir = "led_calibration_results_real"
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 数据加载
    print("\\n步骤1: 加载真实LED数据...")
    processor = LEDDataProcessor(config)
    
    data_file = "B题附件：RGB数值.xlsx"
    led_data = processor.load_led_data(data_file)
    
    if led_data is None:
        print("数据加载失败，程序退出")
        return
    
    # 步骤2: 提取RGB矩阵
    print("\\n步骤2: 提取RGB矩阵数据...")
    rgb_matrices = processor.extract_rgb_matrices()
    
    # 步骤3: 均匀性分析
    print("\\n步骤3: LED均匀性分析...")
    analysis_results = processor.analyze_led_uniformity(rgb_matrices)
    
    # 步骤4: 映射反演校准
    print("\\n步骤4: 应用映射反演校准...")
    calibrator = MappingInversionCalibrator(config)
    calibration_results = calibrator.calibrate_led_system(rgb_matrices)
    
    # 步骤5: 可视化结果
    print("\\n步骤5: 生成可视化结果...")
    visualizer = CalibrationVisualizer(output_dir)
    
    visualizer.plot_led_analysis_summary(
        analysis_results,
        os.path.join(output_dir, 'led_analysis_summary.png')
    )
    
    visualizer.plot_calibration_results(
        calibration_results,
        os.path.join(output_dir, 'calibration_results_comparison.png')
    )
    
    # 步骤6: 保存结果
    print("\\n步骤6: 保存校准结果...")
    
    # 保存分析结果
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    # 保存校准矩阵
    for led_name, results in calibration_results.items():
        np.save(os.path.join(output_dir, f'{led_name}_calibration_matrix.npy'), 
                results['calibration_matrix'])
        np.save(os.path.join(output_dir, f'{led_name}_calibrated_outputs.npy'), 
                results['calibrated_outputs'])
    
    # 生成综合报告
    print("\\n=== 校准完成总结 ===")
    
    total_improvement = 0
    total_excellent_improvement = 0
    
    for led_name, results in calibration_results.items():
        validation = results['validation']
        delta_e_improvement = validation['delta_e_improvement']
        excellent_improvement = validation['excellent_improvement']
        
        print(f"\\n{led_name}:")
        print(f"  ΔE改善: {delta_e_improvement:.1f}%")
        print(f"  优秀像素提升: {excellent_improvement:.1f}%")
        print(f"  优秀率: {validation['excellent_before']} → {validation['excellent_after']}")
        
        total_improvement += delta_e_improvement
        total_excellent_improvement += excellent_improvement
    
    avg_improvement = total_improvement / len(calibration_results)
    avg_excellent_improvement = total_excellent_improvement / len(calibration_results)
    
    print(f"\\n整体效果:")
    print(f"  平均ΔE改善: {avg_improvement:.1f}%")
    print(f"  平均优秀像素提升: {avg_excellent_improvement:.1f}%")
    
    print(f"\\n所有结果已保存至: {output_dir}/")
    print("生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
