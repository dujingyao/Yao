#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED显示屏色彩均匀性校准与优化系统 - 映射反演方法
问题三：基于前两问色彩空间转换映射的LED显示屏校准解决方案

核心思路：映射反演
- 固定输入下(如RGB=220)，若某像素输出色偏为Ci=(Ri,Gi,Bi)
- 目标统一颜色为Ctarget=(R0,G0,B0)
- 需要为该像素重新设计校正输入Cicorr
- 使其通过非理想显示器之后最终输出接近目标值

作者：GitHub Copilot  
日期：2025年7月28日
版本：2.0 - 映射反演版本
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
class CalibrationConfig:
    """校准配置参数"""
    target_rgb: Tuple[int, int, int] = (220, 220, 220)  # 目标RGB值
    display_size: Tuple[int, int] = (64, 64)  # 显示屏尺寸
    max_correction_value: int = 255  # 最大校正值
    min_correction_value: int = 0    # 最小校正值
    delta_e_threshold: float = 1.0   # ΔE阈值
    optimization_method: str = 'SLSQP'  # 优化方法

class ColorSpaceConverter:
    """色彩空间转换器 - 基于前两问建立的转换映射"""
    
    def __init__(self):
        self.setup_color_spaces()
        self.setup_conversion_matrices()
        
    def setup_color_spaces(self):
        """设置色域空间定义"""
        # BT.2020色域顶点 (CIE 1931 xy坐标)
        self.bt2020_vertices = np.array([
            [0.708, 0.292],  # Red
            [0.170, 0.797],  # Green  
            [0.131, 0.046]   # Blue
        ])
        
        # RGBV 4通道色域顶点
        self.rgbv_vertices = np.array([
            [0.708, 0.292],  # R
            [0.170, 0.797],  # G  
            [0.131, 0.046],  # B
            [0.280, 0.330]   # V
        ])
        
        # RGBCX 5通道色域顶点  
        self.rgbcx_vertices = np.array([
            [0.640, 0.330],  # R
            [0.300, 0.600],  # G
            [0.150, 0.060],  # B
            [0.225, 0.329],  # C
            [0.313, 0.329]   # X
        ])
        
    def setup_conversion_matrices(self):
        """设置转换矩阵"""
        # RGB到XYZ转换矩阵 (sRGB标准)
        self.rgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        self.xyz_to_rgb_matrix = np.linalg.inv(self.rgb_to_xyz_matrix)
        
    def rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """RGB到XYZ转换"""
        # 线性化RGB
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
        
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
    
    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """RGB到Lab直接转换"""
        # 归一化RGB到[0,1]
        rgb_norm = rgb / 255.0
        xyz = self.rgb_to_xyz(rgb_norm)
        return self.xyz_to_lab(xyz)
    
    def calculate_delta_e_lab(self, rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
        """计算ΔE*ab色差"""
        lab1 = self.rgb_to_lab(rgb1)
        lab2 = self.rgb_to_lab(rgb2)
        
        delta_lab = lab1 - lab2
        delta_e = np.sqrt(np.sum(delta_lab**2, axis=-1))
        
        return delta_e

class LEDDataAnalyzer:
    """LED数据分析器"""
    
    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.rgb_data = None
        self.color_converter = ColorSpaceConverter()
        
    def load_led_data(self, file_path: str) -> pd.DataFrame:
        """加载LED显示屏RGB数据"""
        try:
            print(f"正在加载LED数据文件: {file_path}")
            
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            # 提取RGB数据
            if len(df.columns) >= 3:
                rgb_columns = df.columns[:3].tolist()
                print(f"使用RGB列: {rgb_columns}")
                
                # 验证数据格式
                expected_pixels = self.config.display_size[0] * self.config.display_size[1]
                if df.shape[0] != expected_pixels:
                    print(f"警告: 像素数({df.shape[0]})与预期64x64={expected_pixels}不匹配")
                
                self.rgb_data = df[rgb_columns].values
                print(f"成功加载RGB数据，形状: {self.rgb_data.shape}")
                print(f"数据范围: R[{self.rgb_data[:, 0].min():.1f}, {self.rgb_data[:, 0].max():.1f}]")
                print(f"         G[{self.rgb_data[:, 1].min():.1f}, {self.rgb_data[:, 1].max():.1f}]")
                print(f"         B[{self.rgb_data[:, 2].min():.1f}, {self.rgb_data[:, 2].max():.1f}]")
                
                return df
            else:
                raise ValueError("数据文件列数不足3列")
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def analyze_uniformity(self) -> Dict:
        """分析LED显示屏均匀性"""
        if self.rgb_data is None:
            raise ValueError("请先加载LED数据")
        
        print("\n=== LED显示屏均匀性分析 ===")
        
        metrics = {}
        target_rgb = np.array(self.config.target_rgb)
        
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            values = self.rgb_data[:, i]
            target_val = target_rgb[i]
            
            # 基本统计
            stats = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'cv': float(values.std() / values.mean()),  # 变异系数
                'uniformity': 1.0 - (values.std() / values.mean()),  # 均匀性指数
                'target_value': float(target_val),
                'deviation_from_target': float(np.abs(values - target_val).mean()),
                'max_deviation': float(np.abs(values - target_val).max())
            }
            
            metrics[channel] = stats
            
            print(f"\n{channel}通道分析:")
            print(f"  均值: {stats['mean']:.2f} (目标: {target_val})")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  变异系数: {stats['cv']:.4f}")
            print(f"  均匀性指数: {stats['uniformity']:.4f}")
            print(f"  与目标偏差: {stats['deviation_from_target']:.2f} ± {stats['max_deviation']:.2f}")
        
        # 计算整体色差
        delta_e_values = self.color_converter.calculate_delta_e_lab(
            self.rgb_data, 
            np.tile(target_rgb, (len(self.rgb_data), 1))
        )
        
        overall_stats = {
            'average_delta_e': float(delta_e_values.mean()),
            'max_delta_e': float(delta_e_values.max()),
            'excellent_pixels': int(np.sum(delta_e_values < 1.0)),  # ΔE < 1
            'good_pixels': int(np.sum((delta_e_values >= 1.0) & (delta_e_values < 3.0))),  # 1 ≤ ΔE < 3
            'poor_pixels': int(np.sum(delta_e_values >= 3.0)),  # ΔE ≥ 3
            'total_pixels': len(delta_e_values)
        }
        
        # 计算质量分布百分比
        total = overall_stats['total_pixels']
        overall_stats['excellent_percentage'] = overall_stats['excellent_pixels'] / total * 100
        overall_stats['good_percentage'] = overall_stats['good_pixels'] / total * 100
        overall_stats['poor_percentage'] = overall_stats['poor_pixels'] / total * 100
        
        metrics['Overall'] = overall_stats
        
        print(f"\n整体色差分析:")
        print(f"  平均ΔE: {overall_stats['average_delta_e']:.3f}")
        print(f"  最大ΔE: {overall_stats['max_delta_e']:.3f}")
        print(f"  优秀像素(ΔE<1): {overall_stats['excellent_pixels']} ({overall_stats['excellent_percentage']:.1f}%)")
        print(f"  良好像素(1≤ΔE<3): {overall_stats['good_pixels']} ({overall_stats['good_percentage']:.1f}%)")
        print(f"  较差像素(ΔE≥3): {overall_stats['poor_pixels']} ({overall_stats['poor_percentage']:.1f}%)")
        
        return metrics

class MappingInversionCalibrator:
    """映射反演校准器 - 核心算法实现"""
    
    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.color_converter = ColorSpaceConverter()
        self.calibration_matrix = None
        self.pixel_response_functions = {}
        
    def build_pixel_response_model(self, rgb_data: np.ndarray) -> Dict:
        """构建像素响应模型"""
        print("\n=== 构建像素响应模型 ===")
        
        # 假设输入为标准220值，分析每个像素的响应特性
        target_input = np.array(self.config.target_rgb)
        print(f"标准输入值: {target_input}")
        
        pixel_models = {}
        
        for pixel_idx in range(len(rgb_data)):
            actual_output = rgb_data[pixel_idx]
            
            # 计算该像素的响应函数偏差
            response_deviation = actual_output - target_input
            response_ratio = actual_output / target_input
            
            pixel_models[pixel_idx] = {
                'input': target_input.copy(),
                'actual_output': actual_output.copy(),
                'deviation': response_deviation.copy(),
                'response_ratio': response_ratio.copy(),
                'linearity_factor': np.mean(response_ratio)  # 假设线性响应
            }
        
        self.pixel_response_functions = pixel_models
        
        # 统计响应特性
        all_ratios = np.array([model['response_ratio'] for model in pixel_models.values()])
        print(f"像素响应比例统计:")
        print(f"  R通道: {all_ratios[:, 0].mean():.3f} ± {all_ratios[:, 0].std():.3f}")
        print(f"  G通道: {all_ratios[:, 1].mean():.3f} ± {all_ratios[:, 1].std():.3f}")
        print(f"  B通道: {all_ratios[:, 2].mean():.3f} ± {all_ratios[:, 2].std():.3f}")
        
        return pixel_models
    
    def solve_inverse_mapping_per_pixel(self, pixel_idx: int, target_output: np.ndarray) -> np.ndarray:
        """为单个像素求解映射反演 - 核心算法"""
        
        if pixel_idx not in self.pixel_response_functions:
            raise ValueError(f"像素{pixel_idx}的响应模型未建立")
        
        pixel_model = self.pixel_response_functions[pixel_idx]
        
        def objective_function(corrected_input):
            """目标函数：最小化输出与目标的色差"""
            # 模拟像素响应：简化的线性模型
            predicted_output = corrected_input * pixel_model['response_ratio']
            
            # 确保输出在合理范围内
            predicted_output = np.clip(predicted_output, 0, 255)
            
            # 计算ΔE色差
            delta_e = self.color_converter.calculate_delta_e_lab(
                predicted_output.reshape(1, -1),
                target_output.reshape(1, -1)
            )[0]
            
            return delta_e
        
        # 初始猜测：基于简单反比例
        initial_guess = target_output / pixel_model['response_ratio']
        initial_guess = np.clip(initial_guess, 
                               self.config.min_correction_value, 
                               self.config.max_correction_value)
        
        # 边界约束
        bounds = [(self.config.min_correction_value, self.config.max_correction_value) for _ in range(3)]
        
        # 优化求解
        try:
            result = minimize(
                objective_function, 
                initial_guess,
                method=self.config.optimization_method,
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return np.clip(result.x, self.config.min_correction_value, self.config.max_correction_value)
            else:
                print(f"警告: 像素{pixel_idx}优化失败，使用初始猜测")
                return initial_guess
                
        except Exception as e:
            print(f"错误: 像素{pixel_idx}优化异常: {e}")
            return initial_guess
    
    def calibrate_all_pixels(self, rgb_data: np.ndarray) -> np.ndarray:
        """对所有像素进行校准 - 主要算法流程"""
        print("\n=== 开始映射反演校准 ===")
        
        # 步骤1: 构建像素响应模型
        self.build_pixel_response_model(rgb_data)
        
        # 步骤2: 设定目标输出
        target_output = np.array(self.config.target_rgb, dtype=float)
        print(f"目标输出颜色: {target_output}")
        
        # 步骤3: 逐像素求解校正输入
        n_pixels = len(rgb_data)
        calibration_matrix = np.zeros((n_pixels, 3))
        
        print(f"开始逐像素校准，总计{n_pixels}个像素...")
        
        # 分批处理以显示进度
        batch_size = max(1, n_pixels // 20)  # 显示20次进度
        
        for i in range(n_pixels):
            calibration_matrix[i] = self.solve_inverse_mapping_per_pixel(i, target_output)
            
            if i % batch_size == 0 or i == n_pixels - 1:
                progress = (i + 1) / n_pixels * 100
                print(f"校准进度: {progress:.1f}% ({i+1}/{n_pixels})")
        
        self.calibration_matrix = calibration_matrix
        
        # 步骤4: 验证校准效果
        print("\n=== 校准效果验证 ===")
        self.validate_calibration_results(rgb_data, calibration_matrix, target_output)
        
        return calibration_matrix
    
    def validate_calibration_results(self, original_rgb: np.ndarray, 
                                   calibration_matrix: np.ndarray, 
                                   target_output: np.ndarray):
        """验证校准结果"""
        
        # 模拟校准后的输出
        calibrated_outputs = []
        
        for i in range(len(original_rgb)):
            corrected_input = calibration_matrix[i]
            pixel_model = self.pixel_response_functions[i]
            
            # 预测输出
            predicted_output = corrected_input * pixel_model['response_ratio']
            predicted_output = np.clip(predicted_output, 0, 255)
            calibrated_outputs.append(predicted_output)
        
        calibrated_outputs = np.array(calibrated_outputs)
        
        # 计算校准后的色差
        target_array = np.tile(target_output, (len(calibrated_outputs), 1))
        delta_e_after = self.color_converter.calculate_delta_e_lab(calibrated_outputs, target_array)
        delta_e_before = self.color_converter.calculate_delta_e_lab(original_rgb, target_array)
        
        # 统计改善效果
        print(f"校准前平均ΔE: {delta_e_before.mean():.3f}")
        print(f"校准后平均ΔE: {delta_e_after.mean():.3f}")
        print(f"ΔE改善: {((delta_e_before.mean() - delta_e_after.mean()) / delta_e_before.mean() * 100):.1f}%")
        
        # 质量分布
        excellent_before = np.sum(delta_e_before < 1.0)
        excellent_after = np.sum(delta_e_after < 1.0)
        
        print(f"优秀像素(ΔE<1):")
        print(f"  校准前: {excellent_before} ({excellent_before/len(delta_e_before)*100:.1f}%)")
        print(f"  校准后: {excellent_after} ({excellent_after/len(delta_e_after)*100:.1f}%)")
        
        return {
            'original_outputs': original_rgb,
            'calibrated_outputs': calibrated_outputs,
            'delta_e_before': delta_e_before,
            'delta_e_after': delta_e_after,
            'improvement_percentage': (delta_e_before.mean() - delta_e_after.mean()) / delta_e_before.mean() * 100
        }

class CalibrationVisualizer:
    """校准结果可视化器"""
    
    def __init__(self, output_dir: str = "led_calibration_results_v2"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_calibration_results(self, validation_results: Dict, save_path: str = None):
        """绘制校准结果对比图"""
        
        original_outputs = validation_results['original_outputs']
        calibrated_outputs = validation_results['calibrated_outputs']
        delta_e_before = validation_results['delta_e_before']
        delta_e_after = validation_results['delta_e_after']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原始RGB分布
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            if len(original_outputs) == 4096:  # 64x64
                matrix_orig = original_outputs[:, i].reshape(64, 64)
                matrix_calib = calibrated_outputs[:, i].reshape(64, 64)
                
                # 原始分布
                im1 = axes[0, i].imshow(matrix_orig, cmap='hot', aspect='equal')
                axes[0, i].set_title(f'Original {channel}', fontsize=14, fontweight='bold')
                plt.colorbar(im1, ax=axes[0, i])
                
                # 校准后分布
                im2 = axes[1, i].imshow(matrix_calib, cmap='hot', aspect='equal')
                axes[1, i].set_title(f'Calibrated {channel}', fontsize=14, fontweight='bold')
                plt.colorbar(im2, ax=axes[1, i])
                
                # 添加统计信息
                orig_std = matrix_orig.std()
                calib_std = matrix_calib.std()
                improvement = (orig_std - calib_std) / orig_std * 100
                
                axes[1, i].text(0.02, 0.02, f'Std改善: {improvement:.1f}%',
                               transform=axes[1, i].transAxes,
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                               fontsize=10, fontweight='bold')
        
        plt.suptitle('LED校准前后RGB分布对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准结果图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'calibration_results_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准结果图保存至: {save_path}")
        
        plt.close()
    
    def plot_delta_e_comparison(self, validation_results: Dict, save_path: str = None):
        """绘制ΔE对比图"""
        
        delta_e_before = validation_results['delta_e_before']
        delta_e_after = validation_results['delta_e_after']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ΔE直方图对比
        axes[0, 0].hist(delta_e_before, bins=50, alpha=0.7, label='校准前', color='red')
        axes[0, 0].hist(delta_e_after, bins=50, alpha=0.7, label='校准后', color='blue')
        axes[0, 0].set_title('ΔE分布直方图', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('ΔE值')
        axes[0, 0].set_ylabel('像素数量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ΔE改善散点图
        axes[0, 1].scatter(delta_e_before, delta_e_after, alpha=0.5, s=1)
        axes[0, 1].plot([0, max(delta_e_before)], [0, max(delta_e_before)], 'r--', label='y=x')
        axes[0, 1].set_title('ΔE改善散点图', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('校准前ΔE')
        axes[0, 1].set_ylabel('校准后ΔE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 如果是64x64数据，显示空间分布
        if len(delta_e_before) == 4096:
            delta_e_before_matrix = delta_e_before.reshape(64, 64)
            delta_e_after_matrix = delta_e_after.reshape(64, 64)
            
            im1 = axes[1, 0].imshow(delta_e_before_matrix, cmap='hot', aspect='equal')
            axes[1, 0].set_title('校准前ΔE空间分布', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[1, 0])
            
            im2 = axes[1, 1].imshow(delta_e_after_matrix, cmap='hot', aspect='equal')
            axes[1, 1].set_title('校准后ΔE空间分布', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ΔE对比图保存至: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'delta_e_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ΔE对比图保存至: {save_path}")
        
        plt.close()

def main():
    """主函数 - 映射反演校准完整流程"""
    print("=== LED显示屏色彩均匀性校准系统 - 映射反演方法 ===")
    print("问题三：基于前两问色彩空间转换映射的校准解决方案")
    print("核心算法：映射反演 + 逐点校正 + ΔE最小化")
    print("=" * 70)
    
    # 配置参数
    config = CalibrationConfig(
        target_rgb=(220, 220, 220),
        display_size=(64, 64),
        delta_e_threshold=1.0,
        optimization_method='SLSQP'
    )
    
    output_dir = "led_calibration_results_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 数据加载与分析
    print("\n步骤1: 加载并分析LED显示屏数据...")
    analyzer = LEDDataAnalyzer(config)
    
    # 尝试加载真实数据
    data_file = "B题附件：RGB数值.xlsx"
    if os.path.exists(data_file):
        df = analyzer.load_led_data(data_file)
        if df is not None and analyzer.rgb_data is not None:
            rgb_data = analyzer.rgb_data
            print("成功加载真实LED数据")
        else:
            print("真实数据加载失败，使用模拟数据")
            rgb_data = generate_simulated_led_data(config)
    else:
        print("数据文件不存在，使用模拟数据")
        rgb_data = generate_simulated_led_data(config)
        # 为模拟数据创建分析器
        analyzer.rgb_data = rgb_data
    
    # 步骤2: 均匀性分析
    print("\n步骤2: LED显示屏均匀性分析...")
    original_metrics = analyzer.analyze_uniformity()
    
    # 步骤3: 映射反演校准
    print("\n步骤3: 应用映射反演校准算法...")
    calibrator = MappingInversionCalibrator(config)
    calibration_matrix = calibrator.calibrate_all_pixels(rgb_data)
    
    # 步骤4: 验证校准效果
    print("\n步骤4: 验证校准效果...")
    validation_results = calibrator.validate_calibration_results(
        rgb_data, calibration_matrix, np.array(config.target_rgb)
    )
    
    # 步骤5: 可视化结果
    print("\n步骤5: 生成可视化结果...")
    visualizer = CalibrationVisualizer(output_dir)
    
    visualizer.plot_calibration_results(
        validation_results,
        os.path.join(output_dir, 'calibration_results_comparison.png')
    )
    
    visualizer.plot_delta_e_comparison(
        validation_results,
        os.path.join(output_dir, 'delta_e_comparison.png')
    )
    
    # 步骤6: 保存结果
    print("\n步骤6: 保存校准结果...")
    
    # 保存校准矩阵
    np.save(os.path.join(output_dir, 'calibration_matrix.npy'), calibration_matrix)
    np.save(os.path.join(output_dir, 'original_rgb_data.npy'), rgb_data)
    np.save(os.path.join(output_dir, 'calibrated_rgb_outputs.npy'), validation_results['calibrated_outputs'])
    
    # 保存综合报告
    report = {
        'config': {
            'target_rgb': config.target_rgb,
            'display_size': config.display_size,
            'optimization_method': config.optimization_method
        },
        'original_metrics': original_metrics,
        'calibration_results': {
            'average_delta_e_before': float(validation_results['delta_e_before'].mean()),
            'average_delta_e_after': float(validation_results['delta_e_after'].mean()),
            'improvement_percentage': float(validation_results['improvement_percentage']),
            'excellent_pixels_before': int(np.sum(validation_results['delta_e_before'] < 1.0)),
            'excellent_pixels_after': int(np.sum(validation_results['delta_e_after'] < 1.0)),
            'total_pixels': len(validation_results['delta_e_before'])
        }
    }
    
    with open(os.path.join(output_dir, 'calibration_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 最终总结
    print("\n=== 校准完成总结 ===")
    print(f"校准方法: 映射反演 + 逐点校正")
    print(f"目标颜色: RGB{config.target_rgb}")
    print(f"平均ΔE改善: {validation_results['improvement_percentage']:.1f}%")
    
    excellent_before = np.sum(validation_results['delta_e_before'] < 1.0)
    excellent_after = np.sum(validation_results['delta_e_after'] < 1.0)
    total_pixels = len(validation_results['delta_e_before'])
    
    print(f"优秀像素提升: {excellent_before} → {excellent_after} (增加{excellent_after - excellent_before}个)")
    print(f"优秀率提升: {excellent_before/total_pixels*100:.1f}% → {excellent_after/total_pixels*100:.1f}%")
    
    print(f"\n所有结果已保存至: {output_dir}/")
    print("生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")
    
    return report

def generate_simulated_led_data(config: CalibrationConfig) -> np.ndarray:
    """生成模拟LED数据用于测试"""
    print("生成模拟LED显示屏数据...")
    
    width, height = config.display_size
    n_pixels = width * height
    target_rgb = np.array(config.target_rgb)
    
    # 设置随机种子确保可重现
    np.random.seed(42)
    
    # 创建空间变化模式的不均匀性
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    rgb_data = np.zeros((n_pixels, 3))
    
    for channel in range(3):
        # 不同通道有不同的空间不均匀性模式
        if channel == 0:  # Red - 正弦波模式
            base_pattern = target_rgb[channel] * (0.85 + 0.25 * np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y))
        elif channel == 1:  # Green - 线性梯度
            base_pattern = target_rgb[channel] * (0.9 + 0.2 * (X + Y) / 2)
        else:  # Blue - 高斯模式
            base_pattern = target_rgb[channel] * (0.8 + 0.3 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.3))
        
        # 添加随机噪声
        noise = np.random.normal(0, target_rgb[channel] * 0.08, (height, width))
        pattern = base_pattern + noise
        
        # 确保在合理范围内
        pattern = np.clip(pattern, target_rgb[channel] * 0.6, target_rgb[channel] * 1.4)
        
        rgb_data[:, channel] = pattern.flatten()
    
    print(f"模拟数据生成完成，形状: {rgb_data.shape}")
    print(f"模拟不均匀性特征:")
    for i, ch in enumerate(['R', 'G', 'B']):
        values = rgb_data[:, i]
        print(f"  {ch}: {values.mean():.1f} ± {values.std():.1f} (CV: {values.std()/values.mean():.3f})")
    
    return rgb_data

if __name__ == "__main__":
    main()
