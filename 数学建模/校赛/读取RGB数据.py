#!/usr/bin/env python3
"""
读取RGB数值.xlsx文件并分析数据结构
用于理解数学建模题目的具体要求
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def read_rgb_data(file_path):
    """读取RGB数据Excel文件"""
    try:
        # 尝试读取Excel文件
        if os.path.exists(file_path):
            print(f"正在读取文件: {file_path}")
            
            # 读取所有工作表
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            print(f"发现工作表: {sheet_names}")
            
            data_dict = {}
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                data_dict[sheet_name] = df
                print(f"\n工作表 '{sheet_name}' 数据结构:")
                print(f"形状: {df.shape}")
                print(f"列名: {list(df.columns)}")
                print("前5行数据:")
                print(df.head())
                print("-" * 50)
            
            return data_dict
        else:
            print(f"文件不存在: {file_path}")
            print("请确保文件路径正确")
            return None
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def analyze_rgb_structure(data_dict):
    """分析RGB数据结构"""
    if not data_dict:
        print("没有数据可供分析")
        return
    
    print("\n=== RGB数据结构分析 ===")
    
    for sheet_name, df in data_dict.items():
        print(f"\n分析工作表: {sheet_name}")
        
        # 基本统计信息
        print(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 检查是否包含RGB相关列
        rgb_columns = [col for col in df.columns if any(c in col.lower() for c in ['r', 'g', 'b', 'red', 'green', 'blue'])]
        if rgb_columns:
            print(f"RGB相关列: {rgb_columns}")
        
        # 检查数值类型列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"数值类型列: {numeric_columns}")
        
        # 数据范围分析
        if numeric_columns:
            print("数值范围:")
            for col in numeric_columns:
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
        
        # 检查是否有缺失值
        missing_count = df.isnull().sum().sum()
        print(f"缺失值总数: {missing_count}")
        
        # 数据预览
        print("数据样例:")
        print(df.head(3).to_string())

def create_rgb_visualization(data_dict, save_dir="rgb_analysis"):
    """创建RGB数据可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    for sheet_name, df in data_dict.items():
        # 寻找RGB相关列
        rgb_cols = []
        for col in df.columns:
            if any(c in col.lower() for c in ['r', 'g', 'b']):
                rgb_cols.append(col)
        
        if len(rgb_cols) >= 3:
            print(f"为工作表 '{sheet_name}' 创建RGB可视化...")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # RGB分布直方图
            for i, col in enumerate(rgb_cols[:3]):
                ax = axes[0, i] if i < 2 else axes[1, 0]
                ax.hist(df[col].dropna(), bins=50, alpha=0.7, color=['red', 'green', 'blue'][i])
                ax.set_title(f'{col} 分布')
                ax.set_xlabel('数值')
                ax.set_ylabel('频次')
            
            # RGB散点图
            if len(rgb_cols) >= 3:
                ax = axes[1, 1]
                scatter = ax.scatter(df[rgb_cols[0]], df[rgb_cols[1]], 
                                   c=df[rgb_cols[2]], cmap='viridis', alpha=0.6)
                ax.set_xlabel(rgb_cols[0])
                ax.set_ylabel(rgb_cols[1])
                ax.set_title(f'{rgb_cols[0]} vs {rgb_cols[1]}')
                plt.colorbar(scatter, ax=ax, label=rgb_cols[2])
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/rgb_analysis_{sheet_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

def identify_problem_type(data_dict):
    """识别问题类型"""
    print("\n=== 问题类型识别 ===")
    
    for sheet_name, df in data_dict.items():
        print(f"\n工作表 '{sheet_name}' 分析:")
        
        # 检查列数和数据类型
        num_cols = len(df.columns)
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        
        print(f"总列数: {num_cols}, 数值列数: {numeric_cols}")
        
        # 基于列数判断可能的问题类型
        if num_cols == 3:
            print("可能是: 3通道RGB数据")
        elif num_cols == 4:
            print("可能是: 4通道RGBV数据")
        elif num_cols == 5:
            print("可能是: 5通道RGBCX数据")
        elif num_cols > 5:
            print("可能是: 多通道数据或包含额外信息")
        
        # 检查数据范围
        if numeric_cols > 0:
            numeric_df = df.select_dtypes(include=[np.number])
            min_vals = numeric_df.min()
            max_vals = numeric_df.max()
            
            if (min_vals >= 0).all() and (max_vals <= 1).all():
                print("数据范围: [0,1] - 标准化颜色值")
            elif (min_vals >= 0).all() and (max_vals <= 255).all():
                print("数据范围: [0,255] - 8位颜色值")
            else:
                print("数据范围: 非标准颜色值范围")

def main():
    """主函数"""
    print("=== RGB数据文件读取和分析工具 ===")
    
    # 可能的文件路径
    possible_paths = [
        "B题附件：RGB数值.xlsx",
        "RGB数值.xlsx",
        "../RGB数值.xlsx", 
        "../../RGB数值.xlsx",
        "/home/yao/RGB数值.xlsx",
        "/home/yao/Desktop/RGB数值.xlsx"
    ]
    
    # 尝试找到文件
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path:
        print(f"找到文件: {file_path}")
        
        # 读取数据
        data_dict = read_rgb_data(file_path)
        
        if data_dict:
            # 分析数据结构
            analyze_rgb_structure(data_dict)
            
            # 识别问题类型
            identify_problem_type(data_dict)
            
            # 创建可视化
            create_rgb_visualization(data_dict)
            
            print("\n=== 分析完成 ===")
            print("生成的文件:")
            print("1. rgb_analysis/ - RGB数据可视化图表")
            print("2. 控制台输出 - 详细的数据结构分析")
        
    else:
        print("未找到 'RGB数值.xlsx' 文件")
        print("请将文件放在以下任一位置:")
        for path in possible_paths:
            print(f"  - {path}")
        
        print("\n或者手动指定文件路径:")
        custom_path = input("请输入Excel文件的完整路径 (回车跳过): ").strip()
        if custom_path and os.path.exists(custom_path):
            data_dict = read_rgb_data(custom_path)
            if data_dict:
                analyze_rgb_structure(data_dict)
                identify_problem_type(data_dict)
                create_rgb_visualization(data_dict)

if __name__ == "__main__":
    main()
