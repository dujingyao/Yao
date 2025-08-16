#!/usr/bin/env python3
"""
简化版RGB数据读取脚本
"""

import pandas as pd
import numpy as np
import sys
import traceback

def main():
    file_path = "B题附件：RGB数值.xlsx"
    
    try:
        print(f"正在读取文件: {file_path}")
        
        # 读取Excel文件
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"发现工作表: {sheet_names}")
        
        for sheet_name in sheet_names:
            print(f"\n=== 工作表: {sheet_name} ===")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            # 显示前几行数据
            print("前5行数据:")
            print(df.head())
            
            # 基本统计信息
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                print("\n数值列统计:")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    print(f"{col}: 范围[{df[col].min():.3f}, {df[col].max():.3f}], 均值{df[col].mean():.3f}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
