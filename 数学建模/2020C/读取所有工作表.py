#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

print("开始读取Excel文件的所有工作表...")

try:
    # 读取Excel文件的所有工作表
    excel_file = pd.ExcelFile('附件1：123家有信贷记录企业的相关数据.xlsx')
    
    print(f"✅ Excel文件加载成功！")
    print(f"📊 工作表列表: {excel_file.sheet_names}")
    print(f"📈 工作表数量: {len(excel_file.sheet_names)}")
    
    # 读取每个工作表
    all_sheets = {}
    for sheet_name in excel_file.sheet_names:
        print(f"\n{'='*60}")
        print(f"📋 正在读取工作表: {sheet_name}")
        print(f"{'='*60}")
        
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            all_sheets[sheet_name] = df
            
            print(f"✅ 工作表读取成功！")
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            print("\n数据类型:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
            
            print(f"\n前5行数据:")
            print(df.head())
            
            print(f"\n描述性统计:")
            print(df.describe())
            
            print(f"\n缺失值统计:")
            print(df.isnull().sum())
            
            # 如果数据不多，显示所有唯一值
            for col in df.columns:
                unique_count = df[col].nunique()
                if unique_count <= 20:  # 如果唯一值不超过20个
                    print(f"\n{col}的唯一值 ({unique_count}个):")
                    print(df[col].value_counts())
            
        except Exception as e:
            print(f"❌ 读取工作表 {sheet_name} 失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 总结信息")
    print(f"{'='*60}")
    print(f"成功读取的工作表数量: {len(all_sheets)}")
    for sheet_name, df in all_sheets.items():
        print(f"  {sheet_name}: {df.shape[0]}行 x {df.shape[1]}列")
    
except Exception as e:
    print(f"❌ 读取失败: {e}")
    import traceback
    traceback.print_exc()
