#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

print("开始读取Excel文件...")

try:
    # 读取Excel文件
    df = pd.read_excel('附件1：123家有信贷记录企业的相关数据.xlsx')
    print(f"✅ 数据加载成功！")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    print("\n数据类型:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print("\n前5行数据:")
    print(df.head())
    
    print("\n描述性统计:")
    print(df.describe())
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"❌ 读取失败: {e}")
    import traceback
    traceback.print_exc()
