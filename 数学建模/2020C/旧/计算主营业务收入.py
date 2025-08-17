#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算企业年度主营业务收入
方法：统计企业作为购方时的销项发票金额总和
"""

import pandas as pd
import numpy as np

def calculate_annual_revenue():
    """计算企业年度主营业务收入"""
    print("📊 开始计算企业年度主营业务收入...")
    
    # 1. 读取销项发票数据
    try:
        sales_data = pd.read_excel('附件1：123家有信贷记录企业的相关数据.xlsx', sheet_name='销项发票信息')
        print(f"✅ 成功读取销项发票数据，共 {len(sales_data)} 条记录")
    except Exception as e:
        print(f"❌ 读取数据失败: {str(e)}")
        return None
        
    # 2. 数据预处理
    # 只保留有效发票
    valid_sales = sales_data[sales_data['发票状态'] == '有效发票']
    
    # 3. 按购方单位代号分组计算金额总和
    annual_revenue = valid_sales.groupby('购方单位代号')['金额'].agg([
        ('年收入', 'sum'),          # 年度总收入
        ('平均单笔收入', 'mean'),   # 平均单笔收入
        ('收入笔数', 'count'),      # 交易笔数
        ('最大收入', 'max'),        # 最大单笔收入
        ('最小收入', 'min')         # 最小单笔收入
    ]).round(2)
    
    # 4. 添加收入统计信息
    annual_revenue['收入标准差'] = valid_sales.groupby('购方单位代号')['金额'].std().round(2)
    annual_revenue['变异系数'] = (annual_revenue['收入标准差'] / annual_revenue['平均单笔收入']).round(4)
    
    # 5. 保存结果
    annual_revenue.to_excel('企业年度主营业务收入统计.xlsx')
    
    # 6. 输出统计信息
    print("\n📈 统计结果:")
    print(f"   - 企业总数: {len(annual_revenue)}家")
    print(f"   - 总收入: {annual_revenue['年收入'].sum()/10000:.2f}亿元")
    print(f"   - 平均年收入: {annual_revenue['年收入'].mean()/10000:.2f}亿元")
    print(f"   - 最高年收入: {annual_revenue['年收入'].max()/10000:.2f}亿元")
    
    # 7. 计算收入分布
    revenue_bins = [0, 1000, 5000, 10000, 50000, float('inf')]
    revenue_labels = ['0-1000万', '1000-5000万', '5000-1亿', '1-5亿', '5亿以上']
    annual_revenue['收入等级'] = pd.cut(annual_revenue['年收入'], 
                                    bins=revenue_bins,
                                    labels=revenue_labels)
    
    distribution = annual_revenue['收入等级'].value_counts().sort_index()
    
    print("\n📊 收入分布:")
    for level, count in distribution.items():
        percentage = count / len(annual_revenue) * 100
        print(f"   {level}: {count}家 ({percentage:.1f}%)")
        
    return annual_revenue

def main():
    """主函数"""
    # 1. 计算年度主营业务收入
    revenue_data = calculate_annual_revenue()
    
    if revenue_data is not None:
        print("\n✅ 数据已保存至: 企业年度主营业务收入统计.xlsx")
        print("   可用于后续信贷风险分析")

if __name__ == "__main__":
    main()