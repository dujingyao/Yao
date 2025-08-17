#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

print("正在加载和处理所有数据...")

class DataProcessor:
    def __init__(self):
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.综合数据 = None
    
    def load_all_data(self):
        """加载所有工作表数据"""
        try:
            excel_file = pd.ExcelFile('附件1：123家有信贷记录企业的相关数据.xlsx')
            
            self.企业信息 = pd.read_excel(excel_file, sheet_name='企业信息')
            self.进项发票 = pd.read_excel(excel_file, sheet_name='进项发票信息')
            self.销项发票 = pd.read_excel(excel_file, sheet_name='销项发票信息')
            
            print("✅ 所有数据加载成功！")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def analyze_invoice_data(self):
        """分析发票数据，计算各种财务指标"""
        print("\n🔍 开始分析发票数据...")
        
        # 处理进项发票数据
        进项汇总 = self.进项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x == '作废发票').sum()
        }).round(2)
        
        # 展平列名
        进项汇总.columns = ['进项总额', '进项发票数量', '进项平均金额', '进项金额标准差', '进项净金额', '进项税额', '进项作废数量']
        进项汇总['进项作废率'] = (进项汇总['进项作废数量'] / 进项汇总['进项发票数量']).round(4)
        
        # 处理销项发票数据
        销项汇总 = self.销项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x.str.strip() == '作废发票').sum()  # 处理可能的空格
        }).round(2)
        
        # 展平列名
        销项汇总.columns = ['销项总额', '销项发票数量', '销项平均金额', '销项金额标准差', '销项净金额', '销项税额', '销项作废数量']
        销项汇总['销项作废率'] = (销项汇总['销项作废数量'] / 销项汇总['销项发票数量']).round(4)
        
        # 合并企业信息
        self.综合数据 = self.企业信息.set_index('企业代号').join([进项汇总, 销项汇总], how='left')
        
        # 计算财务指标
        self._calculate_financial_metrics()
        
        print("✅ 发票数据分析完成！")
        return self.综合数据
    
    def _calculate_financial_metrics(self):
        """计算财务和风险指标"""
        
        # 填充缺失值
        self.综合数据 = self.综合数据.fillna(0)
        
        # 1. 营业状况指标
        self.综合数据['毛利润'] = self.综合数据['销项净金额'] - self.综合数据['进项净金额']
        self.综合数据['毛利率'] = (self.综合数据['毛利润'] / (self.综合数据['销项净金额'] + 1e-6)).round(4)
        
        # 2. 业务活跃度指标
        self.综合数据['总发票数量'] = self.综合数据['进项发票数量'] + self.综合数据['销项发票数量']
        self.综合数据['发票活跃度'] = np.log1p(self.综合数据['总发票数量'])
        
        # 3. 发票质量指标
        self.综合数据['总作废数量'] = self.综合数据['进项作废数量'] + self.综合数据['销项作废数量']
        self.综合数据['整体作废率'] = (self.综合数据['总作废数量'] / (self.综合数据['总发票数量'] + 1e-6)).round(4)
        
        # 4. 现金流指标
        self.综合数据['现金流入'] = self.综合数据['销项总额']
        self.综合数据['现金流出'] = self.综合数据['进项总额']
        self.综合数据['净现金流'] = self.综合数据['现金流入'] - self.综合数据['现金流出']
        
        # 5. 业务稳定性指标
        self.综合数据['收入稳定性'] = 1 / (self.综合数据['销项金额标准差'] / (self.综合数据['销项平均金额'] + 1e-6) + 1)
        self.综合数据['支出稳定性'] = 1 / (self.综合数据['进项金额标准差'] / (self.综合数据['进项平均金额'] + 1e-6) + 1)
        
        # 6. 规模指标
        self.综合数据['业务规模'] = np.log1p(self.综合数据['销项总额'] + self.综合数据['进项总额'])
        
        # 7. 信誉评级数值化
        rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        self.综合数据['信誉评级数值'] = self.综合数据['信誉评级'].map(rating_map)
        
        # 8. 违约标签
        self.综合数据['违约标签'] = (self.综合数据['是否违约'] == '是').astype(int)
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        if self.综合数据 is None:
            print("❌ 请先处理数据！")
            return
        
        print("\n" + "="*80)
        print("📊 中小微企业信贷数据综合分析报告")
        print("="*80)
        
        # 基本统计信息
        print(f"\n📈 基本统计信息:")
        print(f"   企业总数: {len(self.综合数据)}家")
        print(f"   进项发票总数: {self.进项发票.shape[0]:,}张")
        print(f"   销项发票总数: {self.销项发票.shape[0]:,}张")
        
        # 信誉评级分布
        print(f"\n⭐ 信誉评级分布:")
        rating_dist = self.综合数据['信誉评级'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {rating}级: {count:2d}家 ({percentage:5.1f}%)")
        
        # 违约情况分析
        print(f"\n⚠️ 违约情况分析:")
        default_dist = self.综合数据['是否违约'].value_counts()
        for status, count in default_dist.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {status}: {count:2d}家 ({percentage:5.1f}%)")
        
        # 各信誉等级的违约率
        print(f"\n📊 各信誉等级违约率:")
        default_by_rating = self.综合数据.groupby('信誉评级')['违约标签'].agg(['count', 'sum', 'mean'])
        default_by_rating['违约率'] = (default_by_rating['mean'] * 100).round(1)
        for rating, stats in default_by_rating.iterrows():
            print(f"   {rating}级: {stats['sum']:2.0f}/{stats['count']:2.0f} = {stats['违约率']:5.1f}%")
        
        # 财务指标统计
        print(f"\n💰 财务指标统计:")
        financial_metrics = ['销项总额', '进项总额', '毛利润', '毛利率', '净现金流']
        for metric in financial_metrics:
            if metric in self.综合数据.columns:
                stats = self.综合数据[metric].describe()
                print(f"   {metric}:")
                print(f"     平均值: {stats['mean']:>12,.0f}" + ("%" if "率" in metric else "万元"))
                print(f"     中位数: {stats['50%']:>12,.0f}" + ("%" if "率" in metric else "万元"))
                print(f"     标准差: {stats['std']:>12,.0f}" + ("%" if "率" in metric else "万元"))
        
        # 发票质量分析
        print(f"\n📋 发票质量分析:")
        print(f"   平均进项作废率: {self.综合数据['进项作废率'].mean()*100:.2f}%")
        print(f"   平均销项作废率: {self.综合数据['销项作废率'].mean()*100:.2f}%")
        print(f"   整体作废率: {self.综合数据['整体作废率'].mean()*100:.2f}%")
        
        # 业务活跃度分析
        print(f"\n🔥 业务活跃度分析:")
        print(f"   平均月发票数量: {self.综合数据['总发票数量'].mean():.0f}张")
        print(f"   发票数量中位数: {self.综合数据['总发票数量'].median():.0f}张")
        print(f"   最活跃企业发票数: {self.综合数据['总发票数量'].max():,.0f}张")
        
        # 风险预警企业
        print(f"\n🚨 风险预警企业 (D级或高作废率):")
        risk_enterprises = self.综合数据[
            (self.综合数据['信誉评级'] == 'D') | 
            (self.综合数据['整体作废率'] > 0.2)
        ]
        print(f"   风险企业数量: {len(risk_enterprises)}家")
        for idx, (enterprise_id, row) in enumerate(risk_enterprises.head(10).iterrows(), 1):
            print(f"   {idx:2d}. {enterprise_id}: {row['信誉评级']}级, "
                  f"作废率{row['整体作废率']*100:.1f}%, "
                  f"{'违约' if row['违约标签'] else '正常'}")
        
        print("\n" + "="*80)
    
    def export_processed_data(self):
        """导出处理后的数据"""
        if self.综合数据 is not None:
            # 重置索引以便导出企业代号
            export_data = self.综合数据.reset_index()
            export_data.to_excel('企业综合数据分析.xlsx', index=False)
            print(f"✅ 综合数据已导出至: 企业综合数据分析.xlsx")
            
            # 也保存CSV格式
            export_data.to_csv('企业综合数据分析.csv', index=False, encoding='utf-8-sig')
            print(f"✅ 综合数据已导出至: 企业综合数据分析.csv")
            
            return export_data
        return None

# 执行分析
def main():
    processor = DataProcessor()
    
    if processor.load_all_data():
        comprehensive_data = processor.analyze_invoice_data()
        processor.generate_comprehensive_report()
        
        # 导出处理后的数据
        export_data = processor.export_processed_data()
        
        if export_data is not None:
            print(f"\n🎯 处理完成！生成了 {len(export_data)} 家企业的综合分析数据")
            print(f"📊 数据包含 {len(export_data.columns)} 个指标字段")
            
            # 显示一些关键统计
            print(f"\n📋 关键指标概览:")
            key_metrics = ['毛利率', '整体作废率', '发票活跃度', '净现金流', '违约标签']
            for metric in key_metrics:
                if metric in export_data.columns:
                    mean_val = export_data[metric].mean()
                    if metric == '违约标签':
                        print(f"   违约率: {mean_val*100:.1f}%")
                    elif '率' in metric:
                        print(f"   平均{metric}: {mean_val*100:.2f}%")
                    else:
                        print(f"   平均{metric}: {mean_val:.2f}")

if __name__ == "__main__":
    main()
