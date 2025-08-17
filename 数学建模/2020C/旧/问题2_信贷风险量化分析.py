#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题2：对附件2中302家无信贷记录企业的信贷风险进行量化分析
策略：完全基于发票数据构建风险评估模型
目标：在1亿元额度内制定最优信贷方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CreditAnalyzer:
    """无信贷记录企业的信贷风险分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.客户流失率数据 = None
        self.综合数据 = None
        
        # 风险评分维度权重
        self.权重_财务状况 = 0.40  # 重点关注现金流和盈利能力
        self.权重_业务稳定 = 0.35  # 关注经营持续性
        self.权重_发票质量 = 0.25  # 关注交易真实性
        
    def load_data(self):
        """加载数据"""
        try:
            print("📊 加载数据...")
            # 读取企业信息
            self.企业信息 = pd.read_excel('附件2：302家无信贷记录企业的相关数据.xlsx', 
                                     sheet_name='企业信息')
            
            # 读取发票信息
            self.进项发票 = pd.read_excel('附件2：302家无信贷记录企业的相关数据.xlsx', 
                                     sheet_name='进项发票信息')
            self.销项发票 = pd.read_excel('附件2：302家无信贷记录企业的相关数据.xlsx', 
                                     sheet_name='销项发票信息')
            
            # 读取客户流失率数据
            self.客户流失率数据 = pd.read_excel('附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx')
            
            print(f"✅ 数据加载完成:")
            print(f"   - 企业总数: {len(self.企业信息)}家")
            print(f"   - 进项发票: {len(self.进项发票)}条")
            print(f"   - 销项发票: {len(self.销项发票)}条")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def analyze_invoice_features(self):
        """分析发票特征"""
        print("\n🔍 开始分析发票数据...")
        
        # 1. 初始化数据框
        self.综合数据 = self.企业信息.copy()
        
        # 2. 分析销项发票
        销项特征 = self._analyze_sales_invoices()
        self.综合数据 = self.综合数据.merge(销项特征, on='企业代号', how='left')
        
        # 3. 分析进项发票
        进项特征 = self._analyze_purchase_invoices()
        self.综合数据 = self.综合数据.merge(进项特征, on='企业代号', how='left')
        
        # 4. 计算综合指标
        self._calculate_financial_metrics()
        self._calculate_business_stability()
        self._calculate_invoice_quality()
        
        print("✅ 发票分析完成")
        
    def _analyze_sales_invoices(self):
        """分析销项发票"""
        # 创建结果DataFrame
        销项特征 = pd.DataFrame()
        销项特征['企业代号'] = self.企业信息['企业代号'].unique()
        
        # 1. 计算收入指标
        有效发票 = self.销项发票[self.销项发票['发票状态'] == '有效发票']
        收入统计 = 有效发票.groupby('企业代号').agg({
            '金额': ['sum', 'mean', 'std', 'count']
        }).fillna(0)
        收入统计.columns = ['年收入', '平均收入', '收入波动', '收入笔数']
        销项特征 = 销项特征.merge(收入统计, on='企业代号', how='left')
        
        # 2. 计算作废和负数发票比例
        发票状态 = self.销项发票.groupby('企业代号')['发票状态'].value_counts().unstack(fill_value=0)
        总发票数 = 发票状态.sum(axis=1)
        销项特征['销项作废率'] = 发票状态['作废发票'] / 总发票数
        销项特征['销项负数率'] = 发票状态['负数发票'] / 总发票数
        
        # 3. 计算客户集中度
        客户统计 = 有效发票.groupby('企业代号').agg({
            '购方单位代号': lambda x: len(set(x)),  # 不同客户数
            '金额': lambda x: (x / x.sum() ** 2).sum()  # HHI指数
        })
        客户统计.columns = ['客户数量', '客户集中度']
        销项特征 = 销项特征.merge(客户统计, on='企业代号', how='left')
        
        return 销项特征

    def _analyze_purchase_invoices(self):
        """分析进项发票"""
        # 创建结果DataFrame
        进项特征 = pd.DataFrame()
        进项特征['企业代号'] = self.企业信息['企业代号'].unique()
        
        # 1. 计算成本指标
        有效发票 = self.进项发票[self.进项发票['发票状态'] == '有效发票']
        成本统计 = 有效发票.groupby('企业代号').agg({
            '金额': ['sum', 'mean', 'std', 'count']
        }).fillna(0)
        成本统计.columns = ['年采购额', '平均采购', '采购波动', '采购笔数']
        进项特征 = 进项特征.merge(成本统计, on='企业代号', how='left')
        
        # 2. 计算作废和负数发票比例
        发票状态 = self.进项发票.groupby('企业代号')['发票状态'].value_counts().unstack(fill_value=0)
        总发票数 = 发票状态.sum(axis=1)
        进项特征['进项作废率'] = 发票状态['作废发票'] / 总发票数
        进项特征['进项负数率'] = 发票状态['负数发票'] / 总发票数
        
        # 3. 计算供应商集中度
        供应商统计 = 有效发票.groupby('企业代号').agg({
            '销方单位代号': lambda x: len(set(x)),  # 不同供应商数
            '金额': lambda x: (x / x.sum() ** 2).sum()  # HHI指数
        })
        供应商统计.columns = ['供应商数量', '供应商集中度']
        进项特征 = 进项特征.merge(供应商统计, on='企业代号', how='left')
        
        return 进项特征

    def _calculate_financial_metrics(self):
        """计算财务指标"""
        # 1. 计算毛利率
        self.综合数据['毛利率'] = (self.综合数据['年收入'] - self.综合数据['年采购额']) / self.综合数据['年收入']
        self.综合数据['毛利率'] = self.综合数据['毛利率'].fillna(0).clip(-1, 1)
        
        # 2. 计算收入成本比
        self.综合数据['收入成本比'] = self.综合数据['年收入'] / self.综合数据['年采购额']
        self.综合数据['收入成本比'] = self.综合数据['收入成本比'].fillna(0).clip(0, 10)
        
        # 3. 计算现金流稳定性
        self.综合数据['收入稳定性'] = 1 - (self.综合数据['收入波动'] / self.综合数据['平均收入']).fillna(1)
        self.综合数据['采购稳定性'] = 1 - (self.综合数据['采购波动'] / self.综合数据['平均采购']).fillna(1)
        
        # 4. 计算财务状况评分
        self.综合数据['财务状况评分'] = (
            self.综合数据['毛利率'].clip(0, 1) * 0.4 +
            self.综合数据['收入稳定性'].clip(0, 1) * 0.3 +
            self.综合数据['采购稳定性'].clip(0, 1) * 0.3
        ).clip(0, 1)
        
    def _calculate_business_stability(self):
        """计算业务稳定性"""
        # 1. 计算经营规模得分
        self.综合数据['收入规模得分'] = self.综合数据['年收入'] / self.综合数据['年收入'].max()
        self.综合数据['采购规模得分'] = self.综合数据['年采购额'] / self.综合数据['年采购额'].max()
        
        # 2. 计算交易频次得分
        self.综合数据['收入频次得分'] = self.综合数据['收入笔数'] / self.综合数据['收入笔数'].max()
        self.综合数据['采购频次得分'] = self.综合数据['采购笔数'] / self.综合数据['采购笔数'].max()
        
        # 3. 计算客户供应商多样性
        self.综合数据['客户多样性'] = 1 - self.综合数据['客户集中度']
        self.综合数据['供应商多样性'] = 1 - self.综合数据['供应商集中度']
        
        # 4. 计算业务稳定性评分
        self.综合数据['业务稳定性评分'] = (
            self.综合数据['收入规模得分'] * 0.2 +
            self.综合数据['采购规模得分'] * 0.2 +
            self.综合数据['收入频次得分'] * 0.15 +
            self.综合数据['采购频次得分'] * 0.15 +
            self.综合数据['客户多样性'] * 0.15 +
            self.综合数据['供应商多样性'] * 0.15
        ).clip(0, 1)
        
    def _calculate_invoice_quality(self):
        """计算发票质量"""
        # 1. 计算整体作废率
        self.综合数据['整体作废率'] = (
            self.综合数据['销项作废率'] * 0.5 + 
            self.综合数据['进项作废率'] * 0.5
        )
        
        # 2. 计算整体负数发票率
        self.综合数据['整体负数率'] = (
            self.综合数据['销项负数率'] * 0.5 + 
            self.综合数据['进项负数率'] * 0.5
        )
        
        # 3. 计算发票质量评分
        self.综合数据['发票质量评分'] = (
            (1 - self.综合数据['整体作废率']) * 0.6 +
            (1 - self.综合数据['整体负数率']) * 0.4
        ).clip(0, 1)
        
    def build_risk_model(self):
        """构建风险评估模型"""
        print("\n🧮 构建风险评估模型...")
        
        # 1. 计算综合风险评分
        self.综合数据['综合风险评分'] = (
            self.综合数据['财务状况评分'] * self.权重_财务状况 +
            self.综合数据['业务稳定性评分'] * self.权重_业务稳定 +
            self.综合数据['发票质量评分'] * self.权重_发票质量
        )
        
        # 2. 确定风险等级
        risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        risk_labels = ['极高风险', '高风险', '中风险', '较低风险', '低风险']
        
        self.综合数据['风险等级'] = pd.cut(
            self.综合数据['综合风险评分'],
            bins=risk_bins,
            labels=risk_labels,
            include_lowest=True
        )
        
        print("✅ 风险评估模型构建完成")
        
        # 3. 打印风险等级分布
        风险分布 = self.综合数据['风险等级'].value_counts().sort_index()
        print("\n📊 风险评级分布:")
        for level, count in 风险分布.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {level}: {count}家 ({percentage:.1f}%)")
            
    def design_credit_strategy(self, total_budget=10000):
        """设计信贷策略"""
        print(f"\n💰 设计信贷策略 (总预算{total_budget}万元)...")
        
        # 1. 设置基础贷款条件
        def calculate_credit_limit(row):
            """计算贷款额度"""
            risk_score = row['综合风险评分']
            annual_revenue = row['年收入']
            
            if risk_score < 0.2:  # 极高风险，不予贷款
                return 0
                
            # 基础额度 = 年收入 * 风险系数 * 0.3 (最高可贷年收入的30%)
            base_limit = annual_revenue * risk_score * 0.3
            
            # 根据风险等级设置上限
            if risk_score >= 0.8:  # 低风险
                return min(base_limit, 500)  # 最高500万
            elif risk_score >= 0.6:  # 较低风险
                return min(base_limit, 300)  # 最高300万
            elif risk_score >= 0.4:  # 中风险
                return min(base_limit, 200)  # 最高200万
            else:  # 高风险
                return min(base_limit, 100)  # 最高100万
        
        def get_base_rate(risk_score):
            """获取基础利率"""
            if risk_score >= 0.8: return 0.04  # 低风险
            if risk_score >= 0.6: return 0.06  # 较低风险
            if risk_score >= 0.4: return 0.08  # 中风险
            if risk_score >= 0.2: return 0.10  # 高风险
            return 0.12  # 极高风险
        
        # 2. 计算初始贷款方案
        self.综合数据['推荐额度'] = self.综合数据.apply(calculate_credit_limit, axis=1)
        self.综合数据['基础利率'] = self.综合数据['综合风险评分'].apply(get_base_rate)
        
        # 3. 优化利率设置
        self._optimize_interest_rates()
        
        # 4. 优化额度分配
        self._optimize_budget_allocation(total_budget)
        
        print("✅ 信贷策略设计完成")
        
    def _optimize_interest_rates(self):
        """优化贷款利率（考虑客户流失率）"""
        print("\n🎯 优化贷款利率...")
        
        def get_optimal_rate(row):
            """获取最优利率"""
            if row['推荐额度'] == 0:
                return 0
                
            risk_score = row['综合风险评分']
            base_rate = row['基础利率']
            
            # 构建利率-流失率对应关系
            rates = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.15])
            churn_rates = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
            
            # 在基准利率上下0.02范围内寻找最优利率
            best_rate = base_rate
            best_income = 0
            
            for rate in np.arange(max(0.04, base_rate-0.02), 
                                min(0.15, base_rate+0.02), 0.001):
                # 计算该利率下的流失率
                churn_rate = np.interp(rate, rates, churn_rates)
                # 计算预期收益
                income = rate * (1 - churn_rate) * (1 - risk_score)
                
                if income > best_income:
                    best_income = income
                    best_rate = rate
                    
            return best_rate
            
        self.综合数据['推荐利率'] = self.综合数据.apply(get_optimal_rate, axis=1)
        
    def _optimize_budget_allocation(self, total_budget):
        """优化预算分配"""
        # 1. 计算预期收益
        self.综合数据['预期收益率'] = (
            self.综合数据['推荐利率'] * 
            (1 - self.综合数据['综合风险评分'])
        )
        
        # 2. 排序并分配额度
        候选企业 = self.综合数据[self.综合数据['推荐额度'] > 0].copy()
        候选企业 = 候选企业.sort_values('预期收益率', ascending=False)
        
        累计预算 = 0
        最终额度 = []
        
        for idx, row in 候选企业.iterrows():
            if 累计预算 + row['推荐额度'] <= total_budget:
                最终额度.append(row['推荐额度'])
                累计预算 += row['推荐额度']
            elif total_budget - 累计预算 >= 10:  # 确保最小额度
                最终额度.append(total_budget - 累计预算)
                累计预算 = total_budget
            else:
                最终额度.append(0)
                
        候选企业['最终额度'] = 最终额度
        
        # 3. 更新主数据表
        self.综合数据['最终额度'] = 0
        self.综合数据.loc[候选企业.index, '最终额度'] = 候选企业['最终额度']
        
        # 4. 标记贷款决策
        self.综合数据['贷款决策'] = self.综合数据['最终额度'].apply(
            lambda x: '批准' if x > 0 else '拒绝'
        )
        
    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n" + "="*80)
        print("📊 问题2：302家无信贷记录企业信贷风险量化分析报告")
        print("="*80)
        
        # 1. 风险评估结果
        print("\n📈 风险评估结果:")
        risk_dist = self.综合数据['风险等级'].value_counts().sort_index()
        for level, count in risk_dist.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {level}: {count}家 ({percentage:.1f}%)")
            
        # 2. 信贷决策结果
        批准企业 = self.综合数据[self.综合数据['最终额度'] > 0]
        print(f"\n💡 信贷决策结果:")
        print(f"   总申请企业: {len(self.综合数据)}家")
        print(f"   批准贷款: {len(批准企业)}家")
        print(f"   批准率: {len(批准企业)/len(self.综合数据)*100:.1f}%")
        print(f"   总放贷金额: {批准企业['最终额度'].sum():.0f}万元")
        print(f"   平均额度: {批准企业['最终额度'].mean():.0f}万元")
        print(f"   平均利率: {批准企业['推荐利率'].mean()*100:.2f}%")
        
        # 3. 风险等级分布统计
        风险分布 = pd.cut(批准企业['综合风险评分'], 
                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                     labels=['极高风险', '高风险', '中风险', '较低风险', '低风险'])
        
        风险统计 = pd.DataFrame({
            '企业数量': 风险分布.value_counts().sort_index(),
            '放贷金额': 批准企业.groupby(风险分布)['最终额度'].sum(),
            '平均利率': 批准企业.groupby(风险分布)['推荐利率'].mean()
        })
        
        print(f"\n📊 批准企业风险分布:")
        print(风险统计)
        
        # 4. 绘制风险分布图
        plt.figure(figsize=(10, 6))
        sns.barplot(x=风险统计.index, y='企业数量', data=风险统计)
        plt.title('批准贷款企业风险分布')
        plt.xlabel('风险等级')
        plt.ylabel('企业数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('问题2_风险分布.png')
        plt.close()
        
    def export_results(self):
        """导出分析结果"""
        # 1. 选择要导出的字段
        columns = [
            '企业代号', '风险等级', '综合风险评分',
            '财务状况评分', '业务稳定性评分', '发票质量评分',
            '年收入', '毛利率', '收入成本比',
            '客户数量', '供应商数量', '整体作废率',
            '推荐额度', '最终额度', '推荐利率', '贷款决策'
        ]
        
        # 2. 导出到Excel
        结果数据 = self.综合数据[columns].round(4)
        结果数据.to_excel('问题2_信贷风险分析结果.xlsx', index=False)
        print(f"\n✅ 分析结果已导出至: 问题2_信贷风险分析结果.xlsx")

def main():
    """主函数"""
    # 1. 创建分析器实例
    analyzer = CreditAnalyzer()
    
    # 2. 加载数据
    if not analyzer.load_data():
        return
    
    # 3. 特征分析
    analyzer.analyze_invoice_features()
    
    # 4. 风险建模
    analyzer.build_risk_model()
    
    # 5. 制定策略
    analyzer.design_credit_strategy(total_budget=10000)  # 1亿元预算
    
    # 6. 生成报告
    analyzer.generate_analysis_report()
    
    # 7. 导出结果
    analyzer.export_results()
    
    print("\n🎉 问题2分析完成!")
    print("📊 已完成对302家无信贷记录企业的风险量化分析")
    print("💰 已在1亿元额度内制定最优信贷策略")

if __name__ == "__main__":
    main()
