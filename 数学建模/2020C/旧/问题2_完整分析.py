#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题2：对附件2中302家无信贷记录企业的信贷风险进行量化分析
方法：基于发票数据的多维度风险评估
目标：在1亿元额度内制定最优信贷策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnterpriseAnalyzer:
    """企业数据分析器"""
    
    def __init__(self):
        self.data_file = '附件2：302家无信贷记录企业的相关数据.xlsx'
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.综合数据 = None
        
        # AHP权重
        self.权重_财务 = 0.419  # 财务状况权重
        self.权重_业务 = 0.263  # 业务稳定性权重
        self.权重_发票 = 0.160  # 发票质量权重
        self.权重_规模 = 0.097  # 经营规模权重
        self.权重_增长 = 0.061  # 增长潜力权重
    
    def load_data(self):
        """加载数据"""
        try:
            print(f"\n📂 正在读取数据文件: {self.data_file}")
            self.企业信息 = pd.read_excel(self.data_file, sheet_name='企业信息')
            self.进项发票 = pd.read_excel(self.data_file, sheet_name='进项发票信息')
            self.销项发票 = pd.read_excel(self.data_file, sheet_name='销项发票信息')
            
            print("\n✅ 数据加载完成:")
            print(f"   - 企业总数: {len(self.企业信息)}家")
            print(f"   - 进项发票: {len(self.进项发票)}条")
            print(f"   - 销项发票: {len(self.销项发票)}条")
            return True
            
        except Exception as e:
            print(f"\n❌ 数据加载失败:")
            print(f"   - 错误信息: {str(e)}")
            return False

    def preprocess_data(self):
        """数据预处理"""
        print("\n🔄 开始数据预处理...")
        
        # 计算企业收入指标
        print("   - 计算企业收入指标...")
        income_stats = self._calculate_income_metrics()
        
        # 计算企业成本指标
        print("   - 计算企业成本指标...")
        cost_stats = self._calculate_cost_metrics()
        
        # 合并所有指标
        print("   - 合并企业数据...")
        self.综合数据 = self.企业信息.copy()
        self.综合数据 = self.综合数据.merge(income_stats, left_on='企业代号', right_index=True, how='left')
        self.综合数据 = self.综合数据.merge(cost_stats, left_on='企业代号', right_index=True, how='left')
        
        # 填充缺失值
        numeric_cols = ['年收入', '平均收入', '收入波动', '收入笔数', '客户数量',
                       '年采购额', '平均采购', '采购波动', '采购笔数', '供应商数量']
        for col in numeric_cols:
            if col in self.综合数据.columns:
                self.综合数据[col] = self.综合数据[col].fillna(0)
        
        # 计算财务指标
        print("   - 计算财务指标...")
        self._calculate_financial_metrics()
        
        print("✅ 数据预处理完成")
        
    def _calculate_income_metrics(self):
        """计算收入相关指标"""
        valid_sales = self.销项发票[self.销项发票['发票状态'] == '有效发票']
        
        income_stats = valid_sales.groupby('企业代号').agg({
            '金额': ['sum', 'mean', 'std', 'count'],
            '购方单位代号': 'nunique'
        }).round(2)
        
        income_stats.columns = ['年收入', '平均收入', '收入波动', '收入笔数', '客户数量']
        
        # 计算客户集中度
        concentration = self._calculate_concentration(valid_sales, '企业代号', '购方单位代号')
        income_stats['客户集中度'] = concentration
        
        return income_stats
        
    def _calculate_cost_metrics(self):
        """计算成本相关指标"""
        valid_purchases = self.进项发票[self.进项发票['发票状态'] == '有效发票']
        
        cost_stats = valid_purchases.groupby('企业代号').agg({
            '金额': ['sum', 'mean', 'std', 'count'],
            '销方单位代号': 'nunique'
        }).round(2)
        
        cost_stats.columns = ['年采购额', '平均采购', '采购波动', '采购笔数', '供应商数量']
        
        # 计算供应商集中度
        concentration = self._calculate_concentration(valid_purchases, '企业代号', '销方单位代号')
        cost_stats['供应商集中度'] = concentration
        
        return cost_stats

    def _calculate_concentration(self, df, entity_col, partner_col):
        """计算集中度（HHI指数）"""
        partner_shares = df.groupby([entity_col, partner_col])['金额'].sum()
        partner_shares = partner_shares.groupby(level=0).apply(
            lambda x: np.sum((x / x.sum()) ** 2) if x.sum() > 0 else 0)
        return partner_shares

    def _calculate_financial_metrics(self):
        """计算财务指标"""
        # 计算毛利率
        self.综合数据['毛利率'] = np.where(
            self.综合数据['年收入'] > 0,
            (self.综合数据['年收入'] - self.综合数据['年采购额']) / self.综合数据['年收入'],
            0
        )
        self.综合数据['毛利率'] = self.综合数据['毛利率'].clip(-1, 1)
        
        # 计算资产周转率
        self.综合数据['收入成本比'] = np.where(
            self.综合数据['年采购额'] > 0,
            self.综合数据['年收入'] / self.综合数据['年采购额'],
            0
        )
        self.综合数据['收入成本比'] = self.综合数据['收入成本比'].clip(0, 10)
        
        # 计算经营稳定性
        self.综合数据['收入稳定性'] = np.where(
            self.综合数据['平均收入'] > 0,
            1 - (self.综合数据['收入波动'] / self.综合数据['平均收入']),
            0
        )
        self.综合数据['采购稳定性'] = np.where(
            self.综合数据['平均采购'] > 0,
            1 - (self.综合数据['采购波动'] / self.综合数据['平均采购']),
            0
        )
        
        self.综合数据['客户稳定性'] = 1 - self.综合数据['客户集中度'].fillna(0)
        self.综合数据['供应商稳定性'] = 1 - self.综合数据['供应商集中度'].fillna(0)
        
    def calculate_risk_scores(self):
        """计算风险评分"""
        print("\n📊 计算风险评分...")
        
        # 计算各维度风险评分
        financial_score = self._calculate_financial_risk()
        business_score = self._calculate_business_risk()
        invoice_score = self._calculate_invoice_risk()
        scale_score = self._calculate_scale_risk()
        growth_score = self._calculate_growth_risk()
        
        # 计算综合风险评分
        self.综合数据['综合风险评分'] = (
            financial_score * self.权重_财务 +
            business_score * self.权重_业务 +
            invoice_score * self.权重_发票 +
            scale_score * self.权重_规模 +
            growth_score * self.权重_增长
        )
        
        # 确定风险等级
        risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        risk_labels = ['极高风险', '高风险', '中风险', '较低风险', '低风险']
        
        self.综合数据['风险等级'] = pd.cut(
            self.综合数据['综合风险评分'],
            bins=risk_bins,
            labels=risk_labels,
            include_lowest=True
        )
        
        print("✅ 风险评分计算完成")
        
    def _calculate_financial_risk(self):
        """计算财务风险评分"""
        # 标准化财务指标
        normalized_profit = self._normalize_series(self.综合数据['毛利率'])
        normalized_ratio = self._normalize_series(self.综合数据['收入成本比'])
        normalized_stability = self._normalize_series(
            (self.综合数据['收入稳定性'] + self.综合数据['采购稳定性']) / 2
        )
        
        # 计算财务风险评分
        financial_score = (
            normalized_profit * 0.4 +
            normalized_ratio * 0.3 +
            normalized_stability * 0.3
        )
        
        return financial_score
    
    def _calculate_business_risk(self):
        """计算业务风险评分"""
        normalized_customer = self._normalize_series(self.综合数据['客户稳定性'])
        normalized_supplier = self._normalize_series(self.综合数据['供应商稳定性'])
        normalized_transactions = self._normalize_series(
            (self.综合数据['收入笔数'] + self.综合数据['采购笔数']) / 2
        )
        
        business_score = (
            normalized_customer * 0.4 +
            normalized_supplier * 0.3 +
            normalized_transactions * 0.3
        )
        
        return business_score
    
    def _calculate_invoice_risk(self):
        """计算发票风险评分"""
        # 计算作废率
        sales_invalid = self.销项发票[self.销项发票['发票状态'] == '作废发票']
        total_sales = self.销项发票.groupby('企业代号').size()
        invalid_sales = sales_invalid.groupby('企业代号').size()
        invalid_rate = (invalid_sales / total_sales).fillna(0)
        
        # 计算负数发票率
        negative_sales = self.销项发票[self.销项发票['金额'] < 0]
        negative_rate = (negative_sales.groupby('企业代号').size() / total_sales).fillna(0)
        
        # 将结果映射到综合数据
        self.综合数据['作废率'] = self.综合数据['企业代号'].map(invalid_rate).fillna(0)
        self.综合数据['负数发票率'] = self.综合数据['企业代号'].map(negative_rate).fillna(0)
        
        # 计算发票风险评分
        invoice_score = 1 - (
            self._normalize_series(self.综合数据['作废率']) * 0.6 +
            self._normalize_series(self.综合数据['负数发票率']) * 0.4
        )
        
        return invoice_score
    
    def _calculate_scale_risk(self):
        """计算规模风险评分"""
        normalized_income = self._normalize_series(self.综合数据['年收入'])
        normalized_customers = self._normalize_series(self.综合数据['客户数量'])
        normalized_suppliers = self._normalize_series(self.综合数据['供应商数量'])
        
        scale_score = (
            normalized_income * 0.4 +
            normalized_customers * 0.3 +
            normalized_suppliers * 0.3
        )
        
        return scale_score
    
    def _calculate_growth_risk(self):
        """计算增长风险评分"""
        normalized_profit = self._normalize_series(self.综合数据['毛利率'])
        normalized_transactions = self._normalize_series(self.综合数据['收入笔数'])
        
        growth_score = (
            normalized_profit * 0.6 +
            normalized_transactions * 0.4
        )
        
        return growth_score
        
    def _normalize_series(self, series):
        """对数据序列进行归一化"""
        series_clean = series.fillna(0)
        if series_clean.max() == series_clean.min():
            return pd.Series(0, index=series.index)
        return (series_clean - series_clean.min()) / (series_clean.max() - series_clean.min())
        
    def design_credit_strategy(self, total_budget=10000):
        """设计信贷策略"""
        print(f"\n💰 设计信贷策略 (总预算{total_budget}万元)...")
        
        # 计算基础额度
        self._calculate_base_credit_limit()
        
        # 优化利率设置
        self._optimize_interest_rates()
        
        # 优化额度分配
        self._optimize_credit_allocation(total_budget)
        
        print("✅ 信贷策略设计完成")
        
    def _calculate_base_credit_limit(self):
        """计算基础贷款额度"""
        # 基于年收入和风险评分计算基础额度
        self.综合数据['基础额度'] = self.综合数据.apply(
            lambda x: min(100, max(10, x['年收入'] * 0.0003 * x['综合风险评分']))
            if x['综合风险评分'] >= 0.2 else 0, axis=1
        )
        
        # 根据风险等级调整上限
        risk_limits = {
            '低风险': 500,
            '较低风险': 300,
            '中风险': 200,
            '高风险': 100,
            '极高风险': 0
        }
        
        self.综合数据['额度上限'] = self.综合数据['风险等级'].astype(str).map(risk_limits).fillna(0)
        self.综合数据['推荐额度'] = self.综合数据[['基础额度', '额度上限']].min(axis=1)
        
    def _optimize_interest_rates(self):
        """优化贷款利率"""
        # 基于风险评分设置基础利率
        self.综合数据['基础利率'] = 0.04 + (1 - self.综合数据['综合风险评分']) * 0.11
        
        # 考虑风险等级进行调整
        risk_adjustments = {
            '低风险': 0,
            '较低风险': 0.01,
            '中风险': 0.02,
            '高风险': 0.03,
            '极高风险': 0.04
        }
        
        self.综合数据['利率调整'] = self.综合数据['风险等级'].astype(str).map(risk_adjustments).fillna(0.04)
        self.综合数据['推荐利率'] = (self.综合数据['基础利率'] + 
                                 self.综合数据['利率调整']).clip(0.04, 0.15)
        
    def _optimize_credit_allocation(self, total_budget):
        """优化信贷额度分配"""
        # 计算预期收益率
        self.综合数据['预期收益率'] = (
            self.综合数据['推荐利率'] * 
            self.综合数据['综合风险评分']
        )
        
        # 按预期收益率排序
        candidates = self.综合数据[self.综合数据['推荐额度'] > 0].copy()
        candidates = candidates.sort_values('预期收益率', ascending=False)
        
        # 分配额度
        allocated_budget = 0
        final_amounts = []
        
        for idx, row in candidates.iterrows():
            if allocated_budget + row['推荐额度'] <= total_budget:
                final_amounts.append(row['推荐额度'])
                allocated_budget += row['推荐额度']
            elif total_budget - allocated_budget >= 10:
                final_amounts.append(total_budget - allocated_budget)
                allocated_budget = total_budget
            else:
                final_amounts.append(0)
        
        candidates['最终额度'] = final_amounts
        
        # 更新主数据表
        self.综合数据['最终额度'] = 0
        self.综合数据.loc[candidates.index, '最终额度'] = candidates['最终额度']
        
        # 标记贷款决策
        self.综合数据['贷款决策'] = self.综合数据['最终额度'].apply(
            lambda x: '批准' if x > 0 else '拒绝'
        )
        
    def generate_report(self):
        """生成分析报告"""
        print("\n📊 生成分析报告...")
        
        # 风险评级分布
        risk_dist = self.综合数据['风险等级'].value_counts().sort_index()
        print("\n风险评级分布:")
        for level, count in risk_dist.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {level}: {count}家 ({percentage:.1f}%)")
        
        # 信贷决策结果
        approved = self.综合数据[self.综合数据['最终额度'] > 0]
        print(f"\n信贷决策结果:")
        print(f"   总申请企业: {len(self.综合数据)}家")
        print(f"   批准贷款: {len(approved)}家")
        print(f"   批准率: {len(approved)/len(self.综合数据)*100:.1f}%")
        if len(approved) > 0:
            print(f"   总放贷金额: {approved['最终额度'].sum():.0f}万元")
            print(f"   平均额度: {approved['最终额度'].mean():.0f}万元")
            print(f"   平均利率: {approved['推荐利率'].mean()*100:.2f}%")
        
        # 导出结果
        output_cols = [
            '企业代号', '风险等级', '综合风险评分',
            '年收入', '毛利率', '收入成本比',
            '客户数量', '供应商数量',
            '推荐额度', '最终额度', '推荐利率', '贷款决策'
        ]
        
        # 确保所有列都存在
        for col in output_cols:
            if col not in self.综合数据.columns:
                self.综合数据[col] = 0
        
        result_file = 'problem2_credit_analysis_results.xlsx'
        self.综合数据[output_cols].to_excel(result_file, index=False)
        print(f"\n✅ 分析结果已导出至: {result_file}")
        
        # 绘制可视化图表
        self._plot_charts()
        
    def _plot_charts(self):
        """绘制图表"""
        try:
            # 风险分布图
            plt.figure(figsize=(10, 6))
            risk_counts = self.综合数据['风险等级'].value_counts().sort_index()
            
            plt.bar(range(len(risk_counts)), risk_counts.values)
            plt.title('Enterprise Risk Level Distribution')
            plt.xlabel('Risk Level')
            plt.ylabel('Number of Enterprises')
            plt.xticks(range(len(risk_counts)), risk_counts.index, rotation=45)
            
            for i, v in enumerate(risk_counts.values):
                plt.text(i, v, str(v), ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   - 风险分布图已保存: risk_distribution.png")
            
            # 信贷分配图
            approved = self.综合数据[self.综合数据['最终额度'] > 0]
            if len(approved) > 0:
                plt.figure(figsize=(12, 6))
                scatter = plt.scatter(approved['综合风险评分'], approved['最终额度'],
                           alpha=0.6, c=approved['推荐利率'], cmap='viridis')
                
                plt.colorbar(scatter, label='Recommended Interest Rate')
                plt.title('Credit Allocation Strategy')
                plt.xlabel('Risk Score')
                plt.ylabel('Credit Amount (10k CNY)')
                
                plt.tight_layout()
                plt.savefig('credit_allocation.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("   - 信贷分配图已保存: credit_allocation.png")
                
        except Exception as e:
            print(f"   - 图表生成失败: {str(e)}")

def main():
    """主函数"""
    print("="*60)
    print("问题2：302家无信贷记录企业信贷风险量化分析")
    print("="*60)
    
    # 创建分析器实例
    analyzer = EnterpriseAnalyzer()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 数据预处理
    analyzer.preprocess_data()
    
    # 计算风险评分
    analyzer.calculate_risk_scores()
    
    # 制定信贷策略
    analyzer.design_credit_strategy(total_budget=10000)  # 1亿元预算
    
    # 生成报告
    analyzer.generate_report()
    
    print("\n🎉 问题2分析完成!")

if __name__ == "__main__":
    main()
