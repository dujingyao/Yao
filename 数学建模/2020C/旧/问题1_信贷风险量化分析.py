#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1：对附件1中123家企业的信贷风险进行量化分析
给出该银行在年度信贷总额固定时对这些企业的信贷策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CreditRiskAnalyzer:
    """信贷风险量化分析器"""
    
    def __init__(self):
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.客户流失率数据 = None
        self.综合数据 = None
        self.风险评分 = None
        
    def load_data(self):
        """加载附件1和附件3数据"""
        try:
            print("📊 正在加载附件1和附件3数据...")
            
            # 读取附件1各工作表
            self.企业信息 = pd.read_excel('附件1：123家有信贷记录企业的相关数据.xlsx', sheet_name='企业信息')
            self.进项发票 = pd.read_excel('附件1：123家有信贷记录企业的相关数据.xlsx', sheet_name='进项发票信息')
            self.销项发票 = pd.read_excel('附件1：123家有信贷记录企业的相关数据.xlsx', sheet_name='销项发票信息')
            
            # 读取附件3客户流失率数据
            self.客户流失率数据 = pd.read_excel('附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx')
            
            print(f"✅ 数据加载成功:")
            print(f"   - 企业信息: {len(self.企业信息)}家企业")
            print(f"   - 进项发票: {len(self.进项发票)}条记录")
            print(f"   - 销项发票: {len(self.销项发票)}条记录")
            print(f"   - 客户流失率数据: {len(self.客户流失率数据)}条记录")
            
            # 显示客户流失率数据结构
            print(f"\n📈 客户流失率数据预览:")
            print(self.客户流失率数据.head())
            
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def analyze_invoice_features(self):
        """分析发票特征，构建财务指标"""
        print("\n🔍 分析发票数据，构建财务特征...")
        
        # 进项发票汇总
        进项汇总 = self.进项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x == '作废发票').sum()
        })
        进项汇总.columns = ['进项总额', '进项发票数', '进项平均额', '进项标准差', '进项净额', '进项税额', '进项作废数']
        进项汇总['进项作废率'] = 进项汇总['进项作废数'] / 进项汇总['进项发票数']
        
        # 销项发票汇总
        销项汇总 = self.销项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x.str.strip() == '作废发票').sum()
        })
        销项汇总.columns = ['销项总额', '销项发票数', '销项平均额', '销项标准差', '销项净额', '销项税额', '销项作废数']
        销项汇总['销项作废率'] = 销项汇总['销项作废数'] / 销项汇总['销项发票数']
        
        # 合并企业信息和发票数据
        self.综合数据 = self.企业信息.set_index('企业代号').join([进项汇总, 销项汇总], how='left')
        self.综合数据 = self.综合数据.fillna(0)
        
        # 计算关键财务指标
        self._calculate_financial_indicators()
        
        print("✅ 发票特征分析完成")
        
    def _calculate_financial_indicators(self):
        """计算财务和风险指标"""
        
        # 1. 盈利能力指标
        self.综合数据['毛利润'] = self.综合数据['销项净额'] - self.综合数据['进项净额']
        self.综合数据['毛利率'] = self.综合数据['毛利润'] / (self.综合数据['销项净额'] + 1e-6)
        
        # 2. 业务规模指标
        self.综合数据['营业收入'] = self.综合数据['销项净额']
        self.综合数据['业务规模'] = np.log1p(self.综合数据['销项总额'] + self.综合数据['进项总额'])
        
        # 3. 业务活跃度指标
        self.综合数据['总发票数'] = self.综合数据['进项发票数'] + self.综合数据['销项发票数']
        self.综合数据['发票活跃度'] = np.log1p(self.综合数据['总发票数'])
        
        # 4. 发票质量指标
        self.综合数据['总作废数'] = self.综合数据['进项作废数'] + self.综合数据['销项作废数']
        self.综合数据['整体作废率'] = self.综合数据['总作废数'] / (self.综合数据['总发票数'] + 1e-6)
        
        # 5. 现金流指标
        self.综合数据['净现金流'] = self.综合数据['销项总额'] - self.综合数据['进项总额']
        self.综合数据['现金流比率'] = self.综合数据['净现金流'] / (self.综合数据['销项总额'] + 1e-6)
        
        # 6. 业务稳定性指标
        self.综合数据['收入稳定性'] = 1 / (self.综合数据['销项标准差'] / (self.综合数据['销项平均额'] + 1e-6) + 1)
        self.综合数据['支出稳定性'] = 1 / (self.综合数据['进项标准差'] / (self.综合数据['进项平均额'] + 1e-6) + 1)
        
        # 7. 信誉评级数值化
        rating_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        self.综合数据['信誉评级数值'] = self.综合数据['信誉评级'].map(rating_map)
        
        # 8. 违约标签
        self.综合数据['违约标签'] = (self.综合数据['是否违约'] == '是').astype(int)
    
    def build_risk_model(self):
        """构建多维度信贷风险评分模型"""
        print("\n🧮 构建多维度信贷风险评分模型...")
        
        # 定义风险评分维度和权重
        risk_weights = {
            '信誉评级权重': 0.35,
            '历史违约权重': 0.25,
            '财务状况权重': 0.20,
            '发票质量权重': 0.15,
            '业务稳定权重': 0.05
        }
        
        # 1. 信誉评级风险评分 (0-1, 越高风险越大)
        rating_risk_map = {'A': 0.1, 'B': 0.3, 'C': 0.6, 'D': 0.9}
        self.综合数据['信誉风险评分'] = self.综合数据['信誉评级'].map(rating_risk_map)
        
        # 2. 历史违约风险评分
        self.综合数据['违约风险评分'] = self.综合数据['违约标签'] * 0.8 + (1 - self.综合数据['违约标签']) * 0.1
        
        # 3. 财务状况风险评分 (基于毛利率，标准化后取反)
        毛利率_min = self.综合数据['毛利率'].min()
        毛利率_max = self.综合数据['毛利率'].max()
        self.综合数据['财务风险评分'] = 1 - (self.综合数据['毛利率'] - 毛利率_min) / (毛利率_max - 毛利率_min + 1e-6)
        
        # 4. 发票质量风险评分
        self.综合数据['发票风险评分'] = self.综合数据['整体作废率']
        
        # 5. 业务稳定性风险评分 (基于发票活跃度，标准化后取反)
        活跃度_min = self.综合数据['发票活跃度'].min()
        活跃度_max = self.综合数据['发票活跃度'].max()
        self.综合数据['稳定风险评分'] = 1 - (self.综合数据['发票活跃度'] - 活跃度_min) / (活跃度_max - 活跃度_min + 1e-6)
        
        # 综合风险评分 (加权平均)
        self.综合数据['综合风险评分'] = (
            risk_weights['信誉评级权重'] * self.综合数据['信誉风险评分'] +
            risk_weights['历史违约权重'] * self.综合数据['违约风险评分'] +
            risk_weights['财务状况权重'] * self.综合数据['财务风险评分'] +
            risk_weights['发票质量权重'] * self.综合数据['发票风险评分'] +
            risk_weights['业务稳定权重'] * self.综合数据['稳定风险评分']
        )
        
        # 风险等级分类
        def classify_risk_level(score):
            if score <= 0.3:
                return '低风险'
            elif score <= 0.5:
                return '中低风险'
            elif score <= 0.7:
                return '中高风险'
            else:
                return '高风险'
        
        self.综合数据['风险等级'] = self.综合数据['综合风险评分'].apply(classify_risk_level)
        
        print("✅ 风险模型构建完成")
        
        # 显示风险权重设置
        print(f"\n📊 风险评分权重设置:")
        for key, value in risk_weights.items():
            print(f"   {key}: {value}")
    
    def design_credit_strategy(self, total_budget=10000):
        """设计信贷策略"""
        print(f"\n💰 设计信贷策略 (总预算: {total_budget}万元)...")
        
        # 1. 基础贷款额度计算 (基于信誉评级和营业收入)
        def calculate_base_amount(row):
            rating = row['信誉评级']
            income = row['营业收入']
            has_default = row['违约标签']
            
            # D级企业原则上不放贷
            if rating == 'D':
                return 0
            
            # 根据信誉评级确定收入比例
            income_ratios = {
                ('A', 0): 0.25,  # A级无违约
                ('A', 1): 0.15,  # A级有违约
                ('B', 0): 0.20,  # B级无违约
                ('B', 1): 0.12,  # B级有违约
                ('C', 0): 0.10,  # C级无违约
                ('C', 1): 0.08,  # C级有违约
            }
            
            ratio = income_ratios.get((rating, has_default), 0.05)
            base_amount = income * ratio
            
            # 额度上限
            max_amounts = {
                ('A', 0): 100,
                ('A', 1): 80,
                ('B', 0): 80,
                ('B', 1): 60,
                ('C', 0): 50,
                ('C', 1): 30,
            }
            
            max_amount = max_amounts.get((rating, has_default), 0)
            
            return min(max_amount, max(10, base_amount)) if base_amount >= 10 else 0
        
        self.综合数据['基础额度'] = self.综合数据.apply(calculate_base_amount, axis=1)
        
        # 2. 财务状况调整
        def adjust_by_financial(row):
            base = row['基础额度']
            if base == 0:
                return 0
            
            profit_margin = row['毛利率']
            if profit_margin > 0.3:
                return min(100, base * 1.2)
            elif profit_margin < 0:
                return base * 0.7
            else:
                return base
        
        self.综合数据['调整额度'] = self.综合数据.apply(adjust_by_financial, axis=1)
        
        # 3. 最终额度确定 (10-100万元)
        self.综合数据['推荐额度'] = self.综合数据['调整额度'].apply(
            lambda x: min(100, max(10, x)) if x >= 10 else 0
        )
        
        # 4. 利率定价 (结合客户流失率优化)
        self._calculate_optimal_interest_rates()
        
        # 5. 预算分配优化
        self._optimize_budget_allocation(total_budget)
        
        print("✅ 信贷策略设计完成")
    
    def _calculate_optimal_interest_rates(self):
        """结合客户流失率数据计算最优利率"""
        print("\n🎯 结合客户流失率优化利率定价...")
        
        def calculate_interest_rate(row):
            rating = row['信誉评级']
            risk_score = row['综合风险评分']
            has_default = row['违约标签']
            
            if row['推荐额度'] == 0:
                return 0
            
            # 基础利率
            base_rates = {
                ('A', 0): 0.045,
                ('A', 1): 0.055,
                ('B', 0): 0.055,
                ('B', 1): 0.070,
                ('C', 0): 0.080,
                ('C', 1): 0.100,
            }
            
            base_rate = base_rates.get((rating, has_default), 0.12)
            
            # 风险调整
            risk_adjustments = {
                ('A', 0): 0.02,
                ('A', 1): 0.03,
                ('B', 0): 0.03,
                ('B', 1): 0.04,
                ('C', 0): 0.05,
                ('C', 1): 0.05,
            }
            
            risk_adj = risk_adjustments.get((rating, has_default), 0.06)
            initial_rate = base_rate + risk_adj * risk_score
            
            # 结合客户流失率优化利率
            optimal_rate = self._optimize_rate_with_churn(initial_rate, rating)
            
            # 利率约束: 4%-15%
            return max(0.04, min(0.15, optimal_rate))
        
        self.综合数据['推荐利率'] = self.综合数据.apply(calculate_interest_rate, axis=1)
    
    def _optimize_rate_with_churn(self, initial_rate, rating):
        """根据客户流失率数据优化利率"""
        
        # 从附件3数据中找到最接近的利率区间
        rates = self.客户流失率数据.iloc[:, 0].values  # 第一列是利率
        churn_rates = self.客户流失率数据.iloc[:, 1].values  # 第二列是流失率
        
        # 计算不同利率下的预期收益 (考虑客户流失)
        best_rate = initial_rate
        best_net_return = 0
        
        # 在初始利率附近搜索最优利率
        search_range = np.linspace(max(0.04, initial_rate - 0.02), 
                                 min(0.15, initial_rate + 0.02), 20)
        
        for test_rate in search_range:
            # 插值计算该利率对应的客户流失率
            if test_rate <= rates.min():
                churn_rate = churn_rates[0]
            elif test_rate >= rates.max():
                churn_rate = churn_rates[-1]
            else:
                churn_rate = np.interp(test_rate, rates, churn_rates)
            
            # 客户保留率
            retention_rate = 1 - churn_rate / 100
            
            # 计算净收益 = 利率 × 客户保留率
            net_return = test_rate * retention_rate
            
            if net_return > best_net_return:
                best_net_return = net_return
                best_rate = test_rate
        
        return best_rate
    
    def _optimize_budget_allocation(self, total_budget):
        """预算分配优化 (考虑客户流失率)"""
        
        # 计算每个企业的客户流失率
        rates = self.客户流失率数据.iloc[:, 0].values
        churn_rates = self.客户流失率数据.iloc[:, 1].values
        
        def get_churn_rate(interest_rate):
            if interest_rate <= rates.min():
                return churn_rates[0] / 100
            elif interest_rate >= rates.max():
                return churn_rates[-1] / 100
            else:
                return np.interp(interest_rate, rates, churn_rates) / 100
        
        self.综合数据['客户流失率'] = self.综合数据['推荐利率'].apply(get_churn_rate)
        self.综合数据['客户保留率'] = 1 - self.综合数据['客户流失率']
        
        # 计算预期收益 (考虑违约风险和客户流失)
        self.综合数据['违约概率'] = np.minimum(0.95, self.综合数据['综合风险评分'] * 1.1)
        self.综合数据['预期收益'] = (
            self.综合数据['推荐额度'] * 
            self.综合数据['推荐利率'] * 
            (1 - self.综合数据['违约概率']) *
            self.综合数据['客户保留率']  # 新增客户保留率因子
        )
        
        # 风险调整收益率
        self.综合数据['风险调整收益率'] = (
            self.综合数据['预期收益'] / 
            (self.综合数据['推荐额度'] * self.综合数据['综合风险评分'] + 0.01)
        )
        
        # 按风险调整收益率排序，优先分配高收益低风险企业
        候选企业 = self.综合数据[self.综合数据['推荐额度'] > 0].copy()
        候选企业 = 候选企业.sort_values('风险调整收益率', ascending=False)
        
        # 预算分配
        累计预算 = 0
        最终额度 = []
        
        for idx, row in 候选企业.iterrows():
            推荐额度 = row['推荐额度']
            
            if 累计预算 + 推荐额度 <= total_budget:
                最终额度.append(推荐额度)
                累计预算 += 推荐额度
            elif total_budget - 累计预算 >= 10:  # 剩余预算够最小额度
                剩余额度 = total_budget - 累计预算
                最终额度.append(剩余额度)
                累计预算 = total_budget
            else:
                最终额度.append(0)
        
        候选企业['最终额度'] = 最终额度
        
        # 更新综合数据
        self.综合数据['最终额度'] = 0
        self.综合数据.loc[候选企业.index, '最终额度'] = 候选企业['最终额度']
        
        # 更新贷款决策
        self.综合数据['贷款决策'] = self.综合数据['最终额度'].apply(
            lambda x: '批准' if x > 0 else '拒绝'
        )
        
        # 计算最终收益 (考虑客户流失)
        self.综合数据['最终收益'] = (
            self.综合数据['最终额度'] * 
            self.综合数据['推荐利率'] * 
            (1 - self.综合数据['违约概率']) *
            self.综合数据['客户保留率']
        )
    
    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n" + "="*80)
        print("📊 问题1：123家企业信贷风险量化分析报告")
        print("="*80)
        
        # 基本统计
        print(f"\n📈 基本统计信息:")
        print(f"   企业总数: {len(self.综合数据)}家")
        print(f"   有效企业: {len(self.综合数据[self.综合数据['营业收入'] > 0])}家")
        
        # 信誉评级分布
        print(f"\n⭐ 信誉评级分布:")
        rating_dist = self.综合数据['信誉评级'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {rating}级: {count:2d}家 ({percentage:5.1f}%)")
        
        # 违约情况
        print(f"\n⚠️ 历史违约情况:")
        default_stats = self.综合数据.groupby('信誉评级')['违约标签'].agg(['count', 'sum', 'mean'])
        for rating, stats in default_stats.iterrows():
            违约率 = stats['mean'] * 100
            print(f"   {rating}级: {stats['sum']:2.0f}/{stats['count']:2.0f} = {违约率:5.1f}%")
        
        # 风险等级分布
        print(f"\n📊 综合风险等级分布:")
        risk_dist = self.综合数据['风险等级'].value_counts()
        for level, count in risk_dist.items():
            percentage = count / len(self.综合数据) * 100
            print(f"   {level}: {count:2d}家 ({percentage:5.1f}%)")
        
        # 信贷策略结果
        print(f"\n💰 信贷策略结果:")
        批准企业 = self.综合数据[self.综合数据['最终额度'] > 0]
        拒绝企业 = self.综合数据[self.综合数据['最终额度'] == 0]
        
        print(f"   批准放贷: {len(批准企业)}家 ({len(批准企业)/len(self.综合数据)*100:.1f}%)")
        print(f"   拒绝放贷: {len(拒绝企业)}家 ({len(拒绝企业)/len(self.综合数据)*100:.1f}%)")
        print(f"   总放贷金额: {批准企业['最终额度'].sum():,.0f}万元")
        print(f"   平均贷款额度: {批准企业['最终额度'].mean():.1f}万元")
        print(f"   加权平均利率: {(批准企业['最终额度'] * 批准企业['推荐利率']).sum() / 批准企业['最终额度'].sum() * 100:.2f}%")
        
        # 收益分析
        总收益 = 批准企业['最终收益'].sum()
        总投资 = 批准企业['最终额度'].sum()
        投资回报率 = 总收益 / 总投资 * 100 if 总投资 > 0 else 0
        
        print(f"\n📈 收益分析:")
        print(f"   预期年收益: {总收益:.1f}万元")
        print(f"   投资回报率: {投资回报率:.2f}%")
        print(f"   平均违约概率: {批准企业['违约概率'].mean()*100:.2f}%")
        
        # 各等级放贷情况
        print(f"\n📋 各信誉等级放贷情况:")
        for rating in ['A', 'B', 'C', 'D']:
            rating_data = self.综合数据[self.综合数据['信誉评级'] == rating]
            if len(rating_data) > 0:
                批准数 = len(rating_data[rating_data['最终额度'] > 0])
                总数 = len(rating_data)
                批准率 = 批准数 / 总数 * 100
                平均额度 = rating_data[rating_data['最终额度'] > 0]['最终额度'].mean()
                平均利率 = rating_data[rating_data['最终额度'] > 0]['推荐利率'].mean() * 100
                
                print(f"   {rating}级: {批准数}/{总数} ({批准率:.1f}%), "
                      f"平均额度{平均额度:.1f}万元, 平均利率{平均利率:.2f}%")
        
        print("\n" + "="*80)
    
    def export_results(self):
        """导出分析结果"""
        # 选择关键字段导出 (包含客户流失率相关字段)
        export_columns = [
            '企业代号', '信誉评级', '是否违约', '营业收入', '毛利率',
            '发票活跃度', '整体作废率', '综合风险评分', '风险等级',
            '推荐额度', '最终额度', '推荐利率', '客户流失率', '客户保留率',
            '违约概率', '最终收益', '贷款决策'
        ]
        
        结果数据 = self.综合数据.reset_index()[export_columns].round(4)
        
        # 导出Excel
        with pd.ExcelWriter('问题1_信贷风险分析结果_含客户流失率.xlsx') as writer:
            结果数据.to_excel(writer, sheet_name='信贷策略结果', index=False)
            
            # 统计汇总 (包含客户流失率分析)
            批准企业 = self.综合数据[self.综合数据['最终额度'] > 0]
            统计汇总 = pd.DataFrame({
                '指标': ['企业总数', '批准企业数', '拒绝企业数', '批准率(%)', 
                        '总放贷金额(万元)', '平均贷款额度(万元)', '加权平均利率(%)',
                        '平均客户流失率(%)', '平均客户保留率(%)', '总客户保留价值(万元)',
                        '预期年收益(万元)', '投资回报率(%)', '平均违约概率(%)'],
                '数值': [
                    len(self.综合数据),
                    len(批准企业),
                    len(self.综合数据) - len(批准企业),
                    len(批准企业) / len(self.综合数据) * 100,
                    self.综合数据['最终额度'].sum(),
                    批准企业['最终额度'].mean(),
                    (批准企业['最终额度'] * 批准企业['推荐利率']).sum() / 批准企业['最终额度'].sum() * 100,
                    批准企业['客户流失率'].mean() * 100,
                    批准企业['客户保留率'].mean() * 100,
                    (批准企业['最终额度'] * 批准企业['客户保留率']).sum(),
                    self.综合数据['最终收益'].sum(),
                    self.综合数据['最终收益'].sum() / self.综合数据['最终额度'].sum() * 100 if self.综合数据['最终额度'].sum() > 0 else 0,
                    批准企业['违约概率'].mean() * 100
                ]
            })
            统计汇总.to_excel(writer, sheet_name='统计汇总', index=False)
            
            # 客户流失率分析表
            流失率分析 = pd.DataFrame({
                '利率区间': ['4%-6%', '6%-8%', '8%-10%', '10%-12%', '12%以上'],
                '企业数量': [
                    len(批准企业[(批准企业['推荐利率'] >= 0.04) & (批准企业['推荐利率'] < 0.06)]),
                    len(批准企业[(批准企业['推荐利率'] >= 0.06) & (批准企业['推荐利率'] < 0.08)]),
                    len(批准企业[(批准企业['推荐利率'] >= 0.08) & (批准企业['推荐利率'] < 0.10)]),
                    len(批准企业[(批准企业['推荐利率'] >= 0.10) & (批准企业['推荐利率'] < 0.12)]),
                    len(批准企业[批准企业['推荐利率'] >= 0.12])
                ],
                '平均流失率(%)': [
                    批准企业[(批准企业['推荐利率'] >= 0.04) & (批准企业['推荐利率'] < 0.06)]['客户流失率'].mean() * 100,
                    批准企业[(批准企业['推荐利率'] >= 0.06) & (批准企业['推荐利率'] < 0.08)]['客户流失率'].mean() * 100,
                    批准企业[(批准企业['推荐利率'] >= 0.08) & (批准企业['推荐利率'] < 0.10)]['客户流失率'].mean() * 100,
                    批准企业[(批准企业['推荐利率'] >= 0.10) & (批准企业['推荐利率'] < 0.12)]['客户流失率'].mean() * 100,
                    批准企业[批准企业['推荐利率'] >= 0.12]['客户流失率'].mean() * 100
                ]
            })
            流失率分析 = 流失率分析.fillna(0)  # 填充空值
            流失率分析.to_excel(writer, sheet_name='客户流失率分析', index=False)
        
        print(f"✅ 分析结果已导出至: 问题1_信贷风险分析结果_含客户流失率.xlsx")
        print(f"📊 包含客户流失率优化的完整信贷策略分析")

def main():
    """主函数"""
    analyzer = CreditRiskAnalyzer()
    
    # 1. 加载数据
    if not analyzer.load_data():
        return
    
    # 2. 分析发票特征
    analyzer.analyze_invoice_features()
    
    # 3. 构建风险模型
    analyzer.build_risk_model()
    
    # 4. 设计信贷策略
    analyzer.design_credit_strategy(total_budget=10000)  # 假设1亿元预算
    
    # 5. 生成报告
    analyzer.generate_analysis_report()
    
    # 6. 导出结果
    analyzer.export_results()
    
    print(f"\n🎉 问题1分析完成！")
    print(f"📊 已对123家企业进行信贷风险量化分析")
    print(f"💰 已制定年度信贷总额固定时的信贷策略")

if __name__ == "__main__":
    main()
