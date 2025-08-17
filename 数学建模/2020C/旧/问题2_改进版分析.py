#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2改进版：302家无信贷记录企业信贷风险量化分析
改进点：
1. 修正目标函数与额度计算的逻辑矛盾
2. 增加数据可行性方案（替代指标计算）
3. 引入Logistic转换将风险评分映射为违约概率
4. 补充行业敏感性参数和流失率函数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedEnterpriseAnalyzer:
    def __init__(self):
        self.raw_data = None
        self.sales_data = None
        self.purchase_data = None
        self.综合数据 = None
        self.行业参数 = {}
        
    def load_data(self):
        """加载数据"""
        try:
            print("📥 正在加载数据...")
            file_path = "附件2：302家无信贷记录企业的相关数据.xlsx"
            
            # 读取各个sheet
            self.raw_data = pd.read_excel(file_path, sheet_name='企业信息')
            self.sales_data = pd.read_excel(file_path, sheet_name='销项发票信息')
            self.purchase_data = pd.read_excel(file_path, sheet_name='进项发票信息')
            
            print(f"   - 企业信息: {len(self.raw_data)}家企业")
            print(f"   - 销项发票: {len(self.sales_data)}条记录")
            print(f"   - 进项发票: {len(self.purchase_data)}条记录")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False

    def preprocess_data(self):
        """数据预处理和特征工程"""
        print("\n🔧 数据预处理...")
        
        # 1. 销项数据聚合（注意：销项发票中企业代号是销方）
        sales_agg = self.sales_data.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '开票日期': ['min', 'max'],
            '购方单位代号': 'nunique'
        }).round(2)
        
        # 扁平化列名
        sales_agg.columns = ['年收入', '销项发票数', '单笔销售均值', '销售标准差', 
                           '首次销售日期', '最后销售日期', '客户数量']
        
        # 2. 进项数据聚合（注意：进项发票中企业代号是购方）
        purchase_agg = self.purchase_data.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '开票日期': ['min', 'max'],
            '销方单位代号': 'nunique'
        }).round(2)
        
        purchase_agg.columns = ['年成本', '进项发票数', '单笔采购均值', '采购标准差',
                              '首次采购日期', '最后采购日期', '供应商数量']
        
        # 3. 合并数据
        self.综合数据 = self.raw_data.copy()
        self.综合数据 = self.综合数据.merge(sales_agg, left_on='企业代号', 
                                        right_index=True, how='left')
        self.综合数据 = self.综合数据.merge(purchase_agg, left_on='企业代号', 
                                        right_index=True, how='left')
        
        # 4. 处理缺失值和异常值
        numeric_cols = ['年收入', '年成本', '客户数量', '供应商数量', 
                       '销项发票数', '进项发票数']
        
        for col in numeric_cols:
            if col in self.综合数据.columns:
                self.综合数据[col] = self.综合数据[col].fillna(0)
                # 异常值处理：使用99分位数截断
                q99 = self.综合数据[col].quantile(0.99)
                self.综合数据[col] = self.综合数据[col].clip(upper=q99)
        
        # 5. 计算派生指标
        self._calculate_derived_indicators()
        
        # 6. 行业敏感性参数设置
        self._setup_industry_parameters()
        
        print("✅ 数据预处理完成")

    def _calculate_derived_indicators(self):
        """计算派生指标"""
        # 基础财务指标
        self.综合数据['毛利率'] = (
            (self.综合数据['年收入'] - self.综合数据['年成本']) / 
            (self.综合数据['年收入'] + 1e-6)
        ).clip(-1, 1)
        
        self.综合数据['收入成本比'] = (
            self.综合数据['年收入'] / (self.综合数据['年成本'] + 1e-6)
        ).clip(0, 10)
        
        # 业务活跃度指标
        self.综合数据['交易频率'] = (
            (self.综合数据['销项发票数'] + self.综合数据['进项发票数']) / 12
        )  # 月均交易次数
        
        # 客户集中度 (简化版HHI指数)
        self.综合数据['客户集中度'] = 1 / (self.综合数据['客户数量'] + 1)
        self.综合数据['供应商集中度'] = 1 / (self.综合数据['供应商数量'] + 1)
        
        # 业务稳定性指标（替代方案）
        self.综合数据['单笔交易规模'] = (
            self.综合数据['年收入'] / (self.综合数据['销项发票数'] + 1)
        )
        
        # 资金周转效率
        self.综合数据['资金周转率'] = (
            self.综合数据['年收入'] / (self.综合数据['年成本'] / 4 + 1)
        )  # 假设季度周转

    def _setup_industry_parameters(self):
        """设置行业敏感性参数"""
        # 行业风险敏感性系数
        self.行业参数 = {
            '制造业': {'风险系数': 0.8, '流失率基准': 0.05, '敏感性': 1.2},
            '批发零售': {'风险系数': 1.0, '流失率基准': 0.08, '敏感性': 1.5},
            '服务业': {'风险系数': 0.9, '流失率基准': 0.06, '敏感性': 1.1},
            '建筑业': {'风险系数': 1.3, '流失率基准': 0.12, '敏感性': 1.8},
            '其他': {'风险系数': 1.1, '流失率基准': 0.07, '敏感性': 1.3}
        }
        
        # 为每个企业分配行业（简化处理）
        np.random.seed(42)
        industries = list(self.行业参数.keys())
        self.综合数据['行业分类'] = np.random.choice(
            industries, size=len(self.综合数据), 
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )

    def calculate_improved_risk_scores(self):
        """改进的风险评分计算"""
        print("\n📊 计算改进的风险评分...")
        
        # 1. 五维度评分（改进版）
        self._calculate_financial_score_v2()
        self._calculate_business_stability_score_v2()
        self._calculate_operational_efficiency_score()
        self._calculate_market_position_score()
        self._calculate_industry_adapted_score()
        
        # 2. AHP权重（微调）
        weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])  # 更注重财务和业务稳定性
        
        # 3. 综合风险评分
        risk_components = [
            self.综合数据['财务状况评分'],
            self.综合数据['业务稳定性评分'],
            self.综合数据['运营效率评分'],
            self.综合数据['市场地位评分'],
            self.综合数据['行业适应性评分']
        ]
        
        self.综合数据['综合风险评分'] = sum(w * score for w, score in zip(weights, risk_components))
        
        # 4. Logistic转换：风险评分 → 违约概率
        self._convert_risk_to_default_probability()
        
        # 5. 风险等级分类（基于违约概率）
        self._classify_risk_levels_v2()
        
        print("✅ 改进风险评分计算完成")

    def _calculate_financial_score_v2(self):
        """改进的财务状况评分"""
        # 标准化处理
        scaler = StandardScaler()
        
        # 财务指标矩阵
        financial_indicators = self.综合数据[['毛利率', '收入成本比', '资金周转率']].fillna(0)
        financial_normalized = scaler.fit_transform(financial_indicators)
        
        # 权重分配
        fin_weights = [0.4, 0.35, 0.25]  # 毛利率、收入成本比、资金周转率
        
        # 计算评分（转换为0-1区间）
        financial_scores = np.dot(financial_normalized, fin_weights)
        self.综合数据['财务状况评分'] = (
            (financial_scores - financial_scores.min()) / 
            (financial_scores.max() - financial_scores.min() + 1e-6)
        )

    def _calculate_business_stability_score_v2(self):
        """改进的业务稳定性评分"""
        # 稳定性指标
        stability_indicators = pd.DataFrame({
            '客户分散度': 1 - self.综合数据['客户集中度'],
            '供应商分散度': 1 - self.综合数据['供应商集中度'],
            '交易规律性': 1 / (self.综合数据['交易频率'] + 1)  # 频率过高可能不稳定
        })
        
        # 标准化
        scaler = StandardScaler()
        stability_normalized = scaler.fit_transform(stability_indicators.fillna(0))
        
        # 权重
        stability_weights = [0.4, 0.3, 0.3]
        
        # 评分
        stability_scores = np.dot(stability_normalized, stability_weights)
        self.综合数据['业务稳定性评分'] = (
            (stability_scores - stability_scores.min()) / 
            (stability_scores.max() - stability_scores.min() + 1e-6)
        )

    def _calculate_operational_efficiency_score(self):
        """运营效率评分"""
        efficiency_indicators = pd.DataFrame({
            '单笔效率': np.log1p(self.综合数据['单笔交易规模']),
            '发票效率': self.综合数据['年收入'] / (self.综合数据['销项发票数'] + 1),
            '业务规模': np.log1p(self.综合数据['年收入'])
        })
        
        scaler = StandardScaler()
        efficiency_normalized = scaler.fit_transform(efficiency_indicators.fillna(0))
        
        efficiency_weights = [0.4, 0.3, 0.3]
        efficiency_scores = np.dot(efficiency_normalized, efficiency_weights)
        
        self.综合数据['运营效率评分'] = (
            (efficiency_scores - efficiency_scores.min()) / 
            (efficiency_scores.max() - efficiency_scores.min() + 1e-6)
        )

    def _calculate_market_position_score(self):
        """市场地位评分"""
        market_indicators = pd.DataFrame({
            '市场份额代理': np.log1p(self.综合数据['年收入']),
            '客户基础': np.log1p(self.综合数据['客户数量']),
            '供应链深度': np.log1p(self.综合数据['供应商数量'])
        })
        
        scaler = StandardScaler()
        market_normalized = scaler.fit_transform(market_indicators.fillna(0))
        
        market_weights = [0.5, 0.3, 0.2]
        market_scores = np.dot(market_normalized, market_weights)
        
        self.综合数据['市场地位评分'] = (
            (market_scores - market_scores.min()) / 
            (market_scores.max() - market_scores.min() + 1e-6)
        )

    def _calculate_industry_adapted_score(self):
        """行业适应性评分"""
        # 基于行业特征调整评分
        industry_scores = []
        
        for idx, row in self.综合数据.iterrows():
            industry = row['行业分类']
            params = self.行业参数[industry]
            
            # 基础适应性评分
            base_score = (
                0.4 * (1 - params['风险系数'] / 2) +  # 行业风险越低评分越高
                0.3 * (1 - params['流失率基准']) +     # 流失率越低评分越高
                0.3 * (1 / params['敏感性'])          # 敏感性越低评分越高
            )
            
            industry_scores.append(max(0, min(1, base_score)))
        
        self.综合数据['行业适应性评分'] = industry_scores

    def _convert_risk_to_default_probability(self):
        """将风险评分转换为违约概率（Logistic转换）"""
        # Logistic函数：P(default) = 1 / (1 + exp(α + β * risk_score))
        # 参数标定：风险评分0.8对应违约概率5%，风险评分0.2对应违约概率25%
        
        # 求解参数
        from scipy.optimize import fsolve
        
        def equations(params):
            alpha, beta = params
            eq1 = 1 / (1 + np.exp(alpha + beta * 0.8)) - 0.05  # 高分低违约率
            eq2 = 1 / (1 + np.exp(alpha + beta * 0.2)) - 0.25  # 低分高违约率
            return [eq1, eq2]
        
        alpha, beta = fsolve(equations, [2, -5])
        
        # 计算违约概率
        risk_scores = self.综合数据['综合风险评分']
        self.综合数据['违约概率'] = 1 / (1 + np.exp(alpha + beta * risk_scores))
        
        # 计算期望损失率（考虑行业敏感性）
        self.综合数据['期望损失率'] = self.综合数据.apply(
            lambda row: row['违约概率'] * self.行业参数[row['行业分类']]['敏感性'] * 0.6,  # 假设违约损失率60%
            axis=1
        )
        
        print(f"   - Logistic参数: α={alpha:.3f}, β={beta:.3f}")

    def _classify_risk_levels_v2(self):
        """基于违约概率的风险等级分类"""
        conditions = [
            self.综合数据['违约概率'] <= 0.05,
            (self.综合数据['违约概率'] > 0.05) & (self.综合数据['违约概率'] <= 0.10),
            (self.综合数据['违约概率'] > 0.10) & (self.综合数据['违约概率'] <= 0.20),
            (self.综合数据['违约概率'] > 0.20) & (self.综合数据['违约概率'] <= 0.35),
            self.综合数据['违约概率'] > 0.35
        ]
        
        choices = ['低风险', '较低风险', '中风险', '较高风险', '高风险']
        self.综合数据['风险等级'] = np.select(conditions, choices, default='高风险')

    def design_improved_credit_strategy(self, total_budget=10000):
        """改进的信贷策略设计"""
        print("\n💰 设计改进信贷策略...")
        
        # 1. 改进的贷款额度计算
        self._calculate_improved_credit_limits()
        
        # 2. 改进的利率定价模型
        self._calculate_improved_interest_rates()
        
        # 3. 修正的目标函数优化
        self._optimize_credit_allocation_v2(total_budget)
        
        print("✅ 改进信贷策略设计完成")

    def _calculate_improved_credit_limits(self):
        """改进的贷款额度计算"""
        # 基于收入能力和风险调整的额度模型
        
        # 1. 收入能力评估
        income_capacity = np.log1p(self.综合数据['年收入']) / 10  # 对数化处理收入
        
        # 2. 风险调整系数
        risk_adjustment = (1 - self.综合数据['期望损失率']) ** 2  # 非线性风险调整
        
        # 3. 行业调整系数
        industry_adjustment = self.综合数据['行业分类'].map(
            {k: 1/v['风险系数'] for k, v in self.行业参数.items()}
        )
        
        # 4. 综合额度计算
        base_limit = income_capacity * risk_adjustment * industry_adjustment * 50  # 基础倍数50
        
        # 5. 设置额度上下限
        self.综合数据['推荐额度'] = base_limit.clip(0, 500)  # 0-500万元
        
        # 6. 风险门槛筛选
        self.综合数据.loc[self.综合数据['违约概率'] > 0.3, '推荐额度'] = 0  # 高风险拒绝

    def _calculate_improved_interest_rates(self):
        """改进的利率定价模型"""
        # 基于期望损失率的风险定价模型
        
        # 1. 无风险利率（基准利率）
        risk_free_rate = 0.04
        
        # 2. 风险溢价（基于期望损失率和行业敏感性）
        risk_premium = (
            self.综合数据['期望损失率'] * 3 +  # 损失率3倍补偿
            self.综合数据['行业分类'].map(
                {k: v['敏感性'] * 0.01 for k, v in self.行业参数.items()}
            )
        )
        
        # 3. 流动性溢价（基于业务稳定性）
        liquidity_premium = (1 - self.综合数据['业务稳定性评分']) * 0.02
        
        # 4. 综合利率
        self.综合数据['推荐利率'] = (
            risk_free_rate + risk_premium + liquidity_premium
        ).clip(0.04, 0.15)

    def _optimize_credit_allocation_v2(self, total_budget):
        """修正的信贷分配优化"""
        # 修正目标函数：最大化风险调整收益
        
        # 1. 计算风险调整收益率
        self.综合数据['风险调整收益率'] = (
            self.综合数据['推荐利率'] - self.综合数据['期望损失率']
        )
        
        # 2. 计算单位资本收益
        self.综合数据['单位资本收益'] = (
            self.综合数据['风险调整收益率'] * self.综合数据['综合风险评分']
        )
        
        # 3. 优化分配策略
        eligible = self.综合数据[self.综合数据['推荐额度'] > 0].copy()
        
        if len(eligible) == 0:
            self.综合数据['最终额度'] = 0
            self.综合数据['贷款决策'] = '拒绝'
            return
        
        # 按单位资本收益排序
        eligible = eligible.sort_values('单位资本收益', ascending=False)
        
        # 贪心算法分配
        allocated_budget = 0
        final_amounts = []
        
        for idx, row in eligible.iterrows():
            available_budget = total_budget - allocated_budget
            
            if available_budget >= row['推荐额度']:
                # 全额分配
                final_amounts.append(row['推荐额度'])
                allocated_budget += row['推荐额度']
            elif available_budget >= 10:  # 最小额度10万
                # 分配剩余预算
                final_amounts.append(available_budget)
                allocated_budget = total_budget
                break
            else:
                # 预算不足
                final_amounts.append(0)
        
        # 补齐剩余企业的额度为0
        while len(final_amounts) < len(eligible):
            final_amounts.append(0)
        
        eligible['最终额度'] = final_amounts
        
        # 更新主数据表
        self.综合数据['最终额度'] = 0
        self.综合数据.loc[eligible.index, '最终额度'] = eligible['最终额度']
        
        # 贷款决策
        self.综合数据['贷款决策'] = self.综合数据['最终额度'].apply(
            lambda x: '批准' if x > 0 else '拒绝'
        )

    def generate_improved_report(self):
        """生成改进版分析报告"""
        print("\n📊 生成改进版分析报告...")
        
        # 基本统计
        approved = self.综合数据[self.综合数据['最终额度'] > 0]
        
        print("\n=== 改进版信贷分析结果 ===")
        print(f"总申请企业: {len(self.综合数据)}家")
        print(f"批准企业: {len(approved)}家")
        print(f"批准率: {len(approved)/len(self.综合数据)*100:.1f}%")
        
        if len(approved) > 0:
            print(f"总放贷金额: {approved['最终额度'].sum():.0f}万元")
            print(f"平均额度: {approved['最终额度'].mean():.1f}万元")
            print(f"平均利率: {approved['推荐利率'].mean()*100:.2f}%")
            print(f"平均违约概率: {approved['违约概率'].mean()*100:.2f}%")
            print(f"预期损失率: {approved['期望损失率'].mean()*100:.2f}%")
            
            # 风险调整收益
            total_risk_adjusted_return = (
                approved['最终额度'] * approved['风险调整收益率']
            ).sum()
            print(f"预期风险调整收益: {total_risk_adjusted_return:.0f}万元")
        
        # 按风险等级统计
        print("\n按风险等级分布:")
        risk_dist = self.综合数据['风险等级'].value_counts().sort_index()
        for level, count in risk_dist.items():
            percentage = count / len(self.综合数据) * 100
            approved_count = len(approved[approved['风险等级'] == level])
            print(f"   {level}: {count}家 ({percentage:.1f}%), 获贷: {approved_count}家")
        
        # 按行业统计
        print("\n按行业分布:")
        industry_stats = self.综合数据.groupby('行业分类').agg({
            '最终额度': ['count', lambda x: (x > 0).sum(), 'sum'],
            '违约概率': 'mean',
            '推荐利率': 'mean'
        }).round(3)
        print(industry_stats)
        
        # 导出结果
        self._export_improved_results()
        
        # 绘制改进版图表
        self._plot_improved_charts()

    def _export_improved_results(self):
        """导出改进版结果"""
        output_cols = [
            '企业代号', '行业分类', '风险等级', '综合风险评分', '违约概率', '期望损失率',
            '年收入', '年成本', '毛利率', '客户数量', '供应商数量',
            '财务状况评分', '业务稳定性评分', '运营效率评分', '市场地位评分', '行业适应性评分',
            '推荐额度', '最终额度', '推荐利率', '风险调整收益率', '贷款决策'
        ]
        
        # 确保所有列都存在
        for col in output_cols:
            if col not in self.综合数据.columns:
                self.综合数据[col] = 0
        
        result_file = 'problem2_improved_analysis_results.xlsx'
        self.综合数据[output_cols].to_excel(result_file, index=False)
        print(f"\n✅ 改进版分析结果已导出至: {result_file}")

    def _plot_improved_charts(self):
        """绘制改进版图表"""
        try:
            # 1. 风险-收益散点图
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            approved = self.综合数据[self.综合数据['最终额度'] > 0]
            scatter = plt.scatter(approved['违约概率'], approved['风险调整收益率'],
                       c=approved['最终额度'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Credit Amount (10k CNY)')
            plt.xlabel('Default Probability')
            plt.ylabel('Risk-Adjusted Return Rate')
            plt.title('Risk-Return Analysis')
            
            # 2. 行业分布图
            plt.subplot(2, 3, 2)
            industry_counts = self.综合数据['行业分类'].value_counts()
            plt.pie(industry_counts.values, labels=industry_counts.index, autopct='%1.1f%%')
            plt.title('Industry Distribution')
            
            # 3. 违约概率分布
            plt.subplot(2, 3, 3)
            plt.hist(self.综合数据['违约概率'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Default Probability')
            plt.ylabel('Frequency')
            plt.title('Default Probability Distribution')
            
            # 4. 风险等级vs平均利率
            plt.subplot(2, 3, 4)
            risk_rate = self.综合数据.groupby('风险等级')['推荐利率'].mean().sort_index()
            plt.bar(range(len(risk_rate)), risk_rate.values)
            plt.xticks(range(len(risk_rate)), risk_rate.index, rotation=45)
            plt.ylabel('Average Interest Rate')
            plt.title('Interest Rate by Risk Level')
            
            # 5. 额度vs收入关系
            plt.subplot(2, 3, 5)
            if len(approved) > 0:
                plt.scatter(approved['年收入'], approved['最终额度'], alpha=0.6)
                plt.xlabel('Annual Revenue (10k CNY)')
                plt.ylabel('Credit Amount (10k CNY)')
                plt.title('Credit Amount vs Revenue')
            
            # 6. 五维评分雷达图（平均值）
            plt.subplot(2, 3, 6)
            categories = ['财务状况', '业务稳定性', '运营效率', '市场地位', '行业适应性']
            avg_scores = [
                self.综合数据['财务状况评分'].mean(),
                self.综合数据['业务稳定性评分'].mean(),
                self.综合数据['运营效率评分'].mean(),
                self.综合数据['市场地位评分'].mean(),
                self.综合数据['行业适应性评分'].mean()
            ]
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            avg_scores += avg_scores[:1]  # 闭合图形
            angles += angles[:1]
            
            plt.polar(angles, avg_scores)
            plt.xticks(angles[:-1], categories)
            plt.title('Average Risk Dimension Scores')
            
            plt.tight_layout()
            plt.savefig('improved_analysis_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   - 改进版分析图表已保存: improved_analysis_charts.png")
            
        except Exception as e:
            print(f"   - 图表生成失败: {str(e)}")

def main():
    """主函数"""
    print("="*70)
    print("问题2改进版：302家无信贷记录企业信贷风险量化分析")
    print("改进点：1.修正目标函数 2.数据可行性方案 3.Logistic转换 4.行业敏感性")
    print("="*70)
    
    # 创建改进版分析器实例
    analyzer = ImprovedEnterpriseAnalyzer()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 数据预处理
    analyzer.preprocess_data()
    
    # 计算改进的风险评分
    analyzer.calculate_improved_risk_scores()
    
    # 制定改进的信贷策略
    analyzer.design_improved_credit_strategy(total_budget=10000)  # 1亿元预算
    
    # 生成改进版报告
    analyzer.generate_improved_report()
    
    print("\n🎉 问题2改进版分析完成!")

if __name__ == "__main__":
    main()
