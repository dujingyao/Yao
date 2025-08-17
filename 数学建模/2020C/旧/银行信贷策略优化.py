#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行信贷策略优化模型 - 符合实际约束条件
贷款额度：万元为单位，最小10万元
年利率：4%-15%
期限：1年

Author: 数据分析团队
Date: 2025年8月16日
"""

import pandas as pd
import numpy as np
import os

class BankCreditOptimizer:
    """银行信贷策略优化器"""
    
    def __init__(self):
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.综合数据 = None
        self.信贷策略 = None
        
        # 银行约束条件
        self.最小贷款额度 = 10      # 万元
        self.最大贷款额度 = 100     # 万元
        self.最小利率 = 0.04       # 4%
        self.最大利率 = 0.15       # 15%
        self.贷款期限 = 1          # 年
        
    def load_data(self):
        """加载数据"""
        file_path = "附件1：123家有信贷记录企业的相关数据.xlsx"
        
        try:
            print(f"📁 正在加载数据文件: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            
            self.企业信息 = pd.read_excel(excel_file, sheet_name='企业信息')
            self.进项发票 = pd.read_excel(excel_file, sheet_name='进项发票信息')
            self.销项发票 = pd.read_excel(excel_file, sheet_name='销项发票信息')
            
            print(f"✅ 数据加载成功!")
            print(f"   企业信息: {self.企业信息.shape}")
            print(f"   进项发票: {self.进项发票.shape}")
            print(f"   销项发票: {self.销项发票.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def analyze_enterprises(self):
        """分析企业数据"""
        print("\n🔍 开始企业财务分析...")
        
        # 分析进项发票
        进项汇总 = self.进项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x == '作废发票').sum()
        }).round(2)
        
        进项汇总.columns = ['进项总额', '进项发票数量', '进项平均金额', '进项标准差', '进项净金额', '进项税额', '进项作废数量']
        进项汇总['进项作废率'] = (进项汇总['进项作废数量'] / 进项汇总['进项发票数量']).fillna(0)
        
        # 分析销项发票
        销项汇总 = self.销项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x.str.strip() == '作废发票').sum()
        }).round(2)
        
        销项汇总.columns = ['销项总额', '销项发票数量', '销项平均金额', '销项标准差', '销项净金额', '销项税额', '销项作废数量']
        销项汇总['销项作废率'] = (销项汇总['销项作废数量'] / 销项汇总['销项发票数量']).fillna(0)
        
        # 合并数据
        self.综合数据 = self.企业信息.set_index('企业代号').join([进项汇总, 销项汇总], how='left').fillna(0)
        
        # 计算关键财务指标
        self._calculate_financial_metrics()
        
        print("✅ 企业财务分析完成!")
        return True
    
    def _calculate_financial_metrics(self):
        """计算财务指标"""
        
        # 营业收入和成本
        self.综合数据['年营业收入'] = self.综合数据['销项净金额']  # 万元
        self.综合数据['年营业成本'] = self.综合数据['进项净金额']  # 万元
        self.综合数据['年毛利润'] = self.综合数据['年营业收入'] - self.综合数据['年营业成本']
        self.综合数据['毛利率'] = (self.综合数据['年毛利润'] / (self.综合数据['年营业收入'] + 1e-8)).clip(-1, 1)
        
        # 现金流指标
        self.综合数据['年现金流入'] = self.综合数据['销项总额']  # 万元
        self.综合数据['年现金流出'] = self.综合数据['进项总额']  # 万元
        self.综合数据['年净现金流'] = self.综合数据['年现金流入'] - self.综合数据['年现金流出']
        self.综合数据['现金流比率'] = (self.综合数据['年现金流入'] / (self.综合数据['年现金流出'] + 1e-8)).clip(0, 10)
        
        # 业务活跃度
        self.综合数据['年发票总量'] = self.综合数据['进项发票数量'] + self.综合数据['销项发票数量']
        self.综合数据['业务活跃度评分'] = np.log1p(self.综合数据['年发票总量'])
        
        # 发票质量
        self.综合数据['年作废发票总量'] = self.综合数据['进项作废数量'] + self.综合数据['销项作废数量']
        self.综合数据['整体作废率'] = (self.综合数据['年作废发票总量'] / (self.综合数据['年发票总量'] + 1e-8)).clip(0, 1)
        
        # 企业规模
        self.综合数据['企业规模评分'] = np.log1p(self.综合数据['年营业收入'])
        
        # 信誉评级数值化
        rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        self.综合数据['信誉评级数值'] = self.综合数据['信誉评级'].map(rating_map)
        
        # 违约标识
        self.综合数据['历史违约'] = (self.综合数据['是否违约'] == '是').astype(int)
    
    def calculate_risk_score(self):
        """计算综合风险评分"""
        print("⚠️ 计算综合风险评分...")
        
        # 风险权重
        weights = {
            '信誉评级': 0.35,
            '历史违约': 0.25,
            '财务状况': 0.20,
            '发票质量': 0.15,
            '业务稳定': 0.05
        }
        
        # 1. 信誉评级风险
        信誉风险映射 = {'A': 0.1, 'B': 0.3, 'C': 0.6, 'D': 0.9}
        信誉风险得分 = self.综合数据['信誉评级'].map(信誉风险映射)
        
        # 2. 历史违约风险
        违约风险得分 = self.综合数据['历史违约'] * 0.8 + (1 - self.综合数据['历史违约']) * 0.1
        
        # 3. 财务状况风险（毛利率越低风险越高）
        毛利率_标准化 = (self.综合数据['毛利率'] - self.综合数据['毛利率'].min()) / (self.综合数据['毛利率'].max() - self.综合数据['毛利率'].min() + 1e-8)
        财务风险得分 = 1 - 毛利率_标准化.fillna(0.5)
        
        # 4. 发票质量风险
        发票风险得分 = self.综合数据['整体作废率']
        
        # 5. 业务稳定风险
        活跃度_标准化 = (self.综合数据['业务活跃度评分'] - self.综合数据['业务活跃度评分'].min()) / (self.综合数据['业务活跃度评分'].max() - self.综合数据['业务活跃度评分'].min() + 1e-8)
        稳定风险得分 = 1 - 活跃度_标准化.fillna(0.5)
        
        # 综合风险评分
        self.综合数据['综合风险评分'] = (
            weights['信誉评级'] * 信誉风险得分 +
            weights['历史违约'] * 违约风险得分 +
            weights['财务状况'] * 财务风险得分 +
            weights['发票质量'] * 发票风险得分 +
            weights['业务稳定'] * 稳定风险得分
        ).clip(0, 1)
        
        print("✅ 风险评分计算完成!")
    
    def optimize_credit_strategy(self, total_budget=10000):
        """制定符合银行约束的信贷策略"""
        print(f"\n🎯 制定信贷策略")
        print(f"   总预算: {total_budget}万元")
        print(f"   最小贷款额度: {self.最小贷款额度}万元")
        print(f"   利率范围: {self.最小利率:.1%} - {self.最大利率:.1%}")
        print(f"   贷款期限: {self.贷款期限}年")
        
        # 复制数据
        strategy_data = self.综合数据.copy().reset_index()
        
        # 制定每个企业的信贷策略
        strategy_results = []
        
        for idx, row in strategy_data.iterrows():
            企业代号 = row['企业代号']
            信誉评级 = row['信誉评级']
            历史违约 = row['历史违约']
            风险评分 = row['综合风险评分']
            
            # 制定决策
            decision = self._make_bank_credit_decision(信誉评级, 历史违约, 风险评分, row)
            
            strategy_results.append({
                '企业代号': 企业代号,
                '企业名称': row['企业名称'],
                '信誉评级': 信誉评级,
                '是否违约': '是' if 历史违约 else '否',
                '综合风险评分': 风险评分,
                '年营业收入(万元)': row['年营业收入'],
                '毛利率': row['毛利率'],
                '整体作废率': row['整体作废率'],
                '贷款额度(万元)': decision['贷款额度'],
                '年利率': decision['利率'],
                '预期年利息收入(万元)': decision['预期年利息收入'],
                '违约概率': decision['违约概率'],
                '预期年收益(万元)': decision['预期年收益'],
                '风险调整收益': decision['风险调整收益'],
                '决策依据': decision['决策依据']
            })
        
        # 转换为DataFrame
        self.信贷策略 = pd.DataFrame(strategy_results)
        
        # 调整额度以符合预算约束
        self._adjust_budget_constraint(total_budget)
        
        # 统计结果
        获贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] >= self.最小贷款额度]
        拒贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] < self.最小贷款额度]
        
        print(f"✅ 信贷策略制定完成!")
        print(f"   获贷企业: {len(获贷企业)}家")
        print(f"   拒贷企业: {len(拒贷企业)}家")
        print(f"   总放贷金额: {获贷企业['贷款额度(万元)'].sum():.1f}万元")
        
        return self.信贷策略
    
    def _make_bank_credit_decision(self, 信誉评级, 历史违约, 风险评分, enterprise_data):
        """基于银行约束制定单个企业的信贷决策"""
        
        年营业收入 = enterprise_data['年营业收入']
        毛利率 = enterprise_data['毛利率']
        作废率 = enterprise_data['整体作废率']
        
        # 基础决策框架
        if 信誉评级 == 'A':
            if not 历史违约:
                基础额度 = min(200, max(self.最小贷款额度, 年营业收入 * 0.3))
                基础利率 = 0.045 + 0.02 * 风险评分
                决策依据 = "A级优质客户"
            else:
                基础额度 = min(100, max(self.最小贷款额度, 年营业收入 * 0.2))
                基础利率 = 0.055 + 0.03 * 风险评分
                决策依据 = "A级违约客户谨慎放贷"
                
        elif 信誉评级 == 'B':
            if not 历史违约:
                基础额度 = min(150, max(self.最小贷款额度, 年营业收入 * 0.25))
                基础利率 = 0.055 + 0.03 * 风险评分
                决策依据 = "B级标准客户"
            else:
                基础额度 = min(80, max(self.最小贷款额度, 年营业收入 * 0.15))
                基础利率 = 0.070 + 0.04 * 风险评分
                决策依据 = "B级违约客户降额提率"
                
        elif 信誉评级 == 'C':
            if not 历史违约:
                基础额度 = min(80, max(self.最小贷款额度, 年营业收入 * 0.15))
                基础利率 = 0.080 + 0.05 * 风险评分
                决策依据 = "C级客户高风险高利率"
            else:
                基础额度 = min(30, max(self.最小贷款额度, 年营业收入 * 0.1))
                基础利率 = 0.100 + 0.05 * 风险评分
                决策依据 = "C级违约客户严格限制"
        else:  # D级
            基础额度 = 0
            基础利率 = 0
            决策依据 = "D级客户禁止放贷"
        
        # 财务状况调整
        if 基础额度 > 0:
            if 毛利率 > 0.3:
                基础额度 *= 1.2
                决策依据 += "+财务优良"
            elif 毛利率 < 0:
                基础额度 *= 0.7
                基础利率 += 0.01
                决策依据 += "+财务亏损"
            
            if 作废率 > 0.15:
                基础额度 *= 0.8
                基础利率 += 0.01
                决策依据 += "+发票质量差"
        
        # 应用银行约束
        最终额度 = max(0, 基础额度) if 基础额度 >= self.最小贷款额度 else 0
        最终利率 = max(self.最小利率, min(self.最大利率, 基础利率)) if 最终额度 > 0 else 0
        
        # 计算收益
        if 最终额度 > 0:
            年利息收入 = 最终额度 * 最终利率
            违约概率 = min(0.95, 风险评分 * 1.1)  # 校准违约概率
            预期年收益 = 年利息收入 * (1 - 违约概率)
            风险调整收益 = 预期年收益 / (最终额度 * 风险评分 + 0.01)
        else:
            年利息收入 = 0
            违约概率 = 0
            预期年收益 = 0
            风险调整收益 = 0
        
        return {
            '贷款额度': 最终额度,
            '利率': 最终利率,
            '预期年利息收入': 年利息收入,
            '违约概率': 违约概率,
            '预期年收益': 预期年收益,
            '风险调整收益': 风险调整收益,
            '决策依据': 决策依据
        }
    
    def _adjust_budget_constraint(self, total_budget):
        """调整贷款额度以符合预算约束"""
        
        获贷企业_mask = self.信贷策略['贷款额度(万元)'] >= self.最小贷款额度
        current_total = self.信贷策略.loc[获贷企业_mask, '贷款额度(万元)'].sum()
        
        if current_total > total_budget:
            # 超预算，按风险调整收益比例缩减
            print(f"⚖️ 预算调整: 当前总额{current_total:.1f}万元超出预算{total_budget}万元")
            
            获贷企业 = self.信贷策略[获贷企业_mask].copy()
            获贷企业 = 获贷企业.sort_values('风险调整收益', ascending=False)
            
            # 重新分配
            cumulative_amount = 0
            for idx, row in 获贷企业.iterrows():
                if cumulative_amount + row['贷款额度(万元)'] <= total_budget:
                    cumulative_amount += row['贷款额度(万元)']
                else:
                    # 剩余额度分配给最后一家企业
                    remaining = total_budget - cumulative_amount
                    if remaining >= self.最小贷款额度:
                        self.信贷策略.loc[idx, '贷款额度(万元)'] = remaining
                        cumulative_amount = total_budget
                    else:
                        self.信贷策略.loc[idx, '贷款额度(万元)'] = 0
                    break
            
            # 将超出预算的企业额度清零
            if cumulative_amount < total_budget:
                remaining_indices = 获贷企业[获贷企业.index > idx].index
                self.信贷策略.loc[remaining_indices, '贷款额度(万元)'] = 0
        
        # 重新计算收益指标
        self._recalculate_returns()
    
    def _recalculate_returns(self):
        """重新计算收益指标"""
        mask = self.信贷策略['贷款额度(万元)'] > 0
        
        self.信贷策略.loc[mask, '预期年利息收入(万元)'] = (
            self.信贷策略.loc[mask, '贷款额度(万元)'] * 
            self.信贷策略.loc[mask, '年利率']
        )
        
        self.信贷策略.loc[mask, '预期年收益(万元)'] = (
            self.信贷策略.loc[mask, '预期年利息收入(万元)'] * 
            (1 - self.信贷策略.loc[mask, '违约概率'])
        )
        
        self.信贷策略.loc[mask, '风险调整收益'] = (
            self.信贷策略.loc[mask, '预期年收益(万元)'] / 
            (self.信贷策略.loc[mask, '贷款额度(万元)'] * 
             self.信贷策略.loc[mask, '综合风险评分'] + 0.01)
        )
    
    def generate_bank_report(self):
        """生成银行信贷报告"""
        
        print("\n" + "="*100)
        print("🏦 银行中小微企业信贷策略分析报告")
        print("="*100)
        
        获贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] >= self.最小贷款额度]
        拒贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] < self.最小贷款额度]
        
        # 一、基本情况
        print(f"\n📊 一、贷款投放基本情况")
        print(f"   申请企业总数: {len(self.信贷策略)}家")
        print(f"   获贷企业数量: {len(获贷企业)}家 ({len(获贷企业)/len(self.信贷策略)*100:.1f}%)")
        print(f"   拒贷企业数量: {len(拒贷企业)}家 ({len(拒贷企业)/len(self.信贷策略)*100:.1f}%)")
        
        if len(获贷企业) > 0:
            总放贷金额 = 获贷企业['贷款额度(万元)'].sum()
            平均贷款额度 = 获贷企业['贷款额度(万元)'].mean()
            加权平均利率 = np.average(获贷企业['年利率'], weights=获贷企业['贷款额度(万元)'])
            
            print(f"   总放贷金额: {总放贷金额:,.1f}万元")
            print(f"   平均贷款额度: {平均贷款额度:.1f}万元")
            print(f"   加权平均年利率: {加权平均利率:.2%}")
        
        # 二、信誉评级分析
        print(f"\n⭐ 二、各信誉等级投放情况")
        for rating in ['A', 'B', 'C', 'D']:
            rating_data = self.信贷策略[self.信贷策略['信誉评级'] == rating]
            rating_loans = rating_data[rating_data['贷款额度(万元)'] >= self.最小贷款额度]
            
            if len(rating_data) > 0:
                获贷率 = len(rating_loans) / len(rating_data) * 100
                总额度 = rating_loans['贷款额度(万元)'].sum()
                平均利率 = rating_loans['年利率'].mean() if len(rating_loans) > 0 else 0
                
                print(f"   {rating}级: {len(rating_data)}家企业, "
                      f"获贷{len(rating_loans)}家({获贷率:.1f}%), "
                      f"总额度{总额度:.1f}万元, 平均利率{平均利率:.2%}")
        
        # 三、利率分布
        print(f"\n💰 三、利率分布情况")
        if len(获贷企业) > 0:
            利率区间 = [(0.04, 0.06), (0.06, 0.08), (0.08, 0.12), (0.12, 0.15)]
            区间名称 = ["4%-6%", "6%-8%", "8%-12%", "12%-15%"]
            
            for (min_rate, max_rate), 区间名 in zip(利率区间, 区间名称):
                区间企业 = 获贷企业[(获贷企业['年利率'] >= min_rate) & (获贷企业['年利率'] < max_rate)]
                if len(区间企业) > 0:
                    占比 = len(区间企业) / len(获贷企业) * 100
                    总额度 = 区间企业['贷款额度(万元)'].sum()
                    print(f"   {区间名}利率区间: {len(区间企业)}家企业({占比:.1f}%), "
                          f"放贷{总额度:.1f}万元")
        
        # 四、收益预测
        print(f"\n📈 四、预期收益分析")
        if len(获贷企业) > 0:
            预期总利息 = 获贷企业['预期年利息收入(万元)'].sum()
            预期总收益 = 获贷企业['预期年收益(万元)'].sum()
            投资回报率 = 预期总收益 / 获贷企业['贷款额度(万元)'].sum()
            
            print(f"   预期年利息收入: {预期总利息:.1f}万元")
            print(f"   预期年净收益: {预期总收益:.1f}万元")
            print(f"   投资回报率: {投资回报率:.2%}")
            print(f"   平均违约概率: {获贷企业['违约概率'].mean():.2%}")
        
        # 五、风险控制
        print(f"\n🛡️ 五、风险控制情况")
        if len(获贷企业) > 0:
            高风险企业 = 获贷企业[获贷企业['综合风险评分'] > 0.6]
            违约企业贷款 = 获贷企业[获贷企业['是否违约'] == '是']
            
            print(f"   高风险企业贷款: {len(高风险企业)}笔, "
                  f"金额{高风险企业['贷款额度(万元)'].sum():.1f}万元")
            print(f"   历史违约企业贷款: {len(违约企业贷款)}笔, "
                  f"金额{违约企业贷款['贷款额度(万元)'].sum():.1f}万元")
        
        # 六、贷款额度分布
        print(f"\n📊 六、贷款额度分布")
        if len(获贷企业) > 0:
            额度区间 = [(10, 50), (50, 100), (100, 150), (150, 1000)]
            区间名称 = ["10-50万", "50-100万", "100-150万", "150万以上"]
            
            for (min_amt, max_amt), 区间名 in zip(额度区间, 区间名称):
                if 区间名 == "150万以上":
                    区间企业 = 获贷企业[获贷企业['贷款额度(万元)'] >= min_amt]
                else:
                    区间企业 = 获贷企业[(获贷企业['贷款额度(万元)'] >= min_amt) & 
                                    (获贷企业['贷款额度(万元)'] < max_amt)]
                
                if len(区间企业) > 0:
                    占比 = len(区间企业) / len(获贷企业) * 100
                    总额度 = 区间企业['贷款额度(万元)'].sum()
                    print(f"   {区间名}元: {len(区间企业)}家企业({占比:.1f}%), "
                          f"总金额{总额度:.1f}万元")
        
        # 七、重点客户
        print(f"\n🏆 七、前10大贷款客户")
        if len(获贷企业) > 0:
            top_10 = 获贷企业.nlargest(10, '贷款额度(万元)')
            
            print(f"{'序号':<4} {'企业代号':<8} {'信誉':<4} {'额度(万)':<8} {'利率':<6} {'预期收益(万)':<10} {'决策依据'}")
            print("-" * 80)
            
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                print(f"{idx:<4} {row['企业代号']:<8} {row['信誉评级']:<4} "
                      f"{row['贷款额度(万元)']:<8.1f} {row['年利率']:<6.2%} "
                      f"{row['预期年收益(万元)']:<10.1f} {row['决策依据']}")
        
        print(f"\n" + "="*100)
        print(f"📋 银行信贷策略报告完成")
        print(f"="*100)
    
    def export_results(self):
        """导出结果"""
        try:
            # 导出完整策略
            self.信贷策略.to_excel('银行信贷策略_符合约束条件.xlsx', index=False)
            print(f"💾 完整信贷策略已保存: 银行信贷策略_符合约束条件.xlsx")
            
            # 导出获贷企业
            获贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] >= self.最小贷款额度]
            获贷企业.to_excel('获贷企业名单_银行版.xlsx', index=False)
            print(f"💾 获贷企业名单已保存: 获贷企业名单_银行版.xlsx")
            
            # 导出拒贷企业
            拒贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] < self.最小贷款额度]
            拒贷企业.to_excel('拒贷企业名单_银行版.xlsx', index=False)
            print(f"💾 拒贷企业名单已保存: 拒贷企业名单_银行版.xlsx")
            
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False

def main():
    """主函数"""
    print("🏦 银行信贷策略优化系统")
    print("符合银行实际约束条件的信贷决策模型")
    print("=" * 80)
    
    optimizer = BankCreditOptimizer()
    
    try:
        # 1. 加载数据
        if not optimizer.load_data():
            return
        
        # 2. 分析企业
        if not optimizer.analyze_enterprises():
            return
        
        # 3. 计算风险评分
        optimizer.calculate_risk_score()
        
        # 4. 制定信贷策略
        strategy = optimizer.optimize_credit_strategy(total_budget=10000)  # 1亿元预算
        
        # 5. 生成银行报告
        optimizer.generate_bank_report()
        
        # 6. 导出结果
        optimizer.export_results()
        
        print(f"\n🎉 银行信贷策略优化完成!")
        print(f"📄 所有结果已导出到Excel文件")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
