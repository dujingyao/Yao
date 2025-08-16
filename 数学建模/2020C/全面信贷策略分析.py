#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中小微企业信贷策略分析 - 基于决策示例的全面分析
根据典型决策示例制定覆盖123家企业的信贷策略

Author: 数据分析团队
Date: 2025年
"""

import pandas as pd
import numpy as np
import os

class ComprehensiveCreditAnalyzer:
    """全面信贷策略分析器"""
    
    def __init__(self):
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.综合数据 = None
        self.信贷策略 = None
        
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
        print("\n🔍 开始企业分析...")
        
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
        
        # 合并所有数据
        self.综合数据 = self.企业信息.set_index('企业代号').join([进项汇总, 销项汇总], how='left').fillna(0)
        
        # 计算关键指标
        self._calculate_key_metrics()
        
        print("✅ 企业分析完成!")
        return True
    
    def _calculate_key_metrics(self):
        """计算关键指标"""
        
        # 1. 财务健康度指标
        self.综合数据['营业收入'] = self.综合数据['销项净金额']
        self.综合数据['营业成本'] = self.综合数据['进项净金额']
        self.综合数据['毛利润'] = self.综合数据['营业收入'] - self.综合数据['营业成本']
        self.综合数据['毛利率'] = (self.综合数据['毛利润'] / (self.综合数据['营业收入'] + 1e-8)).clip(-1, 1)
        
        # 2. 现金流指标
        self.综合数据['现金流入'] = self.综合数据['销项总额']
        self.综合数据['现金流出'] = self.综合数据['进项总额']
        self.综合数据['净现金流'] = self.综合数据['现金流入'] - self.综合数据['现金流出']
        self.综合数据['现金流比率'] = (self.综合数据['现金流入'] / (self.综合数据['现金流出'] + 1e-8)).clip(0, 10)
        
        # 3. 业务活跃度
        self.综合数据['总发票数量'] = self.综合数据['进项发票数量'] + self.综合数据['销项发票数量']
        self.综合数据['业务活跃度评分'] = np.log1p(self.综合数据['总发票数量'])
        
        # 4. 发票质量评分
        self.综合数据['总作废数量'] = self.综合数据['进项作废数量'] + self.综合数据['销项作废数量']
        self.综合数据['整体作废率'] = (self.综合数据['总作废数量'] / (self.综合数据['总发票数量'] + 1e-8)).clip(0, 1)
        
        # 5. 业务规模评分
        self.综合数据['业务规模评分'] = np.log1p(self.综合数据['营业收入'])
        
        # 6. 信誉评级数值化
        rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        self.综合数据['信誉评级数值'] = self.综合数据['信誉评级'].map(rating_map)
        
        # 7. 违约标识
        self.综合数据['历史违约'] = (self.综合数据['是否违约'] == '是').astype(int)
    
    def calculate_comprehensive_risk_score(self):
        """计算综合风险评分"""
        print("⚠️ 计算综合风险评分...")
        
        # 各项风险权重（参考典型决策示例）
        risk_factors = {
            '信誉评级风险': 0.35,    # 35% - 最重要
            '历史违约风险': 0.25,    # 25% - 历史表现
            '财务状况风险': 0.20,    # 20% - 财务健康度
            '发票质量风险': 0.15,    # 15% - 业务规范性
            '业务稳定风险': 0.05     # 5% - 业务波动性
        }
        
        # 1. 信誉评级风险 (35%)
        信誉风险 = {'A': 0.1, 'B': 0.3, 'C': 0.6, 'D': 0.9}
        信誉评级风险得分 = self.综合数据['信誉评级'].map(信誉风险)
        
        # 2. 历史违约风险 (25%)
        历史违约风险得分 = self.综合数据['历史违约'] * 0.8 + (1 - self.综合数据['历史违约']) * 0.1
        
        # 3. 财务状况风险 (20%)
        # 毛利率越低风险越高
        毛利率_标准化 = (self.综合数据['毛利率'] - self.综合数据['毛利率'].min()) / (self.综合数据['毛利率'].max() - self.综合数据['毛利率'].min() + 1e-8)
        财务状况风险得分 = 1 - 毛利率_标准化.fillna(0.5)  # 反向，毛利率高=风险低
        
        # 4. 发票质量风险 (15%)
        # 作废率越高风险越高
        发票质量风险得分 = self.综合数据['整体作废率']
        
        # 5. 业务稳定风险 (5%)
        # 业务活跃度低=风险高
        活跃度_标准化 = (self.综合数据['业务活跃度评分'] - self.综合数据['业务活跃度评分'].min()) / (self.综合数据['业务活跃度评分'].max() - self.综合数据['业务活跃度评分'].min() + 1e-8)
        业务稳定风险得分 = 1 - 活跃度_标准化.fillna(0.5)
        
        # 综合风险评分
        self.综合数据['综合风险评分'] = (
            risk_factors['信誉评级风险'] * 信誉评级风险得分 +
            risk_factors['历史违约风险'] * 历史违约风险得分 +
            risk_factors['财务状况风险'] * 财务状况风险得分 +
            risk_factors['发票质量风险'] * 发票质量风险得分 +
            risk_factors['业务稳定风险'] * 业务稳定风险得分
        ).clip(0, 1)
        
        # 风险等级分类
        def classify_risk(score):
            if score <= 0.3:
                return '低风险'
            elif score <= 0.5:
                return '中低风险'
            elif score <= 0.7:
                return '中高风险'
            else:
                return '高风险'
        
        self.综合数据['风险等级'] = self.综合数据['综合风险评分'].apply(classify_risk)
        
        print("✅ 综合风险评分计算完成!")
    
    def develop_credit_strategy(self, total_budget=10000):
        """制定信贷策略（覆盖所有123家企业）"""
        print(f"\n🎯 制定全面信贷策略，总预算: {total_budget}万元")
        
        # 复制数据用于策略制定
        strategy_data = self.综合数据.copy().reset_index()
        
        # 基于典型决策示例的策略框架
        strategy_results = []
        
        for idx, row in strategy_data.iterrows():
            企业代号 = row['企业代号']
            信誉评级 = row['信誉评级']
            历史违约 = row['历史违约']
            风险评分 = row['综合风险评分']
            
            # 决策逻辑（基于提供的典型示例）
            decision = self._make_credit_decision(信誉评级, 历史违约, 风险评分, row)
            
            strategy_results.append({
                '企业代号': 企业代号,
                '企业名称': row['企业名称'],
                '信誉评级': 信誉评级,
                '是否违约': '是' if 历史违约 else '否',
                '综合风险评分': 风险评分,
                '风险等级': row['风险等级'],
                '贷款额度(万元)': decision['贷款额度'],
                '利率': decision['利率'],
                '决策依据': decision['决策依据'],
                '业务活跃度评分': row['业务活跃度评分'],
                '毛利率': row['毛利率'],
                '整体作废率': row['整体作废率'],
                '营业收入': row['营业收入'],
                '净现金流': row['净现金流']
            })
        
        # 转换为DataFrame
        self.信贷策略 = pd.DataFrame(strategy_results)
        
        # 调整贷款额度以符合总预算
        self._adjust_loan_amounts(total_budget)
        
        print(f"✅ 信贷策略制定完成!")
        print(f"   覆盖企业数量: {len(self.信贷策略)}家")
        print(f"   获贷企业数量: {len(self.信贷策略[self.信贷策略['贷款额度(万元)'] > 0])}家")
        
        return self.信贷策略
    
    def _make_credit_decision(self, 信誉评级, 历史违约, 风险评分, enterprise_data):
        """基于典型决策示例制定单个企业的信贷决策"""
        
        # 基础决策框架
        if 信誉评级 == 'A':
            if not 历史违约:
                # A级无违约：低风险高额度
                贷款额度 = min(200, max(50, enterprise_data['营业收入'] * 0.3))
                利率 = 0.045 + 0.02 * 风险评分  # 4.5%-6.5%
                决策依据 = "低风险高额度"
            else:
                # A级有违约：谨慎放贷
                贷款额度 = min(100, max(30, enterprise_data['营业收入'] * 0.2))
                利率 = 0.055 + 0.03 * 风险评分  # 5.5%-8.5%
                决策依据 = "A级违约企业谨慎放贷"
                
        elif 信誉评级 == 'B':
            if not 历史违约:
                # B级无违约：中等风险中等额度
                贷款额度 = min(150, max(40, enterprise_data['营业收入'] * 0.25))
                利率 = 0.055 + 0.03 * 风险评分  # 5.5%-8.5%
                决策依据 = "中等风险中等额度"
            else:
                # B级有违约：降低额度提高利率
                贷款额度 = min(80, max(20, enterprise_data['营业收入'] * 0.15))
                利率 = 0.070 + 0.04 * 风险评分  # 7%-11%
                决策依据 = "B级违约企业降额提率"
                
        elif 信誉评级 == 'C':
            if not 历史违约:
                # C级无违约：高风险低额度高利率
                贷款额度 = min(80, max(15, enterprise_data['营业收入'] * 0.15))
                利率 = 0.080 + 0.05 * 风险评分  # 8%-13%
                决策依据 = "高风险低额度高利率"
            else:
                # C级有违约：严格限制
                贷款额度 = min(30, max(10, enterprise_data['营业收入'] * 0.1))
                利率 = 0.100 + 0.05 * 风险评分  # 10%-15%
                决策依据 = "C级违约企业严格限制"
                
        else:  # D级
            # D级企业：原则上禁止放贷
            贷款额度 = 0
            利率 = 0
            决策依据 = "禁止放贷"
        
        # 特殊调整：基于财务状况微调
        if 贷款额度 > 0:
            # 财务状况好的企业可以适当增加额度
            if enterprise_data['毛利率'] > 0.3 and enterprise_data['净现金流'] > 0:
                贷款额度 *= 1.2
                决策依据 += "+财务优良"
            
            # 发票质量差的企业减少额度
            if enterprise_data['整体作废率'] > 0.15:
                贷款额度 *= 0.8
                利率 += 0.01
                决策依据 += "+发票质量差"
        
        # 确保利率在合理范围内
        利率 = max(0.04, min(0.18, 利率))
        
        return {
            '贷款额度': round(贷款额度, 1),
            '利率': 利率,
            '决策依据': 决策依据
        }
    
    def _adjust_loan_amounts(self, total_budget):
        """调整贷款额度以符合总预算"""
        
        # 计算当前总额度
        current_total = self.信贷策略[self.信贷策略['贷款额度(万元)'] > 0]['贷款额度(万元)'].sum()
        
        if current_total > total_budget:
            # 如果超预算，按比例缩减
            adjustment_factor = total_budget / current_total
            mask = self.信贷策略['贷款额度(万元)'] > 0
            self.信贷策略.loc[mask, '贷款额度(万元)'] *= adjustment_factor
            
            print(f"⚖️ 预算调整: 总额度从{current_total:.1f}万元调整为{total_budget}万元")
        
        # 四舍五入
        self.信贷策略['贷款额度(万元)'] = self.信贷策略['贷款额度(万元)'].round(1)
    
    def generate_comprehensive_report(self):
        """生成全面的分析报告"""
        
        print("\n" + "="*100)
        print("📊 中小微企业信贷策略全面分析报告")
        print("="*100)
        
        # 一、总体概况
        print(f"\n📋 一、总体概况")
        print(f"   分析企业总数: {len(self.综合数据)}家")
        print(f"   信贷策略覆盖: {len(self.信贷策略)}家")
        
        获贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] > 0]
        拒贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] == 0]
        
        print(f"   获得贷款企业: {len(获贷企业)}家 ({len(获贷企业)/len(self.信贷策略)*100:.1f}%)")
        print(f"   拒绝放贷企业: {len(拒贷企业)}家 ({len(拒贷企业)/len(self.信贷策略)*100:.1f}%)")
        
        # 二、信誉评级分析
        print(f"\n⭐ 二、信誉评级分布与策略")
        rating_analysis = self.信贷策略.groupby('信誉评级').agg({
            '企业代号': 'count',
            '贷款额度(万元)': ['sum', 'mean', 'count']
        }).round(2)
        
        rating_analysis.columns = ['企业数量', '总贷款额度', '平均贷款额度', '获贷数量']
        rating_analysis['获贷比例'] = (rating_analysis['获贷数量'] / rating_analysis['企业数量'] * 100).round(1)
        
        for rating, stats in rating_analysis.iterrows():
            print(f"   {rating}级: {stats['企业数量']}家企业, "
                  f"获贷{stats['获贷数量']}家({stats['获贷比例']:.1f}%), "
                  f"总额度{stats['总贷款额度']:.1f}万元")
        
        # 三、风险等级分析
        print(f"\n⚠️ 三、风险等级分布")
        risk_analysis = self.信贷策略.groupby('风险等级').agg({
            '企业代号': 'count',
            '贷款额度(万元)': 'sum'
        }).round(1)
        
        for risk_level, stats in risk_analysis.iterrows():
            pct = stats['企业代号'] / len(self.信贷策略) * 100
            print(f"   {risk_level}: {stats['企业代号']}家({pct:.1f}%), "
                  f"贷款额度{stats['贷款额度(万元)']}万元")
        
        # 四、财务健康度分析
        print(f"\n💰 四、财务健康度分析")
        if len(获贷企业) > 0:
            print(f"   获贷企业平均毛利率: {获贷企业['毛利率'].mean()*100:.2f}%")
            print(f"   获贷企业平均营业收入: {获贷企业['营业收入'].mean():,.0f}元")
            print(f"   获贷企业平均净现金流: {获贷企业['净现金流'].mean():,.0f}元")
            print(f"   获贷企业平均作废率: {获贷企业['整体作废率'].mean()*100:.2f}%")
        
        # 五、贷款策略统计
        print(f"\n📈 五、贷款策略统计")
        if len(获贷企业) > 0:
            total_loans = 获贷企业['贷款额度(万元)'].sum()
            avg_loan = 获贷企业['贷款额度(万元)'].mean()
            avg_rate = np.average(获贷企业['利率'], weights=获贷企业['贷款额度(万元)'])
            
            print(f"   总放贷金额: {total_loans:,.1f}万元")
            print(f"   平均单笔贷款: {avg_loan:.1f}万元")
            print(f"   加权平均利率: {avg_rate:.2%}")
            print(f"   最大单笔贷款: {获贷企业['贷款额度(万元)'].max():.1f}万元")
            print(f"   最小单笔贷款: {获贷企业[获贷企业['贷款额度(万元)'] > 0]['贷款额度(万元)'].min():.1f}万元")
        
        # 六、风险控制措施
        print(f"\n🛡️ 六、风险控制措施")
        
        # 高风险企业统计
        high_risk_loans = 获贷企业[获贷企业['风险等级'].isin(['中高风险', '高风险'])]
        if len(high_risk_loans) > 0:
            high_risk_amount = high_risk_loans['贷款额度(万元)'].sum()
            high_risk_ratio = high_risk_amount / 获贷企业['贷款额度(万元)'].sum()
            print(f"   高风险贷款: {len(high_risk_loans)}笔, "
                  f"金额{high_risk_amount:.1f}万元({high_risk_ratio:.1%})")
        
        # 违约企业统计
        default_loans = 获贷企业[获贷企业['是否违约'] == '是']
        if len(default_loans) > 0:
            default_amount = default_loans['贷款额度(万元)'].sum()
            print(f"   违约企业贷款: {len(default_loans)}笔, "
                  f"金额{default_amount:.1f}万元")
        
        # 七、决策依据统计
        print(f"\n📊 七、决策依据分布")
        decision_stats = self.信贷策略['决策依据'].value_counts()
        for decision, count in decision_stats.head(10).items():
            pct = count / len(self.信贷策略) * 100
            print(f"   {decision}: {count}家({pct:.1f}%)")
        
        # 八、典型案例展示
        print(f"\n🏆 八、典型案例展示（前10大贷款企业）")
        top_enterprises = 获贷企业.nlargest(10, '贷款额度(万元)')
        
        print(f"{'序号':<4} {'企业代号':<8} {'信誉评级':<6} {'违约':<4} {'额度(万元)':<10} {'利率':<6} {'决策依据':<20}")
        print("-" * 80)
        
        for idx, (_, row) in enumerate(top_enterprises.iterrows(), 1):
            print(f"{idx:<4} {row['企业代号']:<8} {row['信誉评级']:<6} "
                  f"{row['是否违约']:<4} {row['贷款额度(万元)']:<10.1f} "
                  f"{row['利率']:<6.2%} {row['决策依据']:<20}")
        
        print(f"\n" + "="*100)
        print(f"📋 报告生成完成 - 涵盖全部123家企业的信贷策略")
        print(f"="*100)
    
    def export_results(self):
        """导出分析结果"""
        try:
            # 导出综合企业数据
            self.综合数据.reset_index().to_excel('企业全面分析数据.xlsx', index=False)
            print(f"💾 企业全面分析数据已保存至: 企业全面分析数据.xlsx")
            
            # 导出信贷策略（123家企业完整版）
            if self.信贷策略 is not None:
                self.信贷策略.to_excel('信贷策略完整版_123家企业.xlsx', index=False)
                print(f"💾 完整信贷策略已保存至: 信贷策略完整版_123家企业.xlsx")
                
                # 单独导出获贷企业清单
                获贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] > 0]
                获贷企业.to_excel('获贷企业清单.xlsx', index=False)
                print(f"💾 获贷企业清单已保存至: 获贷企业清单.xlsx")
                
                # 单独导出拒贷企业清单
                拒贷企业 = self.信贷策略[self.信贷策略['贷款额度(万元)'] == 0]
                拒贷企业.to_excel('拒贷企业清单.xlsx', index=False)
                print(f"💾 拒贷企业清单已保存至: 拒贷企业清单.xlsx")
            
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False

def main():
    """主函数"""
    print("🏦 中小微企业信贷策略全面分析系统")
    print("基于典型决策示例的123家企业完整信贷策略")
    print("=" * 80)
    
    # 创建分析器
    analyzer = ComprehensiveCreditAnalyzer()
    
    try:
        # 1. 加载数据
        if not analyzer.load_data():
            return
        
        # 2. 分析企业
        if not analyzer.analyze_enterprises():
            return
        
        # 3. 计算风险评分
        analyzer.calculate_comprehensive_risk_score()
        
        # 4. 制定信贷策略（覆盖全部123家企业）
        strategy = analyzer.develop_credit_strategy(total_budget=10000)  # 总预算1亿元
        
        # 5. 生成全面报告
        analyzer.generate_comprehensive_report()
        
        # 6. 导出结果
        analyzer.export_results()
        
        print(f"\n🎉 全面分析完成!")
        print(f"✅ 已生成覆盖全部123家企业的信贷策略")
        print(f"📄 请查看导出的Excel文件获取详细信息")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
