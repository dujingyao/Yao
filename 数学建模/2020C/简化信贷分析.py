#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中小微企业信贷策略分析 - 基于真实数据
简化版本，直接处理多工作表Excel数据

Author: 数据分析团队
Date: 2025年
"""

import pandas as pd
import numpy as np
import os
import sys

# 尝试导入机器学习库，如果失败则提供简化分析
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
    print("✅ 机器学习库可用")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ 机器学习库不可用，将使用简化分析")

class SimpleCreditAnalyzer:
    """简化版信贷分析器"""
    
    def __init__(self):
        self.企业信息 = None
        self.进项发票 = None
        self.销项发票 = None
        self.综合数据 = None
        
    def load_data(self):
        """加载Excel文件的所有工作表"""
        
        # 尝试多个可能的文件路径
        possible_paths = [
            "附件1：123家有信贷记录企业的相关数据.xlsx",
            "/home/yao/Yao/数学建模/2020C/附件1：123家有信贷记录企业的相关数据.xlsx",
            os.path.join(os.path.dirname(__file__), "附件1：123家有信贷记录企业的相关数据.xlsx"),
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            print("❌ 无法找到数据文件")
            print("请确保以下文件存在:")
            for path in possible_paths:
                print(f"   {path}")
            return False
        
        try:
            print(f"📁 读取文件: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            
            print(f"📊 工作表: {excel_file.sheet_names}")
            
            # 读取各个工作表
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
    
    def process_data(self):
        """处理和分析数据"""
        print("\n🔍 开始数据处理和分析...")
        
        # 分析进项发票
        print("📋 分析进项发票数据...")
        进项汇总 = self.进项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x == '作废发票').sum()
        }).round(2)
        
        进项汇总.columns = ['进项总额', '进项发票数量', '进项平均金额', '进项金额标准差', '进项净金额', '进项税额', '进项作废数量']
        进项汇总['进项作废率'] = (进项汇总['进项作废数量'] / 进项汇总['进项发票数量']).fillna(0)
        
        # 分析销项发票
        print("📋 分析销项发票数据...")
        销项汇总 = self.销项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x.str.strip() == '作废发票').sum()
        }).round(2)
        
        销项汇总.columns = ['销项总额', '销项发票数量', '销项平均金额', '销项金额标准差', '销项净金额', '销项税额', '销项作废数量']
        销项汇总['销项作废率'] = (销项汇总['销项作废数量'] / 销项汇总['销项发票数量']).fillna(0)
        
        # 合并数据
        self.综合数据 = self.企业信息.set_index('企业代号').join([进项汇总, 销项汇总], how='left').fillna(0)
        
        # 计算关键财务指标
        self._calculate_financial_metrics()
        
        print("✅ 数据处理完成!")
        return True
    
    def _calculate_financial_metrics(self):
        """计算财务和风险指标"""
        print("📊 计算财务指标...")
        
        # 1. 营业状况
        self.综合数据['毛利润'] = self.综合数据['销项净金额'] - self.综合数据['进项净金额']
        self.综合数据['毛利率'] = (self.综合数据['毛利润'] / (self.综合数据['销项净金额'] + 1e-8)).clip(-1, 1)
        
        # 2. 现金流
        self.综合数据['净现金流'] = self.综合数据['销项总额'] - self.综合数据['进项总额']
        self.综合数据['现金流比率'] = (self.综合数据['销项总额'] / (self.综合数据['进项总额'] + 1e-8)).clip(0, 10)
        
        # 3. 业务活跃度
        self.综合数据['总发票数量'] = self.综合数据['进项发票数量'] + self.综合数据['销项发票数量']
        self.综合数据['业务活跃度'] = np.log1p(self.综合数据['总发票数量'])
        
        # 4. 发票质量
        self.综合数据['总作废数量'] = self.综合数据['进项作废数量'] + self.综合数据['销项作废数量']
        self.综合数据['整体作废率'] = (self.综合数据['总作废数量'] / (self.综合数据['总发票数量'] + 1e-8)).clip(0, 1)
        
        # 5. 业务规模
        self.综合数据['业务总规模'] = np.log1p(self.综合数据['销项总额'] + self.综合数据['进项总额'])
        self.综合数据['销售规模'] = np.log1p(self.综合数据['销项总额'])
        
        # 6. 信誉评级数值化
        rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        self.综合数据['信誉评级数值'] = self.综合数据['信誉评级'].map(rating_map)
        
        # 7. 违约标签
        self.综合数据['违约标签'] = (self.综合数据['是否违约'] == '是').astype(int)
        
        # 8. 计算风险评分（基于规则的方法）
        self._calculate_rule_based_risk_score()
    
    def _calculate_rule_based_risk_score(self):
        """基于规则计算风险评分"""
        print("⚠️ 计算风险评分...")
        
        # 初始化风险评分
        risk_score = np.zeros(len(self.综合数据))
        
        # 1. 信誉评级权重 (40%)
        rating_risk = {'A': 0.1, 'B': 0.3, 'C': 0.6, 'D': 0.9}
        for rating, score in rating_risk.items():
            mask = self.综合数据['信誉评级'] == rating
            risk_score[mask] += 0.4 * score
        
        # 2. 违约历史权重 (30%)
        risk_score += 0.3 * self.综合数据['违约标签']
        
        # 3. 发票质量权重 (20%)
        # 作废率越高，风险越大
        normalized_废票率 = (self.综合数据['整体作废率'] - self.综合数据['整体作废率'].min()) / (self.综合数据['整体作废率'].max() - self.综合数据['整体作废率'].min() + 1e-8)
        risk_score += 0.2 * normalized_废票率
        
        # 4. 财务状况权重 (10%)
        # 毛利率越低，风险越大（反向）
        normalized_毛利率 = (self.综合数据['毛利率'] - self.综合数据['毛利率'].min()) / (self.综合数据['毛利率'].max() - self.综合数据['毛利率'].min() + 1e-8)
        risk_score += 0.1 * (1 - normalized_毛利率)
        
        # 限制风险评分在0-1之间
        self.综合数据['风险评分'] = np.clip(risk_score, 0, 1)
        
        # 风险等级分类
        self.综合数据['风险等级'] = pd.cut(
            self.综合数据['风险评分'], 
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['低风险', '中低风险', '中高风险', '高风险']
        )
    
    def optimize_credit_strategy(self, total_amount=5000):
        """优化信贷策略"""
        print(f"\n🎯 制定信贷策略，总额度: {total_amount}万元")
        
        # 筛选可放贷企业（排除D级信誉企业）
        eligible = self.综合数据[self.综合数据['信誉评级'] != 'D'].copy()
        print(f"📊 符合条件企业: {len(eligible)}家 (排除{len(self.综合数据) - len(eligible)}家D级企业)")
        
        # 计算预期收益
        self._calculate_expected_returns(eligible)
        
        # 按风险调整收益排序
        eligible = eligible.sort_values('风险调整收益', ascending=False)
        
        # 分配策略：优先考虑低风险高收益企业
        max_enterprises = min(50, len(eligible))  # 最多50家企业
        selected = eligible.head(max_enterprises).copy()
        
        # 根据风险调整收益分配权重
        total_rar = selected['风险调整收益'].sum()
        selected['分配权重'] = selected['风险调整收益'] / total_rar
        selected['建议贷款金额'] = selected['分配权重'] * total_amount
        
        # 设置贷款限额
        selected['建议贷款金额'] = np.clip(selected['建议贷款金额'], 10, 500)  # 10万-500万
        
        # 重新调整确保总额不超标
        actual_total = selected['建议贷款金额'].sum()
        if actual_total > total_amount:
            selected['建议贷款金额'] *= total_amount / actual_total
        
        # 最终筛选
        final_strategy = selected[selected['建议贷款金额'] >= 10].copy()
        
        print(f"✅ 策略制定完成! 共{len(final_strategy)}家企业获得贷款")
        
        return final_strategy
    
    def _calculate_expected_returns(self, data):
        """计算预期收益"""
        # 基于信誉评级和风险评分确定利率
        base_rates = {'A': 0.045, 'B': 0.055, 'C': 0.070}
        
        data['建议利率'] = data['信誉评级'].map(base_rates).fillna(0.08)
        data['建议利率'] += 0.05 * data['风险评分']  # 风险溢价
        data['建议利率'] = data['建议利率'].clip(0.04, 0.15)
        
        # 违约概率
        data['违约概率'] = data['风险评分']
        
        # 预期收益率
        data['预期收益率'] = data['建议利率'] * (1 - data['违约概率'])
        
        # 风险调整收益
        data['风险调整收益'] = data['预期收益率'] / (data['风险评分'] + 0.01)
    
    def generate_report(self, strategy=None):
        """生成分析报告"""
        print("\n" + "="*80)
        print("📊 中小微企业信贷策略分析报告")
        print("="*80)
        
        # 基本统计
        print(f"\n📈 一、数据概览")
        print(f"   企业总数: {len(self.综合数据)}家")
        print(f"   进项发票总数: {len(self.进项发票):,}张")
        print(f"   销项发票总数: {len(self.销项发票):,}张")
        
        # 信誉评级分布
        print(f"\n⭐ 二、信誉评级分布")
        rating_dist = self.综合数据['信誉评级'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            pct = count / len(self.综合数据) * 100
            print(f"   {rating}级: {count:2d}家 ({pct:5.1f}%)")
        
        # 违约情况
        print(f"\n⚠️ 三、违约情况分析")
        default_dist = self.综合数据['是否违约'].value_counts()
        for status, count in default_dist.items():
            pct = count / len(self.综合数据) * 100
            print(f"   {status}: {count:2d}家 ({pct:5.1f}%)")
        
        # 各等级违约率
        print(f"\n📊 四、各信誉等级违约率")
        default_by_rating = self.综合数据.groupby('信誉评级')['违约标签'].agg(['count', 'sum'])
        default_by_rating['违约率'] = (default_by_rating['sum'] / default_by_rating['count'] * 100).round(1)
        for rating, stats in default_by_rating.iterrows():
            print(f"   {rating}级: {stats['sum']:2.0f}/{stats['count']:2.0f} = {stats['违约率']:5.1f}%")
        
        # 财务指标统计
        print(f"\n💰 五、财务指标统计")
        key_metrics = {
            '平均毛利率': self.综合数据['毛利率'].mean() * 100,
            '平均整体作废率': self.综合数据['整体作废率'].mean() * 100,
            '平均业务活跃度': self.综合数据['业务活跃度'].mean(),
            '平均风险评分': self.综合数据['风险评分'].mean()
        }
        
        for metric, value in key_metrics.items():
            if '率' in metric:
                print(f"   {metric}: {value:.2f}%")
            else:
                print(f"   {metric}: {value:.3f}")
        
        # 风险等级分布
        print(f"\n🎯 六、风险等级分布")
        risk_dist = self.综合数据['风险等级'].value_counts()
        for level, count in risk_dist.items():
            pct = count / len(self.综合数据) * 100
            print(f"   {level}: {count}家 ({pct:.1f}%)")
        
        # 信贷策略分析
        if strategy is not None:
            print(f"\n💳 七、信贷策略分析")
            
            total_loans = strategy['建议贷款金额'].sum()
            total_enterprises = len(strategy)
            avg_loan = strategy['建议贷款金额'].mean()
            avg_rate = np.average(strategy['建议利率'], weights=strategy['建议贷款金额'])
            avg_risk = np.average(strategy['风险评分'], weights=strategy['建议贷款金额'])
            
            print(f"   获贷企业数量: {total_enterprises}家")
            print(f"   总放贷金额: {total_loans:.1f}万元")
            print(f"   平均单笔贷款: {avg_loan:.1f}万元")
            print(f"   加权平均利率: {avg_rate:.2%}")
            print(f"   组合平均风险: {avg_risk:.3f}")
            
            # 预期收益
            expected_return = (strategy['建议贷款金额'] * strategy['预期收益率']).sum()
            roi = expected_return / total_loans
            print(f"   预期年收益: {expected_return:.1f}万元")
            print(f"   预期收益率: {roi:.2%}")
            
            # 风险控制
            high_risk_count = len(strategy[strategy['风险评分'] > 0.6])
            high_risk_amount = strategy[strategy['风险评分'] > 0.6]['建议贷款金额'].sum()
            
            print(f"\n🛡️ 八、风险控制分析")
            print(f"   高风险企业: {high_risk_count}家")
            print(f"   高风险金额: {high_risk_amount:.1f}万元 ({high_risk_amount/total_loans:.1%})")
            
            # 前10大贷款企业
            print(f"\n🏆 九、前10大贷款企业")
            top_10 = strategy.nlargest(10, '建议贷款金额')
            for idx, (企业代号, row) in enumerate(top_10.iterrows(), 1):
                print(f"   {idx:2d}. {企业代号}: {row['建议贷款金额']:.1f}万元 "
                      f"(利率{row['建议利率']:.2%}, 风险{row['风险评分']:.3f}, {row['信誉评级']}级)")
        
        print(f"\n" + "="*80)
        print(f"📋 报告生成完成")
        print(f"="*80)

def main():
    """主函数"""
    print("🏦 中小微企业信贷策略分析系统 (简化版)")
    print("基于真实发票数据的风险评估与策略制定")
    print("=" * 60)
    
    # 创建分析器
    analyzer = SimpleCreditAnalyzer()
    
    try:
        # 1. 加载数据
        if not analyzer.load_data():
            print("❌ 数据加载失败，程序退出")
            return
        
        # 2. 处理数据
        if not analyzer.process_data():
            print("❌ 数据处理失败，程序退出")
            return
        
        # 3. 制定信贷策略
        strategy = analyzer.optimize_credit_strategy(total_amount=5000)
        
        # 4. 生成报告
        analyzer.generate_report(strategy)
        
        # 5. 导出结果
        try:
            # 导出综合数据
            analyzer.综合数据.reset_index().to_excel('企业综合分析数据.xlsx', index=False)
            print(f"\n💾 企业综合数据已保存至: 企业综合分析数据.xlsx")
            
            # 导出策略结果
            if strategy is not None:
                strategy.reset_index().to_excel('信贷策略建议.xlsx', index=False)
                print(f"💾 信贷策略已保存至: 信贷策略建议.xlsx")
            
        except Exception as e:
            print(f"⚠️ 文件保存失败: {e}")
        
        print(f"\n🎉 分析完成!")
        print(f"📋 请参考以上报告制定具体的信贷投放策略")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
