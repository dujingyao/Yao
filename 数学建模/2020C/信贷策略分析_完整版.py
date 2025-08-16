# -*- coding: utf-8 -*-
"""
中小微企业信贷策略分析 - 问题1完整版
基于真实数据的信贷风险量化分析与策略制定

Author: 数据分析团队
Date: 2025年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CreditRiskAnalyzer:
    """信贷风险分析器"""
    
    def __init__(self, data_file=None):
        # 自动检测数据文件路径
        if data_file is None:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_file = os.path.join(current_dir, "附件1：123家有信贷记录企业的相关数据.xlsx")
        else:
            self.data_file = data_file
        self.data = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.credit_strategy = None
        
    def load_data(self):
        """
        加载企业数据
        """
        try:
            import os
            print(f"📁 当前工作目录: {os.getcwd()}")
            print(f"📁 数据文件路径: {self.data_file}")
            print(f"📁 文件是否存在: {os.path.exists(self.data_file)}")
            
            # 如果文件不存在，尝试多个可能的路径
            if not os.path.exists(self.data_file):
                possible_paths = [
                    "/home/yao/Yao/数学建模/2020C/附件1：123家有信贷记录企业的相关数据.xlsx",
                    "/home/yao/数学建模/2020C/附件1：123家有信贷记录企业的相关数据.xlsx",
                    "附件1：123家有信贷记录企业的相关数据.xlsx"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.data_file = path
                        print(f"✅ 找到数据文件: {path}")
                        break
                else:
                    raise FileNotFoundError("无法找到数据文件，请检查文件路径")
            
            # 加载Excel文件
            self.data = pd.read_excel(self.data_file)
            print(f"✅ 成功加载数据: {self.data.shape}")
            print(f"📊 数据列名: {list(self.data.columns)}")
            
            # 显示数据基本信息
            print(f"\n📈 数据概览:")
            print(self.data.info())
            print(f"\n📋 前5行数据:")
            print(self.data.head())
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def preprocess_data(self):
        """
        数据预处理和特征工程
        """
        print("🔧 开始数据预处理...")
        
        if self.data is None:
            print("❌ 请先加载数据!")
            return False
        
        # 数据清洗：处理缺失值
        print(f"📊 缺失值统计:")
        missing_info = self.data.isnull().sum()
        for col, missing_count in missing_info[missing_info > 0].items():
            print(f"   {col}: {missing_count}个缺失值")
        
        # 填充缺失值
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # 特征工程
        self._create_risk_features()
        
        # 编码分类变量
        self._encode_categorical_features()
        
        print("✅ 数据预处理完成!")
        return True
    
    def _create_risk_features(self):
        """
        创建风险评估特征
        """
        print("🎯 创建风险评估特征...")
        
        # 获取实际的列名（可能与预期不同）
        columns = self.data.columns.tolist()
        print(f"实际列名: {columns}")
        
        # 根据实际列名进行特征工程
        try:
            # 1. 发票相关特征（假设有发票相关列）
            invoice_cols = [col for col in columns if '发票' in col or 'invoice' in col.lower()]
            if invoice_cols:
                print(f"发现发票相关列: {invoice_cols}")
                
                # 发票质量指标
                if any('有效' in col for col in invoice_cols):
                    effective_col = next(col for col in invoice_cols if '有效' in col)
                    total_col = next((col for col in invoice_cols if '总' in col or '数量' in col), None)
                    if total_col:
                        self.data['发票有效率'] = self.data[effective_col] / (self.data[total_col] + 1e-8)
                
                # 作废发票率
                invalid_cols = [col for col in invoice_cols if '作废' in col or '负数' in col]
                if invalid_cols:
                    total_invalid = self.data[invalid_cols].sum(axis=1)
                    total_col = next((col for col in invoice_cols if '总' in col), None)
                    if total_col:
                        self.data['异常发票率'] = total_invalid / (self.data[total_col] + 1e-8)
            
            # 2. 财务相关特征
            financial_cols = [col for col in columns if any(keyword in col for keyword in ['金额', '资本', '收入', '销项', '进项'])]
            if financial_cols:
                print(f"发现财务相关列: {financial_cols}")
                
                # 营业利润率
                sales_col = next((col for col in financial_cols if '销项' in col or '销售' in col), None)
                purchase_col = next((col for col in financial_cols if '进项' in col or '采购' in col), None)
                if sales_col and purchase_col:
                    self.data['营业利润率'] = (self.data[sales_col] - self.data[purchase_col]) / (self.data[sales_col] + 1e-8)
                
                # 企业规模（基于注册资本）
                capital_col = next((col for col in financial_cols if '资本' in col), None)
                if capital_col:
                    self.data['企业规模'] = np.log1p(self.data[capital_col])
            
            # 3. 供应链相关特征
            supply_cols = [col for col in columns if any(keyword in col for keyword in ['上游', '下游', '供应商', '客户'])]
            if supply_cols:
                print(f"发现供应链相关列: {supply_cols}")
                
                upstream_col = next((col for col in supply_cols if '上游' in col), None)
                downstream_col = next((col for col in supply_cols if '下游' in col), None)
                if upstream_col and downstream_col:
                    self.data['供应链稳定性'] = np.log1p(self.data[upstream_col] + self.data[downstream_col])
                    self.data['客户集中度'] = 1 / (self.data[downstream_col] + 1)
            
            # 4. 信誉评级数值化
            rating_col = next((col for col in columns if '信誉' in col or '评级' in col or 'rating' in col.lower()), None)
            if rating_col:
                print(f"发现信誉评级列: {rating_col}")
                rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
                self.data['信誉评级_数值'] = self.data[rating_col].map(rating_map)
                
        except Exception as e:
            print(f"⚠️ 特征工程部分失败: {e}")
    
    def _encode_categorical_features(self):
        """
        编码分类特征
        """
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['企业ID', '企业名称'] and '编码' not in col:  # 排除标识符列和已编码列
                try:
                    le = LabelEncoder()
                    self.data[f'{col}_编码'] = le.fit_transform(self.data[col].astype(str))
                except:
                    pass
    
    def build_risk_model(self):
        """
        构建信贷风险评估模型
        """
        print("🤖 构建风险评估模型...")
        
        # 自动选择数值特征
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除ID和标识符列
        exclude_patterns = ['ID', 'id', '编号', 'unnamed']
        feature_columns = [col for col in numeric_features 
                          if not any(pattern in col for pattern in exclude_patterns)]
        
        print(f"📊 选择的特征列: {feature_columns}")
        
        if len(feature_columns) == 0:
            print("❌ 没有可用的数值特征!")
            return False
        
        # 准备特征数据
        X = self.data[feature_columns].fillna(0)
        
        # 创建风险标签
        y = self._create_risk_labels()
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        self.risk_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # 如果有足够的样本，进行交叉验证
        if len(X) > 10:
            cv_scores = cross_val_score(self.risk_model, X_scaled, y, cv=min(5, len(X)//2), scoring='roc_auc')
            print(f"📈 模型交叉验证AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 训练模型
        self.risk_model.fit(X_scaled, y)
        
        # 计算风险评分
        risk_probs = self.risk_model.predict_proba(X_scaled)[:, 1]
        self.data['风险评分'] = risk_probs
        
        # 风险等级分类
        self.data['风险等级'] = pd.cut(
            risk_probs, 
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['低风险', '中低风险', '中高风险', '高风险']
        )
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.risk_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 前10个重要特征:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        print("✅ 风险模型构建完成!")
        return True
    
    def _create_risk_labels(self):
        """
        创建风险标签
        """
        # 寻找信誉评级列
        rating_cols = [col for col in self.data.columns if '信誉' in col or '评级' in col or 'rating' in col.lower()]
        
        if rating_cols:
            rating_col = rating_cols[0]
            print(f"📊 基于{rating_col}创建风险标签")
            
            # 基于信誉评级创建风险标签
            risk_labels = []
            for rating in self.data[rating_col].fillna('C'):
                if str(rating).upper() in ['D', '4']:
                    risk_labels.append(1)  # 高风险
                elif str(rating).upper() in ['A', '1', 'AA', 'AAA']:
                    risk_labels.append(0)  # 低风险
                elif str(rating).upper() in ['B', '2']:
                    risk_labels.append(0)  # 低风险
                else:  # C级或其他
                    risk_labels.append(np.random.choice([0, 1], p=[0.7, 0.3]))  # 70%低风险
                    
            return np.array(risk_labels)
        else:
            print("⚠️ 未找到信誉评级列，使用模拟风险标签")
            # 使用模拟标签：30%高风险
            return np.random.binomial(1, 0.3, len(self.data))
    
    def optimize_credit_strategy(self, total_credit_amount=5000):
        """
        优化信贷策略
        
        参数:
            total_credit_amount: 年度信贷总额（万元）
        """
        print(f"🎯 优化信贷策略，总额度: {total_credit_amount}万元")
        
        if self.data is None or '风险评分' not in self.data.columns:
            print("❌ 请先完成风险评估!")
            return False
        
        # 筛选可放贷企业（排除D级信誉企业）
        # 寻找信誉评级列
        rating_cols = [col for col in self.data.columns if '信誉' in col or '评级' in col]
        
        if rating_cols:
            rating_col = rating_cols[0]
            eligible_enterprises = self.data[self.data[rating_col] != 'D'].copy()
        else:
            # 如果没有信誉评级列，排除高风险企业
            eligible_enterprises = self.data[self.data['风险评分'] < 0.8].copy()
        
        if len(eligible_enterprises) == 0:
            print("❌ 没有符合放贷条件的企业!")
            return False
        
        print(f"📊 符合条件的企业: {len(eligible_enterprises)}家")
        
        # 计算每个企业的预期收益和风险
        self._calculate_expected_returns(eligible_enterprises)
        
        # 优化策略
        optimal_strategy = self._solve_optimization_problem(eligible_enterprises, total_credit_amount)
        
        self.credit_strategy = optimal_strategy
        
        print("✅ 信贷策略优化完成!")
        return True
    
    def _calculate_expected_returns(self, data):
        """
        计算预期收益和风险
        """
        # 基于风险评分确定利率（4%-15%区间）
        def calculate_interest_rate(risk_score):
            base_rate = 0.04  # 基础利率4%
            risk_premium = 0.11 * risk_score  # 风险溢价最高11%
            return base_rate + risk_premium
        
        data['建议利率'] = data['风险评分'].apply(calculate_interest_rate)
        
        # 违约概率与风险评分正相关
        data['违约概率'] = data['风险评分']
        
        # 预期收益（考虑违约损失）
        loan_amount = 100  # 假设基准贷款金额100万元
        data['预期收益'] = loan_amount * data['建议利率'] * (1 - data['违约概率'])
        
        # 风险调整收益（收益/风险比）
        data['风险调整收益'] = data['预期收益'] / (data['风险评分'] + 0.01)
    
    def _solve_optimization_problem(self, data, total_amount):
        """
        求解信贷配置优化问题
        """
        print("🧮 求解优化配置...")
        
        # 简化策略：按风险调整收益排序，分配资金
        data_sorted = data.sort_values('风险调整收益', ascending=False).copy()
        
        # 动态分配策略
        n_enterprises = len(data_sorted)
        max_loans = min(50, n_enterprises)  # 最多50家企业获得贷款
        
        # 根据风险调整收益分配权重
        top_enterprises = data_sorted.head(max_loans).copy()
        
        # 计算分配权重（基于风险调整收益）
        total_rar = top_enterprises['风险调整收益'].sum()
        top_enterprises['分配权重'] = top_enterprises['风险调整收益'] / total_rar
        
        # 分配贷款金额
        top_enterprises['建议贷款金额'] = top_enterprises['分配权重'] * total_amount
        
        # 设定最小和最大贷款限额
        min_loan = 10  # 最小10万元
        max_loan = 500  # 最大500万元
        
        top_enterprises['建议贷款金额'] = np.clip(
            top_enterprises['建议贷款金额'], 
            min_loan, 
            max_loan
        )
        
        # 重新标准化以确保总金额不超标
        actual_total = top_enterprises['建议贷款金额'].sum()
        if actual_total > total_amount:
            adjustment_factor = total_amount / actual_total
            top_enterprises['建议贷款金额'] *= adjustment_factor
        
        # 筛选获得贷款的企业（贷款额度>=最小值）
        final_strategy = top_enterprises[top_enterprises['建议贷款金额'] >= min_loan].copy()
        
        return final_strategy
    
    def generate_comprehensive_report(self):
        """
        生成综合分析报告
        """
        print("\n" + "="*80)
        print("📊 中小微企业信贷风险分析与策略报告")
        print("="*80)
        
        if self.data is None:
            print("❌ 没有数据可供分析!")
            return
        
        # === 第一部分：数据概览 ===
        print(f"\n📋 一、数据概览")
        print(f"   企业总数: {len(self.data)}家")
        print(f"   数据维度: {self.data.shape[1]}个特征")
        
        # === 第二部分：风险分析 ===
        print(f"\n⚠️ 二、风险分析")
        
        if '风险等级' in self.data.columns:
            risk_dist = self.data['风险等级'].value_counts()
            print(f"   风险等级分布:")
            for level, count in risk_dist.items():
                pct = count / len(self.data) * 100
                print(f"     {level}: {count}家 ({pct:.1f}%)")
        
        if '风险评分' in self.data.columns:
            avg_risk = self.data['风险评分'].mean()
            print(f"   平均风险评分: {avg_risk:.3f}")
        
        # 信誉评级分布
        rating_cols = [col for col in self.data.columns if '信誉' in col or '评级' in col]
        if rating_cols:
            rating_col = rating_cols[0]
            credit_dist = self.data[rating_col].value_counts()
            print(f"   信誉评级分布:")
            for rating, count in credit_dist.items():
                pct = count / len(self.data) * 100
                print(f"     {rating}级: {count}家 ({pct:.1f}%)")
        
        # 行业风险分析
        industry_cols = [col for col in self.data.columns if '行业' in col or 'industry' in col.lower()]
        if industry_cols and '风险评分' in self.data.columns:
            industry_col = industry_cols[0]
            industry_risk = self.data.groupby(industry_col)['风险评分'].agg(['mean', 'count']).sort_values('mean')
            print(f"   行业风险排名:")
            for industry, stats in industry_risk.head(10).iterrows():
                print(f"     {industry}: 风险{stats['mean']:.3f} ({stats['count']}家)")
        
        # === 第三部分：信贷策略 ===
        if self.credit_strategy is not None:
            strategy = self.credit_strategy
            
            print(f"\n💰 三、信贷策略")
            
            total_loans = strategy['建议贷款金额'].sum()
            total_enterprises = len(strategy)
            avg_loan = strategy['建议贷款金额'].mean()
            avg_rate = np.average(strategy['建议利率'], weights=strategy['建议贷款金额'])
            avg_risk = np.average(strategy['风险评分'], weights=strategy['建议贷款金额'])
            
            print(f"   策略概览:")
            print(f"     获贷企业数量: {total_enterprises}家")
            print(f"     总放贷金额: {total_loans:.1f}万元")
            print(f"     平均单笔贷款: {avg_loan:.1f}万元")
            print(f"     加权平均利率: {avg_rate:.2%}")
            print(f"     组合平均风险: {avg_risk:.3f}")
            
            # 预期收益分析
            total_expected_return = strategy['预期收益'].sum()
            roi = total_expected_return / total_loans
            
            print(f"   收益预测:")
            print(f"     预期总收益: {total_expected_return:.1f}万元")
            print(f"     预期投资回报率: {roi:.2%}")
            
            # 风险控制
            high_risk_ratio = len(strategy[strategy['风险评分'] > 0.6]) / len(strategy)
            print(f"     高风险企业占比: {high_risk_ratio:.1%}")
            
            # 前10大贷款企业
            print(f"\n   前10大贷款企业:")
            top_enterprises = strategy.nlargest(10, '建议贷款金额')
            for idx, (_, row) in enumerate(top_enterprises.iterrows(), 1):
                enterprise_id = row.get('企业ID', f'企业{idx}')
                print(f"     {idx:2d}. {enterprise_id}: "
                      f"{row['建议贷款金额']:.1f}万元 "
                      f"(利率{row['建议利率']:.2%}, "
                      f"风险{row['风险评分']:.3f})")
        
        # === 第四部分：风险控制建议 ===
        print(f"\n🛡️ 四、风险控制建议")
        
        if self.credit_strategy is not None:
            strategy = self.credit_strategy
            
            # 行业集中度风险
            if any('行业' in col for col in strategy.columns):
                industry_col = next(col for col in strategy.columns if '行业' in col)
                industry_concentration = strategy.groupby(industry_col)['建议贷款金额'].sum()
                max_industry_ratio = industry_concentration.max() / strategy['建议贷款金额'].sum()
                
                print(f"   1. 行业集中度控制:")
                if max_industry_ratio > 0.3:
                    print(f"      ⚠️ 单一行业占比{max_industry_ratio:.1%}，建议分散投资")
                else:
                    print(f"      ✅ 行业分散良好，最大占比{max_industry_ratio:.1%}")
            
            # 大额贷款风险
            large_loan_threshold = strategy['建议贷款金额'].quantile(0.9)
            large_loans = strategy[strategy['建议贷款金额'] > large_loan_threshold]
            
            print(f"   2. 大额贷款监控:")
            print(f"      超过{large_loan_threshold:.0f}万元的贷款有{len(large_loans)}笔")
            if len(large_loans) > 0:
                print(f"      建议对这些企业加强贷后管理")
            
            # 风险预警
            high_risk_enterprises = strategy[strategy['风险评分'] > 0.7]
            if len(high_risk_enterprises) > 0:
                print(f"   3. 风险预警:")
                print(f"      {len(high_risk_enterprises)}家高风险企业需要重点关注")
                print(f"      涉及金额: {high_risk_enterprises['建议贷款金额'].sum():.1f}万元")
        
        print(f"\n" + "="*80)
        print(f"📊 报告生成完成")
        print(f"="*80)
    
    def visualize_results(self):
        """
        可视化分析结果
        """
        if self.data is None:
            print("❌ 没有数据可供可视化!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('中小微企业信贷风险分析可视化报告', fontsize=16, fontweight='bold')
        
        # 1. 风险评分分布
        if '风险评分' in self.data.columns:
            axes[0,0].hist(self.data['风险评分'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title('风险评分分布')
            axes[0,0].set_xlabel('风险评分')
            axes[0,0].set_ylabel('企业数量')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. 信誉评级分布
        rating_cols = [col for col in self.data.columns if '信誉' in col or '评级' in col]
        if rating_cols:
            rating_col = rating_cols[0]
            credit_counts = self.data[rating_col].value_counts()
            colors = ['green', 'yellow', 'orange', 'red'][:len(credit_counts)]
            axes[0,1].pie(credit_counts.values, labels=credit_counts.index, 
                         autopct='%1.1f%%', colors=colors)
            axes[0,1].set_title('信誉评级分布')
        
        # 3. 风险评分vs信誉评级
        if '风险评分' in self.data.columns and rating_cols:
            rating_col = rating_cols[0]
            sns.boxplot(data=self.data, x=rating_col, y='风险评分', ax=axes[0,2])
            axes[0,2].set_title('风险评分vs信誉评级')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. 行业风险分析
        industry_cols = [col for col in self.data.columns if '行业' in col]
        if industry_cols and '风险评分' in self.data.columns:
            industry_col = industry_cols[0]
            industry_risk = self.data.groupby(industry_col)['风险评分'].mean().sort_values()
            
            if len(industry_risk) <= 15:
                industry_risk.plot(kind='barh', ax=axes[1,0], color='lightcoral')
            else:
                industry_risk.head(15).plot(kind='barh', ax=axes[1,0], color='lightcoral')
            
            axes[1,0].set_title('各行业平均风险评分')
            axes[1,0].set_xlabel('风险评分')
        
        # 5. 企业规模vs风险关系
        if '企业规模' in self.data.columns and '风险评分' in self.data.columns:
            axes[1,1].scatter(self.data['企业规模'], self.data['风险评分'], 
                            alpha=0.6, color='purple')
            axes[1,1].set_xlabel('企业规模(log)')
            axes[1,1].set_ylabel('风险评分')
            axes[1,1].set_title('企业规模vs风险评分')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. 信贷策略可视化
        if self.credit_strategy is not None:
            strategy = self.credit_strategy
            scatter = axes[1,2].scatter(strategy['风险评分'], strategy['建议贷款金额'], 
                                      c=strategy['建议利率'], cmap='RdYlBu_r', 
                                      alpha=0.7, s=60)
            axes[1,2].set_xlabel('风险评分')
            axes[1,2].set_ylabel('建议贷款金额(万元)')
            axes[1,2].set_title('信贷策略分布')
            axes[1,2].grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=axes[1,2], label='利率')
        
        plt.tight_layout()
        plt.savefig('信贷风险分析报告.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 可视化图表已保存为 '信贷风险分析报告.png'")

def main():
    """
    主函数：执行完整的信贷风险分析流程
    """
    print("🏦 中小微企业信贷策略分析系统")
    print("基于真实数据的风险量化分析")
    print("=" * 60)
    
    # 创建分析器
    analyzer = CreditRiskAnalyzer()
    
    # 执行完整分析流程
    try:
        # 1. 加载数据
        if analyzer.load_data():
            
            # 2. 数据预处理
            if analyzer.preprocess_data():
                
                # 3. 构建风险模型
                if analyzer.build_risk_model():
                    
                    # 4. 优化信贷策略
                    # 假设年度信贷总额为5000万元（可根据实际情况调整）
                    total_credit = 5000
                    if analyzer.optimize_credit_strategy(total_credit):
                        
                        # 5. 生成综合报告
                        analyzer.generate_comprehensive_report()
                        
                        # 6. 可视化分析
                        try:
                            analyzer.visualize_results()
                        except Exception as e:
                            print(f"⚠️ 可视化失败: {e}")
                        
                        print(f"\n🎉 信贷策略分析完成!")
                        print(f"📋 请参考以上报告制定具体的信贷投放策略")
                        print(f"💡 建议定期更新数据并重新分析以优化策略")
                        
                    else:
                        print("❌ 策略优化失败!")
                else:
                    print("❌ 风险模型构建失败!")
            else:
                print("❌ 数据预处理失败!")
        else:
            print("❌ 数据加载失败!")
            print("💡 请确保Excel文件在当前目录且格式正确")
            
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        print("💡 请检查数据格式和文件路径")

if __name__ == "__main__":
    main()
