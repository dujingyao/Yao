#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2深度改进版：302家无信贷记录企业信贷风险量化分析
基于专业建议的四大深度改进：
1. 模型局限性未充分解决 - 行业差异+动态性缺失
2. 关键参数主观性较强 - AHP判断矩阵+机器学习优化
3. 技术细节需优化 - 基础额度公式+增长潜力评分+逻辑约束
4. 风险等级划分粗糙 - 极高风险处理+最低准入评分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import seaborn as sns
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedEnterpriseAnalyzer:
    def __init__(self):
        self.raw_data = None
        self.sales_data = None
        self.purchase_data = None
        self.综合数据 = None
        self.行业参数 = {}
        self.经济指标 = {}
        self.机器学习模型 = {}
        
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

    def setup_economic_indicators(self):
        """设置宏观经济指标（模拟2020年数据）"""
        print("\n📊 设置宏观经济指标...")
        
        # 模拟宏观经济数据
        self.经济指标 = {
            'gdp_growth_rate': 0.023,  # 2020年GDP增长率2.3%
            'inflation_rate': 0.025,   # 通胀率2.5%
            'industry_prosperity': {   # 行业景气度指数
                '制造业': 0.85,
                '批发零售': 0.75,
                '服务业': 0.70,
                '建筑业': 0.80,
                '其他': 0.78
            },
            'credit_spread': 0.015,    # 信用利差1.5%
            'market_volatility': 0.25  # 市场波动率25%
        }
        
        print("   ✅ 宏观经济指标设置完成")

    def advanced_data_preprocessing(self):
        """高级数据预处理和特征工程"""
        print("\n🔧 高级数据预处理...")
        
        # 1. 时间序列处理 - 销项数据
        self.sales_data['开票日期'] = pd.to_datetime(self.sales_data['开票日期'])
        self.sales_data['月份'] = self.sales_data['开票日期'].dt.month
        self.sales_data['季度'] = self.sales_data['开票日期'].dt.quarter
        
        # 2. 时间序列处理 - 进项数据
        self.purchase_data['开票日期'] = pd.to_datetime(self.purchase_data['开票日期'])
        self.purchase_data['月份'] = self.purchase_data['开票日期'].dt.month
        self.purchase_data['季度'] = self.purchase_data['开票日期'].dt.quarter
        
        # 3. 基础聚合统计
        sales_agg = self.sales_data.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '开票日期': ['min', 'max'],
            '购方单位代号': 'nunique',
            '月份': lambda x: len(x.unique()),
            '季度': lambda x: len(x.unique())
        }).round(2)
        
        sales_agg.columns = ['年收入', '销项发票数', '单笔销售均值', '销售标准差', 
                           '首次销售日期', '最后销售日期', '客户数量', '活跃月份数', '活跃季度数']
        
        purchase_agg = self.purchase_data.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '开票日期': ['min', 'max'],
            '销方单位代号': 'nunique',
            '月份': lambda x: len(x.unique()),
            '季度': lambda x: len(x.unique())
        }).round(2)
        
        purchase_agg.columns = ['年成本', '进项发票数', '单笔采购均值', '采购标准差',
                              '首次采购日期', '最后采购日期', '供应商数量', '采购活跃月份数', '采购活跃季度数']
        
        # 4. 合并数据
        self.综合数据 = self.raw_data.copy()
        self.综合数据 = self.综合数据.merge(sales_agg, left_on='企业代号', 
                                        right_index=True, how='left')
        self.综合数据 = self.综合数据.merge(purchase_agg, left_on='企业代号', 
                                        right_index=True, how='left')
        
        # 5. 时间序列特征工程
        self._calculate_time_series_features()
        
        # 6. 高级财务指标计算
        self._calculate_advanced_financial_indicators()
        
        # 7. 行业分类（基于聚类算法）
        self._classify_industries_with_clustering()
        
        # 8. 处理缺失值和异常值
        self._handle_missing_and_outliers()
        
        print("✅ 高级数据预处理完成")

    def _calculate_time_series_features(self):
        """计算时间序列特征"""
        # 季度收入环比增长率
        quarterly_sales = self.sales_data.groupby(['企业代号', '季度'])['价税合计'].sum().reset_index()
        quarterly_growth = quarterly_sales.groupby('企业代号').apply(
            lambda x: x['价税合计'].pct_change().mean()
        ).fillna(0)
        
        self.综合数据['季度收入增长率'] = self.综合数据['企业代号'].map(quarterly_growth)
        
        # 业务连续性指标
        self.综合数据['业务连续性'] = (
            (self.综合数据['活跃月份数'] / 12) * 0.6 + 
            (self.综合数据['活跃季度数'] / 4) * 0.4
        )
        
        # 收入稳定性（变异系数）
        self.综合数据['收入稳定性'] = 1 / (1 + self.综合数据['销售标准差'] / 
                                      (self.综合数据['单笔销售均值'] + 1e-6))

    def _calculate_advanced_financial_indicators(self):
        """计算高级财务指标"""
        # 基础指标
        self.综合数据['毛利率'] = (
            (self.综合数据['年收入'] - self.综合数据['年成本']) / 
            (self.综合数据['年收入'] + 1e-6)
        ).clip(-1, 1)
        
        # 改进的资金周转率（考虑偿债能力）
        self.综合数据['资金周转率'] = (
            self.综合数据['年收入'] / (self.综合数据['年成本'] / 4 + 1e-6)
        ) * (1 + self.综合数据['毛利率'].clip(0, 1))
        
        # 营运能力指标
        self.综合数据['存货周转率'] = self.综合数据['年成本'] / (self.综合数据['年成本'] / 12 + 1e-6)
        
        # 现金流量代理指标
        self.综合数据['现金流量代理'] = (
            self.综合数据['年收入'] * self.综合数据['收入稳定性'] * 
            self.综合数据['业务连续性']
        )

    def _classify_industries_with_clustering(self):
        """基于聚类算法进行行业分类"""
        # 特征矩阵
        features = self.综合数据[['年收入', '年成本', '客户数量', '供应商数量', 
                                '单笔销售均值', '资金周转率']].fillna(0)
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # 行业映射
        industry_mapping = {
            0: '制造业',
            1: '批发零售',
            2: '服务业', 
            3: '建筑业',
            4: '其他'
        }
        
        self.综合数据['行业分类'] = [industry_mapping[c] for c in clusters]

    def _handle_missing_and_outliers(self):
        """处理缺失值和异常值"""
        numeric_cols = ['年收入', '年成本', '客户数量', '供应商数量', 
                       '销项发票数', '进项发票数', '季度收入增长率']
        
        for col in numeric_cols:
            if col in self.综合数据.columns:
                # 缺失值填充
                self.综合数据[col] = self.综合数据[col].fillna(0)
                
                # 异常值处理（3σ原则）
                mean_val = self.综合数据[col].mean()
                std_val = self.综合数据[col].std()
                upper_bound = mean_val + 3 * std_val
                lower_bound = mean_val - 3 * std_val
                
                self.综合数据[col] = self.综合数据[col].clip(lower_bound, upper_bound)

    def setup_dynamic_industry_parameters(self):
        """设置动态行业参数体系"""
        print("\n⚙️ 设置动态行业参数...")
        
        # 基础行业参数
        base_params = {
            '制造业': {'风险系数': 0.8, '流失率基准': 0.05, '敏感性': 1.2, '周期性': 0.7},
            '批发零售': {'风险系数': 1.0, '流失率基准': 0.08, '敏感性': 1.5, '周期性': 1.2},
            '服务业': {'风险系数': 0.9, '流失率基准': 0.06, '敏感性': 1.1, '周期性': 0.9},
            '建筑业': {'风险系数': 1.3, '流失率基准': 0.12, '敏感性': 1.8, '周期性': 1.5},
            '其他': {'风险系数': 1.1, '流失率基准': 0.07, '敏感性': 1.3, '周期性': 1.0}
        }
        
        # 宏观经济调整
        gdp_adjustment = 1 + self.经济指标['gdp_growth_rate']
        
        for industry, params in base_params.items():
            # 景气度调整
            prosperity = self.经济指标['industry_prosperity'][industry]
            
            # 动态调整参数
            adjusted_params = {
                '风险系数': params['风险系数'] * (2 - prosperity),  # 景气度低则风险高
                '流失率基准': params['流失率基准'] * (2 - gdp_adjustment),  # GDP增长低则流失率高
                '敏感性': params['敏感性'] * (1 + self.经济指标['market_volatility']),
                '周期性': params['周期性'],
                '景气度': prosperity
            }
            
            self.行业参数[industry] = adjusted_params
        
        print("   ✅ 动态行业参数设置完成")

    def ml_optimized_risk_scoring(self):
        """机器学习优化的风险评分"""
        print("\n🤖 机器学习优化风险评分...")
        
        # 1. 传统五维评分
        self._calculate_traditional_five_dimensions()
        
        # 2. XGBoost特征重要性分析
        self._xgboost_feature_importance()
        
        # 3. 动态权重优化
        self._optimize_weights_with_ml()
        
        # 4. 集成评分
        self._ensemble_risk_scoring()
        
        print("✅ 机器学习优化风险评分完成")

    def _calculate_traditional_five_dimensions(self):
        """计算传统五维评分"""
        # 财务状况评分（改进版）
        financial_indicators = pd.DataFrame({
            '毛利率': self.综合数据['毛利率'],
            '资金周转率': self.综合数据['资金周转率'],
            '现金流量代理': self.综合数据['现金流量代理']
        })
        
        scaler = StandardScaler()
        financial_normalized = scaler.fit_transform(financial_indicators.fillna(0))
        fin_weights = [0.4, 0.35, 0.25]
        financial_scores = np.dot(financial_normalized, fin_weights)
        self.综合数据['财务状况评分'] = self._minmax_normalize(financial_scores)
        
        # 业务稳定性评分（改进版）
        stability_indicators = pd.DataFrame({
            '客户分散度': 1 / (1 + self.综合数据['客户数量']),
            '供应商分散度': 1 / (1 + self.综合数据['供应商数量']),
            '业务连续性': self.综合数据['业务连续性'],
            '收入稳定性': self.综合数据['收入稳定性']
        })
        
        stability_normalized = scaler.fit_transform(stability_indicators.fillna(0))
        stability_weights = [0.3, 0.25, 0.25, 0.2]
        stability_scores = np.dot(stability_normalized, stability_weights)
        self.综合数据['业务稳定性评分'] = self._minmax_normalize(stability_scores)
        
        # 增长潜力评分（全新设计）
        growth_indicators = pd.DataFrame({
            '季度增长率': self.综合数据['季度收入增长率'].clip(-0.5, 2.0),
            '市场活跃度': np.log1p(self.综合数据['年收入']),
            '业务扩张能力': (self.综合数据['客户数量'] + self.综合数据['供应商数量']) / 2,
            '行业景气度': self.综合数据['行业分类'].map(
                {k: v['景气度'] for k, v in self.行业参数.items()}
            )
        })
        
        growth_normalized = scaler.fit_transform(growth_indicators.fillna(0))
        growth_weights = [0.4, 0.25, 0.2, 0.15]
        growth_scores = np.dot(growth_normalized, growth_weights)
        self.综合数据['增长潜力评分'] = self._minmax_normalize(growth_scores)
        
        # 运营效率评分
        efficiency_indicators = pd.DataFrame({
            '存货周转率': self.综合数据['存货周转率'],
            '单笔交易效率': self.综合数据['年收入'] / (self.综合数据['销项发票数'] + 1),
            '规模效应': np.log1p(self.综合数据['年收入'])
        })
        
        efficiency_normalized = scaler.fit_transform(efficiency_indicators.fillna(0))
        efficiency_weights = [0.4, 0.3, 0.3]
        efficiency_scores = np.dot(efficiency_normalized, efficiency_weights)
        self.综合数据['运营效率评分'] = self._minmax_normalize(efficiency_scores)
        
        # 市场地位评分
        market_indicators = pd.DataFrame({
            '市场份额代理': np.log1p(self.综合数据['年收入']),
            '客户基础': np.log1p(self.综合数据['客户数量']),
            '供应链深度': np.log1p(self.综合数据['供应商数量'])
        })
        
        market_normalized = scaler.fit_transform(market_indicators.fillna(0))
        market_weights = [0.5, 0.3, 0.2]
        market_scores = np.dot(market_normalized, market_weights)
        self.综合数据['市场地位评分'] = self._minmax_normalize(market_scores)

    def _xgboost_feature_importance(self):
        """XGBoost特征重要性分析"""
        # 构造特征矩阵
        features = self.综合数据[[
            '财务状况评分', '业务稳定性评分', '增长潜力评分', 
            '运营效率评分', '市场地位评分'
        ]].fillna(0)
        
        # 构造目标变量（基于年收入和毛利率的综合表现）
        target = (
            np.log1p(self.综合数据['年收入']) * 0.6 + 
            self.综合数据['毛利率'].clip(0, 1) * 0.4
        )
        
        # XGBoost模型
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model.fit(features, target)
        
        # 特征重要性
        importance = xgb_model.feature_importances_
        self.机器学习模型['xgboost_weights'] = importance / importance.sum()
        
        print(f"   - XGBoost权重: {self.机器学习模型['xgboost_weights']}")

    def _optimize_weights_with_ml(self):
        """机器学习优化权重"""
        # 传统AHP权重
        ahp_weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
        
        # XGBoost权重
        xgb_weights = self.机器学习模型['xgboost_weights']
        
        # 集成权重（加权平均）
        self.机器学习模型['optimized_weights'] = 0.6 * ahp_weights + 0.4 * xgb_weights
        
        print(f"   - 优化权重: {self.机器学习模型['optimized_weights']}")

    def _ensemble_risk_scoring(self):
        """集成风险评分"""
        # 使用优化权重计算综合评分
        weights = self.机器学习模型['optimized_weights']
        
        risk_components = np.column_stack([
            self.综合数据['财务状况评分'],
            self.综合数据['业务稳定性评分'],
            self.综合数据['增长潜力评分'],
            self.综合数据['运营效率评分'],
            self.综合数据['市场地位评分']
        ])
        
        self.综合数据['综合风险评分'] = np.dot(risk_components, weights)

    def advanced_logistic_probability_model(self):
        """高级Logistic违约概率模型"""
        print("\n📈 高级Logistic违约概率建模...")
        
        # 1. 多因子Logistic模型
        self._multi_factor_logistic_model()
        
        # 2. 行业调整的期望损失
        self._industry_adjusted_expected_loss()
        
        # 3. 宏观经济调整
        self._macro_economic_adjustment()
        
        # 4. 改进的风险等级划分
        self._improved_risk_classification()
        
        print("✅ 高级Logistic违约概率建模完成")

    def _multi_factor_logistic_model(self):
        """多因子Logistic模型"""
        # 基础Logistic参数
        alpha, beta = 0.483, 3.076
        
        # 基础违约概率
        base_prob = 1 / (1 + np.exp(alpha + beta * self.综合数据['综合风险评分']))
        
        # 行业调整因子
        industry_factor = self.综合数据['行业分类'].map(
            {k: v['敏感性'] for k, v in self.行业参数.items()}
        )
        
        # 增长潜力调整因子
        growth_factor = 1 - 0.2 * self.综合数据['增长潜力评分']
        
        # 综合违约概率
        self.综合数据['违约概率'] = base_prob * industry_factor * growth_factor
        self.综合数据['违约概率'] = self.综合数据['违约概率'].clip(0.01, 0.5)

    def _industry_adjusted_expected_loss(self):
        """行业调整的期望损失"""
        # 基础期望损失
        base_loss_rate = 0.6  # 违约损失率
        
        # 行业周期性调整
        cycle_adjustment = self.综合数据['行业分类'].map(
            {k: v['周期性'] for k, v in self.行业参数.items()}
        )
        
        # 宏观环境调整
        macro_adjustment = 1 + self.经济指标['market_volatility']
        
        # 综合期望损失率
        self.综合数据['期望损失率'] = (
            self.综合数据['违约概率'] * 
            cycle_adjustment * 
            macro_adjustment * 
            base_loss_rate
        ).clip(0.01, 0.8)

    def _macro_economic_adjustment(self):
        """宏观经济调整"""
        # GDP增长率调整
        gdp_adj = 1 - 0.5 * self.经济指标['gdp_growth_rate']
        
        # 信用利差调整
        spread_adj = 1 + 2 * self.经济指标['credit_spread']
        
        # 应用宏观调整
        self.综合数据['违约概率'] *= gdp_adj * spread_adj
        self.综合数据['期望损失率'] *= gdp_adj * spread_adj
        
        # 重新约束范围
        self.综合数据['违约概率'] = self.综合数据['违约概率'].clip(0.01, 0.5)
        self.综合数据['期望损失率'] = self.综合数据['期望损失率'].clip(0.01, 0.8)

    def _improved_risk_classification(self):
        """改进的风险等级划分"""
        # 更细致的风险等级划分
        conditions = [
            self.综合数据['综合风险评分'] >= 0.75,  # 优质企业
            (self.综合数据['综合风险评分'] >= 0.6) & (self.综合数据['综合风险评分'] < 0.75),
            (self.综合数据['综合风险评分'] >= 0.4) & (self.综合数据['综合风险评分'] < 0.6),
            (self.综合数据['综合风险评分'] >= 0.15) & (self.综合数据['综合风险评分'] < 0.4),  # 最低准入评分0.15
            self.综合数据['综合风险评分'] < 0.15
        ]
        
        choices = ['优质', '较低风险', '中风险', '较高风险', '极高风险']
        self.综合数据['风险等级'] = np.select(conditions, choices, default='极高风险')

    def advanced_credit_strategy_design(self, total_budget=10000):
        """高级信贷策略设计"""
        print("\n💰 高级信贷策略设计...")
        
        # 1. 改进的基础额度公式
        self._improved_credit_limit_calculation()
        
        # 2. 动态利率定价模型
        self._dynamic_interest_rate_pricing()
        
        # 3. 约束优化（处理"r≤0"问题）
        self._constraint_optimization()
        
        # 4. 多目标优化分配
        self._multi_objective_allocation(total_budget)
        
        print("✅ 高级信贷策略设计完成")

    def _improved_credit_limit_calculation(self):
        """改进的基础额度公式"""
        # 收入能力评估（结合偿债能力）
        income_capacity = (
            np.log1p(self.综合数据['年收入']) * 
            (1 + self.综合数据['资金周转率']) * 
            (1 + self.综合数据['增长潜力评分'])
        ) / 20
        
        # 风险调整系数（非线性）
        risk_adjustment = (1 - self.综合数据['期望损失率']) ** 1.5
        
        # 行业调整系数
        industry_adjustment = self.综合数据['行业分类'].map(
            {k: 1/v['风险系数'] for k, v in self.行业参数.items()}
        )
        
        # 宏观环境调整
        macro_adjustment = 1 + self.经济指标['gdp_growth_rate']
        
        # 综合推荐额度
        self.综合数据['推荐额度'] = (
            income_capacity * 
            risk_adjustment * 
            industry_adjustment * 
            macro_adjustment * 
            100  # 基础倍数
        ).clip(0, 500)
        
        # 最低准入评分筛选（≥0.15）
        self.综合数据.loc[self.综合数据['综合风险评分'] < 0.15, '推荐额度'] = 0

    def _dynamic_interest_rate_pricing(self):
        """动态利率定价模型"""
        # 无风险利率
        risk_free_rate = 0.04
        
        # 风险溢价（多因子）
        risk_premium = (
            2.5 * self.综合数据['期望损失率'] +  # 损失补偿
            0.5 * (1 - self.综合数据['综合风险评分']) +  # 风险评分调整
            self.经济指标['credit_spread']  # 市场信用利差
        )
        
        # 行业溢价
        industry_premium = self.综合数据['行业分类'].map(
            {k: (v['敏感性'] - 1) * 0.01 for k, v in self.行业参数.items()}
        )
        
        # 流动性溢价
        liquidity_premium = (1 - self.综合数据['业务稳定性评分']) * 0.015
        
        # 期限溢价（假设1年期）
        term_premium = 0.005
        
        # 综合利率
        self.综合数据['推荐利率'] = (
            risk_free_rate + 
            risk_premium + 
            industry_premium + 
            liquidity_premium + 
            term_premium
        ).clip(0.04, 0.18)  # 扩大利率范围

    def _constraint_optimization(self):
        """约束优化（处理利率非负约束）"""
        # 识别风险调整收益为负的情况
        self.综合数据['风险调整收益率'] = (
            self.综合数据['推荐利率'] - self.综合数据['期望损失率']
        )
        
        # 对于风险调整收益率≤0的企业，调整策略
        negative_mask = self.综合数据['风险调整收益率'] <= 0
        
        if negative_mask.sum() > 0:
            print(f"   - 发现{negative_mask.sum()}家企业风险调整收益率≤0，进行策略调整")
            
            # 策略1：提高利率至盈亏平衡点
            self.综合数据.loc[negative_mask, '推荐利率'] = (
                self.综合数据.loc[negative_mask, '期望损失率'] + 0.02
            ).clip(0.04, 0.18)
            
            # 策略2：降低额度
            self.综合数据.loc[negative_mask, '推荐额度'] *= 0.5
            
            # 策略3：极高风险企业直接拒绝
            extreme_risk_mask = (
                (self.综合数据['风险等级'] == '极高风险') | 
                (self.综合数据['期望损失率'] > 0.4)
            )
            self.综合数据.loc[extreme_risk_mask, '推荐额度'] = 0
        
        # 重新计算风险调整收益率
        self.综合数据['风险调整收益率'] = (
            self.综合数据['推荐利率'] - self.综合数据['期望损失率']
        )

    def _multi_objective_allocation(self, total_budget):
        """多目标优化分配"""
        # 计算多维目标函数
        self.综合数据['单位资本收益'] = (
            self.综合数据['风险调整收益率'] * 
            self.综合数据['综合风险评分'] *
            (1 + 0.2 * self.综合数据['增长潜力评分'])  # 增长性奖励
        )
        
        # 候选企业筛选
        eligible = self.综合数据[
            (self.综合数据['推荐额度'] > 0) & 
            (self.综合数据['风险调整收益率'] > 0)
        ].copy()
        
        if len(eligible) == 0:
            self.综合数据['最终额度'] = 0
            self.综合数据['贷款决策'] = '拒绝'
            return
        
        # 按综合收益排序
        eligible = eligible.sort_values('单位资本收益', ascending=False)
        
        # 改进的分配算法（考虑分散化）
        allocated_budget = 0
        final_amounts = []
        industry_allocation = {industry: 0 for industry in self.行业参数.keys()}
        
        for idx, row in eligible.iterrows():
            # 行业集中度控制（单行业不超过40%）
            industry = row['行业分类']
            industry_limit = total_budget * 0.4
            
            if industry_allocation[industry] >= industry_limit:
                final_amounts.append(0)
                continue
            
            # 额度分配
            available_budget = total_budget - allocated_budget
            industry_available = industry_limit - industry_allocation[industry]
            actual_limit = min(row['推荐额度'], industry_available)
            
            if available_budget >= actual_limit and actual_limit >= 10:
                final_amounts.append(actual_limit)
                allocated_budget += actual_limit
                industry_allocation[industry] += actual_limit
            elif available_budget >= 10:
                final_amount = min(available_budget, actual_limit)
                final_amounts.append(final_amount)
                allocated_budget += final_amount
                industry_allocation[industry] += final_amount
            else:
                final_amounts.append(0)
        
        eligible['最终额度'] = final_amounts
        
        # 更新主数据表
        self.综合数据['最终额度'] = 0
        self.综合数据.loc[eligible.index, '最终额度'] = eligible['最终额度']
        
        # 贷款决策
        self.综合数据['贷款决策'] = self.综合数据['最终额度'].apply(
            lambda x: '批准' if x > 0 else '拒绝'
        )

    def _minmax_normalize(self, data):
        """MinMax归一化"""
        return (data - data.min()) / (data.max() - data.min() + 1e-6)

    def generate_advanced_report(self):
        """生成高级分析报告"""
        print("\n📊 生成高级分析报告...")
        
        # 基本统计
        approved = self.综合数据[self.综合数据['最终额度'] > 0]
        
        print("\n=== 深度改进版信贷分析结果 ===")
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
        risk_dist = self.综合数据['风险等级'].value_counts()
        for level, count in risk_dist.items():
            percentage = count / len(self.综合数据) * 100
            approved_count = len(approved[approved['风险等级'] == level])
            print(f"   {level}: {count}家 ({percentage:.1f}%), 获贷: {approved_count}家")
        
        # 按行业统计
        print("\n按行业分布:")
        industry_stats = self.综合数据.groupby('行业分类').agg({
            '最终额度': ['count', lambda x: (x > 0).sum(), 'sum'],
            '违约概率': 'mean',
            '推荐利率': 'mean',
            '增长潜力评分': 'mean'
        }).round(3)
        print(industry_stats)
        
        # 机器学习模型结果
        print(f"\n机器学习优化结果:")
        print(f"XGBoost权重: {self.机器学习模型['xgboost_weights']}")
        print(f"优化权重: {self.机器学习模型['optimized_weights']}")
        
        # 导出结果
        self._export_advanced_results()

    def _export_advanced_results(self):
        """导出高级结果"""
        output_cols = [
            '企业代号', '行业分类', '风险等级', '综合风险评分', '违约概率', '期望损失率',
            '年收入', '年成本', '毛利率', '资金周转率', '季度收入增长率',
            '客户数量', '供应商数量', '业务连续性', '收入稳定性',
            '财务状况评分', '业务稳定性评分', '增长潜力评分', '运营效率评分', '市场地位评分',
            '推荐额度', '最终额度', '推荐利率', '风险调整收益率', '单位资本收益', '贷款决策'
        ]
        
        # 确保所有列都存在
        for col in output_cols:
            if col not in self.综合数据.columns:
                self.综合数据[col] = 0
        
        result_file = 'problem2_advanced_analysis_results.xlsx'
        self.综合数据[output_cols].to_excel(result_file, index=False)
        print(f"\n✅ 深度改进版分析结果已导出至: {result_file}")

def main():
    """主函数"""
    print("="*80)
    print("问题2深度改进版：302家无信贷记录企业信贷风险量化分析")
    print("四大深度改进：1.行业差异+动态性 2.ML优化参数 3.技术细节优化 4.风险等级细化")
    print("="*80)
    
    # 创建深度改进版分析器实例
    analyzer = AdvancedEnterpriseAnalyzer()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 设置宏观经济指标
    analyzer.setup_economic_indicators()
    
    # 高级数据预处理
    analyzer.advanced_data_preprocessing()
    
    # 设置动态行业参数
    analyzer.setup_dynamic_industry_parameters()
    
    # 机器学习优化风险评分
    analyzer.ml_optimized_risk_scoring()
    
    # 高级Logistic违约概率建模
    analyzer.advanced_logistic_probability_model()
    
    # 高级信贷策略设计
    analyzer.advanced_credit_strategy_design(total_budget=10000)
    
    # 生成高级报告
    analyzer.generate_advanced_report()
    
    print("\n🎉 问题2深度改进版分析完成!")

if __name__ == "__main__":
    main()
