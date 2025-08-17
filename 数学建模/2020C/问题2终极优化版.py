#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2最终优化版：基于附件2和附件3的深度改进
立即修正：利率模型中的 Li → Pi，目标函数调整为期望收益
优先补充：行业参数计算逻辑和Logistic拟合代码
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize, curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UltimateEnterpriseAnalyzer:
    def __init__(self):
        self.raw_data = None
        self.sales_data = None
        self.purchase_data = None
        self.interest_loss_data = None
        self.综合数据 = None
        self.行业参数 = {}
        self.流失率模型 = None
        
    def load_data(self):
        """加载所有数据文件"""
        try:
            print("📥 加载附件2和附件3数据...")
            
            # 附件2：企业数据
            file_path2 = "附件2：302家无信贷记录企业的相关数据.xlsx"
            self.raw_data = pd.read_excel(file_path2, sheet_name='企业信息')
            self.sales_data = pd.read_excel(file_path2, sheet_name='销项发票信息')
            self.purchase_data = pd.read_excel(file_path2, sheet_name='进项发票信息')
            
            # 附件3：利率-流失率关系数据
            file_path3 = "附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx"
            self.interest_loss_data = pd.read_excel(file_path3)
            
            print(f"   - 企业信息: {len(self.raw_data)}家")
            print(f"   - 销项发票: {len(self.sales_data)}条")
            print(f"   - 进项发票: {len(self.purchase_data)}条")
            print(f"   - 利率数据: {len(self.interest_loss_data)}组")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False

    def build_interest_loss_model(self):
        """基于附件3构建利率-流失率模型"""
        print("\n📈 构建利率-流失率关系模型...")
        
        # 清理附件3数据
        df = self.interest_loss_data.copy()
        df = df.dropna(subset=['贷款年利率'])
        
        # 提取利率和流失率数据
        rates = df['贷款年利率'].values
        
        # 处理三个信誉等级的流失率
        loss_rates_A = pd.to_numeric(df['客户流失率'], errors='coerce').values
        loss_rates_B = pd.to_numeric(df['Unnamed: 2'], errors='coerce').values
        loss_rates_C = pd.to_numeric(df['Unnamed: 3'], errors='coerce').values
        
        # 去除NaN值
        valid_idx = ~(np.isnan(loss_rates_A) | np.isnan(loss_rates_B) | np.isnan(loss_rates_C))
        rates = rates[valid_idx]
        loss_rates_A = loss_rates_A[valid_idx]
        loss_rates_B = loss_rates_B[valid_idx]
        loss_rates_C = loss_rates_C[valid_idx]
        
        # 定义流失率函数（指数模型）
        def loss_rate_function(r, a, b, c):
            return a * np.exp(b * r) + c
        
        # 拟合三个等级的模型
        models = {}
        for grade, loss_rates in [('A', loss_rates_A), ('B', loss_rates_B), ('C', loss_rates_C)]:
            try:
                popt, _ = curve_fit(loss_rate_function, rates, loss_rates, 
                                  p0=[0.1, 10, 0.01], maxfev=5000)
                models[grade] = popt
                print(f"   - 等级{grade}: 流失率 = {popt[0]:.3f} * exp({popt[1]:.1f} * r) + {popt[2]:.3f}")
            except:
                # 备用线性模型
                coeffs = np.polyfit(rates, loss_rates, 1)
                models[grade] = (coeffs[0], coeffs[1])
                print(f"   - 等级{grade}: 流失率 = {coeffs[0]:.3f} * r + {coeffs[1]:.3f} (线性)")
        
        self.流失率模型 = models
        return models

    def preprocess_data_ultimate(self):
        """终极版数据预处理"""
        print("\n🔧 终极版数据预处理...")
        
        # 1. 销项数据聚合（修正：利用时间序列信息）
        self.sales_data['开票日期'] = pd.to_datetime(self.sales_data['开票日期'])
        self.purchase_data['开票日期'] = pd.to_datetime(self.purchase_data['开票日期'])
        
        # 按月聚合，计算趋势
        sales_monthly = self.sales_data.groupby(['企业代号', 
                                               self.sales_data['开票日期'].dt.to_period('M')])['价税合计'].sum().reset_index()
        
        sales_agg = self.sales_data.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '开票日期': ['min', 'max'],
            '购方单位代号': 'nunique',
            '发票状态': lambda x: (x == '有效发票').mean()
        }).round(2)
        
        sales_agg.columns = ['年收入', '销项发票数', '单笔销售均值', '销售标准差', 
                           '首次销售日期', '最后销售日期', '客户数量', '有效发票率']
        
        # 2. 进项数据聚合
        purchase_agg = self.purchase_data.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '开票日期': ['min', 'max'],
            '销方单位代号': 'nunique',
            '发票状态': lambda x: (x == '有效发票').mean()
        }).round(2)
        
        purchase_agg.columns = ['年成本', '进项发票数', '单笔采购均值', '采购标准差',
                              '首次采购日期', '最后采购日期', '供应商数量', '进项有效率']
        
        # 3. 合并基础数据
        self.综合数据 = self.raw_data.copy()
        self.综合数据 = self.综合数据.merge(sales_agg, left_on='企业代号', 
                                        right_index=True, how='left')
        self.综合数据 = self.综合数据.merge(purchase_agg, left_on='企业代号', 
                                        right_index=True, how='left')
        
        # 4. 计算时间序列特征
        self._calculate_time_series_features()
        
        # 5. 计算增强财务指标
        self._calculate_enhanced_financial_indicators()
        
        # 6. 基于真实数据进行行业分类
        self._classify_industries_by_data()
        
        # 7. 数据质量增强
        self._enhance_data_quality()
        
        print("✅ 终极版数据预处理完成")

    def _calculate_time_series_features(self):
        """计算时间序列特征"""
        print("   - 计算时间序列特征...")
        
        # 营业天数
        self.综合数据['营业天数'] = (
            (pd.to_datetime(self.综合数据['最后销售日期']) - 
             pd.to_datetime(self.综合数据['首次销售日期'])).dt.days + 1
        ).fillna(365)
        
        # 日均收入
        self.综合数据['日均收入'] = self.综合数据['年收入'] / self.综合数据['营业天数']
        
        # 计算月度收入增长率（基于销项发票时间序列）
        growth_rates = []
        for _, row in self.综合数据.iterrows():
            enterprise_id = row['企业代号']
            enterprise_sales = self.sales_data[self.sales_data['企业代号'] == enterprise_id].copy()
            
            if len(enterprise_sales) > 0:
                enterprise_sales['年月'] = enterprise_sales['开票日期'].dt.to_period('M')
                monthly_sales = enterprise_sales.groupby('年月')['价税合计'].sum()
                
                if len(monthly_sales) >= 3:
                    # 计算环比增长率
                    growth_rate = monthly_sales.pct_change().mean()
                    growth_rates.append(growth_rate)
                else:
                    growth_rates.append(0)
            else:
                growth_rates.append(0)
        
        self.综合数据['月均增长率'] = growth_rates

    def _calculate_enhanced_financial_indicators(self):
        """计算增强版财务指标"""
        print("   - 计算增强版财务指标...")
        
        # 修正：处理缺失值
        numeric_cols = ['年收入', '年成本', '客户数量', '供应商数量']
        for col in numeric_cols:
            if col in self.综合数据.columns:
                self.综合数据[col] = self.综合数据[col].fillna(0)
        
        # 基础财务指标（修正：处理无穷大值）
        self.综合数据['毛利率'] = (
            (self.综合数据['年收入'] - self.综合数据['年成本']) / 
            (self.综合数据['年收入'] + 1e-6)
        ).clip(-1, 1).fillna(0)
        
        self.综合数据['收入成本比'] = (
            self.综合数据['年收入'] / (self.综合数据['年成本'] + 1e-6)
        ).clip(0, 10).fillna(1)
        
        # 增强指标：企业偿债能力代理变量
        self.综合数据['流动比率代理'] = (
            self.综合数据['年收入'] / (self.综合数据['年成本'] / 4 + 1e-6)
        ).clip(0, 5).fillna(1)
        
        # 运营效率指标（修正：处理除零）
        self.综合数据['资产周转率代理'] = (self.综合数据['日均收入'] / 1000).clip(0, 1000).fillna(0)
        self.综合数据['客户黏性'] = (
            self.综合数据['年收入'] / (self.综合数据['客户数量'] + 1)
        ).clip(0, 1e8).fillna(0)
        
        # 风险指标
        self.综合数据['客户集中度'] = (1 / (self.综合数据['客户数量'] + 1)).clip(0, 1)
        self.综合数据['供应商集中度'] = (1 / (self.综合数据['供应商数量'] + 1)).clip(0, 1)
        
        # 发票质量指标
        self.综合数据['综合发票质量'] = (
            self.综合数据['有效发票率'] * 0.6 + 
            self.综合数据['进项有效率'] * 0.4
        ).fillna(0.95).clip(0, 1)

    def _classify_industries_by_data(self):
        """基于真实数据特征进行行业分类"""
        print("   - 基于数据特征进行行业分类...")
        
        # 特征工程：用于行业聚类的特征
        features_for_clustering = [
            '毛利率', '收入成本比', '客户黏性', 
            '客户集中度', '月均增长率', '资产周转率代理'
        ]
        
        # 准备聚类数据，添加数据质量检查
        cluster_data = self.综合数据[features_for_clustering].copy()
        
        # 数据清理：处理无穷大值和异常值
        print(f"   - 聚类前数据质量检查...")
        print(f"     无穷大值数量: {np.isinf(cluster_data.values).sum()}")
        print(f"     NaN值数量: {cluster_data.isnull().sum().sum()}")
        
        # 替换无穷大值为NaN，然后用中位数填充
        cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan)
        
        # 用中位数填充NaN值
        for col in cluster_data.columns:
            cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
        
        # 处理极端异常值（使用3倍IQR规则）
        for col in cluster_data.columns:
            Q1 = cluster_data[col].quantile(0.25)
            Q3 = cluster_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            cluster_data[col] = cluster_data[col].clip(lower_bound, upper_bound)
        
        print(f"   - 数据清理完成，最终统计:")
        print(f"     数据形状: {cluster_data.shape}")
        print(f"     数值范围检查通过: {np.all(np.isfinite(cluster_data.values))}")
        
        try:
            # 标准化
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            # K-means聚类（5个行业类别）
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_data_scaled)
            
            print(f"   - 聚类成功，识别出 {len(np.unique(clusters))} 个行业类别")
            
        except Exception as e:
            print(f"   - 聚类失败: {e}，使用备用分类方案")
            # 备用方案：基于收入规模分类
            clusters = pd.cut(
                self.综合数据['年收入'], 
                bins=5, 
                labels=range(5)
            ).fillna(0).astype(int)
            clusters = clusters.values
        
        # 根据聚类结果和特征分析命名行业
        try:
            cluster_centers = kmeans.cluster_centers_
        except:
            # 备用中心计算
            cluster_centers = []
            for i in range(5):
                mask = clusters == i
                if mask.sum() > 0:
                    center = cluster_data[mask].mean()
                else:
                    center = cluster_data.mean()
                cluster_centers.append(center)
        
        # 分析每个聚类的特征
        industry_mapping = {}
        industry_names = ['制造业', '批发零售', '服务业', '建筑业', '科技企业']
        
        for i in range(5):
            cluster_mask = clusters == i
            if cluster_mask.sum() > 0:
                cluster_features = cluster_data[cluster_mask].mean()
                
                # 基于特征特点分配行业名称
                if cluster_features['毛利率'] > 0.3 and cluster_features['客户集中度'] < 0.1:
                    industry_mapping[i] = '科技企业'
                elif cluster_features['收入成本比'] < 1.2 and cluster_features['客户黏性'] > 1000000:
                    industry_mapping[i] = '批发零售'
                elif cluster_features['月均增长率'] > 0.05:
                    industry_mapping[i] = '服务业'
                elif cluster_features['资产周转率代理'] > 500:
                    industry_mapping[i] = '制造业'
                else:
                    industry_mapping[i] = '建筑业'
            else:
                # 如果聚类为空，分配默认行业
                industry_mapping[i] = industry_names[i % len(industry_names)]
        
        # 确保所有行业都有分配
        used_industries = set(industry_mapping.values())
        unused_industries = set(industry_names) - used_industries
        
        for i, unused in enumerate(unused_industries):
            for cluster_id in range(5):
                if cluster_id not in industry_mapping:
                    industry_mapping[cluster_id] = unused
                    break
        
        self.综合数据['行业分类'] = [industry_mapping[c] for c in clusters]
        
        # 基于行业分类计算真实的行业参数
        self._calculate_industry_parameters()

    def _calculate_industry_parameters(self):
        """基于真实数据计算行业参数"""
        print("   - 计算真实行业参数...")
        
        self.行业参数 = {}
        
        for industry in self.综合数据['行业分类'].unique():
            industry_data = self.综合数据[self.综合数据['行业分类'] == industry]
            
            # 计算行业风险系数（基于毛利率稳定性和客户集中度）
            risk_coeff = (
                1 - industry_data['毛利率'].mean() + 
                industry_data['客户集中度'].mean() + 
                abs(industry_data['月均增长率'].mean()) * 2
            )
            risk_coeff = max(0.5, min(2.0, risk_coeff))
            
            # 计算流失率基准（基于客户黏性和增长率）
            loss_base = max(0.02, min(0.15, 
                0.1 - industry_data['客户黏性'].mean() / 10000000 + 
                abs(industry_data['月均增长率'].mean())
            ))
            
            # 计算敏感性系数（基于发票质量和集中度）
            sensitivity = max(0.8, min(2.5,
                1 + (1 - industry_data['综合发票质量'].mean()) + 
                industry_data['客户集中度'].mean()
            ))
            
            self.行业参数[industry] = {
                '风险系数': round(risk_coeff, 2),
                '流失率基准': round(loss_base, 3),
                '敏感性系数': round(sensitivity, 2),
                '企业数量': len(industry_data)
            }
        
        print("   - 行业参数计算完成:")
        for industry, params in self.行业参数.items():
            print(f"     {industry}: 风险系数={params['风险系数']}, "
                  f"流失率={params['流失率基准']}, 敏感性={params['敏感性系数']}, "
                  f"企业数={params['企业数量']}")

    def _enhance_data_quality(self):
        """数据质量增强"""
        print("   - 数据质量增强...")
        
        # 异常值处理：使用99分位数截断
        numeric_cols = ['年收入', '年成本', '客户数量', '供应商数量', '日均收入']
        
        for col in numeric_cols:
            if col in self.综合数据.columns:
                q99 = self.综合数据[col].quantile(0.99)
                q1 = self.综合数据[col].quantile(0.01)
                self.综合数据[col] = self.综合数据[col].clip(lower=q1, upper=q99)
        
        # 缺失值智能填充
        self.综合数据['月均增长率'] = self.综合数据['月均增长率'].fillna(
            self.综合数据.groupby('行业分类')['月均增长率'].transform('median')
        )

    def calculate_ultimate_risk_scores(self):
        """终极版风险评分计算"""
        print("\n📊 计算终极版风险评分...")
        
        # 六维度评分体系（新增：成长性和数据质量）
        self._calculate_financial_score_v3()
        self._calculate_business_stability_score_v3()
        self._calculate_operational_efficiency_score_v3()
        self._calculate_market_position_score_v3()
        self._calculate_growth_potential_score()
        self._calculate_data_quality_score()
        
        # 动态权重（基于RandomForest特征重要性）
        weights = self._calculate_dynamic_weights()
        
        # 综合风险评分
        risk_components = [
            self.综合数据['财务状况评分'],
            self.综合数据['业务稳定性评分'],
            self.综合数据['运营效率评分'],
            self.综合数据['市场地位评分'],
            self.综合数据['成长潜力评分'],
            self.综合数据['数据质量评分']
        ]
        
        self.综合数据['综合风险评分'] = sum(w * score for w, score in zip(weights, risk_components))
        
        # 修正：Logistic转换为违约概率
        self._convert_risk_to_default_probability_v2()
        
        # 立即修正：期望损失率计算
        self._calculate_expected_loss_corrected()
        
        print("✅ 终极版风险评分计算完成")

    def _calculate_financial_score_v3(self):
        """财务状况评分V3"""
        indicators = pd.DataFrame({
            '毛利率': self.综合数据['毛利率'],
            '收入成本比': self.综合数据['收入成本比'],
            '流动比率代理': self.综合数据['流动比率代理'],
            '资产周转率代理': self.综合数据['资产周转率代理']
        })
        
        scaler = StandardScaler()
        indicators_std = scaler.fit_transform(indicators.fillna(0))
        
        weights = [0.3, 0.25, 0.25, 0.2]
        scores = np.dot(indicators_std, weights)
        
        self.综合数据['财务状况评分'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def _calculate_business_stability_score_v3(self):
        """业务稳定性评分V3"""
        indicators = pd.DataFrame({
            '客户分散度': 1 - self.综合数据['客户集中度'],
            '供应商分散度': 1 - self.综合数据['供应商集中度'],
            '收入稳定性': 1 / (abs(self.综合数据['月均增长率']) + 0.01),
            '营业连续性': np.log1p(self.综合数据['营业天数'])
        })
        
        scaler = StandardScaler()
        indicators_std = scaler.fit_transform(indicators.fillna(0))
        
        weights = [0.3, 0.25, 0.25, 0.2]
        scores = np.dot(indicators_std, weights)
        
        self.综合数据['业务稳定性评分'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def _calculate_operational_efficiency_score_v3(self):
        """运营效率评分V3"""
        indicators = pd.DataFrame({
            '客户效率': self.综合数据['客户黏性'],
            '发票效率': self.综合数据['综合发票质量'],
            '日营收效率': self.综合数据['日均收入'],
            '规模效应': np.log1p(self.综合数据['年收入'])
        })
        
        scaler = StandardScaler()
        indicators_std = scaler.fit_transform(indicators.fillna(0))
        
        weights = [0.3, 0.25, 0.25, 0.2]
        scores = np.dot(indicators_std, weights)
        
        self.综合数据['运营效率评分'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def _calculate_market_position_score_v3(self):
        """市场地位评分V3"""
        indicators = pd.DataFrame({
            '市场份额代理': np.log1p(self.综合数据['年收入']),
            '客户基础': np.log1p(self.综合数据['客户数量']),
            '供应链网络': np.log1p(self.综合数据['供应商数量']),
            '交易活跃度': np.log1p(self.综合数据['销项发票数'])
        })
        
        scaler = StandardScaler()
        indicators_std = scaler.fit_transform(indicators.fillna(0))
        
        weights = [0.4, 0.25, 0.2, 0.15]
        scores = np.dot(indicators_std, weights)
        
        self.综合数据['市场地位评分'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def _calculate_growth_potential_score(self):
        """成长潜力评分（新增）"""
        indicators = pd.DataFrame({
            '收入增长率': self.综合数据['月均增长率'].clip(-1, 1),
            '毛利率水平': self.综合数据['毛利率'].clip(-1, 1),
            '市场扩张能力': (self.综合数据['客户数量'] / (self.综合数据['营业天数'] + 1)).clip(0, 100),
            '运营天数': np.log1p(self.综合数据['营业天数']).clip(0, 10)
        })
        
        # 处理无穷大值和异常值
        indicators = indicators.replace([np.inf, -np.inf], np.nan)
        indicators = indicators.fillna(indicators.median())
        
        # 再次检查和处理极端值
        for col in indicators.columns:
            Q1 = indicators[col].quantile(0.25)
            Q3 = indicators[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            indicators[col] = indicators[col].clip(lower_bound, upper_bound)
        
        try:
            scaler = StandardScaler()
            indicators_std = scaler.fit_transform(indicators)
        except:
            # 手动标准化
            indicators_std = (indicators - indicators.mean()) / (indicators.std() + 1e-8)
            indicators_std = indicators_std.fillna(0).values
        
        weights = [0.4, 0.3, 0.2, 0.1]
        scores = np.dot(indicators_std, weights)
        
        self.综合数据['成长潜力评分'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def _calculate_data_quality_score(self):
        """数据质量评分（新增）"""
        indicators = pd.DataFrame({
            '发票质量': self.综合数据['综合发票质量'].clip(0, 1),
            '数据完整性': (1 - (self.综合数据[['年收入', '年成本', '客户数量']].isnull().sum(axis=1) / 3)).clip(0, 1),
            '交易频率': np.log1p(self.综合数据['销项发票数']).clip(0, 15),
            '业务真实性': (self.综合数据['年收入'] / (self.综合数据['销项发票数'] + 1)).clip(0, 1e6)
        })
        
        # 处理无穷大值和异常值
        indicators = indicators.replace([np.inf, -np.inf], np.nan)
        indicators = indicators.fillna(indicators.median())
        
        # 处理极端值
        for col in indicators.columns:
            Q1 = indicators[col].quantile(0.25)
            Q3 = indicators[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            indicators[col] = indicators[col].clip(lower_bound, upper_bound)
        
        try:
            scaler = StandardScaler()
            indicators_std = scaler.fit_transform(indicators)
        except:
            # 手动标准化
            indicators_std = (indicators - indicators.mean()) / (indicators.std() + 1e-8)
            indicators_std = indicators_std.fillna(0).values
        
        weights = [0.4, 0.3, 0.2, 0.1]
        scores = np.dot(indicators_std, weights)
        
        self.综合数据['数据质量评分'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def _calculate_dynamic_weights(self):
        """基于随机森林计算动态权重"""
        print("   - 计算动态权重（RandomForest特征重要性）...")
        
        # 构造目标变量（基于收入规模和毛利率的综合指标）
        target = (
            0.6 * (self.综合数据['年收入'] / self.综合数据['年收入'].max()) +
            0.4 * self.综合数据['毛利率'].fillna(0)
        )
        
        # 特征矩阵
        features = self.综合数据[[
            '财务状况评分', '业务稳定性评分', '运营效率评分',
            '市场地位评分', '成长潜力评分', '数据质量评分'
        ]].fillna(0)
        
        # 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features, target)
        
        # 特征重要性作为权重
        importances = rf.feature_importances_
        
        # 归一化权重
        weights = importances / importances.sum()
        
        print(f"   - 动态权重: {weights.round(3)}")
        return weights

    def _convert_risk_to_default_probability_v2(self):
        """Logistic转换V2（修正参数）"""
        print("   - Logistic违约概率转换...")
        
        # 更精确的参数标定：基于银行实际经验
        # 高风险评分(0.9) -> 低违约概率(2%)
        # 低风险评分(0.1) -> 高违约概率(30%)
        
        from scipy.optimize import fsolve
        
        def equations(params):
            alpha, beta = params
            eq1 = 1 / (1 + np.exp(alpha + beta * 0.9)) - 0.02  # 高分低违约
            eq2 = 1 / (1 + np.exp(alpha + beta * 0.1)) - 0.30  # 低分高违约
            return [eq1, eq2]
        
        try:
            alpha, beta = fsolve(equations, [3, -8])
            print(f"   - 修正Logistic参数: α={alpha:.3f}, β={beta:.3f}")
        except:
            # 备用参数
            alpha, beta = 2.944, -7.313
            print(f"   - 使用备用Logistic参数: α={alpha:.3f}, β={beta:.3f}")
        
        # 计算违约概率
        risk_scores = self.综合数据['综合风险评分']
        self.综合数据['违约概率'] = 1 / (1 + np.exp(alpha + beta * risk_scores))
        
        # 确保违约概率在合理范围内
        self.综合数据['违约概率'] = self.综合数据['违约概率'].clip(0.01, 0.35)

    def _calculate_expected_loss_corrected(self):
        """修正：期望损失率计算"""
        print("   - 修正期望损失率计算...")
        
        # 立即修正：使用违约概率Pi而不是Li
        self.综合数据['期望损失率'] = self.综合数据.apply(
            lambda row: (
                row['违约概率'] *  # 使用Pi而不是Li
                self.行业参数[row['行业分类']]['敏感性系数'] * 
                0.6  # LGD违约损失率
            ), axis=1
        )
        
        # 风险等级分类（基于违约概率）
        self._classify_risk_levels_corrected()

    def _classify_risk_levels_corrected(self):
        """修正风险等级分类"""
        conditions = [
            self.综合数据['违约概率'] <= 0.03,
            (self.综合数据['违约概率'] > 0.03) & (self.综合数据['违约概率'] <= 0.08),
            (self.综合数据['违约概率'] > 0.08) & (self.综合数据['违约概率'] <= 0.15),
            (self.综合数据['违约概率'] > 0.15) & (self.综合数据['违约概率'] <= 0.25),
            self.综合数据['违约概率'] > 0.25
        ]
        
        choices = ['优质', '良好', '一般', '关注', '次级']
        self.综合数据['风险等级'] = np.select(conditions, choices, default='次级')

    def design_ultimate_credit_strategy(self, total_budget=10000):
        """终极版信贷策略设计"""
        print("\n💰 设计终极版信贷策略...")
        
        # 1. 基于附件3的动态利率定价
        self._calculate_market_based_interest_rates()
        
        # 2. 精确额度计算
        self._calculate_precise_credit_limits()
        
        # 3. 修正目标函数优化
        self._optimize_expected_return(total_budget)
        
        print("✅ 终极版信贷策略设计完成")

    def _calculate_market_based_interest_rates(self):
        """基于附件3的市场化利率定价"""
        print("   - 基于市场数据的动态利率定价...")
        
        # 基准利率（无风险利率）
        base_rate = 0.04
        
        # 根据风险等级映射到附件3的信誉等级
        credit_grade_mapping = {
            '优质': 'A',
            '良好': 'A', 
            '一般': 'B',
            '关注': 'B',
            '次级': 'C'
        }
        
        rates = []
        for _, row in self.综合数据.iterrows():
            risk_grade = row['风险等级']
            credit_grade = credit_grade_mapping.get(risk_grade, 'C')
            
            # 基于期望损失率确定利率水平
            expected_loss = row['期望损失率']
            
            # 使用流失率模型确定最优利率
            if credit_grade in self.流失率模型:
                # 目标：最大化 (利率 - 期望损失率) * (1 - 流失率)
                def profit_function(r):
                    if len(self.流失率模型[credit_grade]) == 3:
                        # 指数模型
                        a, b, c = self.流失率模型[credit_grade]
                        loss_rate = a * np.exp(b * r) + c
                    else:
                        # 线性模型
                        k, b = self.流失率模型[credit_grade]
                        loss_rate = k * r + b
                    
                    loss_rate = max(0, min(0.8, loss_rate))  # 限制在合理范围
                    return -(r - expected_loss) * (1 - loss_rate)
                
                # 寻找最优利率
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(profit_function, bounds=(0.04, 0.15), method='bounded')
                optimal_rate = result.x
            else:
                # 备用方案：基于期望损失率的简单定价
                optimal_rate = base_rate + expected_loss * 3 + 0.02
            
            rates.append(max(0.04, min(0.15, optimal_rate)))
        
        self.综合数据['推荐利率'] = rates

    def _calculate_precise_credit_limits(self):
        """精确额度计算"""
        print("   - 精确信贷额度计算...")
        
        # 多因子额度模型
        limits = []
        for _, row in self.综合数据.iterrows():
            # 收入基础额度
            income_base = np.log1p(row['年收入']) / 15
            
            # 风险调整（使用期望损失率）
            risk_adj = (1 - row['期望损失率']) ** 1.5
            
            # 行业调整
            industry_adj = 1 / self.行业参数[row['行业分类']]['风险系数']
            
            # 成长性调整
            growth_adj = 1 + max(-0.2, min(0.3, row['月均增长率']))
            
            # 数据质量调整
            quality_adj = row['数据质量评分']
            
            # 综合额度
            credit_limit = income_base * risk_adj * industry_adj * growth_adj * quality_adj * 80
            
            # 风险门槛
            if row['违约概率'] > 0.25:
                credit_limit = 0
            elif row['风险等级'] == '次级':
                credit_limit = min(credit_limit, 50)
            
            limits.append(max(0, min(500, credit_limit)))
        
        self.综合数据['推荐额度'] = limits

    def _optimize_expected_return(self, total_budget):
        """修正：期望收益优化"""
        print("   - 期望收益优化...")
        
        # 修正目标函数：期望收益 = (利率 - 期望损失率) × 额度
        self.综合数据['期望收益率'] = self.综合数据['推荐利率'] - self.综合数据['期望损失率']
        
        # 单位资本期望收益
        self.综合数据['单位资本收益'] = (
            self.综合数据['期望收益率'] * self.综合数据['综合风险评分']
        )
        
        # 优化分配
        eligible = self.综合数据[self.综合数据['推荐额度'] > 0].copy()
        eligible = eligible.sort_values('单位资本收益', ascending=False)
        
        allocated_budget = 0
        final_amounts = []
        
        for idx, row in eligible.iterrows():
            if allocated_budget + row['推荐额度'] <= total_budget:
                final_amounts.append(row['推荐额度'])
                allocated_budget += row['推荐额度']
            elif total_budget - allocated_budget >= 10:
                final_amounts.append(total_budget - allocated_budget)
                allocated_budget = total_budget
                break
            else:
                final_amounts.append(0)
        
        # 补齐剩余
        while len(final_amounts) < len(eligible):
            final_amounts.append(0)
        
        eligible['最终额度'] = final_amounts
        
        # 更新主数据
        self.综合数据['最终额度'] = 0
        self.综合数据.loc[eligible.index, '最终额度'] = eligible['最终额度']
        
        self.综合数据['贷款决策'] = self.综合数据['最终额度'].apply(
            lambda x: '批准' if x > 0 else '拒绝'
        )

    def generate_ultimate_report(self):
        """生成终极版分析报告"""
        print("\n📊 生成终极版分析报告...")
        
        approved = self.综合数据[self.综合数据['最终额度'] > 0]
        
        print("\n=== 终极版信贷分析结果 ===")
        print(f"总申请企业: {len(self.综合数据)}家")
        print(f"批准企业: {len(approved)}家")
        print(f"批准率: {len(approved)/len(self.综合数据)*100:.1f}%")
        
        if len(approved) > 0:
            print(f"总放贷金额: {approved['最终额度'].sum():.0f}万元")
            print(f"平均额度: {approved['最终额度'].mean():.1f}万元")
            print(f"平均利率: {approved['推荐利率'].mean()*100:.2f}%")
            print(f"平均违约概率: {approved['违约概率'].mean()*100:.2f}%")
            print(f"平均期望损失率: {approved['期望损失率'].mean()*100:.2f}%")
            
            # 期望收益
            total_expected_return = (
                approved['最终额度'] * approved['期望收益率']
            ).sum()
            print(f"预期年收益: {total_expected_return:.0f}万元")
            
            # ROE估算
            roe = total_expected_return / approved['最终额度'].sum() * 100
            print(f"资本收益率(ROE): {roe:.2f}%")
        
        # 风险分布
        print("\n按风险等级分布:")
        risk_dist = self.综合数据['风险等级'].value_counts()
        for level, count in risk_dist.items():
            percentage = count / len(self.综合数据) * 100
            approved_count = len(approved[approved['风险等级'] == level])
            approval_rate = approved_count / count * 100 if count > 0 else 0
            print(f"   {level}: {count}家 ({percentage:.1f}%), 获贷: {approved_count}家 ({approval_rate:.1f}%)")
        
        # 行业分布
        print("\n按行业分布:")
        for industry in self.综合数据['行业分类'].unique():
            industry_data = self.综合数据[self.综合数据['行业分类'] == industry]
            industry_approved = approved[approved['行业分类'] == industry]
            
            print(f"   {industry}: {len(industry_data)}家, 获贷: {len(industry_approved)}家 "
                  f"({len(industry_approved)/len(industry_data)*100:.1f}%)")
        
        # 导出结果
        self._export_ultimate_results()

    def _export_ultimate_results(self):
        """导出终极版结果"""
        output_cols = [
            '企业代号', '企业名称', '行业分类', '风险等级', 
            '综合风险评分', '违约概率', '期望损失率',
            '年收入', '年成本', '毛利率', '月均增长率',
            '客户数量', '供应商数量', '营业天数',
            '财务状况评分', '业务稳定性评分', '运营效率评分', 
            '市场地位评分', '成长潜力评分', '数据质量评分',
            '推荐额度', '最终额度', '推荐利率', '期望收益率', '贷款决策'
        ]
        
        for col in output_cols:
            if col not in self.综合数据.columns:
                self.综合数据[col] = 0
        
        result_file = 'problem2_ultimate_analysis_results.xlsx'
        self.综合数据[output_cols].to_excel(result_file, index=False)
        print(f"\n✅ 终极版分析结果已导出至: {result_file}")

def main():
    """主函数"""
    print("="*80)
    print("问题2终极优化版：基于附件2&3的深度改进")
    print("立即修正：Li→Pi, 期望收益优化")
    print("优先补充：真实行业参数, 市场化利率定价")
    print("="*80)
    
    analyzer = UltimateEnterpriseAnalyzer()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 构建利率-流失率模型
    analyzer.build_interest_loss_model()
    
    # 数据预处理
    analyzer.preprocess_data_ultimate()
    
    # 计算风险评分
    analyzer.calculate_ultimate_risk_scores()
    
    # 制定信贷策略
    analyzer.design_ultimate_credit_strategy(total_budget=10000)
    
    # 生成报告
    analyzer.generate_ultimate_report()
    
    print("\n🎉 问题2终极优化版分析完成!")

if __name__ == "__main__":
    main()
