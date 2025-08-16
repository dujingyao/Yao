# -*- coding: utf-8 -*-
"""
中小微企业信贷策略分析 - 问题1
信贷风险量化分析与策略制定

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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class CreditRiskAnalyzer:
    """信贷风险分析器"""
    
    def __init__(self):
        self.data = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.risk_scores = None
        self.credit_strategy = None
        
    def load_data(self, file_path):
        """
        加载企业数据 - 支持多工作表Excel文件
        
        参数:
            file_path: 数据文件路径
        """
        try:
            # 加载Excel文件的所有工作表
            excel_file = pd.ExcelFile(file_path)
            
            # 读取企业信息
            企业信息 = pd.read_excel(excel_file, sheet_name='企业信息')
            
            # 读取发票数据
            进项发票 = pd.read_excel(excel_file, sheet_name='进项发票信息')
            销项发票 = pd.read_excel(excel_file, sheet_name='销项发票信息')
            
            # 处理发票数据，计算各种财务指标
            self._process_invoice_data(企业信息, 进项发票, 销项发票)
            
            print(f"✅ 成功加载数据: {self.data.shape}")
            print(f"📊 数据列名: {list(self.data.columns)}")
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
        
        # 数据清洗
        self.data = self.data.dropna()
        
        # 特征工程 - 基于发票数据构建风险指标
        self._create_risk_features()
        
        # 编码分类变量
        self._encode_categorical_features()
        
        print("✅ 数据预处理完成!")
        return True
    
    def _process_invoice_data(self, 企业信息, 进项发票, 销项发票):
        """
        处理发票数据，计算财务指标
        """
        # 处理进项发票数据
        进项汇总 = 进项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x == '作废发票').sum()
        }).round(2)
        
        # 展平列名
        进项汇总.columns = ['进项总额', '进项发票数量', '进项平均金额', '进项金额标准差', '进项净金额', '进项税额', '进项作废数量']
        进项汇总['进项作废率'] = (进项汇总['进项作废数量'] / 进项汇总['进项发票数量']).fillna(0)
        
        # 处理销项发票数据
        销项汇总 = 销项发票.groupby('企业代号').agg({
            '价税合计': ['sum', 'count', 'mean', 'std'],
            '金额': 'sum',
            '税额': 'sum',
            '发票状态': lambda x: (x.str.strip() == '作废发票').sum()
        }).round(2)
        
        # 展平列名
        销项汇总.columns = ['销项总额', '销项发票数量', '销项平均金额', '销项金额标准差', '销项净金额', '销项税额', '销项作废数量']
        销项汇总['销项作废率'] = (销项汇总['销项作废数量'] / 销项汇总['销项发票数量']).fillna(0)
        
        # 合并数据
        self.data = 企业信息.set_index('企业代号').join([进项汇总, 销项汇总], how='left').fillna(0)
        
        # 计算综合财务指标
        self._calculate_comprehensive_metrics()
    
    def _calculate_comprehensive_metrics(self):
        """
        计算综合财务和风险指标
        """
        # 1. 营业状况指标
        self.data['毛利润'] = self.data['销项净金额'] - self.data['进项净金额']
        self.data['毛利率'] = (self.data['毛利润'] / (self.data['销项净金额'] + 1e-8)).clip(-1, 1)
        
        # 2. 现金流指标
        self.data['净现金流'] = self.data['销项总额'] - self.data['进项总额']
        self.data['现金流比率'] = (self.data['销项总额'] / (self.data['进项总额'] + 1e-8)).clip(0, 10)
        
        # 3. 业务活跃度指标
        self.data['总发票数量'] = self.data['进项发票数量'] + self.data['销项发票数量']
        self.data['业务活跃度'] = np.log1p(self.data['总发票数量'])
        
        # 4. 发票质量指标
        self.data['总作废数量'] = self.data['进项作废数量'] + self.data['销项作废数量']
        self.data['整体作废率'] = (self.data['总作废数量'] / (self.data['总发票数量'] + 1e-8)).clip(0, 1)
        
        # 5. 业务稳定性指标
        self.data['收入稳定性'] = 1 / (self.data['销项金额标准差'] / (self.data['销项平均金额'] + 1e-6) + 1)
        self.data['支出稳定性'] = 1 / (self.data['进项金额标准差'] / (self.data['进项平均金额'] + 1e-6) + 1)
        
        # 6. 规模指标
        self.data['业务总规模'] = np.log1p(self.data['销项总额'] + self.data['进项总额'])
        self.data['销售规模'] = np.log1p(self.data['销项总额'])
        
        # 7. 信誉评级数值化
        rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        self.data['信誉评级数值'] = self.data['信誉评级'].map(rating_map)
        
        # 8. 违约标签
        self.data['违约标签'] = (self.data['是否违约'] == '是').astype(int)

    def _create_risk_features(self):
        """
        创建风险评估特征 - 更新为使用真实发票数据
        """
        # 此方法现在由 _calculate_comprehensive_metrics 替代
        pass
    
    def _encode_categorical_features(self):
        """
        编码分类特征
        """
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['企业ID', '企业名称']:  # 排除标识符列
                le = LabelEncoder()
                self.data[f'{col}_编码'] = le.fit_transform(self.data[col].astype(str))
    
    def build_risk_model(self):
        """
        构建信贷风险评估模型 - 使用真实发票数据特征
        """
        print("🤖 构建风险评估模型...")
        
        # 特征选择 - 基于真实数据的特征
        feature_columns = [
            '毛利率', '现金流比率', '业务活跃度', '整体作废率',
            '收入稳定性', '支出稳定性', '业务总规模', '销售规模',
            '信誉评级数值', '进项作废率', '销项作废率'
        ]
        
        # 筛选存在的特征
        available_features = [col for col in feature_columns if col in self.data.columns]
        print(f"📊 使用的特征: {available_features}")
        
        if len(available_features) == 0:
            print("❌ 没有可用的特征列!")
            return False
        
        X = self.data[available_features].fillna(0)
        
        # 使用真实的违约标签
        y = self.data['违约标签']
        
        print(f"📈 数据概况: {len(X)}个样本, {len(available_features)}个特征")
        print(f"⚠️ 违约率: {y.mean()*100:.1f}%")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 模型训练
        self.risk_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # 交叉验证
        if len(np.unique(y)) > 1:  # 确保有两个类别
            cv_scores = cross_val_score(self.risk_model, X_scaled, y, cv=5, scoring='roc_auc')
            print(f"📈 模型交叉验证AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 训练最终模型
        self.risk_model.fit(X_scaled, y)
        
        # 计算风险评分
        risk_probs = self.risk_model.predict_proba(X_scaled)[:, 1]
        self.data['风险评分'] = risk_probs
        
        # 风险等级分类
        self.data['风险等级'] = pd.cut(
            risk_probs, 
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['低风险', '中低风险', '中高风险', '高风险']
        )
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            '特征': available_features,
            '重要性': self.risk_model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        print("\n📊 特征重要性排序:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"   {row['特征']}: {row['重要性']:.4f}")
        
        print("✅ 风险模型构建完成!")
        return True
    
    def _create_risk_labels(self):
        """
        创建风险标签
        """
        # 基于信誉评级创建风险标签
        if '信誉评级' in self.data.columns:
            # D级企业为高风险，A/B级为低风险，C级为中风险
            risk_labels = []
            for rating in self.data['信誉评级']:
                if rating == 'D':
                    risk_labels.append(1)  # 高风险
                elif rating in ['A', 'B']:
                    risk_labels.append(0)  # 低风险
                else:
                    risk_labels.append(0)  # 默认低风险
            return np.array(risk_labels)
        else:
            # 如果没有信誉评级，基于其他指标创建
            return np.random.binomial(1, 0.3, len(self.data))  # 示例：30%违约率
    
    def optimize_credit_strategy(self, total_credit_amount):
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
        eligible_enterprises = self.data[self.data['信誉评级'] != 'D'].copy()
        
        if len(eligible_enterprises) == 0:
            print("❌ 没有符合放贷条件的企业!")
            return False
        
        # 计算每个企业的预期收益和风险
        self._calculate_expected_returns(eligible_enterprises)
        
        # 优化算法：最大化风险调整收益
        optimal_strategy = self._solve_optimization(eligible_enterprises, total_credit_amount)
        
        self.credit_strategy = optimal_strategy
        
        print("✅ 信贷策略优化完成!")
        return True
    
    def _calculate_expected_returns(self, data):
        """
        计算预期收益和风险 - 基于真实数据调整
        """
        # 基于风险评分和信誉评级确定利率
        def calculate_interest_rate(risk_score, credit_rating):
            # 基准利率
            base_rates = {'A': 0.045, 'B': 0.055, 'C': 0.070, 'D': 0.100}
            base_rate = base_rates.get(credit_rating, 0.080)
            
            # 风险溢价 (0-8%)
            risk_premium = 0.08 * risk_score
            
            # 最终利率限制在4%-18%之间
            final_rate = base_rate + risk_premium
            return np.clip(final_rate, 0.04, 0.18)
        
        data['建议利率'] = data.apply(lambda x: calculate_interest_rate(x['风险评分'], x['信誉评级']), axis=1)
        
        # 违约概率基于风险评分，但考虑历史数据校准
        # 实际违约率22%，需要校准模型预测
        actual_default_rate = 0.22
        predicted_avg_risk = data['风险评分'].mean()
        calibration_factor = actual_default_rate / (predicted_avg_risk + 1e-8)
        
        data['违约概率'] = (data['风险评分'] * calibration_factor).clip(0, 0.95)
        
        # 根据企业规模确定贷款额度上限
        data['最大贷款额度'] = np.where(
            data['销售规模'] > 15, 1000,  # 大企业最多1000万
            np.where(data['销售规模'] > 10, 500,  # 中型企业最多500万
                    200)  # 小企业最多200万
        )
        
        # 计算预期收益（考虑违约损失）
        loan_amount = 100  # 基准贷款金额100万元
        data['预期收益率'] = data['建议利率'] * (1 - data['违约概率'])
        data['基准预期收益'] = loan_amount * data['预期收益率']
        
        # 风险调整收益（夏普比率的简化版本）
        data['风险调整收益'] = data['预期收益率'] / (data['风险评分'] + 0.01)
    
    def _solve_optimization(self, data, total_amount):
        """
        求解优化问题：在风险约束下最大化收益
        """
        n_enterprises = len(data)
        
        # 决策变量：每个企业的贷款金额
        def objective(x):
            # 目标函数：最大化总的风险调整收益
            total_return = np.sum(x * data['风险调整收益'].values)
            return -total_return  # 最小化负收益 = 最大化收益
        
        # 约束条件
        constraints = [
            # 总额度约束
            {'type': 'eq', 'fun': lambda x: np.sum(x) - total_amount},
            # 风险约束：平均风险评分不超过0.5
            {'type': 'ineq', 'fun': lambda x: 0.5 - np.average(data['风险评分'].values, weights=x+1e-8)}
        ]
        
        # 变量边界：每个企业贷款额度0-1000万元
        bounds = [(0, 1000) for _ in range(n_enterprises)]
        
        # 初始解
        x0 = np.full(n_enterprises, total_amount / n_enterprises)
        
        # 求解优化问题
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_allocation = result.x
            
            # 构建策略结果
            strategy = data.copy()
            strategy['建议贷款金额'] = optimal_allocation
            strategy['实际利率'] = strategy['建议利率']
            
            # 筛选获得贷款的企业
            strategy = strategy[strategy['建议贷款金额'] > 10].copy()  # 最小贷款额度10万
            strategy = strategy.sort_values('风险调整收益', ascending=False)
            
            return strategy
        else:
            print("❌ 优化求解失败!")
            return None
    
    def generate_risk_report(self):
        """
        生成风险分析报告
        """
        if self.data is None:
            print("❌ 没有数据可供分析!")
            return
        
        print("\n" + "="*60)
        print("📊 信贷风险分析报告")
        print("="*60)
        
        # 风险等级分布
        if '风险等级' in self.data.columns:
            risk_distribution = self.data['风险等级'].value_counts()
            print(f"\n🎯 风险等级分布:")
            for level, count in risk_distribution.items():
                percentage = count / len(self.data) * 100
                print(f"   {level}: {count}家 ({percentage:.1f}%)")
        
        # 信誉评级分布
        if '信誉评级' in self.data.columns:
            credit_distribution = self.data['信誉评级'].value_counts()
            print(f"\n⭐ 信誉评级分布:")
            for rating, count in credit_distribution.items():
                percentage = count / len(self.data) * 100
                print(f"   {rating}级: {count}家 ({percentage:.1f}%)")
        
        # 行业风险分析
        if '行业' in self.data.columns and '风险评分' in self.data.columns:
            industry_risk = self.data.groupby('行业')['风险评分'].agg(['mean', 'count']).sort_values('mean')
            print(f"\n🏭 行业风险排名 (风险评分越高越危险):")
            for industry, stats in industry_risk.head(10).iterrows():
                print(f"   {industry}: {stats['mean']:.3f} ({stats['count']}家)")
        
        print("\n" + "="*60)
    
    def generate_strategy_report(self):
        """
        生成信贷策略报告
        """
        if self.credit_strategy is None:
            print("❌ 请先生成信贷策略!")
            return
        
        strategy = self.credit_strategy
        
        print("\n" + "="*60)
        print("💰 信贷策略分析报告")
        print("="*60)
        
        # 基本统计
        total_loans = strategy['建议贷款金额'].sum()
        total_enterprises = len(strategy)
        avg_loan = strategy['建议贷款金额'].mean()
        avg_rate = np.average(strategy['实际利率'], weights=strategy['建议贷款金额'])
        
        print(f"\n📈 策略概览:")
        print(f"   获贷企业数量: {total_enterprises}家")
        print(f"   总放贷金额: {total_loans:.1f}万元")
        print(f"   平均单笔贷款: {avg_loan:.1f}万元")
        print(f"   加权平均利率: {avg_rate:.2%}")
        
        # 风险分析
        avg_risk = np.average(strategy['风险评分'], weights=strategy['建议贷款金额'])
        expected_return = strategy['预期收益'].sum()
        
        print(f"\n⚠️ 风险评估:")
        print(f"   组合平均风险: {avg_risk:.3f}")
        print(f"   预期总收益: {expected_return:.1f}万元")
        print(f"   预期收益率: {expected_return/total_loans:.2%}")
        
        # 前10大贷款企业
        print(f"\n🏆 前10大贷款企业:")
        top_enterprises = strategy.nlargest(10, '建议贷款金额')
        for idx, (_, row) in enumerate(top_enterprises.iterrows(), 1):
            print(f"   {idx:2d}. 企业{row.get('企业ID', '未知')}: "
                  f"{row['建议贷款金额']:.1f}万元 "
                  f"(利率{row['实际利率']:.2%}, "
                  f"风险{row['风险评分']:.3f})")
        
        print("\n" + "="*60)
    
    def visualize_analysis(self):
        """
        可视化分析结果
        """
        if self.data is None:
            print("❌ 没有数据可供可视化!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 风险评分分布
        if '风险评分' in self.data.columns:
            axes[0,0].hist(self.data['风险评分'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title('风险评分分布')
            axes[0,0].set_xlabel('风险评分')
            axes[0,0].set_ylabel('企业数量')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. 信誉评级分布
        if '信誉评级' in self.data.columns:
            credit_counts = self.data['信誉评级'].value_counts()
            axes[0,1].pie(credit_counts.values, labels=credit_counts.index, autopct='%1.1f%%')
            axes[0,1].set_title('信誉评级分布')
        
        # 3. 风险评分与信誉评级关系
        if '风险评分' in self.data.columns and '信誉评级' in self.data.columns:
            sns.boxplot(data=self.data, x='信誉评级', y='风险评分', ax=axes[0,2])
            axes[0,2].set_title('风险评分vs信誉评级')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. 行业风险分析
        if '行业' in self.data.columns and '风险评分' in self.data.columns:
            industry_risk = self.data.groupby('行业')['风险评分'].mean().sort_values()
            if len(industry_risk) <= 15:  # 如果行业不太多，显示所有
                industry_risk.plot(kind='barh', ax=axes[1,0])
            else:  # 否则只显示前15个
                industry_risk.head(15).plot(kind='barh', ax=axes[1,0])
            axes[1,0].set_title('各行业平均风险评分')
            axes[1,0].set_xlabel('风险评分')
        
        # 5. 企业规模与风险关系
        if '企业规模' in self.data.columns and '风险评分' in self.data.columns:
            axes[1,1].scatter(self.data['企业规模'], self.data['风险评分'], alpha=0.6)
            axes[1,1].set_xlabel('企业规模(log)')
            axes[1,1].set_ylabel('风险评分')
            axes[1,1].set_title('企业规模vs风险评分')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. 信贷策略分析
        if self.credit_strategy is not None:
            strategy = self.credit_strategy
            axes[1,2].scatter(strategy['风险评分'], strategy['建议贷款金额'], 
                            c=strategy['实际利率'], cmap='RdYlBu_r', alpha=0.7)
            axes[1,2].set_xlabel('风险评分')
            axes[1,2].set_ylabel('建议贷款金额(万元)')
            axes[1,2].set_title('信贷策略分布')
            axes[1,2].grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(axes[1,2].collections[0], ax=axes[1,2], label='利率')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    主函数：执行完整的信贷风险分析流程
    """
    print("🏦 中小微企业信贷策略分析系统")
    print("=" * 50)
    
    # 创建分析器
    analyzer = CreditRiskAnalyzer()
    
    # 数据文件路径
    data_file = "附件1：123家有信贷记录企业的相关数据.xlsx"
    
    # 执行分析流程
    if analyzer.load_data(data_file):
        if analyzer.preprocess_data():
            if analyzer.build_risk_model():
                
                # 生成风险分析报告
                analyzer.generate_risk_report()
                
                # 优化信贷策略（假设年度信贷总额为5000万元）
                total_credit = 5000  # 万元
                if analyzer.optimize_credit_strategy(total_credit):
                    
                    # 生成策略报告
                    analyzer.generate_strategy_report()
                    
                    print(f"\n🎉 分析完成！")
                    print(f"📄 建议参考生成的报告制定具体的信贷策略")
                
    else:
        print("❌ 分析失败，请检查数据文件路径和格式!")

# 示例用法
if __name__ == "__main__":
    # 如果没有真实数据，可以创建示例数据进行测试
    def create_sample_data():
        """创建示例数据用于测试"""
        np.random.seed(42)
        n_companies = 123
        
        # 生成示例数据
        industries = ['制造业', '服务业', '建筑业', '零售业', '科技业'] * (n_companies // 5 + 1)
        
        data = {
            '企业ID': [f'C{i+1:03d}' for i in range(n_companies)],
            '企业名称': [f'企业{i+1}' for i in range(n_companies)],
            '行业': industries[:n_companies],
            '注册资本': np.random.lognormal(mean=4, sigma=1, size=n_companies),
            '销项发票金额': np.random.lognormal(mean=5, sigma=1, size=n_companies),
            '进项发票金额': np.random.lognormal(mean=4.8, sigma=1, size=n_companies),
            '有效发票数量': np.random.poisson(lam=50, size=n_companies),
            '总发票数量': np.random.poisson(lam=60, size=n_companies),
            '作废发票数量': np.random.poisson(lam=5, size=n_companies),
            '负数发票数量': np.random.poisson(lam=3, size=n_companies),
            '上游企业数量': np.random.poisson(lam=10, size=n_companies),
            '下游企业数量': np.random.poisson(lam=15, size=n_companies),
            '信誉评级': np.random.choice(['A', 'B', 'C', 'D'], size=n_companies, p=[0.2, 0.3, 0.4, 0.1])
        }
        
        # 修正逻辑错误
        for i in range(n_companies):
            # 确保有效发票数量不超过总发票数量
            if data['有效发票数量'][i] > data['总发票数量'][i]:
                data['有效发票数量'][i] = data['总发票数量'][i]
            
            # 确保作废+负数发票不超过总发票数量
            total_invalid = data['作废发票数量'][i] + data['负数发票数量'][i]
            if total_invalid > data['总发票数量'][i]:
                scale_factor = data['总发票数量'][i] / total_invalid
                data['作废发票数量'][i] = int(data['作废发票数量'][i] * scale_factor)
                data['负数发票数量'][i] = int(data['负数发票数量'][i] * scale_factor)
        
        df = pd.DataFrame(data)
        df.to_excel('示例数据_123家企业.xlsx', index=False)
        print("✅ 示例数据已创建: 示例数据_123家企业.xlsx")
        return df
    
    # 创建示例数据并运行分析
    print("📝 创建示例数据进行演示...")
    sample_data = create_sample_data()
    
    # 使用示例数据运行分析
    analyzer = CreditRiskAnalyzer()
    analyzer.data = sample_data
    
    if analyzer.preprocess_data():
        if analyzer.build_risk_model():
            analyzer.generate_risk_report()
            
            if analyzer.optimize_credit_strategy(5000):
                analyzer.generate_strategy_report()
                analyzer.visualize_analysis()
