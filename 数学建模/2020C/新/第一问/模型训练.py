import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=5000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.costs = []
        
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """训练模型"""
        # 添加截距项（偏置）
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # 1. 初始化参数：所有β=0
        self.beta = np.zeros(X.shape[1])
        
        # 梯度下降迭代
        for i in range(self.max_iterations):
            # 2. 计算线性组合 z
            z = X @ self.beta
            
            # 3. 计算预测概率 p
            p = self.sigmoid(z)
            
            # 4. 计算损失函数（交叉熵）
            cost = self.compute_cost(y, p)
            self.costs.append(cost)
            
            # 5. 参数更新（梯度下降）
            gradient = (1/len(y)) * X.T @ (p - y)
            self.beta -= self.learning_rate * gradient
            
            # 每100轮输出一次损失
            if i % 100 == 0:
                print(f"迭代 {i}, 损失: {cost:.6f}")
        
        return self
    
    def compute_cost(self, y, p):
        """计算交叉熵损失函数"""
        # 防止log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        cost = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return cost
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.column_stack([np.ones(X.shape[0]), X])
        z = X @ self.beta
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def add_regularization(self, X, y, l2_lambda=0.01):
        """添加正则化来增强模型泛化性"""
        # 添加截距项（偏置）
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # 初始化参数
        self.beta = np.zeros(X.shape[1])
        
        # 梯度下降迭代（带L2正则化）
        for i in range(self.max_iterations):
            z = X @ self.beta
            p = self.sigmoid(z)
            
            # 带正则化的损失函数
            cost = self.compute_cost(y, p) + l2_lambda * np.sum(self.beta[1:]**2)
            self.costs.append(cost)
            
            # 带正则化的梯度
            gradient = (1/len(y)) * X.T @ (p - y)
            gradient[1:] += 2 * l2_lambda * self.beta[1:]  # 不对截距项正则化
            
            self.beta -= self.learning_rate * gradient
            
            if i % 100 == 0:
                print(f"迭代 {i}, 损失: {cost:.6f}")
        
        return self
    
    def compute_statistics(self, X, y):
        """计算模型统计量"""
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # 计算预测概率
        p = self.predict_proba(X)
        
        # 计算Hessian矩阵（用于标准误差）
        W = np.diag(p * (1 - p))
        H = X_with_intercept.T @ W @ X_with_intercept
        
        try:
            # 计算标准误差
            cov_matrix = np.linalg.inv(H)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # 计算z统计量和p值
            z_scores = self.beta / std_errors
            p_values = 2 * (1 - self._norm_cdf(np.abs(z_scores)))
            
            return std_errors, z_scores, p_values
        except:
            return None, None, None
    
    def _norm_cdf(self, x):
        """标准正态分布的累积分布函数近似"""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))

# 读取标准化后的训练数据
train_data = pd.read_excel('/home/yao/Yao/数学建模/2020C/新/第一问/训练集_标准化.xlsx')

# 准备特征和标签
# 使用多个特征列
feature_columns = ['主营业务收入_标准化', '毛利率', '作废发票率']  # 添加更多特征
target_column = '是否违约'  # 使用实际的标签列名

if all(col in train_data.columns for col in feature_columns + [target_column]):
    print("=== 逻辑回归分析 ===")
    
    # 检查特征数据类型
    print("检查特征数据类型:")
    for col in feature_columns:
        print(f"{col}: {train_data[col].dtype}")
    
    # 特别检查毛利率列的原始数据
    print(f"\n毛利率列前10个值:")
    print(train_data['毛利率'].head(10).tolist())
    print(f"毛利率列唯一值样本:")
    print(train_data['毛利率'].unique()[:10])
    
    # 确保所有特征都是数值类型
    for col in feature_columns:
        if train_data[col].dtype == 'object':
            try:
                # 先看看转换前后的变化
                original_sample = train_data[col].head(5).tolist()
                
                # 特殊处理百分数格式
                if col == '毛利率':
                    # 处理百分数：去除%号并转换为小数
                    train_data.loc[:, col] = train_data[col].astype(str).str.replace('%', '').replace('nan', np.nan)
                    train_data.loc[:, col] = pd.to_numeric(train_data[col], errors='coerce') / 100
                    print(f"已将 {col} 的百分数转换为小数")
                else:
                    train_data.loc[:, col] = pd.to_numeric(train_data[col], errors='coerce')
                    print(f"已将 {col} 转换为数值类型")
                
                converted_sample = train_data[col].head(5).tolist()
                print(f"转换前样本: {original_sample}")
                print(f"转换后样本: {converted_sample}")
            except:
                print(f"无法转换 {col} 为数值类型")
    
    # 检查是否有缺失值
    print("\n检查缺失值:")
    for col in feature_columns:
        missing_count = train_data[col].isnull().sum()
        print(f"{col}: {missing_count} 个缺失值")
    
    # 填充缺失值（用均值或0）
    for col in feature_columns:
        if train_data[col].isnull().any():
            # 先计算非NaN值的均值
            valid_mean = train_data[col].dropna().mean()
            if pd.isna(valid_mean):
                # 如果所有值都是NaN，用0填充
                fill_value = 0.0
                print(f"所有值都是NaN，用0填充 {col}")
            else:
                fill_value = valid_mean
                print(f"已用均值 {fill_value:.6f} 填充 {col} 的缺失值")
            
            train_data.loc[:, col] = train_data[col].fillna(fill_value)
    
    X_train = train_data[feature_columns].values
    y_train = train_data[target_column].values
    
    # 确保X_train是浮点数类型
    X_train = X_train.astype(float)
    
    # 检查并转换标签数据类型
    print(f"\n标签列数据类型: {y_train.dtype}")
    print(f"标签唯一值: {np.unique(y_train)}")
    
    # 如果标签是字符串类型，转换为数值
    if y_train.dtype == 'object':
        # 假设违约标签是'是'/'否'或类似格式
        y_train = (y_train == '是').astype(int)  # 或根据实际情况调整
        print(f"转换后标签唯一值: {np.unique(y_train)}")
    else:
        # 确保是整数类型
        y_train = y_train.astype(int)
    
    # 处理样本不平衡
    print(f"\n样本分布: 违约={np.sum(y_train)}, 未违约={len(y_train)-np.sum(y_train)}")
    
    # 动态学习率
    print("\n=== 训练优化 ===")
    model = LogisticRegression(learning_rate=0.01, max_iterations=5000)
    
    # 使用正则化训练
    model.add_regularization(X_train, y_train, l2_lambda=0.01)
    
    # 输出最终参数
    print(f"\n训练完成！最终参数：")
    print(f"截距 β0: {model.beta[0]:.6f}")
    for i, feature in enumerate(feature_columns):
        print(f"{feature} β{i+1}: {model.beta[i+1]:.6f}")
    
    # 绘制损失函数变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(model.costs)
    plt.title('Loss Function Convergence Curve')  # 使用英文标题避免字体问题
    plt.xlabel('Iterations')
    plt.ylabel('Cross-entropy Loss')
    plt.grid(True)
    plt.savefig('/home/yao/Yao/数学建模/2020C/新/第一问/损失函数曲线.png', dpi=300, bbox_inches='tight')
    # plt.show()  # 注释掉show()避免非交互模式警告
    print("损失函数曲线图已保存")
    
    # 6. 训练模型
    print("\n=== 模型训练 ===")
    model = LogisticRegression(learning_rate=0.01, max_iterations=5000)
    model.fit(X_train, y_train)
    
    # 计算统计量
    std_errors, z_scores, p_values = model.compute_statistics(X_train, y_train)
    
    # 输出详细结果表格
    print(f"\n=== 逻辑回归结果分析 ===")
    print("参数\t\t\tβ值\t\tOR值\t\t预期方向\t合理性\t业务解释")
    print("-" * 80)
    
    # 截距项
    print(f"截距(β₀)\t\t{model.beta[0]:.6f}\t-\t\t-\t\t✓\t基准违约概率≈{1/(1+np.exp(-model.beta[0]))*100:.1f}%")
    
    # 各特征参数
    feature_names = ['主营业务收入_标准化', '毛利率', '作废发票率']
    expected_signs = ['-', '-', '+']  # 预期符号：收入↑违约↓，毛利率↑违约↓，作废率↑违约↑
    
    for i, (feature, expected_sign) in enumerate(zip(feature_names, expected_signs)):
        beta = model.beta[i+1]
        or_value = np.exp(beta)
        actual_sign = '+' if beta > 0 else '-'
        is_reasonable = '✓' if actual_sign == expected_sign else '✗'
        
        if feature == '主营业务收入_标准化':
            interpretation = "收入增加反而提高违约率(OR>1)，违背'收入高→还款能力强'的常识" if beta > 0 else "收入增加降低违约率，符合经济逻辑"
        elif feature == '毛利率':
            interpretation = "毛利率越高，违约率越低(OR<1)，符合企业盈利能力越强风险越低的逻辑"
        else:  # 作废发票率
            interpretation = "作废率升高反降低违约率(OR<1)，与'作废率高→管理混乱→风险高'的常识矛盾" if beta < 0 else "作废率升高提高违约率，符合管理风险逻辑"
        
        print(f"{feature}\t{beta:+.6f}\t{or_value:.4f}\t\t{actual_sign}向\t\t{is_reasonable}\t{interpretation}")
        
        if p_values is not None:
            significance = "***" if p_values[i+1] < 0.001 else "**" if p_values[i+1] < 0.01 else "*" if p_values[i+1] < 0.05 else ""
            print(f"\t\t\t\t\t\t\t\t\tp值: {p_values[i+1]:.4f} {significance}")
    
    # 模型诊断
    print(f"\n=== 模型诊断 ===")
    train_proba = model.predict_proba(X_train)
    train_pred = model.predict(X_train)
    accuracy = np.mean(train_pred == y_train)
    
    print(f"训练集准确率: {accuracy:.4f}")
    print(f"AIC: {2 * len(feature_columns) + 2 * model.costs[-1] * len(y_train):.2f}")
    
    # 问题识别和建议
    print(f"\n=== 模型问题诊断 ===")
    problems = []
    if model.beta[1] > 0:  # 主营业务收入系数为正
        problems.append("1. 主营业务收入系数为正，违反经济逻辑")
    if model.beta[3] < 0:  # 作废发票率系数为负
        problems.append("2. 作废发票率系数为负，不符合风险管理常识")
    
    if problems:
        print("发现的问题:")
        for problem in problems:
            print(f"  {problem}")
        print("\n建议改进措施:")
        print("  • 检查数据质量，特别是异常值处理")
        print("  • 考虑特征工程，如对数变换、分段处理")
        print("  • 增加样本量或考虑其他特征")
        print("  • 尝试不同的模型方法")
    else:
        print("模型符合预期，各参数方向合理")

else:
    print("请检查列名是否正确")
    print(f"现有列名: {list(train_data.columns)}")
