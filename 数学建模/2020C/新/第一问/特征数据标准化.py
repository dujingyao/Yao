import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取训练集数据
train_data = pd.read_excel('/home/yao/Yao/数学建模/2020C/新/第一问/训练集.xlsx')

print("=== 数据质量检查与修复 ===")

# 1. 处理百分数格式的毛利率
if '毛利率' in train_data.columns and train_data['毛利率'].dtype == 'object':
    print("处理毛利率百分数格式...")
    # 移除%符号并转换为数值
    train_data['毛利率'] = train_data['毛利率'].astype(str).str.replace('%', '').replace('nan', np.nan)
    train_data['毛利率'] = pd.to_numeric(train_data['毛利率'], errors='coerce') / 100
    # 限制在合理范围内
    train_data['毛利率'] = train_data['毛利率'].clip(0, 1)

# 2. 修正作废发票率异常值问题
if '作废发票率' in train_data.columns:
    print("修正作废发票率异常值...")
    # 将大于1的值视为百分比形式，除以100
    mask = train_data['作废发票率'] > 1
    train_data.loc[mask, '作废发票率'] = train_data.loc[mask, '作废发票率'] / 100
    # 限制在[0,1]范围内
    train_data['作废发票率'] = train_data['作废发票率'].clip(0, 1)
    print(f"修正了 {mask.sum()} 个异常作废发票率值")

# 3. 处理主营业务收入异常值
if '主营业务收入' in train_data.columns:
    print("处理主营业务收入异常值...")
    # 使用IQR方法移除极端异常值
    Q1 = train_data['主营业务收入'].quantile(0.05)  # 使用更宽松的阈值
    Q3 = train_data['主营业务收入'].quantile(0.95)
    outliers = (train_data['主营业务收入'] < Q1) | (train_data['主营业务收入'] > Q3)
    print(f"发现 {outliers.sum()} 个收入异常值")
    
    # 用中位数替换异常值而不是删除
    median_income = train_data['主营业务收入'].median()
    train_data.loc[outliers, '主营业务收入'] = median_income

# 4. 标准化特征
print("\n=== 特征标准化 ===")
numeric_columns = ['主营业务收入', '毛利率', '作废发票率']

for col in numeric_columns:
    if col in train_data.columns:
        # 处理缺失值
        mean_val = train_data[col].mean()
        train_data[col].fillna(mean_val, inplace=True)
        
        # 标准化
        scaler = StandardScaler()
        train_data[col + '_标准化'] = scaler.fit_transform(train_data[[col]]).flatten()
        print(f"{col} 已标准化 (均值: {train_data[col + '_标准化'].mean():.6f}, 标准差: {train_data[col + '_标准化'].std():.6f})")

# 5. 数据质量检查
print("\n=== 数据质量检查 ===")
print(f"最终样本数量: {len(train_data)}")
print(f"违约样本: {train_data['是否违约'].value_counts()}")

# 保存处理后的数据
train_data.to_excel('/home/yao/Yao/数学建模/2020C/新/第一问/训练集_标准化.xlsx', index=False)
print("标准化后的训练集已保存！")
if '作废发票率' in train_data.columns:
    train_data['作废发票率'] = train_data['作废发票率'].clip(0, 1)
    print("作废发票率已修正到合理范围[0,1]")

# 保存处理后的数据
train_data.to_excel('/home/yao/Yao/数学建模/2020C/新/第一问/训练集_标准化.xlsx', index=False)
print(f"\n标准化后的训练集已保存，共 {len(train_data)} 个样本")

# 显示统计信息
print("\n=== 数据统计信息 ===")
for col in [c for c in train_data.columns if '_标准化' in c]:
    print(f"{col}: 均值={train_data[col].mean():.6f}, 标准差={train_data[col].std():.6f}")
