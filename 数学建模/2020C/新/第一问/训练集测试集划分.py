# 训练集测试集划分
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据文件
data_path = '/home/yao/Yao/数学建模/2020C/ABC评级企业_最终数据.xlsx'
df = pd.read_excel(data_path)

# 进行训练集测试集划分（8:2）
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 保存训练集和测试集
train_data.to_excel('/home/yao/Yao/数学建模/2020C/新/第一问/训练集.xlsx', index=False)
test_data.to_excel('/home/yao/Yao/数学建模/2020C/新/第一问/测试集.xlsx', index=False)

print(f"原始数据集大小: {len(df)}")
print(f"训练集大小: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
print(f"测试集大小: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")
print("数据划分完成，文件已保存！")
