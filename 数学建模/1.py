import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 生成数据表格
# 指标顺序：交通流量、行程车速、饱和度、非直线系数、路网密度、TCI、运行指数
labels = ['交通流量', '行程车速', '饱和度', '非直线系数', '路网密度', 'TCI', '运行指数']
data = {
    '规则型-开放前': [0.60, 0.65, 0.55, 0.70, 0.60, 0.75, 0.68],
    '规则型-开放后': [0.75, 0.80, 0.65, 0.85, 0.72, 0.88, 0.80],
    '带状型-开放前': [0.55, 0.60, 0.50, 0.65, 0.58, 0.70, 0.62],
    '带状型-开放后': [0.62, 0.68, 0.58, 0.72, 0.65, 0.76, 0.70],
    '混合型-开放前': [0.50, 0.55, 0.48, 0.60, 0.55, 0.68, 0.60],
    '混合型-开放后': [0.54, 0.60, 0.52, 0.65, 0.60, 0.72, 0.65]
}

# 创建DataFrame
df = pd.DataFrame(data, index=labels)
print("数据表格：")
print(df)

# 保存数据表格为CSV文件
df.to_csv('data_table.csv', encoding='utf-8-sig')
print("数据表格已保存为 data_table.csv")

# 2. 绘制没有文字的图片
area_types = ['规则型', '带状型', '混合型']
before_vals = [np.mean(data[f'{t}-开放前']) for t in area_types]
after_vals = [np.mean(data[f'{t}-开放后']) for t in area_types]

x = np.arange(len(area_types))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, before_vals, width, color='blue')
plt.bar(x + width / 2, after_vals, width, color='orange')

# 移除所有文字
plt.xticks([])
plt.yticks([])
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend([], frameon=False)  # 不显示图例

# 保存图片
plt.savefig('result_no_text.png', bbox_inches='tight')
print("图片已保存为 result_no_text.png，请手动打开查看。")