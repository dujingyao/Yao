import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from statsmodels.distributions.empirical_distribution import ECDF

# 1. 数据导入
# 从文件"5.dat"中读取数据，假设数据是空格分隔的浮点数
file_path = '5.dat'
with open(file_path, 'r') as file:
    data_str = file.read()

# 将字符串数据分割并转换为浮点数数组
data = np.array([float(num) for num in data_str.split()])

# 打印数据基本信息
print(f"数据点数量: {len(data)}")
print(f"最小值: {np.min(data):.4f}")
print(f"最大值: {np.max(data):.4f}")
print(f"均值: {np.mean(data):.4f}")
print(f"标准差: {np.std(data):.4f}")
print(f"偏度: {stats.skew(data):.4f}")
print(f"峰度: {stats.kurtosis(data):.4f}")

# 2. 数据探索与可视化
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.histplot(data, bins=100, kde=True)
plt.title('数据直方图')
plt.xlabel('数值')
plt.ylabel('频数')

plt.subplot(2, 2, 2)
sns.boxplot(data=data)
plt.title('箱线图')
plt.xlabel('数据分布')

plt.subplot(2, 2, 3)
ecdf = ECDF(data)
plt.plot(ecdf.x, ecdf.y)
plt.title('经验累积分布函数(ECDF)')
plt.xlabel('数值')
plt.ylabel('累积概率')

# 划分数据区间
low_mask = data < 1.0
high_mask = data >= 1.0
data_low = data[low_mask]
data_high = data[high_mask]

print(f"\n低值区域(0-1.0)数据量: {len(data_low)} ({len(data_low)/len(data)*100:.1f}%)")
print(f"高值区域(>1.0)数据量: {len(data_high)} ({len(data_high)/len(data)*100:.1f}%)")

# 3. 低值区域建模
print("\n=== 低值区域(0-1.0)建模 ===")

# Beta分布拟合
def beta_pdf(x, a, b):
    return stats.beta.pdf(x, a, b)

# 使用核密度估计作为拟合目标
kde = stats.gaussian_kde(data_low)
x_sorted = np.sort(data_low)
y_kde = kde(x_sorted)

# 曲线拟合
params_beta, _ = curve_fit(beta_pdf, x_sorted, y_kde, p0=[2, 5])

print(f"Beta分布参数: a={params_beta[0]:.4f}, b={params_beta[1]:.4f}")

# Gamma分布拟合
shape_low, loc_low, scale_low = stats.gamma.fit(data_low, floc=0)
print(f"Gamma分布参数: shape={shape_low:.4f}, scale={scale_low:.4f}")

# 高斯混合模型（双峰）
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_low.reshape(-1, 1))
print("高斯混合模型参数:")
print(f"权重: {gmm.weights_}")
print(f"均值: {gmm.means_.flatten()}")
print(f"标准差: {np.sqrt(gmm.covariances_).flatten()}")

# 4. 高值区域建模
print("\n=== 高值区域(>1.0)建模 ===")

# 对数正态分布拟合
s_high, loc_high, scale_high = stats.lognorm.fit(data_high)
print(f"对数正态分布参数: shape={s_high:.4f}, loc={loc_high:.4f}, scale={scale_high:.4f}")

# Gamma分布拟合
shape_high, loc_high, scale_high = stats.gamma.fit(data_high)
print(f"Gamma分布参数: shape={shape_high:.4f}, loc={loc_high:.4f}, scale={scale_high:.4f}")

# 韦伯分布拟合
c_high, loc_high, scale_high = stats.weibull_min.fit(data_high)
print(f"韦伯分布参数: c={c_high:.4f}, loc={loc_high:.4f}, scale={scale_high:.4f}")

# 5. 模型评估与可视化
x_low = np.linspace(0, 1, 1000)
x_high = np.linspace(1, 100, 1000)

plt.figure(figsize=(15, 10))

# 低值区域拟合图
plt.subplot(2, 1, 1)
sns.histplot(data_low, bins=50, kde=True, stat='density', label='实际数据')
plt.plot(x_low, beta_pdf(x_low, *params_beta), 'r-', label='Beta分布')
plt.plot(x_low, stats.gamma.pdf(x_low, shape_low, loc=0, scale=scale_low), 
         'g--', label='Gamma分布')

# 高斯混合模型
x_gmm = np.linspace(0, 1, 1000)
logprob = gmm.score_samples(x_gmm.reshape(-1, 1))
pdf = np.exp(logprob)
plt.plot(x_gmm, pdf, 'c-', label='高斯混合模型')

plt.title('低值区域(0-1.0)分布拟合')
plt.xlabel('数值')
plt.ylabel('概率密度')
plt.legend()

# 高值区域拟合图
plt.subplot(2, 1, 2)
sns.histplot(data_high, bins=50, kde=True, stat='density', label='实际数据')
plt.plot(x_high, stats.lognorm.pdf(x_high, s_high, loc=loc_high, scale=scale_high), 
         'r-', label='对数正态分布')
plt.plot(x_high, stats.gamma.pdf(x_high, shape_high, loc=loc_high, scale=scale_high), 
         'g--', label='Gamma分布')
plt.plot(x_high, stats.weibull_min.pdf(x_high, c_high, loc=loc_high, scale=scale_high), 
         'c-.', label='韦伯分布')
plt.title('高值区域(>1.0)分布拟合')
plt.xlabel('数值')
plt.ylabel('概率密度')
plt.legend()
plt.tight_layout()

# 6. 异常值检测与处理
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"\n异常值数量: {len(outliers)}")
print(f"异常值范围: {np.min(outliers):.4f} - {np.max(outliers):.4f}")

# 整体数据高斯混合模型
gmm_full = GaussianMixture(n_components=2, random_state=0)
gmm_full.fit(data.reshape(-1, 1))

# 7. 模型应用
def calculate_probability(value):
    """计算累积概率"""
    if value < 1.0:
        # 使用Beta分布
        return stats.beta.cdf(value, *params_beta)
    else:
        # 使用对数正态分布
        return stats.lognorm.cdf(value, s_high, loc=loc_high, scale=scale_high)

# 示例：计算P(X < 0.5)
p_low = calculate_probability(0.5)
print(f"\nP(X < 0.5) = {p_low:.4f}")

# 示例：计算P(5 < X < 10)
if len(data_high) > 0:
    p_low_bound = stats.lognorm.cdf(5, s_high, loc=loc_high, scale=scale_high)
    p_high_bound = stats.lognorm.cdf(10, s_high, loc=loc_high, scale=scale_high)
    p_mid = p_high_bound - p_low_bound
    print(f"P(5 < X < 10) = {p_mid:.4f}")

# 数据自相关分析
if len(data) > 100:
    plt.figure(figsize=(12, 6))
    pd.plotting.autocorrelation_plot(data[:500])
    plt.title('数据自相关图')
    plt.ylabel('自相关系数')
    plt.xlabel('滞后')

plt.show()