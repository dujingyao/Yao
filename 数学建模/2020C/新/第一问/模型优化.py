import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# 读取数据
train_data = pd.read_excel('/home/yao/Yao/数学建模/2020C/新/第一问/训练集_标准化.xlsx')

print("=== 模型优化与对比 ===")

# 准备数据
feature_columns = ['主营业务收入_标准化', '毛利率', '作废发票率']
X = train_data[feature_columns].values
y = (train_data['是否违约'] == '是').astype(int)

# 1. 使用sklearn的逻辑回归进行对比
print("\n1. 使用sklearn逻辑回归进行对比:")
lr_sklearn = SklearnLR(random_state=42, max_iter=5000)
lr_sklearn.fit(X, y)

print("sklearn逻辑回归结果:")
print(f"截距: {lr_sklearn.intercept_[0]:.6f}")
for i, feature in enumerate(feature_columns):
    coef = lr_sklearn.coef_[0][i]
    or_value = np.exp(coef)
    print(f"{feature}: β={coef:.6f}, OR={or_value:.4f}")

# 2. 特征重要性分析
print(f"\n2. 特征重要性分析:")
feature_importance = np.abs(lr_sklearn.coef_[0])
for i, (feature, importance) in enumerate(zip(feature_columns, feature_importance)):
    print(f"{feature}: {importance:.4f}")

# 3. 模型性能评估
y_pred_proba = lr_sklearn.predict_proba(X)[:, 1]
y_pred = lr_sklearn.predict(X)
auc_score = roc_auc_score(y, y_pred_proba)

print(f"\n3. 模型性能:")
print(f"AUC: {auc_score:.4f}")
print(f"准确率: {np.mean(y_pred == y):.4f}")
print("\n分类报告:")
print(classification_report(y, y_pred))

# 4. 建议改进方案
print(f"\n4. 模型改进建议:")
print("• 如果主营业务收入系数不合理，考虑:")
print("  - 对收入进行对数变换")
print("  - 检查收入数据的分布和异常值")
print("  - 考虑收入的分位数特征而非原始值")
print("\n• 如果作废发票率系数不合理，考虑:")
print("  - 重新定义作废发票率的计算方式")
print("  - 检查是否存在数据录入错误")
print("  - 考虑作废发票率的阈值特征")

# 保存优化建议
optimization_results = {
    'sklearn_intercept': lr_sklearn.intercept_[0],
    'sklearn_coefficients': lr_sklearn.coef_[0].tolist(),
    'feature_names': feature_columns,
    'auc_score': auc_score,
    'accuracy': np.mean(y_pred == y)
}

print(f"\n优化分析完成，AUC={auc_score:.4f}")
