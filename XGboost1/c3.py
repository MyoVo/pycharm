import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# 读取数据
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将字符标签转换为整数标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化LightGBM分类器
lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 训练模型
lgbm.fit(X_train, y_train)

# 进行预测
y_pred = lgbm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 计算F1得分
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

# 计算Cohen's Kappa系数
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")
