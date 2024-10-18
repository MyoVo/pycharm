import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import seaborn as sns

# 读取数据
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将字符串标签转换为数值标签
label_mapping = {'Supine': 0, 'Prone': 3, 'Side': 1, 'Foetus': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y = y.map(label_mapping)

# 确保特征数据都是数值类型
X = X.apply(pd.to_numeric, errors='coerce')

# 删除任何包含NaN的行
X = X.dropna()
y = y.loc[X.index]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义随机森林模型和参数空间
rf_model = RandomForestClassifier(random_state=42)

# 定义要优化的参数空间
param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(1, 20),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
}

# 使用 BayesSearchCV 进行贝叶斯优化
opt = BayesSearchCV(rf_model, param_space, n_iter=32, random_state=42, cv=5)

# 训练模型
opt.fit(X_train, y_train)

# 最佳参数
print("最佳参数：", opt.best_params_)

# 使用最佳参数进行预测
y_pred = opt.predict(X_test)

# 输出结果
report = classification_report(y_test, y_pred, digits=4)
accuracy = round(accuracy_score(y_test, y_pred), 4)

print("分类报告：\n", report)
print("准确率：", accuracy)
# 计算Kappa系数
kappa = round(cohen_kappa_score(y_test, y_pred), 4)
print("Kappa系数：", kappa)


