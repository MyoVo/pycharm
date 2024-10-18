import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]  # 所有列，除了最后一列
y = data.iloc[:, -1]  # 最后一列

# 检查标签类型并进行转换
if y.dtype == 'O':  # 如果标签是对象类型（字符串）
    y, unique_labels = pd.factorize(y)
    print("标签对应关系:")
    for index, label in enumerate(unique_labels):
        print(f"{label} -> {index}")
else:
    y = y - 1  # 如果标签是数值类型，将其转换为从0开始的数值

# 将字符标签转换为整数标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# 定义SVM模型和参数空间
svm_model = SVC()

# 定义要优化的参数空间
param_space = {
    'C': Real(0.1, 10.0, prior='log-uniform'),
    'gamma': Real(0.001, 1.0, prior='log-uniform'),
    'kernel': Categorical(['rbf'])
}

# 使用BayesSearchCV进行贝叶斯优化
opt = BayesSearchCV(svm_model, param_space, n_iter=32, random_state=42, cv=5)

# 训练模型
opt.fit(X_train, y_train)

# 最佳参数
print("最佳参数：", opt.best_params_)

# 使用最佳参数进行预测
y_pred_encoded = opt.predict(X_test)

# 将预测标签和实际标签转换回原始标签
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test_original = label_encoder.inverse_transform(y_test)

# 输出分类报告
report = classification_report(y_test_original, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
pd.options.display.float_format = '{:.4f}'.format
print(report_df)

# 输出准确率
accuracy = round(accuracy_score(y_test_original, y_pred), 4)
print("准确率：{:.4f}".format(accuracy))

# 计算Kappa系数
kappa = round(cohen_kappa_score(y_test_original, y_pred), 4)
print("Kappa系数：{:.4f}".format(kappa))

