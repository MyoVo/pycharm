import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import xgboost as xgb
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

# 定义XGBoost模型和参数空间
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 定义要优化的参数空间
param_space = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(1, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'reg_alpha': Real(1e-6, 1.0, prior='log-uniform'),
    'reg_lambda': Real(1e-6, 1.0, prior='log-uniform'),
}

# 使用BayesSearchCV进行贝叶斯优化
opt = BayesSearchCV(xgb_model, param_space, n_iter=32, random_state=42, cv=5)

# 训练模型
opt.fit(X_train, y_train)

# 最佳参数
print("最佳参数：", opt.best_params_)

# 使用最佳参数进行预测
best_model = opt.best_estimator_
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

# 预测并输出分类报告
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df.to_string(formatters={'precision': '{:0.4f}'.format, 'recall': '{:0.4f}'.format, 'f1-score': '{:0.4f}'.format, 'support': '{:0.4f}'.format}))

# 输出准确率
print("准确率：{:.4f}".format(accuracy_score(y_test, y_pred)))

# 计算Kappa系数
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa系数：{:.4f}".format(kappa))

# 可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=reverse_label_mapping.values(), yticklabels=reverse_label_mapping.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
