import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import tensorflow as tf
from SVM_model import train_svm
from ELM_model import tune_elm_model
from XGB_model import train_xgboost
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
# 禁用所有GPU
tf.config.set_visible_devices([], 'GPU')

# 忽略特定的警告
warnings.filterwarnings("ignore", category=FutureWarning, message=".*No further splits with positive gain.*")

# 加载数据
data = pd.read_excel('r2.xlsx')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 标签编码
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 创建一个StratifiedKFold对象，指定折数
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化存储预测概率的数组
svm_probs_train = np.zeros((X_train.shape[0], len(np.unique(y_train))))
elm_probs_train = np.zeros((X_train.shape[0], len(np.unique(y_train))))
xgb_probs_train = np.zeros((X_train.shape[0], len(np.unique(y_train))))

# 在交叉验证的每一个折中进行训练和预测
for train_index, valid_index in kf.split(X_train, y_train):
    X_tr, X_val = X_train[train_index], X_train[valid_index]
    y_tr, y_val = y_train[train_index], y_train[valid_index]

    # 训练每个基模型
    best_svm_model, _ = train_svm(X_tr, y_tr, X_val, y_val)
    best_elm_model, _ = tune_elm_model(X_tr, y_tr, X_val, y_val)
    best_xgb_model, _ = train_xgboost(X_tr, y_tr, X_val, y_val)

    # 预测验证集并存储预测概率
    svm_probs_train[valid_index] = best_svm_model.predict_proba(X_val)
    elm_probs_train[valid_index] = best_elm_model.predict_proba(X_val)
    xgb_probs_train[valid_index] = best_xgb_model.predict_proba(X_val)

    # 计算并输出准确率
    svm_accuracy = accuracy_score(y_val, best_svm_model.predict(X_val))
    elm_accuracy = accuracy_score(y_val, best_elm_model.predict(X_val))
    xgb_accuracy = accuracy_score(y_val, best_xgb_model.predict(X_val))

    print(f"Fold SVM 验证准确率: {svm_accuracy:.4f}")
    print(f"Fold ELM 验证准确率: {elm_accuracy:.4f}")
    print(f"Fold XGB 验证准确率: {xgb_accuracy:.4f}")

# 使用整个训练集训练基模型并生成测试集的预测概率
best_svm_model, _ = train_svm(X_train, y_train, X_test, y_test)
svm_probs_test = best_svm_model.predict_proba(X_test)
svm_test_accuracy = accuracy_score(y_test, best_svm_model.predict(X_test))
print(f"SVM 测试准确率: {svm_test_accuracy:.4f}")

best_elm_model, _ = tune_elm_model(X_train, y_train, X_test, y_test)
elm_probs_test = best_elm_model.predict_proba(X_test)
elm_test_accuracy = accuracy_score(y_test, best_elm_model.predict(X_test))
print(f"ELM 测试准确率: {elm_test_accuracy:.4f}")

best_xgb_model, _ = train_xgboost(X_train, y_train, X_test, y_test)
xgb_probs_test = best_xgb_model.predict_proba(X_test)
xgb_test_accuracy = accuracy_score(y_test, best_xgb_model.predict(X_test))
print(f"XGB 测试准确率: {xgb_test_accuracy:.4f}")

# 将所有基模型的预测概率拼接起来，作为元分类器的输入
stacking_features_train = np.hstack((svm_probs_train, elm_probs_train, xgb_probs_train))
stacking_features_test = np.hstack((svm_probs_test, elm_probs_test, xgb_probs_test))

# 训练元分类器
meta_classifier = LogisticRegression()
meta_classifier.fit(stacking_features_train, y_train)

# 使用元分类器进行预测
final_stacking_pred_train = meta_classifier.predict(stacking_features_train)
final_stacking_pred_test = meta_classifier.predict(stacking_features_test)

# 获取训练集和测试集的准确率
stacking_train_accuracy = accuracy_score(y_train, final_stacking_pred_train)
stacking_test_accuracy = accuracy_score(y_test, final_stacking_pred_test)

print(f"Stacking 模型训练准确率: {stacking_train_accuracy:.4f}")
print(f"Stacking 模型测试准确率: {stacking_test_accuracy:.4f}")

# 获取类别标签的字符串形式
target_names = [str(cls) for cls in encoder.classes_]

# 打印分类报告
print("Stacking 模型分类报告:\n",
      classification_report(y_test, final_stacking_pred_test, target_names=target_names, digits=4))

# 生成并显示混淆矩阵
cm = confusion_matrix(y_test, final_stacking_pred_test)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix (Stacking)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('stacking1.svg', format='svg', bbox_inches='tight')
plt.show()

print("训练集类别分布:")
print(pd.Series(y_train).value_counts())

print("测试集类别分布:")
print(pd.Series(y_test).value_counts())

