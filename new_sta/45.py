import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
import warnings
import seaborn as sns
import os
from sklearn.utils.multiclass import unique_labels
from scipy.special import softmax
import contextlib
from hpelm import ELM
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore", message="Covariance matrix is not full rank", category=UserWarning)

# 禁用所有GPU
tf.config.set_visible_devices([], 'GPU')

# ELM 包装器
class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=100, activation='sigm', epochs=500):
        self.n_hidden = n_hidden
        self.activation = activation
        self.epochs = epochs

    def fit(self, X, y):
        # 存储训练集中看到的唯一类标签
        self.classes_ = unique_labels(y)
        y_one_hot = to_categorical(y, num_classes=len(self.classes_))
        self.elm = ELM(X.shape[1], y_one_hot.shape[1])
        self.elm.add_neurons(self.n_hidden, self.activation)
        for _ in range(self.epochs):
            self.elm.train(X, y_one_hot, 'c')
        return self

    def predict(self, X):
        return self.elm.predict(X).argmax(axis=1)

    def predict_proba(self, X):
        # 使用 scipy 的 softmax 函数来计算概率
        predictions = self.elm.predict(X)
        return softmax(predictions, axis=1)

    def get_params(self, deep=False):
        return {'n_hidden': self.n_hidden, 'activation': self.activation, 'epochs': self.epochs}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.sys.stdout
        os.sys.stdout = devnull
        try:
            yield
        finally:
            os.sys.stdout = old_stdout

# 加载数据
data = pd.read_excel('6.xlsx')

# 获取特征名称
feature_names = data.columns[:-1].tolist()

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

# SVM贝叶斯优化
param_space_svm = {
'C': Real(0.1, 20.0, prior='log-uniform'),
'gamma': Real(0.001, 1.0, prior='log-uniform'),
'kernel': Categorical(['rbf'])
}

opt_svm = BayesSearchCV(SVC(probability=True), param_space_svm, n_iter=32, cv=5, scoring='accuracy')
opt_svm.fit(X_train, y_train)
best_svm = opt_svm.best_estimator_
print(f"最佳SVM参数: {opt_svm.best_params_}")

# XGBoost贝叶斯优化
param_space_xgb = {
    'learning_rate': Real(0.01, 0.5),
    'max_depth': Integer(3, 20),
    'n_estimators': Integer(100, 500),

}

opt_xgb = BayesSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_space_xgb, n_iter=32, cv=5, scoring='accuracy')
opt_xgb.fit(X_train, y_train)
best_xgb = opt_xgb.best_estimator_
print(f"最佳XGBoost参数: {opt_xgb.best_params_}")

# ELM贝叶斯优化
param_space_elm = {
    'n_hidden': Integer(100, 500),
    'activation': Categorical(['sigm', 'tanh']),
}

opt_elm = BayesSearchCV(ELMClassifier(), param_space_elm, n_iter=32, cv=5, scoring='accuracy')
opt_elm.fit(X_train, y_train)
best_elm = opt_elm.best_estimator_
print(f"最佳ELM参数: {opt_elm.best_params_}")


# 定义Stacking模型
stacking_model = StackingClassifier(
    estimators=[ ('elm', best_elm), ('xgb', best_xgb), ('dndt', best_svm)],
    final_estimator=LogisticRegression(),
    cv=5
)

# 训练Stacking模型
with suppress_stdout():
    stacking_model.fit(X_train, y_train)

# 基模型的准确率
base_models = [ best_elm, best_xgb,best_svm]
model_names = ['SVM', 'elm', 'XGB']

# 在训练集上进行预测
y_train_pred = stacking_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nStacking Model Training Set Accuracy: {train_accuracy:.4f}")

# 在测试集上进行预测
y_test_pred = stacking_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Stacking Model Test Set Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import cohen_kappa_score
test_kappa = cohen_kappa_score(y_test, y_test_pred)
print(f"Stacking Model Test Set Kappa: {test_kappa:.4f}")
'''
from sklearn.model_selection import GridSearchCV
# 定义Stacking模型
stacking_model = StackingClassifier(
    estimators=[('elm', best_elm), ('xgb', best_xgb), ('svm', best_svm)],
    final_estimator=LogisticRegression(),
    cv=5
)
# 对逻辑回归分类器进行正则化调参
param_grid = {

    'final_estimator__C': [ 0.01, 0.1, 1, 10, 100]
}

# 使用交叉验证进行参数搜索
grid_search = GridSearchCV(stacking_model, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

# 获取最佳参数的堆叠模型
best_stacking_model = grid_search.best_estimator_

# 在训练集上进行预测
y_train_pred = best_stacking_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nBest Stacking Model Training Set Accuracy: {train_accuracy:.4f}")

# 在测试集上进行预测
y_test_pred = best_stacking_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Best Stacking Model Test Set Accuracy: {test_accuracy:.4f}")

# 计算Kappa系数
from sklearn.metrics import cohen_kappa_score
test_kappa = cohen_kappa_score(y_test, y_test_pred)
print(f"Best Stacking Model Test Set Kappa: {test_kappa:.4f}")



# 输出基模型的准确率和混淆矩阵
models = {'SVM': best_svm, 'XGBoost': best_xgb, 'ELM': best_elm}

for name, model in models.items():
    # 在训练集上进行预测
    y_train_pred_base = model.predict(X_train)
    train_accuracy_base = accuracy_score(y_train, y_train_pred_base)
    print(f"\n{name} Training Set Accuracy: {train_accuracy_base:.4f}")
    print(f"{name} Training Set Confusion Matrix:\n{confusion_matrix(y_train, y_train_pred_base)}")

    # 在测试集上进行预测
    y_test_pred_base = model.predict(X_test)
    test_accuracy_base = accuracy_score(y_test, y_test_pred_base)
    print(f"{name} Test Set Accuracy: {test_accuracy_base:.4f}")
    print(f"{name} Test Set Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred_base)}")


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

train_sizes = np.linspace(0.1, 0.9, 9)  # 确定训练集大小
train_errors = []
test_errors = []

for train_size in train_sizes:
    # 随机选择指定比例的训练数据，这里的test_size计算以确保train_size是所需的比例
    X_train_subset, _, y_train_subset, _ = train_test_split(
        X_train, y_train, train_size=train_size, stratify=y_train, random_state=42)

    skf = StratifiedKFold(n_splits=5)
    fold_train_errors = []
    fold_test_errors = []

    for train_index, val_index in skf.split(X_train_subset, y_train_subset):
        X_fold_train, X_fold_val = X_train_subset[train_index], X_train_subset[val_index]
        y_fold_train, y_fold_val = y_train_subset[train_index], y_train_subset[val_index]

        stacking_model.fit(X_fold_train, y_fold_train)

        y_fold_train_pred = stacking_model.predict(X_fold_train)
        fold_train_errors.append(1 - accuracy_score(y_fold_train, y_fold_train_pred))

        y_fold_val_pred = stacking_model.predict(X_fold_val)
        fold_test_errors.append(1 - accuracy_score(y_fold_val, y_fold_val_pred))

    train_errors.append(np.mean(fold_train_errors))
    test_errors.append(np.mean(fold_test_errors))

plt.figure()
plt.plot(train_sizes * 100, train_errors, label='Training set', marker='o', linestyle='-', color='blue')
plt.plot(train_sizes * 100, test_errors, label='Test set', marker='^', linestyle='-', color='orange')
plt.xlabel('Training set in percent (%)')
plt.ylabel('Misclassification error')
plt.title('Convergence curve of the stacking ensemble model')
plt.legend()
plt.grid(False)
plt.savefig('convergence_curve.svg', format='svg', bbox_inches='tight')
plt.show()

print("Training set sizes (%):", [int(size * 100) for size in train_sizes])
print("Training errors:", train_errors)
print("Test errors:", test_errors)
print(f"\nStacking Model Training Set Accuracy: {train_accuracy:.4f}")
print(f"Stacking Model Test Set Accuracy: {test_accuracy:.4f}")

# 绘制百分比混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix for Stacking Model (Percentage)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('stacking_confusion_matrix_percentage.svg', format='svg', bbox_inches='tight')
plt.show()

import shap

# 创建类别映射
class_mapping = {0: "Supine", 1: "Side", 2: "Foetus", 3: "Prone"}
mapped_class_names = [class_mapping[i] for i in best_stacking_model.classes_]

# 计算SHAP值
explainer = shap.KernelExplainer(best_stacking_model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test)

# 输出每个特征的平均绝对SHAP值
print("Average Absolute SHAP values per feature for each class:")
for i, class_name in enumerate(mapped_class_names):
    print(f"Class: {class_name}")
    avg_shap = np.abs(shap_values[i]).mean(axis=0)
    for feature_name, shap_value in zip(feature_names, avg_shap):
        print(f"{feature_name}: {shap_value}")

# 计算并打印每个特征在所有类别中的汇总SHAP值
total_shap_values = np.sum([np.abs(shap_values[i]) for i in range(len(mapped_class_names))], axis=0)
total_mean_shap_values = total_shap_values.mean(axis=0)

print("\nTotal SHAP values per feature across all classes:")
for feature_name, shap_value in zip(feature_names, total_mean_shap_values):
    print(f"{feature_name}: {shap_value}")

# 绘制条形图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar",
                  class_names=mapped_class_names, show=False)

# 保存并显示图形
plt.savefig('feature_importance_bar.svg', format='svg', bbox_inches='tight')
plt.show()

# 为每个输出类别绘制点图并保存
for i in range(len(mapped_class_names)):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[i], X_test, feature_names=feature_names, plot_type="dot", show=False)
    plt.title(f'SHAP Dot Plot for Class {mapped_class_names[i]}')
    plt.savefig(f'shap_dot_plot_class_{mapped_class_names[i]}.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

'''