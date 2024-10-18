from sklearn.preprocessing import StandardScaler, LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from MLP_model import tune_mlp_model
from XGB_model import train_xgboost
from Tabnet_model import train_tabnet
from Stacking_model import stack_models
import warnings
from scipy.special import softmax
from RF_model import train_random_forest
from SVM_model import train_svm
from DNDT_model import tune_dndf_model
from sklearn.metrics import classification_report, roc_auc_score
from ELM_model import tune_elm_model
import tensorflow as tf
# 禁用所有GPU
tf.config.set_visible_devices([], 'GPU')

# 打印所有可用的设备
print("All available devices:")
for device in tf.config.list_physical_devices():
    print(device)

# 打印是否使用了GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

# 调整 ELM 模型
best_elm_model, elm_accuracy = tune_elm_model(X_train, y_train, X_test, y_test)

# 训练 TabNet 模型并获取预测概率
tabnet_model, tabnet_accuracy = train_tabnet(X_train, y_train, X_test, y_test)

# 训练 DNDF 模型
dndf_model_model, dndf_accuracy = tune_dndf_model(X_train, y_train, X_test, y_test)
# 训练 XGBoost 模型
best_xgboost_model, xgboost_accuracy = train_xgboost(X_train, y_train, X_test, y_test)

# 训练 RF 模型
best_rf_model, rf_accuracy = train_random_forest(X_train, y_train, X_test, y_test)

# 训练 svm 模型
best_svm_model, svm_accuracy = train_svm(X_train, y_train, X_test, y_test)

# 调整 MLP 模型
best_mlp_model, mlp_accuracy = tune_mlp_model(X_train, y_train, X_test, y_test)

# 获取各个模型在测试集上的预测概率
xgboost_probs = best_xgboost_model.predict_proba(X_test)

svm_probs = best_svm_model.predict_proba(X_test)

# 尝试获取 DNDF 模型的预测概率和预测类别
try:
    dndf_probs = dndf_model_model.predict_proba(X_test)
    dndf_preds = np.argmax(dndf_probs, axis=1)  # 将概率转换为类别预测
except AttributeError:
    # 如果模型没有predict_proba方法，使用predict方法作为替代
    dndf_preds = dndf_model_model.predict(X_test)
    # 创建一个one-hot编码的概率矩阵，假设预测正确的类别具有1.0的概率，其余为0
    dndf_probs = np.zeros((len(dndf_preds), y_encoded.max() + 1))
    dndf_probs[np.arange(len(dndf_preds)), dndf_preds] = 1.0
# 首先确保类标签是字符串格式
target_names = [str(cls) for cls in encoder.classes_]
# 接下来可以直接使用 dndf_preds 生成分类报告
print("DNDF 模型分类报告:\n", classification_report(y_test, dndf_preds, target_names=target_names))

rf_probs = best_rf_model.predict_proba(X_test)

# 创建 X_test_dict
X_test_dict = {'dummy': X_test}
tabnet_raw_output = tabnet_model.predict(X_test_dict)
tabnet_probs = softmax(tabnet_raw_output, axis=1)

mlp_probs = best_mlp_model.predict_proba(X_test)

elm_probs = best_elm_model.predict_proba(X_test)

# 使用 stacking 模型
meta_accuracy = stack_models(mlp_probs, xgboost_probs,rf_probs,elm_probs,svm_probs,tabnet_probs, dndf_probs,y_test)

# 打印各模型准确率和 stacking 模型准确率
print("XGBoost 模型准确率:", xgboost_accuracy)
print("RF准确率:",rf_accuracy)
print("DNDT准确率:",dndf_accuracy)
print("SVM准确率:",svm_accuracy)
print("TabNet 模型准确率:", tabnet_accuracy)
print("MLP 模型准确率:", mlp_accuracy)
print("ELM 模型准确率:", elm_accuracy)
print("Stacking 模型准确率:", meta_accuracy)

# XGBoost 模型分类报告
xgboost_pred = best_xgboost_model.predict(X_test)

# 随机森林模型分类报告
rf_pred = best_rf_model.predict(X_test)

# SVM 模型分类报告
svm_pred = best_svm_model.predict(X_test)

# TabNet 模型分类报告
tabnet_pred = np.argmax(tabnet_probs, axis=1)

# MLP 模型分类报告
mlp_pred = best_mlp_model.predict(X_test)

# MLP 模型分类报告
elm_pred = best_elm_model.predict(X_test)

# 假定 stack_models 返回了预测值，如果没有，则需要调整 stack_models 函数返回预测值
stacking_pred = stack_models(mlp_probs, xgboost_probs,rf_probs,elm_probs,svm_probs,tabnet_probs, dndf_probs, y_test)

# 将类标签转换为字符串格式的列表
target_names = [str(cls) for cls in encoder.classes_]

# 现在使用这个字符串列表作为 target_names 参数
print("XGBoost 模型分类报告:\n", classification_report(y_test, xgboost_pred, target_names=target_names, digits=4))
print("RF 模型分类报告:\n", classification_report(y_test, rf_pred, target_names=target_names, digits=4))
print("SVM 模型分类报告:\n", classification_report(y_test, svm_pred, target_names=target_names, digits=4))
print("DNDT 模型分类报告:\n", classification_report(y_test, dndf_preds, target_names=target_names, digits=4))
print("TabNet 模型分类报告:\n", classification_report(y_test, tabnet_pred, target_names=target_names, digits=4))
print("MLP 模型分类报告:\n", classification_report(y_test, mlp_pred, target_names=target_names, digits=4))
print("ELM 模型分类报告:\n", classification_report(y_test, elm_pred, target_names=target_names, digits=4))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# 假设这里的 xgboost_probs, rf_probs, 等变量已经在你的环境中定义好了
probabilities = np.column_stack([
    xgboost_probs[:, 1],
    rf_probs[:, 1],
    svm_probs[:, 1],
    dndf_probs[:, 1],
    tabnet_probs[:, 1],
    mlp_probs[:, 1],
    elm_probs[:, 1]

])
# 计算Spearman相关系数矩阵
corr, _ = spearmanr(probabilities)

# 创建DataFrame方便热图显示
model_names = ['Bo-XGB', 'Bo-RF', 'Bo-SVM', 'Bo-DNDT', 'Bo-TabNet', 'Bo-MLP','Bo-ELM']
corr_df = pd.DataFrame(corr, index=model_names, columns=model_names)

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='YlGnBu', fmt=".2f", vmin=0.5, vmax=1)
plt.title('Spearman Correlation Matrix')
# 保存图形为SVG文件
plt.savefig('re1.svg', format='svg', bbox_inches='tight')
plt.show()

from sklearn.metrics import cohen_kappa_score

# XGBoost 模型的 Kappa 系数
xgboost_kappa = cohen_kappa_score(y_test, xgboost_pred)
print("XGBoost 模型 Kappa 系数:", xgboost_kappa)

# RF 模型的 Kappa 系数
rf_kappa = cohen_kappa_score(y_test, rf_pred)
print("RF 模型 Kappa 系数:", rf_kappa)

# SVM 模型的 Kappa 系数
svm_kappa = cohen_kappa_score(y_test, svm_pred)
print("SVM 模型 Kappa 系数:", svm_kappa)

# DNDF 模型的 Kappa 系数
dndf_kappa = cohen_kappa_score(y_test, dndf_preds)
print("DNDF 模型 Kappa 系数:", dndf_kappa)

# TabNet 模型的 Kappa 系数
tabnet_kappa = cohen_kappa_score(y_test, tabnet_pred)
print("TabNet 模型 Kappa 系数:", tabnet_kappa)

# MLP 模型的 Kappa 系数
mlp_kappa = cohen_kappa_score(y_test, mlp_pred)
print("MLP 模型 Kappa 系数:", mlp_kappa)

elm_kappa = cohen_kappa_score(y_test, elm_pred)
print("MLP 模型 Kappa 系数:", elm_kappa)

# 计算 AUC 值
xgboost_auc = roc_auc_score(y_test, xgboost_probs, multi_class='ovo', average='macro')
rf_auc = roc_auc_score(y_test, rf_probs, multi_class='ovo', average='macro')
svm_auc = roc_auc_score(y_test, svm_probs, multi_class='ovo', average='macro')
dndf_auc = roc_auc_score(y_test, dndf_probs, multi_class='ovo', average='macro')
tabnet_auc = roc_auc_score(y_test, tabnet_probs, multi_class='ovo', average='macro')
mlp_auc = roc_auc_score(y_test, mlp_probs, multi_class='ovo', average='macro')
elm_auc = roc_auc_score(y_test, elm_probs, multi_class='ovo', average='macro')

# 打印各模型的 AUC 值
print("XGBoost 模型 AUC 值:", xgboost_auc)
print("RF 模型 AUC 值:", rf_auc)
print("SVM 模型 AUC 值:", svm_auc)
print("DNDF 模型 AUC 值:", dndf_auc)
print("TabNet 模型 AUC 值:", tabnet_auc)
print("MLP 模型 AUC 值:", mlp_auc)
print("ELM 模型 AUC 值:", elm_auc)


