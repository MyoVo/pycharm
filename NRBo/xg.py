import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, log_loss
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from SVM_module import optimize_svm_params, train_and_evaluate_svm
from XGb import optimize_xgboost_params, train_and_evaluate_xgboost
from fully_connected_nn import optimize_nn_params, train_and_evaluate_nn  # 替换为全连接神经网络模块
from rbf import optimize_random_forest_params,train_and_evaluate_random_forest
from DT import optimize_decision_tree_params, train_and_evaluate_decision_tree

# 加载数据集
data = pd.read_excel('789.xlsx')

# 划分数据集
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = y - 1

'''
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
'''
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

N = 10
MaxIt = 10

# 调用各个模型的优化函数
best_params_svm, best_score_svm = optimize_svm_params(X_train_scaled, y_train, N, MaxIt)
best_params_xgboost, best_score_xgboost = optimize_xgboost_params(X_train_scaled, y_train, N, MaxIt)
best_params_fcnn, best_score_fcnn = optimize_nn_params(X_train_scaled, y_train, N, MaxIt)  # 替换为全连接神经网络模型
best_params_rf, best_score_rf = optimize_random_forest_params(X_train_scaled, y_train, N, MaxIt)
best_params_dt, best_score_dt = optimize_decision_tree_params(X_train_scaled, y_train, N, MaxIt)

# 训练和评估各个模型
optimized_svm_model, accuracy_svm, report_svm = train_and_evaluate_svm(X_train_scaled, y_train, X_test_scaled, y_test, best_params_svm)
optimized_xgb_model, accuracy_xgb, report_xgb = train_and_evaluate_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, best_params_xgboost)
optimized_fcnn_model, accuracy_fcnn, report_fcnn = train_and_evaluate_nn(X_train_scaled, y_train, X_test_scaled, y_test, best_params_fcnn)  # 替换为全连接神经网络模型
optimized_rf_model, accuracy_rf, report_rf = train_and_evaluate_random_forest(X_train_scaled, y_train, X_test_scaled, y_test, best_params_rf)
optimized_dt_model, accuracy_dt, report_dt = train_and_evaluate_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test, best_params_dt)

def plot_confusion_matrix_custom(y_true, y_pred, classes, title, normalize=False):
    conf_matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

# 绘制各个模型的混淆矩阵
plot_confusion_matrix_custom(y_test, optimized_svm_model.predict(X_test_scaled), classes=np.unique(y_test),
                             title='SVM Confusion Matrix', normalize=True)
plot_confusion_matrix_custom(y_test, optimized_xgb_model.predict(X_test_scaled), classes=np.unique(y_test),
                             title='XGBoost Confusion Matrix', normalize=True)
plot_confusion_matrix_custom(y_test, optimized_fcnn_model.predict(X_test_scaled), classes=np.unique(y_test),
                             title='Fully Connected NN Confusion Matrix', normalize=True)  # 替换为全连接神经网络模型
plot_confusion_matrix_custom(y_test, optimized_rf_model.predict(X_test_scaled), classes=np.unique(y_test),
                             title='Random Forest Confusion Matrix', normalize=True)
plot_confusion_matrix_custom(y_test, optimized_dt_model.predict(X_test_scaled), classes=np.unique(y_test),
                             title='Decision Tree Confusion Matrix', normalize=True)

base_models = [('SVM', optimized_svm_model),
               ('XGBoost', optimized_xgb_model),
               ('Fully Connected NN', optimized_fcnn_model),  # 替换为全连接神经网络模型
               ('Random Forest', optimized_rf_model),
               ('Decision Tree', optimized_dt_model)]

base_models_accuracies = {name: accuracy for name, accuracy in [('SVM', accuracy_svm),
                                                                ('XGBoost', accuracy_xgb),
                                                                ('Fully Connected NN', accuracy_fcnn),  # 替换为全连接神经网络模型
                                                                ('Random Forest', accuracy_rf),
                                                                ('Decision Tree', accuracy_dt)]}

# 选择最好的基模型作为元模型
best_base_model_name = max(base_models_accuracies, key=base_models_accuracies.get)
best_base_model = dict(base_models)[best_base_model_name]


stacking_model = StackingClassifier(estimators=base_models, final_estimator=best_base_model)

# 用交叉验证对堆叠模型进行评估
cv_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

# 输出每一折的准确率
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

# 训练堆叠模型
stacking_model.fit(X_train_scaled, y_train)

# 对测试集进行预测
y_pred_stacking = stacking_model.predict(X_test_scaled)

# 评估堆叠模型
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print("Stacking Model Accuracy:", accuracy_stacking)
report_stacking = classification_report(y_test, y_pred_stacking, digits=4)
print("Classification Report for Stacking Model:")
print(report_stacking)

# 绘制堆叠模型的混淆矩阵
plot_confusion_matrix_custom(y_test, y_pred_stacking, classes=np.unique(y_test),
                             title='Stacking Confusion Matrix', normalize=True)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

# 多类问题，需要首先将标签二值化
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_binarized.shape[1]

# 对于每个模型，计算每个类的ROC曲线和AUC
def compute_roc_auc(model, X_test_scaled, y_test_binarized):
    # 计算每个类的概率或决策函数
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_scaled)
    else:
        y_score = model.predict_proba(X_test_scaled)

    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc

# 绘制所有ROC曲线
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
models = [optimized_svm_model, optimized_xgb_model, optimized_fcnn_model,  # 替换为全连接神经网络模型
          optimized_rf_model, optimized_dt_model, stacking_model]
model_names = ['NRBO-SVM', 'NRBO-XGBoost', 'NRBO-Fully Connected NN',  # 替换为全连接神经网络模型
               'NRBO-Random Forest', 'NRBO-DT', 'Stacking']

for model, color, name in zip(models, colors, model_names):
    fpr, tpr, roc_auc = compute_roc_auc(model, X_test_scaled, y_test_binarized)
    plt.plot(fpr["micro"], tpr["micro"], color=color, lw=1.5,
             label='ROC curve of {0} (area = {1:.2f})'.format(name, roc_auc["micro"]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right", fontsize='x-small')  # 缩小文本框大小
plt.tight_layout()  # 调整布局以防止文本重叠
plt.show()

from sklearn.metrics import cohen_kappa_score

# 计算Kappa值
kappa_svm = cohen_kappa_score(y_test, optimized_svm_model.predict(X_test_scaled))
kappa_xgb = cohen_kappa_score(y_test, optimized_xgb_model.predict(X_test_scaled))
kappa_fcnn = cohen_kappa_score(y_test, optimized_fcnn_model.predict(X_test_scaled))  # 替换为全连接神经网络模型
kappa_rf = cohen_kappa_score(y_test, optimized_rf_model.predict(X_test_scaled))
kappa_dt = cohen_kappa_score(y_test, optimized_dt_model.predict(X_test_scaled))
kappa_stacking = cohen_kappa_score(y_test, y_pred_stacking)

# 输出每个模型的Kappa值
print("Kappa Score for SVM:", kappa_svm)
print("Kappa Score for XGBoost:", kappa_xgb)
print("Kappa Score for Fully Connected NN:", kappa_fcnn)  # 替换为全连接神经网络模型
print("Kappa Score for Random Forest:", kappa_rf)
print("Kappa Score for Decision Tree:", kappa_dt)
print("Kappa Score for Stacking Model:", kappa_stacking)
