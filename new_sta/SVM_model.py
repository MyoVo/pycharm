import numpy as np
from sklearn.svm import SVC
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def train_svm(X_train, y_train, X_test, y_test):
    # 定义 SVM 模型
    svm_model = SVC(probability=True)

    # 定义要调整的超参数空间
    param_grid = {
        'C': (0.1, 10.0, 'log-uniform'),
        'gamma': (0.001, 1.0, 'log-uniform'),
        'kernel': ['rbf']
    }

    # 使用 BayesSearchCV 进行贝叶斯优化搜索
    bayes_search = BayesSearchCV(estimator=svm_model, search_spaces=param_grid, n_iter=32, cv=3, scoring='accuracy', n_jobs=-1)
    bayes_search.fit(X_train, y_train)

    # 获取最佳参数的 SVM 模型
    best_svm_model = bayes_search.best_estimator_

    # 输出最佳参数
    print("Best parameters:", bayes_search.best_params_)

    # 使用最佳参数的 SVM 模型进行预测
    svm_probs = best_svm_model.predict_proba(X_test)

    # 计算准确率
    y_pred = np.argmax(svm_probs, axis=1)
    svm_accuracy = accuracy_score(y_test, y_pred)
    #print('SVM Accuracy:', svm_accuracy)

    # 绘制百分比混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 计算百分比
    plt.figure(figsize=(8, 6))
    # 绘制热图并指定自定义标签
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Supine', 'Side', 'Foetus', 'Prone'],
                yticklabels=['Supine', 'Side', 'Foetus', 'Prone'])
    plt.title('Confusion Matrix (Bo-SVM)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 保存图形为SVG文件
    plt.savefig('SVM.svg', format='svg', bbox_inches='tight')
    plt.show()

    return best_svm_model, svm_accuracy
