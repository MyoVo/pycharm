import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from nrbo_module import NRBO

# 参数搜索范围
param_ranges_svm = {
    'C': [0.1, 10],
    'gamma': [0.001, 1],
    'kernel': [1, 2, 3]  # 0: 'linear', 1: 'poly', 2: 'rbf'
}

def svm_fobj(params, X_train, y_train):
    # 确保 params 是一个字典
    if not isinstance(params, dict):
        params = {'C': params[0], 'gamma': params[1], 'kernel': int(params[2])}

    # 从 params 字典中提取参数
    C = max(param_ranges_svm['C'][0], min(param_ranges_svm['C'][1], params['C']))
    gamma = max(param_ranges_svm['gamma'][0], min(param_ranges_svm['gamma'][1], params['gamma']))
    kernel = ['linear', 'poly', 'rbf'][int(params['kernel']) % 3]

    # 使用提取的参数构建 SVM 模型
    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)

    # 使用交叉验证评估模型性能
    scores = cross_val_score(svm_model, X_train, y_train, cv=5)

    # 返回交叉验证得分的负值
    return -np.max(scores)

def optimize_svm_params(X_train, y_train, N, MaxIt):
    # 使用 param_ranges_svm 中的上下界
    LB = [param_ranges_svm['C'][0], param_ranges_svm['gamma'][0], param_ranges_svm['kernel'][0]]
    UB = [param_ranges_svm['C'][1], param_ranges_svm['gamma'][1], param_ranges_svm['kernel'][1]]

    dim = 3  # 使用待优化参数的维度

    _, best_params, best_score = NRBO(N, MaxIt, LB, UB, dim, svm_fobj,"SVM Model", X_train, y_train)

    return best_params, best_score

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, best_params, normalize=True):
    # 裁剪参数
    C, gamma, kernel_idx = [max(param_ranges_svm['C'][0], min(param_ranges_svm['C'][1], value)) for value in
                            best_params[:3]]
    kernel = ['linear', 'poly', 'rbf'][int(kernel_idx) % 3]

    # 输出最优参数
    print("最终参数 C:", C)
    print("最终参数 gamma:", gamma)
    print("最终参数 kernel:", kernel)

    # 使用最优参数训练 SVM 模型
    optimized_svm_model = SVC(C=C, gamma=gamma, kernel=kernel , probability= True)
    optimized_svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = optimized_svm_model.predict(X_test)

    # 输出准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    #report = classification_report(y_test, y_pred)
    # 这里修改了classification_report的调用，增加了digits参数
    report = classification_report(y_test, y_pred, digits=4)

    print("svm测试集准确率:", accuracy)
    print("svm分类报告:\n", report)

    return optimized_svm_model, accuracy, report



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap='viridis', normalize=True):
    """
    绘制混淆矩阵图形
    """
    if not title:
        title = 'Confusion Matrix'

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # 配置标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 在矩阵方块内显示百分比
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.show()

