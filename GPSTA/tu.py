import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def plot_confusion_matrix(y_true, y_pred, title, normalize=False):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 如果选择了归一化，将混淆矩阵中的值除以每行的总和
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建一个图形，并使用 seaborn 的热力图功能绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%' if normalize else 'd', cmap='Blues')

    # 添加标题和轴标签
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 显示图形
    plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize


def plot_performance_curves(models, labels, X_test, y_test, n_classes):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = {model_label: {metric: [] for metric in metrics} for model_label in labels}

    # Ensure y_test is in integer label format
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1) + 1
    else:
        y_true = y_test

    for model, model_label in zip(models, labels):
        y_pred = np.argmax(model.predict(X_test), axis=1) + 1  # Adjust to labels 1,2,3,4
        y_true_binarized = label_binarize(y_true, classes=[1, 2, 3, 4])

        for i in range(n_classes):
            # Calculate metrics for the current class
            y_true_i = y_true_binarized[:, i]
            y_pred_i = label_binarize(y_pred, classes=[1, 2, 3, 4])[:, i]

            scores[model_label]['Accuracy'].append(accuracy_score(y_true_i, y_pred_i))
            scores[model_label]['Precision'].append(precision_score(y_true_i, y_pred_i, zero_division=0))
            scores[model_label]['Recall'].append(recall_score(y_true_i, y_pred_i, zero_division=0))
            scores[model_label]['F1 Score'].append(f1_score(y_true_i, y_pred_i, zero_division=0))

    plt.figure(figsize=(15, 8))  # Set the size of the chart
    for metric in metrics:
        for model_label in labels:
            # Create abbreviated labels, e.g., 'Baseline Model Accuracy' -> 'BMA'
            abbreviated_label = ''.join(word[0] for word in (model_label.split() + metric.split()))
            plt.plot(range(1, n_classes + 1), scores[model_label][metric], marker='o', label=abbreviated_label)

    plt.title('Model Performance by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.xticks(range(1, n_classes + 1))
    plt.show()


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(model, X_test, y_test, n_classes):
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Ensure y_test is in integer label format
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    y_pred = model.predict(X_test)
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i + 1} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curves')
    plt.legend(loc="lower right")
    plt.show()

from sklearn.metrics import precision_recall_curve
# 定义性能曲线绘制函数
def plot_stacking_model_performance(history, y_true, y_pred_prob):
    plt.figure(figsize=(12, 6))
    n_classes = 4

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制每个类别的ROC曲线
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 计算每个类别的Precision和Recall
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_pred_prob[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # 绘制每个类别的PR曲线
    plt.subplot(1, 2, 2)
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {pr_auc[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()



