from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def tune_mlp_model(X_train, y_train, X_test, y_test):
    # 定义 MLP 模型
    mlp_model = MLPClassifier(max_iter=500)

    # 定义要调整的超参数空间
    param_grid = {
        'hidden_layer_sizes': Integer(50, 500),  # 调整为适当的范围
        'activation': Categorical(['relu', 'tanh']),
        'solver': Categorical(['adam', 'sgd']),
        'alpha': Real(0.0001, 0.01, prior='log-uniform')
    }

    # 使用 BayesSearchCV 进行贝叶斯优化搜索
    optimizer_kwargs = {'base_estimator': "GP"}  # 使用高斯过程作为基础估计器
    bayes_search = BayesSearchCV(mlp_model, param_grid, n_iter=30, cv=5, n_jobs=-1, optimizer_kwargs=optimizer_kwargs)

    # 尝试捕捉参数越界的异常
    try:
        bayes_search.fit(X_train, y_train)
    except ValueError as e:
        print("参数越界错误：", e)
        return None

    # 获取最佳参数的 MLP 模型
    best_mlp_model = bayes_search.best_estimator_

    # 使用最佳参数的 MLP 模型进行预测
    y_pred = best_mlp_model.predict(X_test)

    # 计算准确率
    mlp_accuracy = accuracy_score(y_test, y_pred)

    # 输出最佳参数和准确率
    print("Best Parameters:", bayes_search.best_params_)
    print("Best Score:", bayes_search.best_score_)
    print("MLP Model Accuracy:", mlp_accuracy)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 绘制百分比混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    # 绘制热图并指定自定义标签
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Supine', 'Side', 'Foetus', 'Prone'],
                yticklabels=['Supine', 'Side', 'Foetus', 'Prone'])
    plt.title('Confusion Matrix (Bo-MLP)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 保存图形为SVG文件
    plt.savefig('mlp.svg', format='svg', bbox_inches='tight')
    plt.show()

    return best_mlp_model, mlp_accuracy

