import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def train_random_forest(X_train, y_train, X_test, y_test):
    # 定义随机森林模型
    random_forest_model = RandomForestClassifier()

    # 定义要调整的超参数空间
    param_grid = {
        'n_estimators': (50, 500),
        'max_depth': (3, 7),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    }

    # 使用 BayesSearchCV 进行贝叶斯优化搜索
    bayes_search = BayesSearchCV(estimator=random_forest_model, search_spaces=param_grid, n_iter=32, cv=3, scoring='accuracy', n_jobs=-1)
    bayes_search.fit(X_train, y_train)

    # 获取最佳参数的随机森林模型
    best_random_forest_model = bayes_search.best_estimator_

    # 输出最佳参数
    print("Best parameters:", bayes_search.best_params_)

    # 使用最佳参数的随机森林模型进行预测
    random_forest_probs = best_random_forest_model.predict_proba(X_test)

    # 计算准确率
    y_pred = np.argmax(random_forest_probs, axis=1)
    random_forest_accuracy = accuracy_score(y_test, y_pred)
    print('Random Forest Accuracy:', random_forest_accuracy)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 绘制百分比混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 计算百分比
    plt.figure(figsize=(8, 6))
    # 绘制热图并指定自定义标签
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Supine', 'Side', 'Foetus', 'Prone'],
                yticklabels=['Supine', 'Side', 'Foetus', 'Prone'])
    plt.title('Confusion Matrix (Bo-RF)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 保存图形为SVG文件
    plt.savefig('RF.svg', format='svg', bbox_inches='tight')
    plt.show()

    return best_random_forest_model, random_forest_accuracy
