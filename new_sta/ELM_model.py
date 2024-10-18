from hpelm import ELM
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

class SklearnCompatibleELM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=10, activation_func='sigm'):  # 改为 'sigm'
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.model = None
        self.onehot_encoder = OneHotEncoder(categories='auto', sparse=False)

    def fit(self, X, y):
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            y = self.onehot_encoder.fit_transform(y.reshape(-1, 1))
        self.model = ELM(X.shape[1], y.shape[1])
        self.model.add_neurons(self.n_neurons, self.activation_func)
        self.model.train(X, y, 'c')
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        predictions = self.model.predict(X)
        probabilities = softmax(predictions, axis=1)
        return probabilities

    def score(self, X, y):
        predictions = self.predict_proba(X)
        y_pred = np.argmax(predictions, axis=1)
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        return accuracy_score(y, y_pred)


def tune_elm_model(X_train, y_train, X_test, y_test):
    # 使用新的包装器创建 ELM 模型
    elm_model = SklearnCompatibleELM()

    # 定义要调整的超参数空间
    param_grid = {
        'n_neurons': Integer(50, 500),
        'activation_func': Categorical(['sigm', 'tanh', 'lin'])  # 改为 'sigm'
    }

    # 使用 BayesSearchCV 进行贝叶斯优化搜索
    optimizer_kwargs = {'base_estimator': "GP"}
    bayes_search = BayesSearchCV(elm_model, param_grid, n_iter=30, cv=5, n_jobs=-1, optimizer_kwargs=optimizer_kwargs)

    # 进行模型拟合
    try:
        bayes_search.fit(X_train, y_train)
    except ValueError as e:
        print("参数越界错误：", e)
        return None

    # 获取最佳模型和预测结果
    best_elm_model = bayes_search.best_estimator_
    y_pred = best_elm_model.predict(X_test)

    # 如果 y_pred 是一维数组，我们不能使用 y_pred.shape[1]
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # 如果 y_test 是独热编码，需要转换为类标签索引
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    # 计算和输出准确率
    elm_accuracy = accuracy_score(y_test, y_pred)
    print("Best Parameters:", bayes_search.best_params_)
    #print("Best Score:", bayes_search.best_score_)
    #print("ELM Model Accuracy:", elm_accuracy)

    # 绘制百分比混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    # 绘制热图并指定自定义标签
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Supine', 'Side', 'Foetus', 'Prone'],
                yticklabels=['Supine', 'Side', 'Foetus', 'Prone'])
    plt.title('Confusion Matrix (Bo-ELM)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 保存图形为SVG文件
    plt.savefig('ELM.svg', format='svg', bbox_inches='tight')
    plt.show()


    return best_elm_model, elm_accuracy


