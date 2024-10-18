
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
def stack_models(mlp_probs, xgboost_probs,rf_probs,elm_probs,svm_probs,tabnet_probs, dndf_probs, y_test):
    # 将六个模型的概率预测结果合并
    combined_probs = np.hstack((mlp_probs, xgboost_probs,rf_probs,elm_probs,svm_probs,tabnet_probs, dndf_probs))

    # 定义MLP的参数空间
    param_space = {
        'hidden_layer_sizes': Integer(50, 500),  # 隐藏层的大小
        'alpha': Real(1e-4, 1e-1, prior='log-uniform'),  # L2正则化
        'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform'),  # 初始学习率
    }

    # 创建一个BayesSearchCV对象
    mlp_bayes_search = BayesSearchCV(
        MLPClassifier(max_iter=2000),
        param_space,
        n_iter=32,  # 贝叶斯优化迭代次数
        scoring=make_scorer(accuracy_score),  # 使用准确率评分
        cv=5,  # 交叉验证的折数
        n_jobs=-1,  # 使用所有可用的CPU核心
        verbose=1
    )

    # 使用拟合数据训练模型
    mlp_bayes_search.fit(combined_probs, y_test)

    # 获取最佳模型
    best_mlp = mlp_bayes_search.best_estimator_

    # 使用最佳模型进行预测
    meta_predictions = best_mlp.predict(combined_probs)

    # 计算准确率
    meta_accuracy = accuracy_score(y_test, meta_predictions)


    # 返回准确率和最佳参数
    return meta_accuracy, mlp_bayes_search.best_params_










