import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, roc_auc_score
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 定义调优 MLP 模型的函数
def tune_mlp_model(X_train, y_train):
    # 定义 MLP 模型
    mlp_model = MLPClassifier(max_iter=1000, validation_fraction=0.1, n_iter_no_change=10, early_stopping=True)

    # 定义要调整的超参数空间
    param_grid = {
        'hidden_layer_sizes': Integer(50, 500),
        'activation': Categorical(['relu', 'tanh']),
        'solver': Categorical(['adam', 'sgd']),
        'alpha': Real(0.0001, 0.01, prior='log-uniform')
    }

    # 使用 BayesSearchCV 进行贝叶斯优化搜索
    bayes_search = BayesSearchCV(mlp_model, param_grid, n_iter=30, cv=5, n_jobs=-1)

    # 尝试捕捉参数越界的异常
    try:
        bayes_search.fit(X_train, y_train)
    except ValueError as e:
        print("参数越界错误：", e)
        return None

    # 获取最佳参数的 MLP 模型
    best_mlp_model = bayes_search.best_estimator_

    # 输出最佳参数和最佳得分
    print("Best Parameters:", bayes_search.best_params_)
    print("Best Score:", bayes_search.best_score_)

    return best_mlp_model

# 定义计算六个指标的函数
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 计算AUC
    y_prob = model.predict_proba(X_test)
    if len(np.unique(y_test)) > 2:  # 多分类问题
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:  # 二分类问题
        auc = roc_auc_score(y_test, y_prob[:, 1])

    # 计算精确率、召回率、F1得分和Kappa指数
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    kappa = cohen_kappa_score(y_test, y_pred)

    # 输出测试集的六个指标
    print("MLP Model Accuracy:", accuracy)
    print("AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Kappa:", kappa)

    return accuracy, auc, precision, recall, f1, kappa

# 读取数据
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将字符标签转换为整数标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 调用函数调优 MLP 模型并获取最佳模型
best_mlp_model = tune_mlp_model(X_train, y_train)

# 计算并输出测试集的六个指标
if best_mlp_model is not None:
    accuracy, auc, precision, recall, f1, kappa = calculate_metrics(best_mlp_model, X_test, y_test)
    print("测试集准确率：", accuracy)
    print("测试集AUC：", auc)
    print("测试集Precision：", precision)
    print("测试集Recall：", recall)
    print("测试集F1 Score：", f1)
    print("测试集Kappa：", kappa)
