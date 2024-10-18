import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shap.explainers.tf_utils import tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from hpelm import ELM
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax
import warnings

warnings.filterwarnings("ignore")

class SklearnCompatibleELM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=10, activation_func='sigm'):
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.model = None
        self.onehot_encoder = OneHotEncoder(categories='auto', sparse_output=False)

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

# 读取数据
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]  # 所有列，除了最后一列
y = data.iloc[:, -1]  # 最后一列

# 将字符串标签转换为数值标签
label_mapping = {'Supine': 0, 'Prone': 3, 'Side': 1, 'Foetus': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y = y.map(label_mapping)

# 将字符标签转换为整数标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义ELM模型和参数空间
elm_model = SklearnCompatibleELM()

# 定义要优化的参数空间
param_space = {
    'n_neurons': Integer(100, 500),
    'activation_func': Categorical(['sigm', 'tanh'])
}

# 使用BayesSearchCV进行贝叶斯优化
opt = BayesSearchCV(elm_model, param_space, n_iter=32, random_state=42, cv=5)

# 训练模型
opt.fit(X_train, y_train)

# 最佳参数
print("最佳参数：", opt.best_params_)

# 使用最佳参数进行预测
y_pred_encoded = opt.predict(X_test)

# 将预测标签和实际标签转换回原始标签
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test_original = label_encoder.inverse_transform(y_test)

# 输出分类报告
report = classification_report(y_test_original, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
pd.options.display.float_format = '{:.4f}'.format
print(report_df)

# 输出准确率
accuracy = round(accuracy_score(y_test_original, y_pred), 4)
print("准确率：{:.4f}".format(accuracy))

# 计算Kappa系数
kappa = round(cohen_kappa_score(y_test_original, y_pred), 4)
print("Kappa系数：{:.4f}".format(kappa))


# 保存预测结果到文件
np.savetxt("elm_predictions.csv", y_pred_encoded, delimiter=",")
