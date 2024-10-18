import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Average
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
# 禁用所有GPU
tf.config.set_visible_devices([], 'GPU')

class DeepNeuralDecisionForest(BaseEstimator, ClassifierMixin):
    def __init__(self, num_trees=3, tree_depth=3, num_classes=2, learning_rate=0.001):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = [self.build_tree(x) for _ in range(self.num_trees)]

        if len(outputs) > 1:
            output = Average()(outputs)
        else:
            output = outputs[0]

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def build_tree(self, inputs):
        x = inputs
        for _ in range(self.tree_depth):
            x = Dense(self.num_classes, activation='softmax', use_bias=False)(x)
        return x

    def fit(self, X_train, y_train, batch_size=32, epochs=100):
        self.build_model(X_train.shape[1:])
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        self.model.fit(X_train, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=0)

    def predict(self, X):
        pred = self.model.predict(X)
        return pred.argmax(axis=1)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if sample_weight is not None:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return accuracy_score(y, y_pred)

    def get_params(self, deep=False):
        return {'num_trees': self.num_trees, 'tree_depth': self.tree_depth,
                'num_classes': self.num_classes, 'learning_rate': self.learning_rate}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

# 读取数据
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 检查标签类型并进行转换
if y.dtype == 'O':  # 如果标签是对象类型（字符串）
    y, unique_labels = pd.factorize(y)
    print("标签对应关系:")
    for index, label in enumerate(unique_labels):
        print(f"{label} -> {index}")
else:
    y = y - 1  # 如果标签是数值类型，将其转换为从0开始的数值

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义DNDT模型和参数空间
dndt_model = DeepNeuralDecisionForest(num_classes=len(np.unique(y_train)))

# 定义要优化的参数空间
param_space = {
        'num_trees': Integer(1, 10),
        'tree_depth': Integer(1, 7),
        'learning_rate': Real(0.0001, 0.01, prior='log-uniform'),
        'batch_size': Categorical([16, 32, 64])
    }

# 使用BayesSearchCV进行贝叶斯优化
opt = BayesSearchCV(dndt_model, param_space, n_iter=32, random_state=42, cv=5)

# 训练模型
opt.fit(X_train, y_train)

# 最佳参数
print("最佳参数：", opt.best_params_)

# 使用最佳参数进行预测
y_pred = opt.predict(X_test)

# 将预测标签和实际标签转换回原始标签
y_test_original = pd.Series(y_test).map(lambda x: unique_labels[x])
y_pred_original = pd.Series(y_pred).map(lambda x: unique_labels[x])

# 输出结果
report = classification_report(y_test_original, y_pred_original, digits=4)
print("分类报告：\n", report)

accuracy = round(accuracy_score(y_test_original, y_pred_original), 4)
print("准确率：", accuracy)

# 计算Kappa系数
kappa = round(cohen_kappa_score(y_test_original, y_pred_original), 4)
print("Kappa系数：{:.4f}".format(kappa))
# 最佳参数
print("最佳参数：", opt.best_params_)
# 保存预测结果到文件
np.savetxt("dndt_predictions.csv", y_pred, delimiter=",")
