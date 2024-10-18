import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from hpelm import ELM
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Average
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

tf.config.set_visible_devices([], 'GPU')

class SklearnCompatibleELM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=10, activation_func='sigm'):
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.model = None
        self.onehot_encoder = OneHotEncoder(categories='auto', sparse_output=False)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
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

# 定义兼容Scikit-learn的DNDF模型
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

    def fit(self, X_train, y_train, batch_size=32, epochs=10):
        self.classes_ = np.unique(y_train)
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

def tune_dndf_model(X_train, y_train):
    param_grid = {
        'num_trees': Integer(1, 10),
        'tree_depth': Integer(1, 7),
        'learning_rate': Real(0.0001, 0.01, prior='log-uniform'),
        'batch_size': Categorical([16, 32, 64])
    }
    bayes_search = BayesSearchCV(estimator=DeepNeuralDecisionForest(num_classes=len(set(y_train))),
                                 search_spaces=param_grid, n_iter=30, cv=5, n_jobs=1, scoring='accuracy')
    bayes_search.fit(X_train, y_train)
    best_params = bayes_search.best_params_
    best_dndf_model = DeepNeuralDecisionForest(num_trees=best_params['num_trees'],
                                               tree_depth=best_params['tree_depth'],
                                               learning_rate=best_params['learning_rate'],
                                               num_classes=len(set(y_train)))
    best_dndf_model.build_model(X_train.shape[1:])
    best_dndf_model.fit(X_train, y_train, batch_size=int(best_params['batch_size']), epochs=100)
    return best_dndf_model

# 读取数据
file_path = '8.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1].values  # 所有列，除了最后一列
y = data.iloc[:, -1].values  # 最后一列

# 将字符串标签转换为数值标签
label_mapping = {'Supine': 0, 'Prone': 3, 'Side': 1, 'Foetus': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y = pd.Series(y).map(label_mapping).values

# 将字符标签转换为整数标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义ELM模型和参数空间
elm_model = SklearnCompatibleELM()

# 定义要优化的参数空间
param_space_elm = {
    'n_neurons': Integer(100, 500),
    'activation_func': Categorical(['sigm', 'tanh'])
}

# 使用BayesSearchCV进行贝叶斯优化
opt_elm = BayesSearchCV(elm_model, param_space_elm, n_iter=32, random_state=42, cv=5)
opt_elm.fit(X_train, y_train)

# 最佳参数
print("ELM最佳参数：", opt_elm.best_params_)

# 定义SVM模型
svm_model = SVC(C=10.0, gamma=1.0, kernel='rbf', probability=True)

# 调整DNDF模型参数
best_dndf_model = tune_dndf_model(X_train, y_train)

# 定义Stacking模型，逻辑回归作为元学习器
stacking_model = StackingClassifier(
    estimators=[
        ('elm', opt_elm.best_estimator_),
        ('svm', svm_model),
        ('dndf', best_dndf_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# 训练Stacking模型
stacking_model.fit(X_train, y_train)

# 预测
y_pred_encoded = stacking_model.predict(X_test)

# 将预测标签和实际标签转换回原始标签
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test_original = label_encoder.inverse_transform(y_test)

# 输出分类报告
report = classification_report(y_test_original, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
pd.options.display.float_format = '{:.4f}'.format
print(report_df)
