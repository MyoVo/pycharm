import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Average
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from hpelm import ELM
from scipy.special import softmax
from tabnet import TabNetClassifier
import warnings

warnings.filterwarnings("ignore")

# 禁用所有GPU
tf.config.set_visible_devices([], 'GPU')

class DeepNeuralDecisionForest(BaseEstimator, ClassifierMixin):
    def __init__(self, num_trees=1, tree_depth=1, num_classes=4, learning_rate=0.004951144203594119, batch_size=16):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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

    def fit(self, X_train, y_train, epochs=10):
        self.build_model(X_train.shape[1:])
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        self.model.fit(X_train, y_train_cat, batch_size=self.batch_size, epochs=epochs, verbose=0)

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
                'num_classes': self.num_classes, 'learning_rate': self.learning_rate, 'batch_size': self.batch_size}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

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

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_decision_steps=3, relaxation_factor=1.0, sparsity_coefficient=1e-5):
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.model = None

    def fit(self, X, y):
        feature_columns = [tf.feature_column.numeric_column('dummy', shape=(X.shape[1],))]
        self.model = TabNetClassifier(feature_columns=feature_columns, num_classes=len(np.unique(y)),
                                      num_features=X.shape[1], feature_dim=64, output_dim=32,
                                      num_decision_steps=self.num_decision_steps,
                                      relaxation_factor=self.relaxation_factor,
                                      sparsity_coefficient=self.sparsity_coefficient)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy())
        X_dict = {'dummy': X}
        self.model.fit(X_dict, y, epochs=100, batch_size=8, verbose=0)
        return self

    def predict(self, X):
        X_dict = {'dummy': X}
        y_pred_probs = self.model.predict(X_dict)
        y_pred_probs = softmax(y_pred_probs, axis=1)
        return np.argmax(y_pred_probs, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# 读取数据
data = pd.read_excel('7.xlsx')

# 获取特征名称
feature_names = data.columns[:-1].tolist()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将字符串标签转换为数值标签
label_mapping = {'Supine': 0, 'Prone': 3, 'Side': 1, 'Foetus': 2}
y_series = pd.Series(y)  # 将y转换为Pandas Series对象
y = y_series.map(label_mapping).values  # 映射并转换回NumPy数组

# 如果需要进一步转换为整数标签，可以使用LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 计算特征和标签之间的互信息
mi = mutual_info_classif(X, y)

# 输出每个特征的互信息值，并排序
mi_dict = {feature_names[i]: mi[i] for i in range(len(feature_names))}
sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

print("互信息值（从大到小排序）：")
for feature, value in sorted_mi:
    print(f"{feature}: {value:.4f}")


# 根据互信息对特征进行排序
sorted_indices = np.argsort(mi)[::-1]
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_X = X[:, sorted_indices]

# 数据标准化
scaler = StandardScaler()
sorted_X_scaled = scaler.fit_transform(sorted_X)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(sorted_X_scaled, y, test_size=0.2, random_state=42)

# 初始化分类误差列表
xgb_errors = []
rf_errors = []
svm_errors = []
dndt_errors = []
elm_errors = []
tabnet_errors = []

# 随机森林最佳参数
rf_best_params = {
    'max_depth': 19,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 298
}

# DNDT最佳参数
dndt_best_params = {
    'num_trees': 1,
    'tree_depth': 1,
    'learning_rate': 0.004951144203594119,
    'batch_size': 16
}

# ELM最佳参数
elm_best_params = {
    'n_neurons': 500,
    'activation_func': 'tanh'
}

# 从大到小依次添加特征进行分类，并计算分类误差
errors_dict = {'XGBoost': [], 'Random Forest': [], 'SVM': [], 'DNDT': [], 'ELM': []}

for i in range(1, X_train.shape[1] + 1):
    # XGBoost分类器
    xgb_classifier = XGBClassifier(max_depth=7, n_estimators=300, random_state=42)
    xgb_classifier.fit(X_train[:, :i], y_train)
    y_pred_xgb = xgb_classifier.predict(X_test[:, :i])
    error_xgb = 1 - accuracy_score(y_test, y_pred_xgb)
    xgb_errors.append(error_xgb)
    errors_dict['XGBoost'].append((sorted_feature_names[:i], error_xgb))

    # 随机森林分类器
    rf_classifier = RandomForestClassifier(
        max_depth=rf_best_params['max_depth'],
        min_samples_leaf=rf_best_params['min_samples_leaf'],
        min_samples_split=rf_best_params['min_samples_split'],
        n_estimators=rf_best_params['n_estimators'],
        random_state=42
    )
    rf_classifier.fit(X_train[:, :i], y_train)
    y_pred_rf = rf_classifier.predict(X_test[:, :i])
    error_rf = 1 - accuracy_score(y_test, y_pred_rf)
    rf_errors.append(error_rf)
    errors_dict['Random Forest'].append((sorted_feature_names[:i], error_rf))

    # SVM分类器
    svm_classifier = SVC(C=4.82, gamma=1, kernel='rbf', random_state=42)
    svm_classifier.fit(X_train[:, :i], y_train)
    y_pred_svm = svm_classifier.predict(X_test[:, :i])
    error_svm = 1 - accuracy_score(y_test, y_pred_svm)
    svm_errors.append(error_svm)
    errors_dict['SVM'].append((sorted_feature_names[:i], error_svm))

    # DNDT分类器
    dndt_classifier = DeepNeuralDecisionForest(num_classes=len(np.unique(y_train)),
                                               num_trees=dndt_best_params['num_trees'],
                                               tree_depth=dndt_best_params['tree_depth'],
                                               learning_rate=dndt_best_params['learning_rate'],
                                               batch_size=dndt_best_params['batch_size'])
    dndt_classifier.fit(X_train[:, :i], y_train, epochs=10)
    y_pred_dndt = dndt_classifier.predict(X_test[:, :i])
    error_dndt = 1 - accuracy_score(y_test, y_pred_dndt)
    dndt_errors.append(error_dndt)
    errors_dict['DNDT'].append((sorted_feature_names[:i], error_dndt))

    # ELM分类器
    elm_classifier = SklearnCompatibleELM(n_neurons=elm_best_params['n_neurons'],
                                          activation_func=elm_best_params['activation_func'])
    elm_classifier.fit(X_train[:, :i], y_train)
    y_pred_elm = elm_classifier.predict(X_test[:, :i])
    error_elm = 1 - accuracy_score(y_test, y_pred_elm)
    elm_errors.append(error_elm)
    errors_dict['ELM'].append((sorted_feature_names[:i], error_elm))


    # TabNet分类器
    '''
    tabnet_classifier = TabNetWrapper(num_decision_steps=3, relaxation_factor=1.0, sparsity_coefficient=1e-5)
    tabnet_classifier.fit(X_train[:, :i], y_train)
    y_pred_tabnet = tabnet_classifier.predict(X_test[:, :i])
    error_tabnet = 1 - accuracy_score(y_test, y_pred_tabnet)
    tabnet_errors.append(error_tabnet)
    errors_dict['TabNet'].append((sorted_feature_names[:i], error_tabnet))
    '''

# 可视化分类误差
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(xgb_errors) + 1), xgb_errors, marker='o', label='XGBoost')
plt.plot(range(1, len(rf_errors) + 1), rf_errors, marker='x', label='Random Forest')
plt.plot(range(1, len(svm_errors) + 1), svm_errors, marker='s', label='SVM')
plt.plot(range(1, len(dndt_errors) + 1), dndt_errors, marker='^', label='DNDT')
plt.plot(range(1, len(elm_errors) + 1), elm_errors, marker='v', label='ELM')
#plt.plot(range(1, len(tabnet_errors) + 1), tabnet_errors, marker='d', label='TabNet')
plt.xticks(range(1, len(xgb_errors) + 1), sorted_feature_names, rotation=90)
plt.xlabel('Number of Features')
plt.ylabel('Classification Error')
plt.title('Classification Error vs. Number of Features Added')
plt.legend()
plt.grid()
plt.show()

# 找到分类误差最小的特征数量
optimal_feature_count_xgb = np.argmin(xgb_errors) + 1
optimal_feature_count_rf = np.argmin(rf_errors) + 1
optimal_feature_count_svm = np.argmin(svm_errors) + 1
optimal_feature_count_dndt = np.argmin(dndt_errors) + 1
optimal_feature_count_elm = np.argmin(elm_errors) + 1
#optimal_feature_count_tabnet = np.argmin(tabnet_errors) + 1
print(f"XGBoost最佳特征数量: {optimal_feature_count_xgb}")
print(f"随机森林最佳特征数量: {optimal_feature_count_rf}")
print(f"SVM最佳特征数量: {optimal_feature_count_svm}")
print(f"DNDT最佳特征数量: {optimal_feature_count_dndt}")
print(f"ELM最佳特征数量: {optimal_feature_count_elm}")
#print(f"TabNet最佳特征数量: {optimal_feature_count_tabnet}")

# 输出每个模型在每个特征数量下的分类误差
for model_name, errors in errors_dict.items():
    print(f"\n{model_name}分类器的分类误差：")
    for features, error in errors:
        print(f"特征数量: {len(features)}, 特征: {features}, 分类误差: {error}")