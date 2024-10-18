import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
from keras import Model, Input
from keras.layers import Dense, Average
from keras.utils import to_categorical
import tensorflow as tf
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
tf.config.set_visible_devices([], 'GPU')  # 禁用所有GPU

# 定义 DNDT 模型类
class DeepNeuralDecisionForest(BaseEstimator, ClassifierMixin):
    def __init__(self, num_trees=1, tree_depth=1, num_classes=4, learning_rate=0.0006174697383521571, batch_size=16):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.classes_ = None

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

    def fit(self, X, y, epochs=100):
        self.build_model(X.shape[1:])
        y_cat = to_categorical(y, num_classes=self.num_classes)
        self.model.fit(X, y_cat, batch_size=self.batch_size, epochs=epochs, verbose=0)
        self.classes_ = np.unique(y)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.argmax(pred, axis=1)

    def predict_proba(self, X):
        pred = self.model.predict(X)
        return pred

    def get_params(self, deep=False):
        return {'num_trees': self.num_trees, 'tree_depth': self.tree_depth,
                'num_classes': self.num_classes, 'learning_rate': self.learning_rate, 'batch_size': self.batch_size}

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

# 将字符标签转换为整数标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义基模型
xgb_base = XGBClassifier(learning_rate=0.1692504891932796, max_depth=1, n_estimators=500, use_label_encoder=False)
dndt = DeepNeuralDecisionForest(num_trees=7, tree_depth=1, num_classes=len(np.unique(y_train)), learning_rate=0.0024505706365593586, batch_size=16)
svm = SVC(C=4.7290805470559185, gamma=0.4466278825224283, kernel='rbf', probability=True, random_state=42)

# 训练并评估基模型
xgb_base.fit(X_train, y_train)
dndt.fit(X_train, y_train)
svm.fit(X_train, y_train)

xgb_base_pred = xgb_base.predict(X_test)
dndt_pred = dndt.predict(X_test)
svm_pred = svm.predict(X_test)

xgb_base_acc = accuracy_score(y_test, xgb_base_pred)
dndt_acc = accuracy_score(y_test, dndt_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"XGBoost 基模型的准确率: {xgb_base_acc:.4f}")
print(f"DNDT 基模型的准确率: {dndt_acc:.4f}")
print(f"SVM 基模型的准确率: {svm_acc:.4f}")

# 定义元学习器
xgb_meta = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=500, use_label_encoder=False, eval_metric='mlogloss')
# 定义参数网格
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# 创建逻辑回归模型
lr = LogisticRegression(max_iter=1000)

# 使用 GridSearchCV 进行超参数调整
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")

# 使用最佳参数创建逻辑回归模型
best_lr = grid_search.best_estimator_

# 定义 Stacking 模型
stacking_model = StackingClassifier(
    estimators=[
        ('xgb_base', xgb_base),
        ('dndt', dndt),
        ('svm', svm)
    ],
    final_estimator=best_lr,
    cv=5
)

# 训练 Stacking 模型
stacking_model.fit(X_train, y_train)

# 预测
y_pred = stacking_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Stacking 模型的准确率: {accuracy:.4f}")
print(f"Stacking 模型的Kappa系数: {kappa:.4f}")
print(f"Stacking 模型的分类报告:\n{report}")

