import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.layers import Input, Dense, Average
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tabnet import TabNetClassifier

# 禁用所有GPU
tf.config.set_visible_devices([], 'GPU')

class DeepNeuralDecisionForest(BaseEstimator, ClassifierMixin):
    def __init__(self, num_trees=1, tree_depth=1, num_classes=4, learning_rate=0.01):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

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
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def build_tree(self, inputs):
        x = inputs
        for _ in range(self.tree_depth):
            x = Dense(self.num_classes, activation='softmax', use_bias=False)(x)
        return x

    def fit(self, X, y, batch_size=16, epochs=10):
        self.build_model(X.shape[1:])
        y_cat = to_categorical(y, num_classes=self.num_classes)
        self.model.fit(X, y_cat, batch_size=batch_size, epochs=epochs, verbose=0)
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        return pred.argmax(axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'num_trees': self.num_trees,
            'tree_depth': self.tree_depth,
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_columns, num_classes, num_features, feature_dim, output_dim, num_decision_steps, relaxation_factor, sparsity_coefficient):
        self.feature_columns = feature_columns
        self.num_classes = num_classes
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.model = TabNetClassifier(
            feature_columns=self.feature_columns,
            num_classes=self.num_classes,
            num_features=self.num_features,
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            num_decision_steps=self.num_decision_steps,
            relaxation_factor=self.relaxation_factor,
            sparsity_coefficient=self.sparsity_coefficient
        )

    def fit(self, X, y):
        X_dict = {'dummy': X}
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        self.model.fit(X_dict, y, epochs=10, batch_size=16, verbose=0)
        return self

    def predict(self, X):
        X_dict = {'dummy': X}
        y_pred_probs = self.model.predict(X_dict)
        y_pred = y_pred_probs.argmax(axis=1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'feature_columns': self.feature_columns,
            'num_classes': self.num_classes,
            'num_features': self.num_features,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'num_decision_steps': self.num_decision_steps,
            'relaxation_factor': self.relaxation_factor,
            'sparsity_coefficient': self.sparsity_coefficient
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = TabNetClassifier(
            feature_columns=self.feature_columns,
            num_classes=self.num_classes,
            num_features=self.num_features,
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            num_decision_steps=self.num_decision_steps,
            relaxation_factor=self.relaxation_factor,
            sparsity_coefficient=self.sparsity_coefficient
        )
        return self

# 读取Excel文件
file_path = 'a2.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]  # 所有列，除了最后一列
y = data.iloc[:, -1]  # 最后一列
y = y - 1

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算互信息
mi_scores = mutual_info_classif(X_scaled, y)

# 获取特征名称
feature_names = X.columns

# 创建特征及其互信息值的字典
mi_scores_dict = {feature_names[i]: mi_scores[i] for i in range(len(feature_names))}

# 按互信息值从大到小排序
sorted_mi_scores = sorted(mi_scores_dict.items(), key=lambda item: item[1], reverse=True)

# 打印排序后的互信息值
print("互信息值（从大到小排序）：")
for feature, score in sorted_mi_scores:
    print(f"{feature}: {score}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化准确率列表
stacking_accuracies = []
selected_features = []

for feature, _ in sorted_mi_scores:
    selected_features.append(feature)

    # 选择当前特征集
    feature_indices = [feature_names.get_loc(f) for f in selected_features]
    X_train_selected = X_train[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]

    # 构建基模型
    feature_columns = [tf.feature_column.numeric_column('dummy', shape=(X_train_selected.shape[1],))]
    base_models = [
        ('mlp', MLPClassifier(
            activation='relu',
            alpha=0.0004865681396304516,
            hidden_layer_sizes=(464,),
            solver='adam',
            random_state=42
        )),
        ('dndt', DeepNeuralDecisionForest(num_trees=1, tree_depth=1, num_classes=4, learning_rate=0.01)),
        ('tabnet', TabNetWrapper(
            feature_columns=feature_columns,
            num_classes=len(np.unique(y_train)),
            num_features=X_train_selected.shape[1],
            feature_dim=64,
            output_dim=32,
            num_decision_steps=3,
            relaxation_factor=1.029035429382165,
            sparsity_coefficient=1.977816065999646e-05
        ))
    ]

    # 构建堆叠模型，元模型为MLP
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=MLPClassifier(max_iter=1000),
        passthrough=True
    )

    # 训练堆叠模型
    stacking_model.fit(X_train_selected, y_train)
    y_pred_stacking = stacking_model.predict(X_test_selected)
    stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
    stacking_accuracies.append(stacking_accuracy)

    # 输出当前特征集和准确率
    print(f"特征: {selected_features} -> Stacking 准确率: {stacking_accuracy:.4f}")

# 绘制准确率图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(stacking_accuracies) + 1), stacking_accuracies, label='Stacking', color='purple', marker='o')

plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy of Stacking Model vs. Number of Features')
plt.legend()
plt.grid(True)
# 保存图形为SVG文件
plt.savefig('stacking_accuracy.svg', format='svg', bbox_inches='tight')
plt.show()
