import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from xgboost import XGBClassifier
from sklearn.svm import SVC
from keras import Model, Input
from keras.layers import Dense, Average
from keras.utils import to_categorical
import tensorflow as tf
import warnings

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
xgb_base = XGBClassifier(learning_rate=0.1692504891932796, max_depth=1, n_estimators=500, use_label_encoder=False, eval_metric='mlogloss')
dndt = DeepNeuralDecisionForest(num_trees=3, tree_depth=1, num_classes=len(np.unique(y_train)), learning_rate=0.0024505706365593586, batch_size=16)
svm = SVC(C=4.7290805470559185, gamma=0.4466278825224283, kernel='rbf', probability=True, random_state=42)

# 训练基模型
xgb_base.fit(X_train, y_train)
svm.fit(X_train, y_train)
dndt.fit(X_train, y_train)

# 使用StackingClassifier
meta_model = XGBClassifier(learning_rate=0.1, max_depth=1, n_estimators=500, use_label_encoder=False, eval_metric='mlogloss')

lr = LogisticRegression(C=10, penalty='l2', solver='liblinear')

stacking_model = StackingClassifier(
    estimators=[
        ('xgb_base', xgb_base),
        ('svm', svm),
        ('dndt', dndt)
    ],
    final_estimator=meta_model
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

# 获取基模型的特征重要性
xgb_importance = xgb_base.feature_importances_

# 使用Permutation Importance计算SVM和DNDT特征重要性
def permutation_importance(model, X, y, metric, n_repeats=30):
    baseline_score = metric(y, model.predict(X))
    importances = np.zeros(X.shape[1])

    for col in range(X.shape[1]):
        save_col = X[:, col].copy()
        scores = np.zeros(n_repeats)
        for i in range(n_repeats):
            X[:, col] = np.random.permutation(X[:, col])
            scores[i] = metric(y, model.predict(X))
            X[:, col] = save_col
        importances[col] = baseline_score - np.mean(scores)
    return importances

svm_importance = permutation_importance(svm, X_test, y_test, accuracy_score)
svm_importance /= svm_importance.sum() if svm_importance.sum() != 0 else 1

dndt_importance = permutation_importance(dndt, X_test, y_test, accuracy_score)
dndt_importance /= dndt_importance.sum() if dndt_importance.sum() != 0 else 1

# 计算相对贡献
importance_df = pd.DataFrame({
    'Features': data.columns[:-1],
    'XGBoost': xgb_importance,
    'SVM': svm_importance,
    'DNDT': dndt_importance
})
importance_df.set_index('Features', inplace=True)
importance_df['Average'] = importance_df.mean(axis=1)

# 打印特征重要性
print("Feature Importance for Four-Class Classification:")
print(importance_df)

# 第一个Sankey图：特征对基模型的贡献
label_with_values_base = [f"{feature}: {value:.2f}" for feature, value in zip(importance_df.index, importance_df['Average'])]
fig_base = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label_with_values_base + ["XGBoost", "SVM", "DNDT"],
        color=["lightgrey"]*len(label_with_values_base) + ["blue", "chocolate", "green"]
    ),
    link=dict(
        source=list(range(len(importance_df.index)))*3,
        target=[len(importance_df.index)]*len(importance_df.index) + [len(importance_df.index)+1]*len(importance_df.index) + [len(importance_df.index)+2]*len(importance_df.index),
        value=list(importance_df['XGBoost']) + list(importance_df['SVM']) + list(importance_df['DNDT']),
        color=['rgba(31, 119, 180, 0.8)']*len(importance_df['XGBoost']) +
              ['rgba(255, 127, 14, 0.8)']*len(importance_df['SVM']) +
              ['rgba(44, 160, 44, 0.8)']*len(importance_df['DNDT'])
    )
)])

fig_base.update_layout(
    title_text="",
    font_size=10,
    font=dict(size=20)
)

fig_base.show()

# 获取元学习器的特征重要性
meta_model_fitted = stacking_model.final_estimator_

# 从 Logistic Regression 中获取特征重要性
meta_feature_importances = np.abs(meta_model_fitted.coef_[0])

# 计算每个原始特征对标签的贡献度
num_classes = len(np.unique(y_train))
num_features = X.shape[1]
final_feature_importance = np.zeros((num_features, num_classes))

# 计算每个基模型对每个类别的贡献
base_feature_importances = {
    'XGBoost': xgb_importance,
    'SVM': svm_importance,
    'DNDT': dndt_importance
}

# 针对每个类别单独计算特征重要性
for j in range(num_classes):
    for i, model_name in enumerate(base_feature_importances.keys()):
        final_feature_importance[:, j] += meta_feature_importances[i] * base_feature_importances[model_name] * (y_train == j).mean()

# 标准化最终的特征重要性
final_feature_importance /= final_feature_importance.sum(axis=0)

# 创建最终的特征重要性数据框
final_importance_df = pd.DataFrame(final_feature_importance, columns=label_encoder.classes_, index=data.columns[:-1])

# 打印最终特征重要性
print("Final Feature Importance in Stacking Model by Class:")
print(final_importance_df)

# 第二个Sankey图：特征对每个类别的贡献
label_with_values_final = [f"{feature}: {value:.2f}" for feature, value in zip(final_importance_df.index, final_importance_df.mean(axis=1))]
source_indices = np.tile(np.arange(len(final_importance_df.index)), num_classes)
target_indices = np.repeat(np.arange(len(final_importance_df.index), len(final_importance_df.index) + num_classes), len(final_importance_df.index))
values = final_importance_df.values.flatten()

fig_final = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label_with_values_final + list(label_encoder.classes_),
        color=["lightgrey"]*len(final_importance_df.index) + ["blue", "chocolate", "green", "purple"]
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values,
        color=['rgba(31, 119, 180, 0.8)']*len(final_importance_df.iloc[:, 0]) +
              ['rgba(255, 127, 14, 0.8)']*len(final_importance_df.iloc[:, 1]) +
              ['rgba(44, 160, 44, 0.8)']*len(final_importance_df.iloc[:, 2]) +
              ['rgba(148, 103, 189, 0.8)']*len(final_importance_df.iloc[:, 3])
    )
)])

fig_final.update_layout(
    title_text="",
    font_size=10,
    font=dict(size=20)
)

fig_final.show()
