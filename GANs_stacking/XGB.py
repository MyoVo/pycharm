import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tabnet import TabNetClassifier
from scipy.special import softmax
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# 读取数据
file_path = 'r2.xlsx'
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]  # 所有列，除了最后一列
y = data.iloc[:, -1]  # 最后一列
y = y - 1

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义要优化的超参数空间
dim_num_decision_steps = Integer(low=3, high=10, name='num_decision_steps')
dim_relaxation_factor = Real(low=1.0, high=2.0, name='relaxation_factor')
dim_sparsity_coefficient = Real(low=1e-5, high=1e-1, prior='log-uniform', name='sparsity_coefficient')

dimensions = [dim_num_decision_steps, dim_relaxation_factor, dim_sparsity_coefficient]

feature_columns = [tf.feature_column.numeric_column('dummy', shape=(X_train.shape[1],))]

@use_named_args(dimensions=dimensions)
def fitness(num_decision_steps, relaxation_factor, sparsity_coefficient):
    # 使用给定的超参数创建 TabNet 模型
    tabnet_model = TabNetClassifier(feature_columns=feature_columns, num_classes=len(np.unique(y_train)),
                                    num_features=X_train.shape[1], feature_dim=64, output_dim=32,
                                    num_decision_steps=num_decision_steps, relaxation_factor=relaxation_factor,
                                    sparsity_coefficient=sparsity_coefficient)
    tabnet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy())

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train_dict = {'dummy': X_train_split}
    X_val_dict = {'dummy': X_val}

    # 使用早停法来避免过拟合
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = tabnet_model.fit(X_train_dict, y_train_split, epochs=1000, batch_size=8,
                               validation_data=(X_val_dict, y_val), callbacks=[early_stopping], verbose=0)

    # 计算验证集上的最佳损失
    best_loss = min(history.history['val_loss'])
    return best_loss

# 使用贝叶斯优化来找到最佳超参数
search_result = gp_minimize(func=fitness, dimensions=dimensions, n_calls=30, x0=[3, 1.0, 1e-4])

print("Best parameters:", search_result.x)
print("Best validation loss:", search_result.fun)

# 使用最佳超参数重新训练模型
best_num_decision_steps, best_relaxation_factor, best_sparsity_coefficient = search_result.x
best_model = TabNetClassifier(feature_columns=feature_columns, num_classes=len(np.unique(y_train)),
                              num_features=X_train.shape[1], feature_dim=128, output_dim=32,
                              num_decision_steps=best_num_decision_steps, relaxation_factor=best_relaxation_factor,
                              sparsity_coefficient=best_sparsity_coefficient)
best_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy())
best_model.fit({'dummy': X_train}, y_train, epochs=1000, batch_size=8, verbose=0)

# 模型评估和预测
y_pred_probs = best_model.predict({'dummy': X_test})
y_pred_probs = softmax(y_pred_probs, axis=1)
y_pred = np.argmax(y_pred_probs, axis=1)
tabnet_accuracy = accuracy_score(y_test, y_pred)
print("Final model accuracy:", tabnet_accuracy)
print("Classification report:\n", classification_report(y_test, y_pred))
