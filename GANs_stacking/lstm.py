import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# 加载数据集
file_path = "rr.xlsx"  # 替换为您的文件路径
data = pd.read_excel(file_path, header=None)

# 分离标签和心率数据
labels = data.iloc[0, :].values
heart_rate_data = data.iloc[1:, :].values  # 其余行是心率数据

# 标准化特征
scaler = StandardScaler()
heart_rate_data_scaled = scaler.fit_transform(heart_rate_data.T).T

# 创建滑动窗口数据
def create_sequences(data, labels, time_steps=10):
    X, y = [], []
    for i in range(data.shape[1] - time_steps):
        X.append(data[:, i:i + time_steps])  # 从data中提取长度为time_steps的窗口
        y.append(labels[i + time_steps])  # 相应的标签
    return np.array(X), np.array(y)

# 定义时间步长
time_steps = 10

# 创建序列数据
X, y = create_sequences(heart_rate_data_scaled, labels, time_steps)

# 打印输入 LSTM 之前的前几行数据
print("输入 LSTM 之前的前几行数据（X）：")
print(X[:3])  # 打印前3个样本
print("\n对应的标签（y）：")
print(y[:3])  # 打印前3个标签

# 调整数据形状以适应LSTM模型 (样本数量, 时间步长, 特征数量)
X = X.reshape((X.shape[0], time_steps, heart_rate_data_scaled.shape[0]))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# 预测
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # 将概率转化为类别

# 计算准确率
test_accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Test Accuracy: {test_accuracy}')