import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# 读取Excel数据
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

# 将标签转换为one-hot编码（用于评估，不是模型训练所需）
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# 定义MLP模型
mlp = MLPClassifier(max_iter=500)

# 定义超参数搜索空间
param_space = {
    'hidden_layer_sizes': Integer(50, 200),  # 隐藏层神经元数
    'alpha': Real(1e-5, 1e-2, prior='log-uniform'),  # L2正则化参数
    'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform')  # 初始学习率
}

# 使用贝叶斯优化进行超参数搜索
opt = BayesSearchCV(
    mlp,
    search_spaces=param_space,
    n_iter=30,  # 迭代次数
    random_state=42,
    cv=3,  # 3折交叉验证
    scoring='accuracy',
    n_jobs=-1
)

# 训练贝叶斯优化模型
opt.fit(X_train, y_train)

# 最佳参数
print("最佳超参数配置:", opt.best_params_)

# 在测试集上进行预测
y_pred = opt.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 绘制目标函数随迭代次数的变化图
iterations = range(1, len(opt.cv_results_['mean_test_score']) + 1)
mean_test_scores = opt.cv_results_['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(iterations, mean_test_scores, marker='o')
plt.title('Objective Function Value over Iterations (Accuracy)', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Mean Test Accuracy', fontsize=14)
plt.grid(True)
plt.show()
