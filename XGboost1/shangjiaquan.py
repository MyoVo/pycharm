import numpy as np
import math

# 输入原始数据
models = ["XGB", "RF", "ELM", "SVM", "DNDT", "MLP",'Tabnet']
metrics = ["Accuracy", "Precision", "F1_Score", "Recall", "Kappa"]
data = np.array([
    [0.9050, 0.8916, 0.8987, 0.9038, 0.8708],
    [0.9019, 0.9005, 0.8976, 0.8967, 0.8732],
    [0.8856, 0.8781, 0.8839, 0.8891, 0.8575],
    [0.8967, 0.8881, 0.8976, 0.8924, 0.8610],
    [0.8923, 0.8951, 0.8896, 0.8941, 0.8662],
    [0.8895, 0.8854, 0.8792, 0.8812, 0.8595],
    [0.8802, 0.8794, 0.8873, 0.8831, 0.8501]
])

# 标准化处理
def normalize(data):
    normalized_data = []
    for j in range(data.shape[1]):
        column = data[:, j]
        min_val = np.min(column)
        max_val = np.max(column)
        normalized_column = (column - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(column)
        normalized_data.append(normalized_column)
    return np.array(normalized_data).T

normalized_data = normalize(data)

# 计算比例 p_ij
def calculate_proportions(data):
    proportions = []
    for j in range(data.shape[1]):
        column_sum = np.sum(data[:, j])
        proportions.append(data[:, j] / column_sum if column_sum != 0 else np.zeros_like(data[:, j]))
    return np.array(proportions).T

proportions = calculate_proportions(normalized_data)

# 计算信息熵 H_j
def calculate_entropy(proportions):
    entropy = []
    for j in range(proportions.shape[1]):
        H_j = -np.sum(proportions[:, j] * np.log(proportions[:, j] + 1e-9)) / np.log(len(models))
        entropy.append(H_j)
    return np.array(entropy)

entropy = calculate_entropy(proportions)

# 计算权重 W_j
def calculate_weights(entropy):
    sum_entropy = np.sum(1 - entropy)
    weights = (1 - entropy) / sum_entropy
    return weights

weights = calculate_weights(entropy)

# 计算加权综合得分 S_i
def calculate_weighted_scores(data, weights):
    weighted_scores = np.dot(data, weights)
    return weighted_scores

weighted_scores = calculate_weighted_scores(normalized_data, weights)

# 输出结果
for i in range(len(models)):
    print(f"模型 {models[i]} 的加权综合得分: {weighted_scores[i]:.4f}")

# 选择得分最高的模型
best_model_index = np.argmax(weighted_scores)
print(f"最佳模型是: 模型 {models[best_model_index]}")
