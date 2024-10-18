import numpy as np
import matplotlib.pyplot as plt

# 输入原始数据
models = ["XGBoost", "RF", "ELM", "Tabnet", "SVM", "DNDT", "MLP"]
metrics = ["Accuracy", "Precision", "F1_Score", "Recall", "Kappa"]
data = np.array([
    [0.9215, 0.9039, 0.9102, 0.9070, 0.8972],  # XGBoost
    [0.9126, 0.9083, 0.8926, 0.9003, 0.8910],  # RF
    [0.8931, 0.8849, 0.8961, 0.8904, 0.8721],  # ELM
    [0.9108, 0.8920, 0.9056, 0.8987, 0.8891],  # Tabnet
    [0.9079, 0.8949, 0.9097, 0.9022, 0.8862],  # SVM
    [0.9115, 0.9013, 0.9171, 0.9091, 0.8846],  # DNDT
    [0.8907, 0.8724, 0.8965, 0.8843, 0.8693]   # MLP
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
print("标准化后的数据:\n", normalized_data)

# 计算比例 p_ij
def calculate_proportions(data):
    proportions = []
    for j in range(data.shape[1]):
        column_sum = np.sum(data[:, j])
        proportions.append(data[:, j] / column_sum if column_sum != 0 else np.zeros_like(data[:, j]))
    return np.array(proportions).T

proportions = calculate_proportions(normalized_data)
print("比例 p_ij:\n", proportions)

# 计算信息熵 H_j
def calculate_entropy(proportions):
    entropy = []
    for j in range(proportions.shape[1]):
        H_j = -np.sum(proportions[:, j] * np.log(proportions[:, j] + 1e-9)) / np.log(len(models))
        entropy.append(H_j)
    return np.array(entropy)

entropy = calculate_entropy(proportions)
print("信息熵 H_j:\n", entropy)

# 计算权重 W_j
def calculate_weights(entropy):
    sum_entropy = np.sum(1 - entropy)
    weights = (1 - entropy) / sum_entropy
    return weights

weights = calculate_weights(entropy)
print("权重 W_j:\n", weights)


def calculate_weighted_scores(data, weights):
    weighted_scores = np.dot(data, weights)
    return weighted_scores

weighted_scores = calculate_weighted_scores(normalized_data, weights)
print("加权综合得分 S_i:\n", weighted_scores)

# 输出结果
for i in range(len(models)):
    print(f"模型 {models[i]} 的加权综合得分: {weighted_scores[i]:.4f}")

# 选择得分最高的模型
best_model_index = np.argmax(weighted_scores)
print(f"最佳模型是: 模型 {models[best_model_index]}")

# 绘制柱状图
colors = ['lightcoral', 'peru', 'tan', 'blue', 'lightgreen', 'skyblue', 'lightpink']  # 自定义颜色
plt.figure(figsize=(10, 8))
bars = plt.barh(models, weighted_scores, color=colors)
plt.xlabel('')
plt.ylabel('')
plt.title('')
# 添加网格线，且网格线不覆盖图形
plt.grid(axis='x', linestyle='--', linewidth=0.5, zorder=0)
plt.barh(models, weighted_scores, color=colors, zorder=3)

# 设置坐标轴字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=19)

# 设置x轴刻度
plt.xticks(np.arange(0, 1.1, 0.1))
# 确保y轴线在最上层
ax = plt.gca()
ax.spines['left'].set_zorder(3)
# 保存图形为SVG文件
plt.savefig('shangjiaquan.png', format='png', bbox_inches='tight', dpi=150)
plt.show()
