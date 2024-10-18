import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义混淆矩阵的百分比值
confusion_matrix_percentage = np.array([
    [94.02, 1.88, 2.48, 1.62],
    [1.31, 97.04, 0.62, 1.03],
    [1.92, 2.24, 89.37, 6.47],
    [1.75, 0.72, 7.63, 89.90]
])

# 标签
labels = ["Foetus", "Prone", "Side", "Supine"]

# 创建带百分号的注释
annotations = np.empty_like(confusion_matrix_percentage).astype(str)
for i in range(confusion_matrix_percentage.shape[0]):
    for j in range(confusion_matrix_percentage.shape[1]):
        annotations[i, j] = f"{confusion_matrix_percentage[i, j]:.2f}%"

# 绘制混淆矩阵
plt.figure(figsize=(10, 8), dpi=100)  # 设置图像大小为10x10英寸，分辨率为100 DPI
sns.heatmap(confusion_matrix_percentage, annot=annotations, fmt="", cmap="Blues", cbar=True,
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 15})

plt.title("Confusion Matrix ", fontsize=18)
plt.xlabel("Predicted Label", fontsize=16)
plt.ylabel("True Label", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 保存图像
plt.savefig('xgb.png', format='png', bbox_inches='tight', dpi=100)  # 保存为1000x1000分辨率的PNG图像

plt.show()
