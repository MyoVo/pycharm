import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个数据字典来模拟相关性数据
data = {
    'Bo-XGB': [1.0, 0.90, 0.82, 0.71, 0.75, 0.86, 0.66],
    'Bo-RF': [0.90, 1.0, 0.80, 0.68, 0.69, 0.89, 0.64],
    'Bo-SVM': [0.82, 0.80, 1.0, 0.71, 0.67, 0.75, 0.68],
    'Bo-DNDT': [0.71, 0.68, 0.71, 1.0, 0.66, 0.71, 0.66],
    'Bo-TabNet': [0.75, 0.69, 0.67, 0.66, 1.0, 0.72, 0.65],
    'Bo-ELM': [0.66, 0.64, 0.68, 0.66, 0.65, 0.69, 1.0]
}

# 将数据字典转换为DataFrame
correlation_matrix = pd.DataFrame(data, index=['Bo-XGB', 'Bo-RF', 'Bo-SVM', 'Bo-DNDT', 'Bo-TabNet', 'Bo-MLP', 'Bo-ELM'])

# 使用Seaborn绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='twilight_shifted', vmin=0.5, vmax=1.0, fmt=".2f")
plt.title('Spearman Correlation Matrix')
# 保存图形为SVG文件
plt.savefig('Tab.svg', format='svg', bbox_inches='tight')
plt.show()
