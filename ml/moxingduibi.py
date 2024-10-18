import matplotlib.pyplot as plt

# 数据
models = ['XGBoost', 'RF', 'ELM', 'Tabnet', 'SVM', 'DNDT', 'MLP', 'Stacking']
scores = [0.9070, 0.9003, 0.8904, 0.8987, 0.9022, 0.9091, 0.8843, 0.9362]

# 设置图形大小
plt.figure(figsize=(10, 6))

# 设置条形颜色，Stacking为绿色，其他为蓝色
colors = ['dodgerblue'] * (len(models) - 1) + ['limegreen']

# 绘制垂直条形图
bars = plt.bar(models, scores, color=colors)

# 设置y轴的范围
plt.ylim(0.80, 0.96)

# 添加标题和标签
plt.title('')
plt.xlabel('')
plt.ylabel('')

# 调整x轴的标签角度
#plt.xticks(rotation=45, ha='right')


# 设置坐标轴字体大小
plt.xticks(fontsize=18)
plt.yticks(fontsize=19)
# 显示网格
plt.grid(True, axis='y', zorder=0)

# 让条形图在网格线之上
for bar in bars:
    bar.set_zorder(3)

# 确保x轴线在最上层
ax = plt.gca()
ax.spines['bottom'].set_zorder(3)

plt.savefig('f1.png', format='png', bbox_inches='tight', dpi=150)
# 显示图形
plt.show()
