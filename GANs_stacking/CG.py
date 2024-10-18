import matplotlib.pyplot as plt

# 定义训练集大小的百分比
training_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# 训练误差
training_errors = [0.12098360655737654, 0.082269938650306789, 0.04588555858310626,
                   0.040429009193054126, 0.032058823529411797,
                   0.02450646698434311, 0.0305869311551925,
                   0.0143950995405819, 0.029139745916515464]

# 测试误差
test_errors = [0.19575856443719408, 0.12561174551386622,
               0.1174551386623165, 0.09624796084828713,
               0.0780913539967374, 0.0804078303425776,
               0.07177814029363783, 0.057096247960848334, 0.0516215334420881]


plt.figure(figsize=(10, 6))
# 举例：使用三角形标记和虚线
plt.plot(training_sizes, training_errors, label='Training set', marker='x',
         linestyle='-', color='aquamarine',markersize='9', linewidth=2)
plt.plot(training_sizes, test_errors, label='Test set', marker='^',
         linestyle='-.', color='coral',markersize='9', linewidth=2)

# 添加图表标题和轴标签
plt.title('Convergence curve of the stacking ensemble model')
plt.xlabel('Training set in percent (%)')
plt.ylabel('Misclassification error')

# 添加图例
plt.legend()
plt.savefig('train test.svg', format='svg', bbox_inches='tight')
# 显示图表
plt.show()
