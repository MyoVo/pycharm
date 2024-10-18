from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = 'a2.xlsx'
data = pd.read_excel(file_path)

# 设置X为所有列，除了最后一列
X = data.iloc[:, :-1]  # 使用 .iloc[:, :-1] 选择除了最后一列之外的所有列作为特征

# 设置y为最后一列
y = data.iloc[:, -1]  # 使用 .iloc[:, -1] 选择最后一列作为标签

# 特征名称可以从DataFrame中获取除了最后一列外的列名
feature_names = data.columns[:-1]  # 使用 .columns 获取列名，并排除最后一列

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
importances = rf.feature_importances_

# 将特征重要性与特征名称结合，排序显示
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=True)  # 改为升序以匹配从下到上的条形图

# 创建足够长的颜色列表
base_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
colors = base_colors * (len(feature_importance) // 4) + base_colors[:len(feature_importance) % 4]

# 可视化特征重要性
plt.figure(figsize=(10, 8))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
# 保存图形为SVG文件
plt.savefig('f.svg', format='svg', bbox_inches='tight')
plt.show()
