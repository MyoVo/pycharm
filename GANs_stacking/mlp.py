import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载示例数据集（这里使用的是Iris数据集）
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林分类器
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 绘制特征重要性条形图
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# 绘制详细的SHAP值图
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
