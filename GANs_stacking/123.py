import pandas as pd
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# 加载数据
file_path = 'a2.xlsx'  # 请替换为你的文件路径
data = pd.read_excel(file_path)

# 分离特征和标签
features = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列
labels = data.iloc[:, -1]  # 所有行，最后一列

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 定义模型
models = {
    'SVM': SVC(kernel='linear', probability=True),
    'XGBoost': XGBClassifier(),

}

# 逐步添加特征
add_results = []

for i in range(1, len(features.columns) + 1):
    selected_features = features.columns[:i]
    X_selected = features[selected_features].values

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_selected, labels_encoded, test_size=0.2, random_state=42)

    accuracies = []
    for model_name, model in models.items():
        if model_name == 'ELM':
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = np.mean(preds == y_test)
        else:
            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            accuracy = scores.mean()
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies)
    add_results.append((i, avg_accuracy))

add_results_df = pd.DataFrame(add_results, columns=['FeatureCount', 'AvgAccuracy'])

# 可视化逐步添加特征的结果
plt.figure(figsize=(14, 8))
sns.lineplot(data=add_results_df, x='FeatureCount', y='AvgAccuracy', marker='o')
plt.title('Average Accuracy vs. Number of Features (Adding Features)')
plt.xlabel('Number of Features')
plt.ylabel('Average Accuracy')
plt.show()
