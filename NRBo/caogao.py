import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
# 加载数据集
data = pd.read_excel('u1.xlsx')

# 划分数据集
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = y - 1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建标准化器对象
scaler = StandardScaler()

# 对训练集进行标准化
X_train_scaled = scaler.fit_transform(X_train)

# 使用相同的标准化器对测试集进行标准化
X_test_scaled = scaler.transform(X_test)

# 基模型
estimators = [
    ('svm', SVC(C=6.2 ,gamma=1.7,kernel='rbf')),
    ('xgboost', xgb.XGBClassifier(reg_lambda=1.97,max_depth=15,learning_rate=0.1,subsample=0.62,reg_alpha=0.23,n_estimators=424)),
    ('bpnn', MLPClassifier(hidden_layer_sizes=6,learning_rate_init=0.009,activation="relu",solver="adam",max_iter=3000)),
    ('dt', DecisionTreeClassifier(max_depth=15,min_samples_split=4,min_samples_leaf=1)),
    ('rf', RandomForestClassifier(n_estimators=45,max_depth=13,min_samples_split=5,min_samples_leaf=1)),
]

'''
estimators = [
    ('svm', SVC()),
    ('xgboost', xgb.XGBClassifier()),
    ('bpnn', MLPClassifier()),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier()),
]
'''
# 元模型

meta_model = xgb.XGBClassifier()

# 堆叠分类器
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model)

# 训练堆叠分类器
stacking_clf.fit(X_train_scaled, y_train)

# 在测试集上进行预测
predictions = stacking_clf.predict(X_test_scaled)

# 输出分类报告
report = classification_report(y_test, predictions, digits=4)
print("Classification Report:\n", report)

# 计算 Kappa 值
kappa = cohen_kappa_score(y_test, predictions)
print("Kappa:", format(kappa, '.4f'))