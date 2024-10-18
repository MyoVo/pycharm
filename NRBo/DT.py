import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from nrbo_module import NRBO

# 参数搜索范围 for Decision Tree
param_ranges_decision_tree = {
    'max_depth': [1, 10],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 10]
}

def decision_tree_fobj(params, X_train, y_train):
    # 确保 params 是一个字典
    if not isinstance(params, dict):
        params = {'max_depth': int(params[0]), 'min_samples_split': int(params[1]), 'min_samples_leaf': int(params[2])}

    # 从 params 字典中提取参数并确保在范围内
    max_depth = max(param_ranges_decision_tree['max_depth'][0], min(param_ranges_decision_tree['max_depth'][1], params['max_depth']))
    min_samples_split = max(param_ranges_decision_tree['min_samples_split'][0], min(param_ranges_decision_tree['min_samples_split'][1], params['min_samples_split']))
    min_samples_leaf = max(param_ranges_decision_tree['min_samples_leaf'][0], min(param_ranges_decision_tree['min_samples_leaf'][1], params['min_samples_leaf']))

    decision_tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf, random_state=42)
    scores = cross_val_score(decision_tree_model, X_train, y_train, cv=5)

    # 返回交叉验证得分的最大值
    return -np.max(scores)


def optimize_decision_tree_params(X_train, y_train, N, MaxIt):
    # 使用 param_ranges_decision_tree 中的上下界
    LB = [param_ranges_decision_tree['max_depth'][0], param_ranges_decision_tree['min_samples_split'][0],
          param_ranges_decision_tree['min_samples_leaf'][0]]
    UB = [param_ranges_decision_tree['max_depth'][1], param_ranges_decision_tree['min_samples_split'][1],
          param_ranges_decision_tree['min_samples_leaf'][1]]

    dim = 3  # 待优化参数的维度

    _, best_params, best_score = NRBO(N, MaxIt, LB, UB, dim, decision_tree_fobj, "dt Model",X_train, y_train)

    return best_params, best_score

def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test, best_params):
    # 从 best_params 中提取参数
    max_depth, min_samples_split, min_samples_leaf = best_params[:3]

    # 输出最优参数
    print("最终 max_depth 参数:", max_depth)
    print("最终 min_samples_split 参数:", min_samples_split)
    print("最终 min_samples_leaf 参数:", min_samples_leaf)

    # 使用最优参数训练决策树模型
    optimized_decision_tree_model = DecisionTreeClassifier(max_depth=int(max_depth),
                                                           min_samples_split=int(min_samples_split),
                                                           min_samples_leaf=int(min_samples_leaf))
    optimized_decision_tree_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = optimized_decision_tree_model.predict(X_test)

    # 输出准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("dt测试集准确率:", accuracy)
    print("dt分类报告:\n", report)

    return optimized_decision_tree_model, accuracy, report
