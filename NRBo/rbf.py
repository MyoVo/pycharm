from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from nrbo_module import NRBO
import numpy as np
# 参数搜索范围 for 随机森林
param_ranges_rf = {
    'n_estimators': [10, 200],
    'max_depth': [1, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def random_forest_fobj(params, X_train, y_train):
    # 确保 params 是一个字典
    if not isinstance(params, dict):
        params = {'n_estimators': int(params[0]), 'max_depth': int(params[1]),
                  'min_samples_split': int(params[2]), 'min_samples_leaf': int(params[3])}

    # 从 params 字典中提取参数并确保在范围内
    n_estimators = max(param_ranges_rf['n_estimators'][0], min(param_ranges_rf['n_estimators'][1], params['n_estimators']))
    max_depth = params['max_depth'] if params['max_depth'] is None else max(param_ranges_rf['max_depth'][0], min(param_ranges_rf['max_depth'][1], params['max_depth']))
    min_samples_split = max(param_ranges_rf['min_samples_split'][0], min(param_ranges_rf['min_samples_split'][1], params['min_samples_split']))
    min_samples_leaf = max(param_ranges_rf['min_samples_leaf'][0], min(param_ranges_rf['min_samples_leaf'][1], params['min_samples_leaf']))

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    scores = cross_val_score(rf_model, X_train, y_train, cv=5)

    # 返回交叉验证得分的最大值
    return -np.max(scores)


def optimize_random_forest_params(X_train, y_train, N, MaxIt):
    # 使用 param_ranges_rf 中的上下界
    LB = [param_ranges_rf['n_estimators'][0], param_ranges_rf['max_depth'][0],
          param_ranges_rf['min_samples_split'][0], param_ranges_rf['min_samples_leaf'][0]]
    UB = [param_ranges_rf['n_estimators'][1], param_ranges_rf['max_depth'][1],
          param_ranges_rf['min_samples_split'][1], param_ranges_rf['min_samples_leaf'][1]]

    dim = 4  # 待优化参数的维度

    _, best_params, best_score = NRBO(N, MaxIt, LB, UB, dim, random_forest_fobj,"rbf Model", X_train, y_train)

    return best_params, best_score

def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test, best_params):
    # 从 best_params 中提取参数
    n_estimators, max_depth, min_samples_split, min_samples_leaf = best_params[:4]

    # 将 max_depth 转换为整数类型
    max_depth = int(max_depth)

    # 输出最优参数
    print("最终 n_estimators 参数:", n_estimators)
    print("最终 max_depth 参数:", max_depth)
    print("最终 min_samples_split 参数:", min_samples_split)
    print("最终 min_samples_leaf 参数:", min_samples_leaf)

    # 使用最优参数训练随机森林模型
    optimized_rf_model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth,
                                                 min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf))
    optimized_rf_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = optimized_rf_model.predict(X_test)

    # 输出准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("随机森林测试集准确率:", accuracy)
    print("随机森林分类报告:\n", report)

    return optimized_rf_model, accuracy, report
