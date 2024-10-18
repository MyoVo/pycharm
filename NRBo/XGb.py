import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from nrbo_module import NRBO

# 参数搜索范围 for XGBoost
param_ranges_xgboost = {
    'n_estimators': [10, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [1, 10],
    'subsample': [0.5, 1.0],
    'reg_alpha': [0.001, 1.0],
    'reg_lambda': [0.001, 3.0]
}

def xgboost_fobj(params, X_train, y_train):
    # 确保 params 是一个字典
    if not isinstance(params, dict):
        params = {'n_estimators': int(params[0]), 'learning_rate': params[1],
                  'max_depth': int(params[2]), 'subsample': params[3],
                  'reg_alpha': params[4], 'reg_lambda': params[5]}

    # 从 params 字典中提取参数并确保在范围内
    n_estimators = max(param_ranges_xgboost['n_estimators'][0], min(param_ranges_xgboost['n_estimators'][1], params['n_estimators']))
    learning_rate = max(param_ranges_xgboost['learning_rate'][0], min(param_ranges_xgboost['learning_rate'][1], params['learning_rate']))
    max_depth = max(param_ranges_xgboost['max_depth'][0], min(param_ranges_xgboost['max_depth'][1], params['max_depth']))
    subsample = max(param_ranges_xgboost['subsample'][0], min(param_ranges_xgboost['subsample'][1], params['subsample']))
    reg_alpha = max(param_ranges_xgboost['reg_alpha'][0], min(param_ranges_xgboost['reg_alpha'][1], params['reg_alpha']))
    reg_lambda = max(param_ranges_xgboost['reg_lambda'][0], min(param_ranges_xgboost['reg_lambda'][1], params['reg_lambda']))

    xgb_model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                              max_depth=max_depth, subsample=subsample,
                              reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    scores = cross_val_score(xgb_model, X_train, y_train, cv=5)

    # 返回交叉验证得分的最大值
    return -np.max(scores)




def optimize_xgboost_params(X_train, y_train, N, MaxIt):
    # 使用 param_ranges_xgboost 中的上下界
    LB = [param_ranges_xgboost['n_estimators'][0], param_ranges_xgboost['learning_rate'][0],
          param_ranges_xgboost['max_depth'][0], param_ranges_xgboost['subsample'][0],
          param_ranges_xgboost['reg_alpha'][0], param_ranges_xgboost['reg_lambda'][0]]
    UB = [param_ranges_xgboost['n_estimators'][1], param_ranges_xgboost['learning_rate'][1],
          param_ranges_xgboost['max_depth'][1], param_ranges_xgboost['subsample'][1],
          param_ranges_xgboost['reg_alpha'][1], param_ranges_xgboost['reg_lambda'][1]]

    dim = 6  # 待优化参数的维度

    _, best_params, best_score = NRBO(N, MaxIt, LB, UB, dim, xgboost_fobj, "xgb Model",X_train, y_train)

    return best_params, best_score

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, best_params):
    # 从 best_params 中提取参数
    n_estimators, learning_rate, max_depth, subsample, reg_alpha, reg_lambda = best_params[:6]

    # 输出最优参数
    print("最终 n_estimators 参数:", n_estimators)
    print("最终 learning_rate 参数:", learning_rate)
    print("最终 max_depth 参数:", max_depth)
    print("最终 subsample 参数:", subsample)
    print("最终 reg_alpha 参数:", reg_alpha)
    print("最终 reg_lambda 参数:", reg_lambda)

    # 使用最优参数训练 XGBoost 模型
    optimized_xgb_model = XGBClassifier(n_estimators=int(n_estimators), learning_rate=learning_rate,
                                        max_depth=int(max_depth), subsample=subsample,
                                        reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    optimized_xgb_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = optimized_xgb_model.predict(X_test)

    # 输出准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("xgb测试集准确率:", accuracy)
    print("xgb分类报告:\n", report)

    return optimized_xgb_model, accuracy, report
