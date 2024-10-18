import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from nrbo_module import NRBO
import warnings

# 禁用所有警告消息
warnings.filterwarnings("ignore")

# 参数搜索范围 for BP神经网络
param_ranges_nn = {
    'hidden_layer_sizes': [1, 5],  # 隐藏层大小
    'learning_rate_init': [0.001, 0.1],  # 初始学习率
    'activation': [0, 1, 2],  # 0: 'relu', 1: 'tanh', 2: 'logistic'
    'solver': [0, 1, 2]  # 0: 'adam', 1: 'sgd', 2: 'lbfgs'
}
def nn_fobj(params, X_train, y_train):
    # 确保 params 是一个字典
    if not isinstance(params, dict):
        params = {'hidden_layer_sizes': int(params[0]), 'learning_rate_init': params[1],
                  'activation': ['relu', 'tanh', 'logistic'][int(params[2]) % 3],
                  'solver': ['adam', 'sgd', 'lbfgs'][int(params[3]) % 3]}

    # 从 params 字典中提取参数并确保在范围内
    hidden_layer_sizes = max(param_ranges_nn['hidden_layer_sizes'][0], min(param_ranges_nn['hidden_layer_sizes'][1], params['hidden_layer_sizes']))
    learning_rate_init = max(param_ranges_nn['learning_rate_init'][0], min(param_ranges_nn['learning_rate_init'][1], params['learning_rate_init']))
    activation = params['activation']
    solver = params['solver']

    nn_model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), learning_rate_init=learning_rate_init,
                             activation=activation, solver=solver,max_iter=3000, random_state=42)
    scores = cross_val_score(nn_model, X_train, y_train, cv=5)

    # 返回交叉验证得分的最大值
    return -np.max(scores)


def optimize_nn_params(X_train, y_train, N, MaxIt):
    # 使用 param_ranges_nn 中的上下界
    LB = [param_ranges_nn['hidden_layer_sizes'][0], param_ranges_nn['learning_rate_init'][0],
          param_ranges_nn['activation'][0], param_ranges_nn['solver'][0]]
    UB = [param_ranges_nn['hidden_layer_sizes'][1], param_ranges_nn['learning_rate_init'][1],
          param_ranges_nn['activation'][2], param_ranges_nn['solver'][2]]

    dim = 4  # 待优化参数的维度

    _, best_params, best_score = NRBO(N, MaxIt, LB, UB, dim, nn_fobj,"bpnn Model", X_train, y_train)

    return best_params, best_score

def train_and_evaluate_nn(X_train, y_train, X_test, y_test, best_params):
    # 从 best_params 中提取参数
    hidden_layer_sizes, learning_rate_init, activation, solver = best_params[:4]

    # 转换 hidden_layer_sizes 参数为元组类型
    hidden_layer_sizes = (int(hidden_layer_sizes),)

    # 输出最优参数
    print("最终 hidden_layer_sizes 参数:", hidden_layer_sizes)
    print("最终 learning_rate_init 参数:", learning_rate_init)
    print("最终 activation 参数:", ['relu', 'tanh', 'logistic'][int(activation) % 3])
    print("最终 solver 参数:", ['adam', 'sgd', 'lbfgs'][int(solver) % 3])

    # 使用最优参数训练 BP神经网络 模型
    optimized_nn_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init,
                                       activation=['relu', 'tanh', 'logistic'][int(activation) % 3],
                                       solver=['adam', 'sgd', 'lbfgs'][int(solver) % 3], random_state=42)
    optimized_nn_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = optimized_nn_model.predict(X_test)

    # 输出准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("bpnn测试集准确率:", accuracy)
    print("bpnn分类报告:\n", report)

    return optimized_nn_model, accuracy, report
