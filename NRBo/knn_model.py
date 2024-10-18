import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from nrbo_module import NRBO  # 导入 NRBO 模块（假设这是一个超参数优化的模块）

# 参数搜索范围
param_ranges_knn = {
    'n_neighbors': [1, 5],  # K 的取值范围
    'weights': [0, 1]  # 0: 统一权重，1: 距离加权
}

def knn_fobj(params, X_train, y_train):
    # 确保 params 是一个字典
    if not isinstance(params, dict):
        params = {'n_neighbors': int(params[0]), 'weights': params[1]}  # 不再转换为整数

    # 从 params 字典中提取参数
    n_neighbors = int(max(param_ranges_knn['n_neighbors'][0], min(param_ranges_knn['n_neighbors'][1], params['n_neighbors'])))
    weights = ['uniform', 'distance'][int(params['weights']) % 2]  # 强制转换为字符串类型

    # 使用提取的参数构建 KNN 模型
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    # 使用交叉验证评估模型性能
    scores = cross_val_score(knn_model, X_train, y_train, cv=5)

    # 返回交叉验证得分的负值
    return -np.max(scores)


def optimize_knn_params(X_train, y_train, N, MaxIt):
    # 使用 param_ranges_knn 中的上下界
    LB = [param_ranges_knn['n_neighbors'][0], param_ranges_knn['weights'][0]]
    UB = [param_ranges_knn['n_neighbors'][1], param_ranges_knn['weights'][1]]

    dim = 2  # 使用待优化参数的维度

    _, best_params, best_score = NRBO(N, MaxIt, LB, UB, dim, knn_fobj, "knn Model",X_train, y_train)

    return best_params, best_score

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, best_params):
    # 裁剪参数
    n_neighbors, weights_idx = [int(max(param_ranges_knn['n_neighbors'][0], min(param_ranges_knn['n_neighbors'][1], value)))
                                for value in best_params[:2]]
    weights = ['uniform', 'distance'][int(weights_idx) % 2]

    # 输出最优参数
    print("最终参数 n_neighbors:", n_neighbors)
    print("最终参数 weights:", weights)

    # 使用最优参数训练 KNN 模型
    optimized_knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    optimized_knn_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = optimized_knn_model.predict(X_test)

    # 输出准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("KNN 测试集准确率:", accuracy)
    print("KNN 分类报告:\n", report)

    return optimized_knn_model, accuracy, report
