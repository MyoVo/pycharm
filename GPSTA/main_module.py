import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tu import plot_confusion_matrix, plot_performance_curves ,plot_stacking_model_performance,plot_roc_curve
from genetic_ import create_standard_nn, train_and_optimize_with_genetic_algorithm
from yi import create_meta_model
import seaborn as sns
import matplotlib.pyplot as plt
def calculate_performance_metrics(model, X_test, y_test, n_classes):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

# 读取数据集
data = pd.read_excel('vs1.xlsx')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y_categorical = to_categorical(y-1, num_classes=4)  # 四分类问题
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42,shuffle=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
all_train_losses = []  # 存储每个fold的训练损失历史
all_val_losses = []  # 存储每个fold的验证损失历史
all_train_accuracies = []  # 存储每个fold的训练准确率历史
all_val_accuracies = []  # 存储每个fold的验证准确率历史
# 五折交叉验证
for train_index, val_index in kf.split(X_train_scaled):
    # 创建模型
    model = create_standard_nn(X_train_scaled.shape[1], y_train.shape[1])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练和验证模型
    history = model.fit(X_train_scaled[train_index], y_train[train_index], epochs=100, verbose=0,
                        validation_data=(X_train_scaled[val_index], y_train[val_index]))

    # 存储训练和验证损失
    all_train_losses.append(history.history['loss'])
    all_val_losses.append(history.history['val_loss'])

    # 存储训练和验证准确率
    all_train_accuracies.append(history.history['accuracy'])
    all_val_accuracies.append(history.history['val_accuracy'])

    _, accuracy = model.evaluate(X_train_scaled[val_index], y_train[val_index], verbose=0)
    accuracies.append(accuracy)

    # 在验证集上预测
    y_val_true = np.argmax(y_train[val_index], axis=1)
    y_val_pred = np.argmax(model.predict(X_train_scaled[val_index]), axis=1)

    # 计算性能指标
    accuracy_val = accuracy_score(y_val_true, y_val_pred)
    recall_val = recall_score(y_val_true, y_val_pred, average='weighted')
    f1_val = f1_score(y_val_true, y_val_pred, average='weighted')
    precision_val = precision_score(y_val_true, y_val_pred, average='weighted')
    print(f'Fold  Accuracy: {accuracy_val:.4f}')
    print(f'Fold  Recall: {recall_val:.4f}')
    print(f'Fold  F1 Score: {f1_val:.4f}')
    print(f'Fold  Precision: {precision_val:.4f}')
    print("--------------------")

# 计算交叉验证的平均准确率和标准差
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'5-Fold Cross-Validation Accuracy: {average_accuracy:.4f} (±{std_accuracy:.4f})')

# 计算平均训练和验证损失
#avg_train_loss = np.mean(all_train_losses, axis=0)
#avg_val_loss = np.mean(all_val_losses, axis=0)
# 计算平均训练损失和准确率
avg_train_loss = np.mean(all_train_losses, axis=0)
avg_train_accuracy = np.mean(all_train_accuracies, axis=0)

# 计算平均验证损失和准确率
avg_val_loss = np.mean(all_val_losses, axis=0)
avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

# 绘制平均训练和验证损失曲线
plt.plot(avg_train_loss, label='Average Training Loss')
plt.plot(avg_val_loss, label='Average Validation Loss')

plt.title('Average Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制平均训练和验证准确率曲线
plt.plot(avg_train_accuracy, label='Average Training Accuracy', linestyle='-',color='darkred')
plt.plot(avg_val_accuracy, label='Average Validation Accuracy', linestyle='-',color='darkblue')

plt.title('Average Training and Validation Loss/Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# 在整个训练集上训练模型
final_model = create_standard_nn(X_train_scaled.shape[1], y_train.shape[1])
final_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
final_model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 在测试集上评估模型
test_loss, test_accuracy = final_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Set Accuracy: {test_accuracy:.4f}')

# 计算并输出标准神经网络性能指标
standard_nn_metrics = calculate_performance_metrics(model, X_test_scaled, y_test, 5)
print("Standard Neural Network Metrics:")
for metric_name, metric_value in standard_nn_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# 存储标准神经网络的初始参数
initial_params = {
    'neurons_layer1': model.layers[0].units,
    'neurons_layer2': model.layers[1].units,
    'learning_rate': 0.001,  # 适用于您的初始学习率
    'optimizer': Adam(learning_rate=0.001),  # 适用于您的优化器和初始学习率
    'epochs': 100  # 适用于您的初始训练周期数
}

# 输出标准神经网络的初始参数
print("\nInitial Parameters for Standard Neural Network:")
for param_name, param_value in initial_params.items():
    print(f"{param_name}: {param_value}")

n_classes = 4

# 预测测试集的标签
standard_nn_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
# 调用混淆矩阵绘制函数
plot_confusion_matrix(np.argmax(y_test, axis=1), standard_nn_predictions, title='Standard Neural Network Confusion Matrix', normalize=True)

print("+++++++++++------------------------------------------------")

best_params_list = []
optimized_models = []  # 初始化模型列表

# 运行遗传算法优化五次
for iteration in range(5):
    print(f"\nIteration {iteration + 1}")

    # 调用第一个模块并运行遗传算法优化
    best_params = train_and_optimize_with_genetic_algorithm(X_train_scaled, y_train, X_test_scaled, y_test,
                                                            population_size=20, generations=3)
    # 使用优化的参数重新创建模型并运行五次
    optimized_model = create_standard_nn(num_features=X_train_scaled.shape[1], num_classes=y_train.shape[1])
    optimized_model.layers[0].units = best_params['neurons_layer1']
    optimized_model.layers[1].units = best_params['neurons_layer2']
    optimized_model.compile(optimizer=best_params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    history = optimized_model.fit(X_train_scaled, y_train, epochs=best_params['epochs'], verbose=0,
                                  validation_data=(X_test_scaled, y_test))
    test_accuracy = history.history['val_accuracy'][-1]

    # 存储模型
    optimized_models.append(optimized_model)
    # 添加最佳参数到列表
    best_params_list.append(best_params)

# 输出每次迭代的最佳参数
for iteration, best_params in enumerate(best_params_list):
    print(f"\nBest Parameters in Iteration {iteration + 1}:")
    for param_name, param_value in best_params.items():
        print(f"{param_name}: {param_value}")


from sklearn.model_selection import KFold
# 准备 KFold 实例
kf = KFold(n_splits=5)

# 假设我们有5个基模型，每个模型输出4个特征
num_base_models = 5
num_features_per_model = 4
input_shape_for_meta_model = num_base_models * num_features_per_model

# 其他参数
output_shape = y_train.shape[1]  # 类别数量
neurons_layer2 = 64  # 假设隐藏层有64个神经元

# 存储每个折的性能指标
fold_performance_metrics = []
base_model_histories = []  # 用于存储每个基模型的训练历史
meta_model_histories = []  # 用于存储每个折中元模型的训练历史


for train_index, val_index in kf.split(X_train_scaled):
    # 分割数据
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 存储基模型的预测
    base_model_predictions = []
    fold_base_model_histories = []  # 用于存储当前折中基模型的训练历史

    # 训练并预测每个基模型
    for model in optimized_models:
        history = model.fit(X_train_fold, y_train_fold, epochs=200, verbose=0)
        predictions = model.predict(X_val_fold)
        base_model_predictions.append(predictions)
        fold_base_model_histories.append(history)

    base_model_histories.append(fold_base_model_histories)

    # 合并基模型的预测，作为新特征
    stacked_features = np.column_stack(base_model_predictions)

    # 创建并训练元模型
    meta_model = create_meta_model(input_shape_for_meta_model, output_shape, neurons_layer2)
    # 存储训练历史
    # 修改这一行以包含验证数据
    history_meta = meta_model.fit(stacked_features, y_val_fold, epochs=200, verbose=1, validation_split=0.2)

    # 将训练历史追加到列表中
    meta_model_histories.append(history_meta)
    meta_model.fit(stacked_features, y_val_fold, epochs=200, verbose=0)  # 根据需要调整 epochs 和 verbose

    # 在验证集上评估元模型
    val_predictions = meta_model.predict(stacked_features)
    val_predictions_classes = np.argmax(val_predictions, axis=1)
    y_val_true = np.argmax(y_val_fold, axis=1)

    # 计算并存储性能指标
    accuracy = accuracy_score(y_val_true, val_predictions_classes)
    precision = precision_score(y_val_true, val_predictions_classes, average='weighted')
    recall = recall_score(y_val_true, val_predictions_classes, average='weighted')
    f1 = f1_score(y_val_true, val_predictions_classes, average='weighted')

    fold_performance_metrics.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })


# 用于存储每个折中的训练和验证准确率
all_train_accuracies = []
all_val_accuracies = []

# 从每个折中的历史对象中提取数据
for history in meta_model_histories:
    # 请确认下面的键与您的历史数据中的键匹配
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    all_train_accuracies.append(train_accuracy)
    all_val_accuracies.append(val_accuracy)

# 将所有折中的数据转换为numpy数组以便计算平均值
all_train_accuracies = np.array(all_train_accuracies)
all_val_accuracies = np.array(all_val_accuracies)

# 计算平均训练和验证准确率
avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

# 绘制准确率曲线图
plt.figure(figsize=(10, 6))
plt.plot(avg_train_accuracy, label='Training Accuracy',color='darkred')
plt.plot(avg_val_accuracy, label='Validation Accuracy',color='darkblue')
plt.title('Average Training and Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# 计算平均性能指标
average_performance = {metric: np.mean([fold_metrics[metric] for fold_metrics in fold_performance_metrics]) for metric in fold_performance_metrics[0]}
print("Average Performance Metrics Across Folds:")
for metric, value in average_performance.items():
    print(f"{metric}: {value:.4f}")

# 首先，使用所有的基模型对测试集进行预测
base_model_test_predictions = [model.predict(X_test_scaled) for model in optimized_models]

# 将所有基模型的预测结果合并为新的特征集
stacked_test_features = np.column_stack(base_model_test_predictions)

# 现在，使用元模型对这些新特征进行预测
meta_model_test_predictions = meta_model.predict(stacked_test_features)
meta_model_test_predictions_classes = np.argmax(meta_model_test_predictions, axis=1)

# 计算测试集上的准确率
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), meta_model_test_predictions_classes)
print(f'Test Set Accuracy of Meta Model: {test_accuracy:.4f}')

# 预测测试集的标签
standard_nn_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
# 调用混淆矩阵绘制函数
plot_confusion_matrix(np.argmax(y_test, axis=1), standard_nn_predictions, title='Stacking Confusion Matrix', normalize=True)

# Plot loss curves for each fold
for fold, history_meta in enumerate(meta_model_histories):
    plt.plot(history_meta.history['loss'], label=f'Fold {fold + 1}')

plt.title('Stacking Model Loss Curves Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation loss curves for the final stacking model
plt.plot(history_meta.history['loss'], label='Training Loss')
plt.plot(history_meta.history['val_loss'], label='Validation Loss')
plt.title('Stacking Model Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 计算并输出标准神经网络性能指标
standard_nn_metrics = calculate_performance_metrics(model, X_test_scaled, y_test, n_classes)
print("Standard Neural Network Metrics:")
for metric_name, metric_value in standard_nn_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# 计算并输出堆叠模型的性能指标
stacked_model_metrics = calculate_performance_metrics(meta_model, stacked_test_features, y_test, n_classes)
print("\nStacked Model Metrics:")
for metric_name, metric_value in stacked_model_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# 绘制性能指标的比较曲线图
plt.figure(figsize=(12, 6))

# 设置性能指标的标签和值
labels = list(standard_nn_metrics.keys())
standard_nn_scores = list(standard_nn_metrics.values())
stacked_model_scores = list(stacked_model_metrics.values())

# 为了绘制曲线图，我们需要一个x轴的位置序列
x = np.arange(len(labels))

# 绘制曲线
plt.plot(x, standard_nn_scores, marker='o', label='Standard NN')
plt.plot(x, stacked_model_scores, marker='s', label='Stacked Model')

# 添加标题和标签
plt.ylabel('Scores')
plt.title('Performance Metrics Comparison')
plt.xticks(x, labels)
plt.legend()

# 显示图表
plt.show()