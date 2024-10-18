import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.special import softmax
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

class MultiClassAUC(Metric):
    def __init__(self):
        self._name = "auc"  # 使用内部变量名 _name
        self._maximize = True  # AUC 越高越好

    def __call__(self, y_true, y_score):
        return roc_auc_score(y_true, y_score, multi_class="ovo", average="macro")


def train_tabnet_(X_train, y_train, X_test, y_test):
    # 定义要优化的超参数空间
    dim_num_decision_steps = Integer(low=3, high=10, name='n_steps')
    dim_relaxation_factor = Real(low=1.0, high=2, name='gamma')
    dim_sparsity_coefficient = Real(low=1e-5, high=1e-1, prior='log-uniform', name='lambda_sparse')

    dimensions = [dim_num_decision_steps, dim_relaxation_factor, dim_sparsity_coefficient]

    @use_named_args(dimensions=dimensions)
    def fitness(n_steps, gamma, lambda_sparse):
        # 创建 TabNet 模型
        tabnet_model = TabNetClassifier(
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0,

        )

        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        # 训练模型
        tabnet_model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            max_epochs=100,
            patience=20,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0,
            drop_last=False
        )

        # 检查历史记录中的键并获取最佳损失
        print(tabnet_model.history.keys())
        if 'val_loss' in tabnet_model.history:
            best_loss = np.min(tabnet_model.history['val_loss'])
        else:
            raise ValueError("Validation loss not found in model history.")

        return best_loss

    # 使用贝叶斯优化找到最佳超参数
    search_result = gp_minimize(func=fitness, dimensions=dimensions, n_calls=30, x0=[3, 1.0, 1e-4])

    print("Best parameters:", search_result.x)
    print("Best validation loss:", search_result.fun)

    # 使用最佳超参数重新训练模型
    best_num_decision_steps, best_relaxation_factor, best_sparsity_coefficient = search_result.x
    best_model = TabNetClassifier(
        n_steps=best_num_decision_steps,
        gamma=best_relaxation_factor,
        lambda_sparse=best_sparsity_coefficient,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,

    )

    best_model.fit(X_train, y_train, max_epochs=1000, batch_size=64, virtual_batch_size=32,
                   num_workers=0, drop_last=False)

    # 模型评估和预测
    y_pred_probs = best_model.predict(X_test)
    y_pred_probs = softmax(y_pred_probs, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    tabnet_accuracy = accuracy_score(y_test, y_pred)
    print("Final model accuracy:", tabnet_accuracy)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 绘制百分比混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Supine', 'Side', 'Foetus', 'Prone'],
                yticklabels=['Supine', 'Side', 'Foetus', 'Prone'])
    plt.title('Confusion Matrix (Bys-Tabnet)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('Tab.svg', format='svg', bbox_inches='tight')
    plt.show()

    return best_model, tabnet_accuracy
