from scipy.stats import wilcoxon
import numpy as np
# 模拟的数据（替换为你的实际数据）
accuracy_model1 = [0.9280, 0.9318, 0.9091, 0.9167, 0.9205]
accuracy_model2 = [1, 1, 0.9848, 1, 1]

precision_model1 = [0.9288, 0.9332, 0.9147, 0.9197, 0.9230]
precision_model2 = [1, 1, 0.9850, 0.82, 0.86]

recall_model1 = [0.9280, 0.9318, 0.9091, 0.9167, 0.9205]
recall_model2 = [1, 1, 0.9848, 1, 1]

f1_model1 = [0.9281, 0.9320, 0.9063, 0.9165, 0.9203]
f1_model2 = [1, 1, 0.9848, 1, 1]

# 每个性能指标的Wilcoxon符号秩检验和解释
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    # 计算Wilcoxon符号秩检验
    stat, p_value = wilcoxon(eval(f"{metric}_model1"), eval(f"{metric}_model2"))

    # 输出Wilcoxon统计量和P值
    print(f"Wilcoxon统计量 ({metric}): {stat}")
    print(f"P 值 ({metric}): {p_value}")

    # 判断显著性
    if p_value < 0.05:
        print("显著性水平为0.05，拒绝原假设，即两个模型在", metric, "方面存在显著差异。")
    else:
        print("显著性水平为0.05，无法拒绝原假设，即两个模型在", metric, "方面没有显著差异。")

    # 计算效应大小（根据实际情况选择合适的效应大小指标）
    effect_size = stat / np.sqrt(len(eval(f"{metric}_model1")))
    print(f"效应大小 ({metric}): {effect_size}")

    print("------------------------")
