import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CubicSpline
# 函数：计算 k-最近邻距离
def compute_dk(dataset, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(dataset)
    distances, _ = nbrs.kneighbors(dataset)
    return distances[:, k-1]

# 函数：计算基于 BADk 方法的极值，使用方差
def compute_BADk_extremes_with_variance(dk, c1=1.5, c2=1.5):
    Q1 = np.percentile(dk, 25)
    Q3 = np.percentile(dk, 75)
    var_X = np.var(dk)

    LE = Q1 - c1 * var_X
    UE = Q3 + c2 * (Q3 - np.median(dk))
    return LE, UE, var_X

# 函数：识别异常值
def identify_outliers(dk, LE, UE):
    outliers = (dk < LE) | (dk > UE)
    return outliers

# 函数：使用三次条样插值替换异常值
def replace_outliers_with_interpolation(dataset, outliers):
    replaced_dataset = dataset.copy()
    for i in range(dataset.shape[1]):
        feature = dataset[:, i]
        feature[outliers] = np.nan
        non_nan_indices = np.where(~np.isnan(feature))[0]
        nan_indices = np.where(np.isnan(feature))[0]
        if len(non_nan_indices) > 1:
            cs = CubicSpline(non_nan_indices, feature[non_nan_indices], bc_type='clamped')
            replaced_feature = cs(nan_indices)
            replaced_dataset[:, i][nan_indices] = replaced_feature
    return replaced_dataset

dataset = pd.read_excel('v4.xlsx')

# 应用 BADk 方法
k = 5  # k 的示例值
dk = compute_dk(dataset.values, k)

# 计算使用方差的极值
LE_with_variance, UE_with_variance, var_X = compute_BADk_extremes_with_variance(dk, c1=1.5, c2=1.5)

# 识别异常值
outliers_with_variance = identify_outliers(dk, LE_with_variance, UE_with_variance)

# 计算异常值数量
num_outliers_with_variance = np.sum(outliers_with_variance)
# 使用三次条样插值替换异常值
replaced_dataset = replace_outliers_with_interpolation(dataset.values, outliers_with_variance)
# 打印信息
print(f'使用方差的极值 (LE): {LE_with_variance}, 使用方差的极值 (UE): {UE_with_variance}')
print(f'使用方差的异常值数量: {num_outliers_with_variance}')
print(f'数据集的方差: {var_X}')

# 绘制箱线图并标出异常值
plt.boxplot(dataset.values, flierprops=dict(markerfacecolor='deepskyblue', marker='o', markersize=8))
plt.title('Boxplot of the Dataset with Outliers Highlighted')
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()