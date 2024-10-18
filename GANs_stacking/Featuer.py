import pandas as pd
from itertools import combinations

# 读取Excel文件
df = pd.read_excel('104.xlsx')

# 定义特征列表
features = ['P1', 'P2', 'P3', 'P4']

# 计算统计特征
df['mean_pressure'] = df[features].mean(axis=1)
df['max_pressure'] = df[features].max(axis=1)
df['min_pressure'] = df[features].min(axis=1)
df['std_pressure'] = df[features].std(axis=1)

# 生成特征组合
#for (f1, f2) in combinations(features, 2):
    #df[f'{f1}_{f2}'] = df[f1] * df[f2]

# 假设标签列名为 'label'
feature_columns = [col for col in df.columns if col != 'lable']
df = df[feature_columns + ['lable']]  # 重新排列列，将标签列放在最后

# 保存到Excel文件
df.to_excel('105.xlsx', index=False)
