import pandas as pd

# 读取数据
data = pd.read_excel('103.xlsx')

# 定义函数计算每个特征的统计值
def calculate_feature_stats(df, feature_name):
    stats = pd.DataFrame()
    stats[f'{feature_name}_max'] = [df[feature_name].max()] * len(df)
    stats[f'{feature_name}_min'] = [df[feature_name].min()] * len(df)
    stats[f'{feature_name}_sum'] = [df[feature_name].sum()] * len(df)
    stats[f'{feature_name}_std'] = [df[feature_name].std()] * len(df)
    stats[f'{feature_name}_q1'] = [df[feature_name].quantile(0.25)] * len(df)
    stats[f'{feature_name}_q3'] = [df[feature_name].quantile(0.75)] * len(df)
    stats[f'{feature_name}_mean'] = [df[feature_name].mean()] * len(df)

    return stats

# 根据序列号分组（第一列是序列号）
grouped = data.groupby(data.columns[0])  # 假设第一列是序列号

result = pd.DataFrame()

# 对每个序列号分组计算特征的统计值
for name, group in grouped:
    feature_stats = pd.DataFrame()

    for feature in ['p1', 'p2', 'p3', 'p4']:  # 请根据实际特征列名调整
        stats = calculate_feature_stats(group, feature)
        feature_stats = pd.concat([feature_stats, stats], axis=1)

    # 添加序列号和标签列
    sequence_number = pd.Series([name] * len(group), name='Sequence')  # 保留序列号
    labels = group.iloc[:, -1]  # 假设标签在最后一列

    # 合并原始特征、统计特征、序列号和标签
    final_group_data = pd.concat([sequence_number.reset_index(drop=True),
                                  group[['p1', 'p2', 'p3', 'p4']].reset_index(drop=True),
                                  feature_stats.reset_index(drop=True),
                                  labels.reset_index(drop=True)], axis=1)

    # 将每个序列号的结果合并到最终结果
    result = pd.concat([result, final_group_data], axis=0)

# 保存结果到 Excel 文件
output_path = '4.xlsx'
result.to_excel(output_path, index=False)
