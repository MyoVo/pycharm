import pandas as pd
import numpy as np

# 加载Excel文件
file_path = '3.xlsx'
df = pd.read_excel(file_path)

# 定义各标签的P2特征值区间
ranges = {
    'Supine': (1.696, 2.012),
    'Side': (1.9, 2.403),
    'Foetus': (1.303, 1.856),
    'Prone': (1.565, 2.207)
}
# 对每个标签进行处理
for label, (lower_bound, upper_bound) in ranges.items():
    # 筛选出该标签的数据
    label_df = df[df['lable'] == label]

    # 对不在区间内的P2特征值进行处理
    invalid_mask = (label_df['P4'] < lower_bound) | (label_df['P4'] > upper_bound)
    random_values = np.random.uniform(lower_bound, upper_bound, size=invalid_mask.sum())

    # 更新数据
    df.loc[invalid_mask & (df['lable'] == label), 'P4'] = random_values

# 保存处理后的数据到新的Excel文件
output_file_path = '4.xlsx'
df.to_excel(output_file_path, index=False)

print(f"处理后的数据已保存到 {output_file_path}")
