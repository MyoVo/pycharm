import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font

# 加载数据集
file_path = '6.xlsx'  # 确保6.xlsx文件在当前目录中
df = pd.read_excel(file_path)

# 随机向数据集中添加4%的噪声，确保每个标签都有噪声
num_rows, num_cols = df.shape
total_values = num_rows * (num_cols - 1)  # 总值的数量，不包括标签列
num_noise = int(total_values * 0.04)  # 需要添加噪声的值的数量

# 创建数据集副本以标记噪声
df_with_noise = df.copy()
noise_indices = []

# 确保每个标签都有噪声
for label in df.iloc[:, 0].unique():
    label_indices = df[df.iloc[:, 0] == label].index
    label_noise_indices = np.random.choice(label_indices, size=int(num_noise / len(df.iloc[:, 0].unique())),
                                           replace=False)

    for row in label_noise_indices:
        col = np.random.randint(1, num_cols)  # 随机选择一个列，跳过标签列
        original_value = df_with_noise.iloc[row, col]
        noise_value = original_value + np.random.normal(0, 0.1 * original_value)  # 添加噪声
        df_with_noise.iloc[row, col] = noise_value
        noise_indices.append((row + 2, col + 1))  # 记录噪声位置，+2行是因为Excel从1开始且有标题，+1列是因为Excel从1开始

# 先将修改后的数据集保存到一个新的 Excel 文件
output_file_path = '7.xlsx'
df_with_noise.to_excel(output_file_path, index=False)

# 重新打开文件以添加格式
wb = openpyxl.load_workbook(output_file_path)
ws = wb.active

# 将添加噪声的单元格字体设为红色
for row, col in noise_indices:
    cell = ws.cell(row=row, column=col)  # 使用记录的噪声位置
    cell.font = Font(color="FF0000")

# 保存最终文件
wb.save(output_file_path)

print("文件已保存为 '6_with_noise.xlsx'")
