import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the provided data into a DataFrame
data = {
    "Model": ["LightGBM", "AdaBoost", "Bagging", "Hard Voting", "Soft Voting", "Stacking"],

       "Kappa": [0.9020, 0.8791, 0.8864, 0.8802, 0.8933, 0.9235],

       "F1-score": [0.9139, 0.8977, 0.9012, 0.9059, 0.9142, 0.9362],

       "Accuracy": [0.9266, 0.9054, 0.9049, 0.9103, 0.9201, 0.9448],



}

df = pd.DataFrame(data)

# Transpose the DataFrame to have ACC, F1, Kappa on the x-axis
df_transposed = df.set_index('Model').T

# Plotting the line graph with different markers for each model
plt.figure(figsize=(10, 6))

markers = ['*', 's', 'D', '^', 'v', 'x']
for marker, model in zip(markers, df_transposed.columns):
    plt.plot(df_transposed.index, df_transposed[model], marker=marker, label=model)

plt.xlabel("")
plt.ylabel("Score", fontsize=20)
plt.title("")
plt.legend()
plt.grid(True)
plt.xticks(df_transposed.index, rotation=45)

# 设置坐标轴字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=19)

# Shift x-axis to the right a bit by adding padding
plt.xlim(-0.1, len(df_transposed.index) - 0.5)

plt.tight_layout()

plt.savefig('jicheng.png', format='png', bbox_inches='tight', dpi=150)  # 保存为1000x1000分辨率的PNG图像
# Display the plot
plt.show()
