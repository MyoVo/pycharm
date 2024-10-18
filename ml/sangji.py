import plotly.graph_objects as go
import pandas as pd

# 将你的特征重要性数据转换为DataFrame
data = {
    'Features': ['P<sub>1</sub>', 'P<sub>2</sub>', 'P<sub>3</sub>', 'P<sub>4</sub>', 'P<sub>mean</sub>', 'P<sub>max</sub>', 'P<sub>min</sub>', 'P<sub>std</sub>'],
    'XGBoost': [0.171324, 0.192374, 0.267117, 0.256748, 0.008610, 0.018624, 0.074489, 0.010715],
    'SVM': [0.199789, 0.246283, 0.118047, 0.051098, 0.113971, 0.037361, 0.199864, 0.033587],
    'DNDT': [0.255517, 0.248367, 0.179143, 0.087303, 0.071836, 0.024197, 0.083866, 0.049770],
    'Average': [0.208877, 0.229008, 0.188102, 0.131716, 0.064806, 0.026727, 0.119406, 0.031357]
}

importance_df = pd.DataFrame(data)
importance_df.set_index('Features', inplace=True)

# 只显示特征名称而不包含值的标签
label_without_values_base = list(importance_df.index)

fig_base = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label_without_values_base + ["XGBoost", "SVM", "DNDT"],
        color=["lightgrey"]*len(label_without_values_base) + ["blue", "chocolate", "green"]
    ),
    link=dict(
        source=list(range(len(importance_df.index)))*3,
        target=[len(importance_df.index)]*len(importance_df.index) + [len(importance_df.index)+1]*len(importance_df.index) + [len(importance_df.index)+2]*len(importance_df.index),
        value=list(importance_df['XGBoost']) + list(importance_df['SVM']) + list(importance_df['DNDT']),
        color=['rgba(31, 119, 180, 0.8)']*len(importance_df['XGBoost']) +
              ['rgba(255, 127, 14, 0.8)']*len(importance_df['SVM']) +
              ['rgba(44, 160, 44, 0.8)']*len(importance_df['DNDT'])
    )
)])

fig_base.update_layout(
    title_text="",
    font_size=15,
    font=dict(size=30),
)

# 可视化图像
fig_base.show()